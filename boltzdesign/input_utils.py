import os
import logging
import warnings
from pathlib import Path
from io import StringIO
import requests
import yaml
import pandas as pd
import pypdb
from prody import *
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from prody import parsePDB


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self, main_dir: str = None):
        self.CUDA_DEVICE = "1"
        if main_dir is None:
            self.MAIN_DIR = Path.cwd() / 'inputs'
        else:
            self.MAIN_DIR = Path(main_dir)
        self.PDB_DIR = self.MAIN_DIR / 'PDB'
        self.MSA_DIR = self.MAIN_DIR / 'MSA'
        self.YAML_DIR = self.MAIN_DIR / 'yaml'
        self.DESIGN_DIR = self.MAIN_DIR / 'designs'

    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.MAIN_DIR, self.PDB_DIR, self.MSA_DIR, self.YAML_DIR, self.DESIGN_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

# Utility functions
def download_pdb(pdb_code: str, save_path: Path) -> bool:
    """Download PDB file from RCSB.
    
    Args:
        pdb_code: PDB identifier
        save_path: Directory to save the PDB file
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        file_path = save_path / f"{pdb_code}.pdb"
        file_path.write_text(response.text)
        logger.info(f"PDB file {pdb_code}.pdb downloaded successfully!")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to download {pdb_code}.pdb: {str(e)}")
        return False

def get_chains_sequence(pdb_path: Path) -> dict:
    """Extract protein sequences from PDB file.
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        dict: Dictionary mapping chain IDs to sequences
    """
    aa_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    chain_sequences = {}
    prev_res_nums = {}
    
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    chain_id = line[21]
                    res_name = line[17:20].strip()
                    res_num = line[22:26].strip()
                    
                    if res_name not in aa_dict:
                        continue
                    
                    if chain_id not in chain_sequences:
                        chain_sequences[chain_id] = []
                        prev_res_nums[chain_id] = None
                    
                    if res_num != prev_res_nums[chain_id]:
                        chain_sequences[chain_id].append(aa_dict[res_name])
                        prev_res_nums[chain_id] = res_num
        
        return {chain: ''.join(seq) for chain, seq in chain_sequences.items()}
    except Exception as e:
        logger.error(f"Error processing PDB file {pdb_path}: {str(e)}")
        return {}


def get_pdb_components(pdb_id):
    """
    Split a protein-ligand pdb into protein and ligand components
    :param pdb_id:
    :return:
    """
    pdb = parsePDB(pdb_id)
    protein = pdb.select('protein')
    ligand = pdb.select('not protein and not water')
    return protein, ligand

_CCD_CACHE = None


def _load_ccd_pickle():
    """Lazy-load Boltz's local CCD pickle (`~/.boltz/ccd.pkl`). Maps 3-letter
    residue code -> RDKit Mol with bond orders. Returns {} on failure."""
    global _CCD_CACHE
    if _CCD_CACHE is not None:
        return _CCD_CACHE
    import pickle
    path = os.path.expanduser('~/.boltz/ccd.pkl')
    try:
        with open(path, 'rb') as f:
            _CCD_CACHE = pickle.load(f)
    except Exception as e:
        warnings.warn(f"Failed to load Boltz CCD pickle from {path}: {e}")
        _CCD_CACHE = {}
    return _CCD_CACHE


def is_ccd_code(name):
    """True if `name` is a residue code present in Boltz's CCD pickle (the same
    dictionary Boltz's parser resolves a YAML `ccd:` entry against). Used to
    decide whether a small-molecule identifier (e.g. ``HEM``) should be emitted
    as a CCD ligand -- canonical resname + atom names + exact bond orders, which
    resname/atom-name-based features (motif ligand carry-along, atom_pairs)
    require -- rather than a perceived SMILES (generic resname ``LIG`` with
    element-renamed atoms). Non-CCD identifiers (e.g. ``6TR``) return False and
    stay on the SMILES path."""
    if not isinstance(name, str) or not name:
        return False
    return name in _load_ccd_pickle()


def get_ligand_smiles(ligand, res_name):
    """
    Get SMILES string for a ligand. Offline-first to dodge RCSB network
    failures: Boltz CCD pickle -> RDKit perception on the local PDB ligand
    block -> None (caller filters None entries).
    :param ligand: ligand AtomGroup as generated by prody
    :param res_name: residue name of ligand to extract
    :return: SMILES string or None
    """
    # 1) Boltz local CCD pickle (covers most standard ligands; offline; exact
    #    bond orders).
    ccd = _load_ccd_pickle()
    if res_name in ccd:
        try:
            return Chem.MolToSmiles(ccd[res_name])
        except Exception as e:
            warnings.warn(f"CCD pickle entry for {res_name} unreadable: {e}")

    # 2) RDKit perception on the local PDB ligand block. Always works as long
    #    as the ligand atoms exist in the user's PDB; bond orders are
    #    perceived from geometry (occasionally imperfect for novel ligands,
    #    but adequate for the YAML pipeline).
    try:
        sub_mol = ligand.select(f"resname {res_name}")
        if sub_mol is None or sub_mol.numAtoms() == 0:
            return None
        buf = StringIO()
        writePDBStream(buf, sub_mol)
        mol = Chem.MolFromPDBBlock(buf.getvalue(), sanitize=True, removeHs=False)
        if mol is None:
            mol = Chem.MolFromPDBBlock(buf.getvalue(),
                                       sanitize=False, removeHs=False)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception as e:
        warnings.warn(f"RDKit perception failed for {res_name}: {e}")

    return None

def get_ligand_from_pdb(pdb_name):
    """
    Get dictionary mapping ligand names to their SMILES strings from a PDB file
    :param pdb_name: id from the pdb, doesn't need to have an extension
    :return: dict mapping ligand residue names to SMILES strings
    """
    # Common ions and small molecules to ignore
    IGNORE_LIST = {'HOH', 'H2O', 'NA', 'CA', 'MG', 'CL', 'SO4', 'PO4', 'K', 'ZN', 'CU', 'FE', 'MN',
                   'NI', 'CO', 'CD', 'GOL', 'PEG', 'EDO', 'DMS', 'ACT', 'FMT', 'MES', 'HEM', 'TRS',
                   'ACE', 'BME', 'PGE', 'MPD', 'TLA', 'EOH', 'IPA', 'PCA', 'PG4', 'DTT', 'IMD'}
    
    _, ligand = get_pdb_components(pdb_name)
    res_name_list = list(set(ligand.getResnames()) - IGNORE_LIST)
    
    # If no valid ligands found
    if not res_name_list:
        return {}
    
    # Create dictionary mapping ligand names to SMILES strings
    ligand_dict = {}
    for res in res_name_list:
        smiles = get_ligand_smiles(ligand, res)
        if smiles:
            ligand_dict[res] = smiles
            
    return ligand_dict

def get_nucleotide_from_pdb(pdb_path):
    """Extract nucleotide sequence from PDB file"""
    parser = PDBParser(QUIET=True)  # Suppress PDB warnings
    pdb_code = os.path.basename(pdb_path).split('.')[0]
    structure = parser.get_structure(pdb_code, pdb_path)
    
    sequences = {}
    for chain in structure.get_chains():
        seq = ""
        is_dna = False
        for residue in chain:
            resname = residue.get_resname()
            if resname in ['DA', 'DT', 'DC', 'DG']:  # DNA nucleotides
                is_dna = True
                seq += resname[1]  # Remove the 'D' prefix
            elif resname in ['A', 'U', 'C', 'G']:  # RNA nucleotides
                seq += resname
        if seq:
            sequences[chain.id] = {'seq': seq, 'is_dna': is_dna}
            
    return sequences

def process_modifications(modifications: str, modifications_positions: str):
    """Process modifications data"""
    if modifications and modifications_positions:
        mod_list = [mod.strip() for mod in modifications.split(',')]
        pos_list = [int(pos.strip()) for pos in modifications_positions.split(',')]
        
        if len(mod_list) != len(pos_list):
            raise ValueError("Number of modifications and positions must match.")
        
        modifications_data = []
        for mod, pos in zip(mod_list, pos_list):
            modifications_data.append({
                'position': pos,
                'ccd': mod
            })
    else:
        modifications_data = None
    return modifications_data

def setup_constraints(contact_pairs, binder_id: str):
    """Build the Boltz1 pocket-constraint block.

    contact_pairs : list of (chain, residue_int) -- already parsed and chain-
        prefixed. May span multiple target chains; they all union into a single
        pocket constraint (Boltz1 takes one inference_pocket tensor, and the
        ``contacts`` list naturally enumerates (chain, res) pairs across any
        number of target chains).
    """
    if not contact_pairs:
        return None
    return {
        'pocket': {
            'binder': binder_id,
            'contacts': [[chain, res] for chain, res in contact_pairs],
        }
    }


def process_design_constraints(target_id_map: dict, modifications: str, modifications_positions: str, modification_target: str, contact_pairs, binder_id: str):
    """Process design constraints and modifications.

    contact_pairs is the already-parsed list of (chain, res) from
    parse_contact_residues_spec; chain IDs are YAML chain IDs so no
    target_id_map lookup is needed for them. target_id_map is still consulted
    for the modification_target (which uses the input-id -> YAML-chain mapping).
    """
    if not (contact_pairs or modifications):
        return None, None

    modification_target = target_id_map.get(modification_target, '') if modifications else ''

    modifications = {
        'data': process_modifications(modifications, modifications_positions),
        'target': modification_target,
    }
    constraints = setup_constraints(contact_pairs, binder_id)

    return constraints, modifications
    
def build_chain_dict(targets: list, target_type, binder_id: str, constraints: dict = None, modifications: dict = None, modification_target: str = None) -> dict:
    # Build chain dictionary
    chain_dict = {binder_id: {'type': 'protein', 'sequence': 'X' * 100}}
    # Map target types to their YAML representation
    type_map = {
        'protein': {'type': 'protein', 'sequence': True, 'msa': 'empty'},
        'small_molecule': {'type': 'ligand', 'smiles': True},
        'metal': {'type': 'ligand', 'ccd': True},
        'dna': {'type': 'dna', 'sequence': True},
        'rna': {'type': 'rna', 'sequence': True}
    }
    # target_type may be a single string (broadcast to every target) or a
    # per-target list (multi-target mode), one entry per target, in order.
    if isinstance(target_type, str):
        target_types = [target_type] * len(targets)
    else:
        target_types = list(target_type)
        if len(target_types) != len(targets):
            raise ValueError(
                f"target_type list has {len(target_types)} entries but there are "
                f"{len(targets)} targets; they must match one-to-one and in order.")
    yaml_target_ids = []

    for i, target in enumerate(targets):
        # Get letters in order, removing binder_id
        available_letters = ''.join(c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c != binder_id)
        target_id = available_letters[i]

        yaml_target_ids.append(target_id)
        target_info = {'id': target_id}

        # Explicit ligand-key override: a target given as {'ccd': code} or
        # {'smiles': str} states its YAML key directly. This is how a
        # small_molecule whose --target_mols identifier is a valid CCD code
        # (e.g. HEM) is emitted as a CCD ligand -- canonical resname + atom
        # names + exact bond orders, required by resname/atom-name features
        # (motif ligand carry-along --motif_ligand_residues, atom_pairs) --
        # instead of a perceived SMILES (generic resname 'LIG', element-renamed
        # atoms). The caller decides (it knows the identifier is a code, not a
        # SMILES that merely collides with one, e.g. 'CO' methanol vs CCD CO).
        if isinstance(target, dict) and ('ccd' in target or 'smiles' in target):
            target_info['type'] = 'ligand'
            key = 'ccd' if 'ccd' in target else 'smiles'
            target_info[key] = target[key]
        else:
            # Add appropriate fields based on target type.
            type_info = type_map[target_types[i]]
            for field, value in type_info.items():
                if value is True:
                    target_info[field] = target
                elif value:
                    target_info[field] = value

        chain_dict[target_id] = target_info
        
    return chain_dict, yaml_target_ids

def generate_yaml_for_target_binder(name:str, target_type, targets: list, config="", binder_id='A', constraints: dict = None, modifications: dict = None, modification_target: str = None, use_msa: bool = False) -> dict:
    """
    Generate YAML content for a small molecule binder with multiple targets and create the YAML file.
    
    Args:
        name (str): Name/PDB code for the target
        target_type (str | list): Target type, one of
            ('small_molecule', 'dna', 'rna', 'metal', 'protein'). A single string
            is broadcast to every target; a list assigns one type per target
            (multi-target mode), in the same order as ``targets``.
        targets (list): List of target information (SMILES, sequences, or CCD codes)
        binder_id (str): ID of the binder
        config (Config): Configuration object
        constraints (dict): Optional constraints to add to YAML
        modifications (dict): Optional modifications to add to YAML
        modification_target (str): Optional modification target to add to YAML
        use_msa (bool): Whether to use MSA for proteins
        
    Returns:
        tuple: YAML content dictionary and output path
    """ 

    chain_dict, yaml_target_ids = build_chain_dict(targets, target_type, binder_id, constraints, modifications, modification_target)
    # Build sequences list for YAML
    sequences = []
    for chain_id, info in chain_dict.items():
        if not isinstance(info, dict) or 'type' not in info:
            continue
            
        entry = {}
        if info['type'] == 'ligand':
            key = 'smiles' if 'smiles' in info else 'ccd'
            entry = {
                "ligand": {
                    "id": [chain_id],
                    key: info[key]
                }
            }
        elif info['type'] in ['dna', 'rna']:
            entry = {
                info['type']: {
                    "id": [chain_id],
                    "sequence": info['sequence']
                }
            }
        else:  # protein
            msa_path = (config.MSA_DIR / f"{name}_{chain_id}_env/msa.npz" 
                       if use_msa and not all(x == 'X' for x in info['sequence']) 
                       else "empty")

            if msa_path != "empty":
                process_msa(chain_id, info['sequence'], name, config)
                print(f"Processed MSA for {name} chain {chain_id}")
            
            entry = {
                "protein": {
                    "id": [chain_id],
                    "sequence": info['sequence'],
                    "msa": str(msa_path)
                }
            }
            
            if modifications and chain_id in yaml_target_ids and chain_id == modification_target:
                entry["protein"]["modifications"] = modifications
                
        sequences.append(entry)
    
    # Create and write YAML content
    yaml_content = {"version": 1, "sequences": sequences}
    if constraints:
        yaml_content["constraints"] = [constraints]

    output_path = config.YAML_DIR / f"{name}.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Created YAML file for {name}")
    
    return yaml_content, output_path

    
def process_msa(chain_id: str, sequence: str, pdb_code: str, config: Config) -> bool:
    """Process MSA for a single chain."""
    msa_chain_dir = config.MSA_DIR / f"{pdb_code}_{chain_id}"
    env_dir = msa_chain_dir.with_name(f"{msa_chain_dir.name}_env")
    env_dir.mkdir(exist_ok=True)
    
    # Run MSA
    unpaired_msa = run_mmseqs2(
        [sequence],
        str(msa_chain_dir),
        use_env=True,
        use_pairing=False,
        host_url="https://api.colabfold.com",
        pairing_strategy="greedy"
    )
    
    # Save MSA results
    msa_a3m_path = env_dir / "msa.a3m"
    msa_a3m_path.write_text(unpaired_msa[0])
    
    # Process MSA if not already processed
    msa_npz_path = env_dir / "msa.npz"
    if not msa_npz_path.exists():
        msa = parse_a3m(
            msa_a3m_path,
            taxonomy=None,
            max_seqs=4096,
        )
        msa.dump(msa_npz_path)
    
    logger.info(f"Processed MSA for {pdb_code} chain {chain_id}")
    return True