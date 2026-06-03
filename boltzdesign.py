import os
import sys
import argparse
import yaml
import json
import shutil
import pickle
import glob
import numpy as np
import random
import logging
import subprocess
import pandas as pd
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
sys.path.append(f'{os.getcwd()}/boltzdesign')

from boltzdesign_utils import *
from ligandmpnn_utils import *
from alphafold_utils import *
from input_utils import *
from utils import *
import torch


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_gpu_environment(gpu_id):
    """Setup GPU environment variables"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="BoltzDesign: Protein Design Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Design binder for DNA target
  python boltzdesign_generalized.py --target_name 5zmc --target_types dna --pdb_target_ids C,D --target_mols SAM --binder_id A
        """
    )

    # Required arguments
    parser.add_argument('--target_name', type=str, required=True,
                        help='Target name/PDB code (e.g., 5zmc)')
    # Target configuration
    parser.add_argument('--target_types', type=str, nargs='+', default=['protein'],
                        help='Per-target types, one entry per target in chain order '
                             '(e.g. --target_types protein small_molecule). Length 1 is '
                             'single-target mode; the chosen type seeds the base default '
                             'config (default_{ppi,sm,na,pep,metal}_config.yaml) and labels '
                             'the output folder. Length >1 is multi-target: protein/dna/rna '
                             'targets draw identifiers from --pdb_target_ids (in order), '
                             'small_molecule/metal targets from --target_mols (in order); '
                             'in --input_type custom mode all identifiers come from '
                             '--custom_target_inputs (in order). The old quoted / comma '
                             'forms ("protein small_molecule" or "protein,small_molecule") '
                             'still work. Default "protein".')
    parser.add_argument('--input_type', type=str, choices=['pdb', 'custom'], default='pdb',
                        help='Input type: pdb code or custom input')
    parser.add_argument('--pdb_path', type=str, default='',
                        help='Path to a local PDB file (if specify use custom pdb, else fetch from RCSB)')
    parser.add_argument('--pdb_target_ids', type=str, nargs='+', default=[],
                        help='Target chain IDs in the PDB, e.g. --pdb_target_ids C D '
                             '(old "C,D" form still works)')
    parser.add_argument('--target_mols', type=str, nargs='+', default=[],
                        help='Target ligand identifiers (CCD codes or non-CCD tokens), '
                             'e.g. --target_mols HEM ZN (old "HEM,ZN" form still works)')
    parser.add_argument('--custom_target_inputs', type=str, nargs='+', default=[],
                        help='Custom target sequences / SMILES / CCD codes, one per target '
                             'in --target_types order. Each value is its own token, so '
                             'SMILES with shell-special characters quote naturally: '
                             '--custom_target_inputs "ATAT" "GCGC" or '
                             '--custom_target_inputs "<protein_seq>" "ZN" or '
                             '--custom_target_inputs "[O-]C(=O)C(N)CC[S+](C)CC3OC(n2cnc1c(ncnc12)N)C(O)C3O". '
                             'Old comma form still works.')
    parser.add_argument('--custom_target_ids', type=str, nargs='+', default=[],
                        help='Custom target chain IDs, e.g. --custom_target_ids A B '
                             '(old "A,B" form still works)')
    parser.add_argument('--binder_id', type=str, default='A',
                        help='Binder chain ID')
    parser.add_argument('--use_msa', type=str2bool, default=False,
                        help='Use MSA (if False, runs in single-sequence mode)')
    parser.add_argument('--msa_max_seqs', type=int, default=4096,
                        help='Maximum MSA sequences')
    parser.add_argument('--suffix', type=str, default='0',
                        help='Suffix for the output directory')
    
    # Modifications
    parser.add_argument('--modifications', type=str, nargs='+', default=[],
                        help='Modifications (per-residue CCD codes), e.g. '
                             '--modifications SEP SEP (old "SEP,SEP" form still works)')
    parser.add_argument('--modifications_wt', type=str, nargs='+', default=[],
                        help='Wild-type AAs matching --modifications, e.g. '
                             '--modifications_wt S S (old "S,S" form still works)')
    parser.add_argument('--modifications_positions', type=str, nargs='+', default=[],
                        help='Modification positions, e.g. --modifications_positions 10 20 '
                             '(old "10,20" form still works)')
    parser.add_argument('--modification_target', type=str, default='',
                        help='Target ID for modifications (e.g., "A")')
    
    # Constraints
    parser.add_argument('--contact_residues', type=str, nargs='+', default=[],
                        help='Pocket-conditioning contact residues, one token per '
                             'contact: CHAIN:RESNUM (YAML chain ID, 1-indexed). '
                             'Multi-target pockets supported by mixing chains: '
                             '--contact_residues B:99 B:100 C:200 says "binder should '
                             'contact residues 99,100 of chain B AND residue 200 of '
                             'chain C." The model receives a single union pocket '
                             'constraint. Old "99,100,109" form (bare numbers + '
                             '--constraint_target) is no longer accepted -- prefix '
                             'each token with its chain.')

    # Design parameters
    parser.add_argument('--length_min', type=int, default=100,
                        help='Minimum binder length')
    parser.add_argument('--length_max', type=int, default=150,
                        help='Maximum binder length')
    parser.add_argument('--optimizer_type', type=str, choices=['SGD', 'AdamW'], default='SGD',
                        help='Optimizer type')
    
    # Iteration parameters
    parser.add_argument('--pre_iteration', type=int, default=30,
                        help='Pre-iteration steps')
    parser.add_argument('--soft_iteration', type=int, default=75,
                        help='Soft iteration steps')
    parser.add_argument('--temp_iteration', type=int, default=50,
                        help='Temperature iteration steps')
    parser.add_argument('--hard_iteration', type=int, default=5,
                        help='Hard iteration steps')
    parser.add_argument('--semi_greedy_steps', type=int, default=2,
                        help='Semi-greedy steps')
    parser.add_argument('--recycling_steps', type=int, default=0,
                        help='Recycling steps')
    parser.add_argument('--sampling_steps', type=int, default=200,
                        help='Number of EDM diffusion sampling steps (total step count) in the '
                             'structure module. Lower = faster/coarser. Only affects the full '
                             'pipeline (distogram_only=False, trajectory snapshots, and the final '
                             'structure prediction). Default 200.')
    
    # Advanced configuration
    parser.add_argument('--use_default_config', type=str2bool, default=True,
                        help='Use default configuration (recommended)')
    parser.add_argument('--mask_target_prerun', type=str2bool, nargs='+', default=[],
                        help='Per-target booleans: mask this target during the warm-up '
                             '(pre_iteration) stage so the binder evolves topology before '
                             'seeing it. One value per --target_types entry; length 1 '
                             'broadcasts. Unset => per-type YAML default (sm/metal: True; '
                             'protein/dna/rna/peptide: False). Renamed from --mask_ligand.')
    parser.add_argument('--optimize_contact_per_binder_pos', type=str2bool, nargs='+', default=[],
                        help='Per-target booleans: when True, every binder position must '
                             'reach --num_inter_contacts to that target; when False the '
                             'aggregate count is used. Length 1 broadcasts. Unset => per-'
                             'type YAML default (ppi/na: True; sm/metal/pep: False).')
    parser.add_argument('--distogram_only', type=str2bool, default=True,
                        help='Only use distogram for optimization')
    parser.add_argument('--use_heun', type=str2bool, default=False,
                        help="Use Heun's second-order corrector in the diffusion sampler "
                             "(applies only when distogram_only=False or during trajectory snapshots; "
                             "doubles per-step NFE)")
    parser.add_argument('--step_scale', type=float, default=1.638,
                        help="Diffusion sampler step length / over-relaxation factor (eta): each Euler "
                             "(and Heun) update is scaled by step_scale * (sigma_t - t_hat). "
                             "Higher = larger denoising steps. Set to 1.0 for the velocity-consistent "
                             "ODE step that, with --gamma_0 0, enables stable few-step inference. "
                             "Baked into the model at load time. Default 1.638.")
    parser.add_argument('--gamma_0', type=float, default=0.605,
                        help="EDM churn level (gamma_0): the default schedule injects stochastic noise at "
                             "high sigma. Set to 0.0 to switch to a noise-free ODE sampler. Combined with "
                             "--step_scale 1.0 and a small --sampling_steps, this reproduces the "
                             "manuscript's few-step (even 1-step) inference recipe. Only affects the full "
                             "pipeline (distogram_only=False, trajectory snapshots, final prediction). "
                             "Baked into the model at load time. Default 0.605.")
    parser.add_argument('--deterministic_sampler', type=str2bool, default=False,
                        help="Make the reverse diffusion sampler deterministic: disable per-step "
                             "random augmentation (centering kept), skip the Kabsch/SVD alignment, "
                             "and zero the EDM churn noise. Trades single-sample structural fidelity "
                             "for a smooth, low-variance, SVD-free sequence->coords map; intended "
                             "for stable rg/confidence-loss backprop with --attach_coords True. "
                             "Baked into the model at load time. Default False.")
    parser.add_argument('--attach_coords', type=str2bool, default=False,
                        help="Keep sample_atom_coords attached to the autograd graph so "
                             "confidence-head losses (plddt/pae/i_pae) backprop through the "
                             "diffusion sampler. Only meaningful when distogram_only=False; "
                             "increases memory/compute substantially.")
    parser.add_argument('--design_algorithm', type=str, choices=['3stages', '3stages_extra'], 
                        default='3stages', help='Design algorithm')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for optimization')
    parser.add_argument('--learning_rate_pre', type=float, default=0.1, 
                        help='Learning rate for pre iterations (warm-up stage)')
    parser.add_argument('--e_soft', type=float, default=0.8,
                        help='Softmax temperature for 3stages')
    parser.add_argument('--e_soft_1', type=float, default=0.8,
                        help='Initial softmax temperature for 3stages_extra')
    parser.add_argument('--e_soft_2', type=float, default=1.0,
                        help='Additional softmax temperature for 3stages_extra')
    
    # Interaction parameters
    parser.add_argument('--inter_chain_cutoff', type=float, nargs='+', default=[],
                        help='Per-target inter-chain distance cutoff in Angstrom. One '
                             'value per --target_types entry; length 1 broadcasts. Unset '
                             '=> per-type YAML default (20 A for all bundled presets).')
    parser.add_argument('--intra_chain_cutoff', type=int, default=14,
                        help='Intra-chain (binder-internal) distance cutoff. Singular.')
    parser.add_argument('--num_inter_contacts', type=int, nargs='+', default=[],
                        help='Per-target minimum inter-chain contacts. One value per '
                             '--target_types entry; length 1 broadcasts. Unset => per-type '
                             'YAML default (ppi/na/pep: 2; sm: 1; metal: 4).')
    parser.add_argument('--num_intra_contacts', type=int, default=2,
                        help='Number of intra-chain (binder-internal) contacts. Singular.')
    

    # loss parameters
    parser.add_argument('--con_loss', type=float, default=1.0,
                        help='Contact loss weight')
    parser.add_argument('--i_con_loss', type=float, nargs='+', default=[],
                        help='Per-target inter-chain contact loss weight. One value per '
                             '--target_types entry; length 1 broadcasts. Unset => 1.0 for '
                             'every target (reproduces the prior global default). Setting '
                             '0 for a target disables its contact term.')
    parser.add_argument('--plddt_loss', type=float, default=0.1,
                        help='Binder-pLDDT loss weight. Singular (binder-only).')
    parser.add_argument('--pae_loss', type=float, default=0.4,
                        help='Binder-internal PAE loss weight. Singular (binder-only).')
    parser.add_argument('--i_pae_loss', type=float, nargs='+', default=[],
                        help='Per-target inter-chain PAE loss weight. One value per '
                             '--target_types entry; length 1 broadcasts. Unset => 0.1 for '
                             'every target (reproduces the prior global default).')
    parser.add_argument('--rg_loss', type=float, default=0.3,
                        help='Radius of gyration loss weight (default 0.3, matching '
                             'BindCraft weights_rg / use_rg_loss=true). Only contributes '
                             'a gradient when --distogram_only False and --attach_coords True; '
                             'otherwise it is reported but inert (sampler coords are detached).')
    parser.add_argument('--com_loss', type=float, default=0.0,
                        help='Binder centroid-of-mass loss weight. Pulls the binder CA '
                             'COM toward the average of HETATM ORI atoms parsed from '
                             '--pdb_path and/or --motif_pdb. Each ORI is rigid-body '
                             'transformed from its source-PDB frame into the live '
                             'co-fold frame via Kabsch alignment (target chains for '
                             '--pdb_path; motif backbone for --motif_pdb). Default 0 = '
                             'off. Only computed in full mode (--distogram_only False); '
                             'carries a gradient only with --attach_coords True (mirrors '
                             '--rg_loss). Insert ORI HETATMs into your input PDB as '
                             '"HETATM .. ORI  ORI z   1   x  y  z" lines.')
    parser.add_argument('--target_plddt_loss', type=float, nargs='+', default=[],
                        help='Per-target pLDDT loss weight (boosts confidence in each '
                             'target\'s own placement, e.g. a heme/Zn cofactor). One value '
                             'per --target_types entry; length 1 broadcasts. Unset => 0.0 '
                             'for every target (off by default; replaces the removed '
                             '--target_plddt_chains gate -- set the per-target weight to '
                             '0 to exclude a target). Only computed in full mode '
                             '(--distogram_only False).')
    parser.add_argument('--helix_loss_max', type=float, default=0.0,
                        help='Maximum helix loss weights')
    parser.add_argument('--helix_loss_min', type=float, default=-0.3,
                        help='Minimum helix loss weights')
    parser.add_argument('--motif_distogram_loss', type=float, default=1.0,
                        help='Motif scaffolding distogram-CCE loss weight '
                             '(ColabDesign `partial` dgram_cce). Only used when '
                             '--motif_pdb is set. Formerly --motif_loss.')
    parser.add_argument('--motif_coords_loss', type=float, default=1.0,
                        help='Motif scaffolding coord-RMSD loss weight. Kabsch-'
                             'aligned RMSD on the sampled atom coords; the rigid '
                             'transform is fit from the protein motif backbone '
                             '(N,CA,C,CB) ONLY, then applied to any ligand atoms '
                             'specified by --motif_ligand_residues (decoupled '
                             'alignment), and the combined RMSD is the optimized '
                             'scalar. Active only when --motif_pdb is set AND '
                             '--distogram_only False (needs sample_atom_coords). '
                             'Carries a gradient only with --attach_coords True; '
                             'otherwise reported but inert, mirroring --rg_loss.')
    parser.add_argument('--motif_fape_loss', type=float, default=1.0,
                        help='Motif scaffolding sidechain FAPE loss weight '
                             '(AlphaFold-style frame-aligned point error of motif '
                             'sidechain atoms in the motif backbone frames). Active '
                             'only when --motif_pdb is set AND --distogram_only '
                             'False AND shared sidechain heavy atoms exist (the '
                             'binder is built as UNK, so by default only CB/CG). '
                             'Gradient only with --attach_coords True.')

    # Motif scaffolding (ColabDesign `partial` protocol). Design a binder that
    # also retains a given structural motif (e.g. a catalytic/cofactor-binding
    # site) so the binder is itself an enzyme / small-molecule binder. Inactive
    # unless --motif_pdb is provided.
    parser.add_argument('--motif_pdb', type=str, default='',
                        help='Reference PDB/CIF containing the motif to scaffold')
    parser.add_argument('--motif_residues', type=str, nargs='+', default=[],
                        help='Chain-prefixed motif residue selection, one token per '
                             'residue/range, e.g. --motif_residues A57 A102 A195 (or '
                             'A10-14 B57 C195 for multi-chain). Each token is '
                             'CHAIN+RESNUM (author/PDB) in the motif PDB, ranges '
                             'allowed. Bare residue numbers are rejected -- prefix '
                             'the chain. One-to-one with --motif_binder_positions in '
                             'declaration order. Old "A57,A102,A195" form still works.')
    parser.add_argument('--motif_binder_positions', type=str, nargs='+', default=[],
                        help='1-indexed binder positions the motif maps to (same count/'
                             'order as --motif_residues), e.g. --motif_binder_positions '
                             '30 75 110. Default: N-terminal contiguous block. Old '
                             '"30,75,110" form still works.')
    parser.add_argument('--fix_motif_seq', type=str2bool, default=True,
                        help='Pin the motif residues to the reference sequence '
                             '(retained + grad-frozen). False = scaffold '
                             'geometry only, sequence stays designable.')
    parser.add_argument('--motif_ligand_residues', type=str, nargs='+', default=[],
                        help='Optional ligand residues in --motif_pdb to carry '
                             'along with the motif under --motif_coords_loss, one '
                             'token per residue: --motif_ligand_residues B1 (or '
                             'B1 C401). Each is CHAIN+RESNUM (author/PDB) in the '
                             'motif PDB. The Kabsch transform is fit on the protein '
                             'motif backbone (unchanged by this flag), then applied '
                             'to these ligand heavy atoms so their RMSD reports their '
                             'placement RELATIVE to the motif framework -- the '
                             'natural objective for hemoprotein scaffolding. '
                             'Predicted ligand is matched in the designed system by '
                             'resname (chain IDs may differ between motif PDB and '
                             'the YAML-built design). Independent of '
                             '--motif_binder_positions: ligand residues do NOT '
                             'consume binder slots; they live in their own target '
                             'chain (added via --target_mols). Old "B1,C401" form '
                             'still works.')

    # Explicit atom/residue distance restraints. Independent of the general
    # binder-target contact loss, and usable between ANY two chains -- including
    # target<->target (e.g. holding a cofactor near a specific target residue).
    # Inactive unless --atom_pairs is set.
    parser.add_argument('--atom_pairs', type=str, nargs='+', default=[],
                        help='Distance restraints, one pair per token. Two forms: '
                             '4-field WINDOW "epA, epB, lo, hi" (flat-bottom in [lo,hi]) '
                             'or 3-field POINT "epA, epB, d_ref" (treated as lo=hi=d_ref; '
                             'with --atom_pair_distogram_loss_type expected this becomes '
                             'the bin-center MSE sum_k p_k * (mid_pts_k - d_ref)^2 on the '
                             'predicted distogram). Each endpoint is CHAIN:SEL[@ATOM]: '
                             'SEL is a 1-indexed residue position for polymer chains or an '
                             'atom name for ligand chains; @ATOM pins a named atom (coord '
                             'loss only). All distances in Angstrom. Examples: '
                             '--atom_pairs "C:FE1, B:145, 0, 6" "C:FE1, B:50@NE2, 1.8, 2.6" '
                             '(windows) or --atom_pairs "C:FE1, B:94, 8" (point). Old '
                             '";"-separated string still works.')
    parser.add_argument('--atom_pair_distogram_loss', type=float, default=1.0,
                        help='Weight for the token-level distogram window loss '
                             'over --atom_pairs (fast-mode safe; ligand atoms '
                             'exact, polymer residues at Cβ). Inert unless '
                             '--atom_pairs is set.')
    parser.add_argument('--atom_pair_coords_loss', type=float, default=1.0,
                        help='Weight for the atom-level coord distance restraint '
                             'over --atom_pairs (flat-bottom). Active only with '
                             '--distogram_only False; carries a gradient only '
                             'with --attach_coords True (mirrors '
                             '--motif_coords_loss). Inert unless --atom_pairs set.')
    parser.add_argument('--atom_pair_distogram_loss_type', type=str, default='expected',
                        choices=['expected', 'prob', 'contact'],
                        help="How the --atom_pairs distogram restraint is computed: "
                             "'expected' (default) = bin-center MSE on the predicted "
                             "distribution -- for a POINT TARGET (3-field spec, "
                             "lo=hi=d_ref) this is sum_k p_k * (mid_pts_k - d_ref)^2 "
                             "(= Var + bias^2, so drives BOTH the expected distance "
                             "toward d_ref AND the prediction toward a sharp delta "
                             "there); for a WINDOW (4-field spec, lo < hi) this is "
                             "the flat-bottom relu(lo-E[d])^2+relu(E[d]-hi)^2 on the "
                             "expected distance. 'prob' = -log P(d in [lo,hi]) over "
                             "the in-window probability mass (its gradient collapses "
                             "by ~20 A). 'contact' = BoltzDesign1's own categorical "
                             "contact loss (cutoff=hi, lo ignored): a robust attractive "
                             "gradient that is large when far and tapers near -- "
                             "use this when pulling a far pair into contact. All "
                             "three are capped by the distogram's ~24.5 A ceiling; "
                             "for a truly unbounded long-range pull use the coord "
                             "loss in full mode (--distogram_only False "
                             "--attach_coords True). Inert unless --atom_pairs set.")


    # LigandMPNN parameters
    parser.add_argument('--num_designs', type=int, default=2,
                        help='Number of designs per PDB for LigandMPNN')
    parser.add_argument('--cutoff', type=int, default=4,
                        help='Cutoff distance for interface residues (Angstroms)')
    parser.add_argument('--i_ptm_cutoff', type=float, default=0.5,
                        help='iPTM cutoff for redesign')
    parser.add_argument('--complex_plddt_cutoff', type=float, default=0.7,
                        help='Complex pLDDT cutoff for high confidence designs')
    
    # System configuration
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--design_samples', type=int, default=1,
                        help='Number of design samples')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='Working directory (default: current directory)')
    parser.add_argument('--high_iptm', type=str2bool, default=True,
                        help='Disable high iPTM designs')
    # Paths
    parser.add_argument('--boltz_checkpoint', type=str,
        default='/opt/boltz1_weights/boltz1_conf.ckpt',
        help='Path to Boltz checkpoint')
    parser.add_argument('--ccd_path', type=str,
        default='/opt/boltz1_weights/ccd.pkl',
        help='Path to CCD file')
    parser.add_argument('--alphafold_dir', type=str,
        default='/opt/alphafold3',
        help='AlphaFold directory')
    parser.add_argument('--af3_docker_name', type=str,
        default='alphafold3',
        help='Docker name')
    parser.add_argument('--af3_database_settings', type=str,
        default='~/alphafold3/alphafold3_data_save',
        help='AlphaFold3 database settings')
    parser.add_argument('--af3_hmmer_path', type=str,
        default='/home/jupyter-yehlin/.conda/envs/alphafold3_venv',
        help='AlphaFold3 hmmer path, required for RNA MSA generation')
    # Control flags
    parser.add_argument('--run_boltz_design', type=str2bool, default=True,
                        help='Run Boltz design step')
    parser.add_argument('--run_ligandmpnn', type=str2bool, default=True,
                        help='Run LigandMPNN redesign step')
    parser.add_argument('--run_alphafold', type=str2bool, default=True,
                        help='Run AlphaFold validation step')
    parser.add_argument('--run_rosetta', type=str2bool, default=True,
                        help='Run Rosetta energy calculation (protein targets only)')
    parser.add_argument('--redo_boltz_predict', type=str2bool, default=False,
                        help='Redo Boltz prediction')


    ## Visualization
    parser.add_argument('--show_animation', type=str2bool, default=True,
                        help='Show animation')
    parser.add_argument('--save_trajectory', type=str2bool, default=False,
                        help='Save trajectory')
    parser.add_argument('--save_intermediate_structures', type=str2bool, default=False,
                        help='Dump the per-epoch folded structure as a PDB when '
                             'the diffusion sampler runs (i.e. --distogram_only False '
                             'or --save_trajectory True). Writes to '
                             '<output>/intermediate_structures/{stage}_epoch{NNNN}.pdb.')
    return parser.parse_args()


class YamlConfig:
    """Configuration class for managing directories"""
    def __init__(self, main_dir: str = None):
        if main_dir is None:
            self.MAIN_DIR = Path.cwd() / 'inputs'
        else:
            self.MAIN_DIR = Path(main_dir)
        self.PDB_DIR = self.MAIN_DIR / 'PDB'
        self.MSA_DIR = self.MAIN_DIR / 'MSA'
        self.YAML_DIR = self.MAIN_DIR / 'yaml'
    
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.MAIN_DIR, self.PDB_DIR, self.MSA_DIR, self.YAML_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


def load_boltz_model(args, device):
    """Load Boltz model"""
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    
    boltz_model = get_boltz_model(args.boltz_checkpoint, predict_args, device, use_heun=args.use_heun, attach_coords=args.attach_coords, step_scale=args.step_scale, deterministic_sampler=args.deterministic_sampler, gamma_0=args.gamma_0)
    boltz_model.train()
    return boltz_model, predict_args

def _split_csv(value):
    """Normalize a list-valued arg to a list of stripped, non-empty tokens.

    Accepts three forms so every caller works regardless of how the user
    passed the flag:
      - empty / None -> []
      - str ("a,b" or "a b") -> split on whitespace and commas
      - list (from nargs="+") -> flatten each element on whitespace + commas
    Splitting inside list elements preserves back-compat with the old quoted
    "a,b" / "a b" forms when those land as a single nargs token.
    """
    if not value:
        return []
    items = [value] if isinstance(value, str) else list(value)
    out = []
    for item in items:
        for tok in str(item).replace(",", " ").split():
            out.append(tok)
    return out


def parse_contact_residues_spec(tokens):
    """Parse chain-prefixed --contact_residues tokens into [(chain, res), ...].

    Each token is CHAIN:RESNUM where CHAIN is a single uppercase letter (YAML
    chain ID) and RESNUM is a 1-indexed integer. Accepts the legacy quoted /
    comma forms via _split_csv (so "B:99 B:100 C:200", "B:99,B:100,C:200" and
    --contact_residues B:99 B:100 C:200 all parse identically). Bare numbers
    are rejected with a clear migration error pointing to the new format.
    """
    out = []
    for tok in _split_csv(tokens):
        if ":" not in tok:
            raise ValueError(
                f"--contact_residues token '{tok}' is missing a chain prefix. "
                f"Use CHAIN:RESNUM (e.g. B:99). The bare-number form + "
                f"--constraint_target is no longer supported -- prefix the "
                f"chain on every token.")
        chain, res = tok.split(":", 1)
        chain, res = chain.strip(), res.strip()
        if len(chain) != 1 or not chain.isupper():
            raise ValueError(
                f"--contact_residues token '{tok}': chain prefix must be one "
                f"uppercase letter (got '{chain}')")
        try:
            res_int = int(res)
        except ValueError:
            raise ValueError(
                f"--contact_residues token '{tok}': residue '{res}' is not an "
                f"integer")
        out.append((chain, res_int))
    return out


def _split_semi(value):
    """Normalize an --atom_pairs-style arg to a list of full-pair strings.

    Each returned string is one complete pair (commas preserved); downstream
    parsers (parse_atom_pairs_spec) handle the comma-tuples inside. Always
    splits on ';' inside every element, so:
      - str ("p1; p2") -> ['p1', 'p2']
      - native nargs ["p1", "p2"] -> ['p1', 'p2']
      - quoted-legacy nargs ["p1; p2"] -> ['p1', 'p2']
    """
    if not value:
        return []
    items = [value] if isinstance(value, str) else list(value)
    out = []
    for item in items:
        for s in str(item).split(";"):
            s = s.strip()
            if s:
                out.append(s)
    return out


def get_all_target_types(args):
    """Return the per-target type list for this run, parsed from --target_types.

    Length 1 = single-target mode (reproduces the old ``--target_type`` path);
    length >1 = multi-target mode. All targets are contacted symmetrically by
    the design loop -- there is no "primary" target; this list only describes
    how each target's YAML entry is built and whether any target is a protein
    (for the MSA path).
    """
    types = _split_csv(args.target_types)
    if not types:
        raise ValueError("--target_types is empty; provide at least one target type")
    valid = {'protein', 'rna', 'dna', 'small_molecule', 'metal'}
    bad = [t for t in types if t not in valid]
    if bad:
        raise ValueError(f"--target_types contains unknown type(s): {bad}; "
                         f"valid: {sorted(valid)}")
    return types


def get_target_chain_ids(args):
    """Per-target chain IDs in --target_types order, skipping --binder_id.

    Mirrors the chain-letter assignment in build_chain_dict / generate_yaml_config:
    targets get 'A','B','C',... minus the binder letter, in declaration order. So
    --binder_id A + --target_types protein small_molecule -> ['B','C'].
    """
    types_list = get_all_target_types(args)
    letters = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c != args.binder_id]
    return letters[:len(types_list)]


# Per-target settings sourced from the per-type default YAML. Each entry is
# (cli-flag attribute name on args, key inside the YAML file). The user's CLI
# value -- if provided -- wins; otherwise the YAML for each target's type is
# consulted (so `--target_types protein small_molecule` picks ppi's value for
# target 0 and sm's value for target 1).
_PER_TARGET_YAML_SETTINGS = (
    ('mask_target_prerun', 'mask_target_prerun'),
    ('num_inter_contacts', 'num_inter_contacts'),
    ('optimize_contact_per_binder_pos', 'optimize_contact_per_binder_pos'),
    ('inter_chain_cutoff', 'inter_chain_cutoff'),
)

# Per-target loss-weight settings: (CLI flag name on args, kwarg name expected
# by boltz_hallucination, default value). Each default matches the prior global
# behavior: i_con_loss 1.0 / i_pae_loss 0.1 reproduce the old loss_scales;
# target_plddt_loss 0.0 keeps the term off unless the user opts in (replaces
# the removed --target_plddt_chains gate -- setting a per-target weight to 0 is
# the new "off for this target" knob).
_PER_TARGET_LOSS_WEIGHTS = (
    ('i_con_loss', 'i_con_loss_weights', 1.0),
    ('i_pae_loss', 'i_pae_loss_weights', 0.1),
    ('target_plddt_loss', 'target_plddt_loss_weights', 0.0),
)


def _broadcast_to_targets(values, n, flag_name):
    """Validate a per-target CLI list: length 1 broadcasts to n; length n is
    used as-is; anything else is an error. Empty list -> caller will substitute
    the auto-derived defaults."""
    if not values:
        return None
    if len(values) == 1:
        return list(values) * n
    if len(values) != n:
        raise ValueError(
            f"--{flag_name} has {len(values)} entries but --target_types lists "
            f"{n} target(s); provide either 1 (broadcast) or {n} values.")
    return list(values)


def derive_per_target_settings(args, work_dir):
    """Build the per-target value lists for every per-target setting.

    For each setting, the user's CLI list (with length-1 broadcast) wins; if
    they passed nothing, the per-type YAML is consulted per target. Returns a
    dict keyed by the same attribute names on args.
    """
    types_list = get_all_target_types(args)
    n = len(types_list)
    settings = {}

    # YAML-sourced: cache load_design_config per unique type so we read each
    # file at most once.
    _yaml_cache = {}
    def _yaml_for(t):
        if t not in _yaml_cache:
            _yaml_cache[t] = load_design_config(t, work_dir)
        return _yaml_cache[t]

    for cli_name, yaml_key in _PER_TARGET_YAML_SETTINGS:
        user_val = getattr(args, cli_name)
        bcast = _broadcast_to_targets(user_val, n, cli_name)
        if bcast is not None:
            settings[cli_name] = bcast
        else:
            settings[cli_name] = [_yaml_for(t)[yaml_key] for t in types_list]

    # Loss weights (constant defaults; CLI name differs from kwarg name).
    for cli_name, kwarg_name, default in _PER_TARGET_LOSS_WEIGHTS:
        user_val = getattr(args, cli_name)
        bcast = _broadcast_to_targets(user_val, n, cli_name)
        settings[kwarg_name] = bcast if bcast is not None else [default] * n

    return settings


def load_design_config(target_type, work_dir):
    """
    Load design configuration based on target type.
    Modified so that config files are always loaded from the script's directory,
    instead of using work_dir/boltzdesign/configs.
    """
    # Determine the directory where this script (boltzdesign.py) lives:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The configs directory is under script_dir/boltzdesign/configs/
    config_dir = os.path.join(script_dir, 'boltzdesign', 'configs')
    
    if target_type=='small_molecule':
        config_path = os.path.join(config_dir, "default_sm_config.yaml")
    elif target_type=='metal':
        config_path = os.path.join(config_dir, "default_metal_config.yaml")
    elif target_type=='dna' or target_type=='rna':
        config_path = os.path.join(config_dir, "default_na_config.yaml")

    elif target_type=='protein':
        config_path = os.path.join(config_dir, "default_ppi_config.yaml")
    else:
        raise ValueError(f"Unknown target type: {target_type}")
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_explicit_args():
    # Get all command-line arguments (excluding the script name)
    explicit_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            # Handle --arg=value and --arg value
            if '=' in arg:
                explicit_args.add(arg.split('=')[0].lstrip('-').replace('-', '_'))
            else:
                explicit_args.add(arg.lstrip('-').replace('-', '_'))
    return explicit_args

def update_config_with_args(config, args):
    """Update configuration with command line arguments"""
    # Always update these basic parameters regardless of use_default_config
    work_dir = args.work_dir or os.getcwd()
    basic_params = {
    'binder_chain': args.binder_id,
    # Single-sequence (no-MSA) path unless at least one target is a protein,
    # in which case the protein-MSA path is used.
    'non_protein_target': not any(t == 'protein' for t in get_all_target_types(args)),
    'pocket_conditioning': bool(args.contact_residues),
    # Per-target ID list for the design loop's per-target masks.
    'target_chain_ids': get_target_chain_ids(args),
    # COM loss inputs: pdb_path is read inside boltz_hallucination to extract
    # ORI HETATMs + their target-chain Kabsch anchor; com_loss_weight gates
    # the loss. Either one being None/0 keeps the loss off, byte-identical to
    # pre-feature runs.
    'pdb_path': args.pdb_path,
    'com_loss_weight': args.com_loss,
    }
    # Per-target settings: user CLI lists (with length-1 broadcast) win;
    # otherwise pulled from the per-type default YAML for each target.
    basic_params.update(derive_per_target_settings(args, work_dir))

    # Update basic parameters
    explicit_args = get_explicit_args()
    config.update(basic_params)

    # For advanced parameters, only update those that are explicitly set by the user
    # (i.e., different from their default values in argparse)
    parser = argparse.ArgumentParser()
    _, defaults = parser.parse_known_args([])  # Get default values

    advanced_params = {
        'distogram_only': args.distogram_only,
        'attach_coords': args.attach_coords,
        'design_algorithm': args.design_algorithm,
        'learning_rate': args.learning_rate,
        'learning_rate_pre': args.learning_rate_pre,
        'e_soft': args.e_soft,
        'e_soft_1': args.e_soft_1,
        'e_soft_2': args.e_soft_2,
        'length_min': args.length_min,
        'length_max': args.length_max,
        'intra_chain_cutoff': args.intra_chain_cutoff,
        'num_intra_contacts': args.num_intra_contacts,
        'helix_loss_max': args.helix_loss_max,
        'helix_loss_min': args.helix_loss_min,
        'optimizer_type': args.optimizer_type,
        'pre_iteration': args.pre_iteration,
        'soft_iteration': args.soft_iteration,
        'temp_iteration': args.temp_iteration,
        'hard_iteration': args.hard_iteration,
        'semi_greedy_steps': args.semi_greedy_steps,
        'msa_max_seqs': args.msa_max_seqs,
        'recycling_steps': args.recycling_steps,
        'sampling_steps': args.sampling_steps,
        'motif_pdb': args.motif_pdb,
        # Re-emit the list-valued nargs="+" flags as the legacy comma/`;`-separated
        # strings the downstream parsers in boltzdesign_utils consume (parse_motif_*,
        # parse_atom_pairs_spec). This keeps the parser surface in utils unchanged.
        'motif_residues': ",".join(_split_csv(args.motif_residues)),
        'motif_binder_positions': ",".join(_split_csv(args.motif_binder_positions)),
        'fix_motif_seq': args.fix_motif_seq,
        'motif_ligand_residues': ",".join(_split_csv(args.motif_ligand_residues)),
        'atom_pairs': "; ".join(_split_semi(args.atom_pairs)),
        'atom_pair_distogram_loss_type': args.atom_pair_distogram_loss_type,
    }

    for param_name, param_value in advanced_params.items():
        if param_name in explicit_args:
            print(f"Updating {param_name} to {param_value}")
            config[param_name] = param_value
    return config
    
def run_boltz_design_step(args, config, boltz_model, yaml_dir, main_dir, version_name):
    """Run the Boltz design step"""
    print("Starting Boltz design step...")
    
    # Per-target loss weights (i_con_loss, i_pae_loss, target_plddt_loss) are
    # applied inside get_model_loss before the per-target sum, so the global
    # loss_scales multiplier is 1.0 for those keys (no double-weighting). All
    # other losses keep their global weight here.
    loss_scales = {
        'con_loss': args.con_loss,
        'i_con_loss': 1.0,
        'plddt_loss': args.plddt_loss,
        'pae_loss': args.pae_loss,
        'i_pae_loss': 1.0,
        'rg_loss': args.rg_loss,
        'com_loss': args.com_loss,
        'target_plddt_loss': 1.0,
        'motif_distogram_loss': args.motif_distogram_loss,
        'motif_coords_loss': args.motif_coords_loss,
        'motif_fape_loss': args.motif_fape_loss,
        'atom_pair_distogram_loss': args.atom_pair_distogram_loss,
        'atom_pair_coords_loss': args.atom_pair_coords_loss,
    }
    
    boltz_path = shutil.which("boltz")
    if boltz_path is None:
        raise FileNotFoundError("The 'boltz' command was not found in the system PATH.")
    
    run_boltz_design(
        boltz_path=boltz_path,
        main_dir=main_dir,
        yaml_dir=os.path.dirname(yaml_dir),
        boltz_model=boltz_model,
        ccd_path=args.ccd_path,
        design_samples=args.design_samples,
        version_name=version_name,
        config=config,
        loss_scales=loss_scales,
        show_animation=args.show_animation,
        save_trajectory=args.save_trajectory,
        save_intermediate_structures=args.save_intermediate_structures,
        redo_boltz_predict=args.redo_boltz_predict,
    )
    
    print("Boltz design step completed!")

def run_ligandmpnn_step(args, main_dir, version_name, ligandmpnn_dir, yaml_dir, work_dir):
    """Run the LigandMPNN redesign step"""
    print("Starting LigandMPNN redesign step...")
    # Setup LigandMPNN config
    yaml_path = f"{work_dir}/LigandMPNN/run_ligandmpnn_logits_config.yaml"
    with open(yaml_path, "r") as f:
        mpnn_config = yaml.safe_load(f)
    
    for key, value in mpnn_config.items():
        if isinstance(value, str) and "${CWD}" in value:
            mpnn_config[key] = value.replace("${CWD}", work_dir)
    
    if not Path(mpnn_config["checkpoint_soluble_mpnn"]).exists():
        raise FileNotFoundError("LigandMPNN checkpoint file not found!")
    
    with open(yaml_path, "w") as f:
        yaml.dump(mpnn_config, f, default_flow_style=False)
    
    # Setup directories
    boltzdesign_dir = f"{main_dir}/{version_name}/results_final"
    pdb_save_dir = f"{main_dir}/{version_name}/pdb"
    
    lmpnn_redesigned_dir = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned')
    lmpnn_redesigned_fa_dir = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned_fa')
    lmpnn_redesigned_yaml_dir = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned_yaml')
    
    os.makedirs(ligandmpnn_dir, exist_ok=True)
    # Convert CIF to PDB and run LigandMPNN
    convert_cif_files_to_pdb(boltzdesign_dir, pdb_save_dir, high_iptm=args.high_iptm, i_ptm_cutoff=args.i_ptm_cutoff)

    if not any(f.endswith('.pdb') for f in os.listdir(pdb_save_dir)):
        print("No successful designs from BoltzDesign")
        sys.exit(1)
    
    run_ligandmpnn_redesign(
        ligandmpnn_dir, pdb_save_dir, shutil.which("boltz"),
        os.path.dirname(yaml_dir), yaml_path, top_k=args.num_designs, cutoff=args.cutoff,
        non_protein_target=not any(t == 'protein' for t in get_all_target_types(args)), binder_chain=args.binder_id,
        target_chains="all", out_dir=lmpnn_redesigned_fa_dir,
        lmpnn_yaml_dir=lmpnn_redesigned_yaml_dir, results_final_dir=lmpnn_redesigned_dir
    )
    
    # Filter high confidence designs
    filter_high_confidence_designs(args, ligandmpnn_dir, lmpnn_redesigned_dir, lmpnn_redesigned_yaml_dir)
    
    print("LigandMPNN redesign step completed!")
    return ligandmpnn_dir

def filter_high_confidence_designs(args, ligandmpnn_dir, lmpnn_redesigned_dir, lmpnn_redesigned_yaml_dir):
    """Filter and save high confidence designs"""
    print("Filtering high confidence designs...")
    
    yaml_dir_success_designs_dir = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned_high_iptm')
    yaml_dir_success_boltz_yaml = os.path.join(yaml_dir_success_designs_dir, 'yaml')
    yaml_dir_success_boltz_cif = os.path.join(yaml_dir_success_designs_dir, 'cif')
    
    os.makedirs(yaml_dir_success_boltz_yaml, exist_ok=True)
    os.makedirs(yaml_dir_success_boltz_cif, exist_ok=True)
    
    successful_designs = 0
    
    # Process designs
    for root in os.listdir(lmpnn_redesigned_dir):
        root_path = os.path.join(lmpnn_redesigned_dir, root, 'predictions')
        if not os.path.isdir(root_path):
            continue
        
        for subdir in os.listdir(root_path):
            json_path = os.path.join(root_path, subdir, f'confidence_{subdir}_model_0.json')
            yaml_path = os.path.join(lmpnn_redesigned_yaml_dir, f'{subdir}.yaml')
            cif_path = os.path.join(lmpnn_redesigned_dir, f'boltz_results_{subdir}', 'predictions', subdir, f'{subdir}_model_0.cif')
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                design_name = json_path.split('/')[-2]
                length = int(subdir[subdir.find('length') + 6:subdir.find('_model')])
                iptm = data.get('iptm', 0)
                complex_plddt = data.get('complex_plddt', 0)
                
                print(f"{design_name} length: {length} complex_plddt: {complex_plddt:.2f} iptm: {iptm:.2f}")
                
                if iptm > args.i_ptm_cutoff and complex_plddt > args.complex_plddt_cutoff:
                    shutil.copy(yaml_path, os.path.join(yaml_dir_success_boltz_yaml, f'{subdir}.yaml'))
                    shutil.copy(cif_path, os.path.join(yaml_dir_success_boltz_cif, f'{subdir}.cif'))
                    print(f"✅ {design_name} copied")
                    successful_designs += 1
            
            except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Skipping {subdir}: {e}")
                continue
    
    if successful_designs == 0:
        print("Error: No LigandMPNN/ProteinMPNN redesigned designs passed the confidence thresholds")
        sys.exit(1)


def calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, binder_chain):
    """Calculate RMSD between holo and apo structures and update confidence CSV.
    
    Args:
        af_pdb_dir (str): Directory containing holo PDB files
        af_pdb_dir_apo (str): Directory containing apo PDB files
    """
    confidence_csv_path = af_pdb_dir + '/high_iptm_confidence_scores.csv'
    if os.path.exists(confidence_csv_path):
        df_confidence_csv = pd.read_csv(confidence_csv_path)
        for pdb_name in os.listdir(af_pdb_dir):
            if pdb_name.endswith('.pdb'):
                pdb_path = os.path.join(af_pdb_dir, pdb_name)
                pdb_path_apo = os.path.join(af_pdb_dir_apo, pdb_name)
                xyz_holo, _ = get_CA_and_sequence(pdb_path, chain_id=binder_chain)
                xyz_apo, _ = get_CA_and_sequence(pdb_path_apo, chain_id='A')
                rmsd = np_rmsd(np.array(xyz_holo), np.array(xyz_apo))
                df_confidence_csv.loc[df_confidence_csv['file'] == pdb_name.split('.pdb')[0]+'.cif', 'rmsd'] = rmsd
                print(f"{pdb_path} rmsd: {rmsd}")
        df_confidence_csv.to_csv(confidence_csv_path, index=False)
        
        
def run_alphafold_step(args, ligandmpnn_dir, work_dir, mod_to_wt_aa):
    """Run AlphaFold validation step"""
    print("Starting AlphaFold validation step...")
    types_list = get_all_target_types(args)

    alphafold_dir = os.path.expanduser(args.alphafold_dir)
    afdb_dir = os.path.expanduser(args.af3_database_settings)
    hmmer_path = os.path.expanduser(args.af3_hmmer_path)
    print("alphafold_dir", alphafold_dir)
    print("afdb_dir", afdb_dir)
    print("hmmer_path", hmmer_path)
    
    # Create AlphaFold directories
    af_input_dir = f'{ligandmpnn_dir}/02_design_json_af3'
    af_output_dir = f'{ligandmpnn_dir}/02_design_final_af3'
    af_input_apo_dir = f'{ligandmpnn_dir}/02_design_json_af3_apo'
    af_output_apo_dir = f'{ligandmpnn_dir}/02_design_final_af3_apo'
    
    for dir_path in [af_input_dir, af_output_dir, af_input_apo_dir, af_output_apo_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Process YAML files
    yaml_dir_success_boltz_yaml = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned_high_iptm', 'yaml')
    
    process_yaml_files(
        yaml_dir_success_boltz_yaml,
        af_input_dir,
        af_input_apo_dir,
        target_type=('multi' if len(types_list) > 1 else types_list[0]),
        binder_chain=args.binder_id,
        mod_to_wt_aa=mod_to_wt_aa,
        afdb_dir=afdb_dir,
        hmmer_path=hmmer_path
    )
    # Run AlphaFold on holo state
    subprocess.run([
        f'{work_dir}/boltzdesign/alphafold.sh',
        af_input_dir,
        af_output_dir,
        str(args.gpu_id),
        alphafold_dir,
        args.af3_docker_name
    ], check=True)
    
    # Run AlphaFold on apo state
    subprocess.run([
        f'{work_dir}/boltzdesign/alphafold.sh',
        af_input_apo_dir,
        af_output_apo_dir,
        str(args.gpu_id),
        alphafold_dir,
        args.af3_docker_name
    ], check=True)
    
    print("AlphaFold validation step completed!")

    af_pdb_dir = f"{ligandmpnn_dir}/03_af_pdb_success"
    af_pdb_dir_apo = f"{ligandmpnn_dir}/03_af_pdb_apo"
    
    convert_cif_files_to_pdb(af_output_dir, af_pdb_dir, af_dir=True, high_iptm=args.high_iptm)
    if not any(f.endswith('.pdb') for f in os.listdir(af_pdb_dir)):
        print("No successful designs from AlphaFold")
        sys.exit(1)
    convert_cif_files_to_pdb(af_output_apo_dir, af_pdb_dir_apo, af_dir=True)
    calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, args.binder_id)

    return af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo

def run_rosetta_step(args, ligandmpnn_dir, af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo):
    """Run Rosetta energy calculation (protein targets only)"""
    if not any(t == 'protein' for t in get_all_target_types(args)):
        print("Skipping Rosetta step (no protein target)")
        return
    
    print("Starting Rosetta energy calculation...")
    af_pdb_rosetta_success_dir = f"{ligandmpnn_dir}/af_pdb_rosetta_success"
    from pyrosetta_utils import measure_rosetta_energy
    measure_rosetta_energy(
        af_pdb_dir, af_pdb_dir_apo, af_pdb_rosetta_success_dir,
        binder_holo_chain=args.binder_id, binder_apo_chain='A'
    )
    
    print("Rosetta energy calculation completed!")

def setup_environment():
    """Setup environment and parse arguments"""
    args = parse_arguments()
    # Validate --target_types eagerly so a typo fails at startup, not deep in
    # the pipeline. Read the list via get_all_target_types(args) wherever
    # needed -- no derived args.target_type attribute is maintained.
    get_all_target_types(args)
    work_dir = args.work_dir or os.getcwd()
    os.chdir(work_dir)
    setup_gpu_environment(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return args

def get_target_ids(args):
    """Get target IDs from either PDB or custom input.

    Required for modifications (target_id_map maps the user's --modification_target
    to its YAML chain letter); contact_residues carries its own chain prefix per
    token so it no longer needs the map.
    """
    target_ids = _split_csv(args.pdb_target_ids if args.input_type == "pdb"
                            else args.custom_target_ids)

    if args.modifications and not target_ids:
        input_type = "PDB" if args.input_type == "pdb" else "Custom"
        raise ValueError(f"{input_type} target IDs must be provided when using modifications")

    return target_ids

def assign_chain_ids(target_ids_list, binder_chain='A'):
    """Maps target IDs to unique chain IDs, skipping binder_chain."""
    letters = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c != binder_chain]
    return {id: letters[i] for i, id in enumerate(target_ids_list)}


def initialize_pipeline(args):
    """Initialize models and configurations"""
    work_dir = args.work_dir or os.getcwd()
    boltz_model, _ = load_boltz_model(args, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    config_obj = YamlConfig(main_dir=f'{work_dir}/inputs/{get_all_target_types(args)[0]}_{args.target_name}_{args.suffix}')
    config_obj.setup_directories()
    return boltz_model, config_obj

def generate_yaml_config(args, config_obj):
    """Generate YAML configuration based on input type"""
    if args.contact_residues or args.modifications:
        target_ids_list = get_target_ids(args)
        target_id_map = assign_chain_ids(target_ids_list, args.binder_id)
        print(f"Mapped target IDs: {list(target_id_map.values())}")
        # Modifications still flow as legacy CSV strings; contact_residues is
        # parsed to a chain-prefixed (chain, res) list which setup_constraints
        # emits directly as the pocket-constraint contacts array (one union
        # block across all referenced target chains -- the model takes one
        # inference_pocket).
        constraints, modifications = process_design_constraints(
            target_id_map,
            ",".join(_split_csv(args.modifications)),
            ",".join(_split_csv(args.modifications_positions)),
            args.modification_target,
            parse_contact_residues_spec(args.contact_residues),
            args.binder_id,
        )
    else:
        constraints, modifications = None, None
    # Per-target type list, in chain order. Length-1 -> single-target mode (the
    # legacy --target_type path); all targets are contacted symmetrically by
    # the design loop regardless.
    types_list = get_all_target_types(args)
    multi = len(types_list) > 1
    target = []

    if args.input_type == "pdb":
        pdb_target_ids = _split_csv(args.pdb_target_ids)
        target_mols = _split_csv(args.target_mols)
        if args.pdb_path:
            pdb_path = Path(args.pdb_path)
            print("load local pdb from", pdb_path)
            if not pdb_path.is_file():
                raise FileNotFoundError(f"Could not find local PDB: {args.pdb_path}")
        else:
            print("fetch pdb from RCSB")
            download_pdb(args.target_name, config_obj.PDB_DIR)
            pdb_path = config_obj.PDB_DIR / f"{args.target_name}.pdb"

        # Lookups are built lazily and cached so a mixed target set only parses
        # what it needs. The ligand source mirrors the single-type fix: a local
        # file when --pdb_path is given (target_name may be a user label, not a
        # real PDB ID), else the target_name-as-PDB-ID auto-fetch.
        _nuc = _lig = _chains = None
        def _ligand_lookup():
            return get_ligand_from_pdb(str(pdb_path) if args.pdb_path else args.target_name)

        if multi:
            # Protein/dna/rna draw identifiers from --pdb_target_ids; small
            # molecule/metal from --target_mols. Each is consumed in order.
            n_pdb = sum(1 for t in types_list if t in ('protein', 'dna', 'rna'))
            n_mol = sum(1 for t in types_list if t in ('small_molecule', 'metal'))
            if n_pdb != len(pdb_target_ids):
                raise ValueError(
                    f"--target_types needs {n_pdb} protein/dna/rna identifier(s) "
                    f"from --pdb_target_ids, got {len(pdb_target_ids)}: {pdb_target_ids}")
            if n_mol != len(target_mols):
                raise ValueError(
                    f"--target_types needs {n_mol} small_molecule/metal identifier(s) "
                    f"from --target_mols, got {len(target_mols)}: {target_mols}")

            pi = mi = 0  # cursors into pdb_target_ids / target_mols
            for t in types_list:
                if t == 'protein':
                    if _chains is None:
                        _chains = get_chains_sequence(pdb_path)
                    target.append(_chains[pdb_target_ids[pi]]); pi += 1
                elif t in ('dna', 'rna'):
                    if _nuc is None:
                        _nuc = get_nucleotide_from_pdb(pdb_path)
                    target.append(_nuc[pdb_target_ids[pi]]['seq']); pi += 1
                elif t == 'small_molecule':
                    _tm = target_mols[mi]
                    # CCD code -> CCD ligand ({'ccd': code}); else perceive
                    # SMILES (see single-type branch below for the rationale).
                    if is_ccd_code(_tm):
                        target.append({'ccd': _tm})
                    else:
                        if _lig is None:
                            _lig = _ligand_lookup()
                        print(_tm, _lig.keys())
                        target.append(_lig[_tm])
                    mi += 1
                elif t == 'metal':
                    target.append(target_mols[mi]); mi += 1  # CCD code, used as-is
                else:
                    raise ValueError(f"Unsupported target type: {t}")
        else:
            if types_list[0] in ['rna', 'dna']:
                _nuc = get_nucleotide_from_pdb(pdb_path)
                for target_id in pdb_target_ids:
                    target.append(_nuc[target_id]['seq'])
            elif types_list[0] == 'small_molecule':
                for target_mol in target_mols:
                    # A valid CCD code (e.g. HEM) is emitted as a CCD ligand
                    # ({'ccd': code} -> canonical resname + atom names, needed
                    # for the motif ligand carry-along / atom_pairs) and never
                    # goes through PDB SMILES perception (which also skips HEM
                    # et al. via get_ligand_from_pdb's IGNORE_LIST). Non-CCD
                    # codes (e.g. 6TR) fall back to perceived SMILES from the PDB.
                    if is_ccd_code(target_mol):
                        target.append({'ccd': target_mol})
                    else:
                        if _lig is None:
                            _lig = _ligand_lookup()
                        print(target_mol, _lig.keys())
                        target.append(_lig[target_mol])
            elif types_list[0] == 'protein':
                _chains = get_chains_sequence(pdb_path)
                for target_id in pdb_target_ids:
                    target.append(_chains[target_id])
            else:
                raise ValueError(f"Unsupported target type: {types_list[0]}")
    else:
        target_inputs = _split_csv(args.custom_target_inputs)
        target = target_inputs or [args.target_name]
        if multi and len(types_list) != len(target):
            raise ValueError(
                f"--target_types lists {len(types_list)} type(s) but "
                f"--custom_target_inputs has {len(target)} entr(ies); they must "
                f"match one-to-one and in order.")

    # str (broadcast) for single-type, per-target list for multi-type.
    target_types_arg = types_list if multi else types_list[0]

    return generate_yaml_for_target_binder(
        args.target_name,
        target_types_arg,
        target,
        config=config_obj,
        binder_id=args.binder_id,
        constraints=constraints,
        modifications=modifications['data'] if modifications else None,
        modification_target=modifications['target'] if modifications else None,
        use_msa=args.use_msa
    )

def setup_pipeline_config(args):
    """Setup pipeline configuration"""
    work_dir = args.work_dir or os.getcwd()
    config = load_design_config(get_all_target_types(args)[0], work_dir)
    return update_config_with_args(config, args)

def setup_output_directories(args):
    """Setup output directories"""
    work_dir = args.work_dir or os.getcwd()
    main_dir = f'{work_dir}/outputs'
    os.makedirs(main_dir, exist_ok=True)
    return {
        'main_dir': main_dir,
        'version': f'{get_all_target_types(args)[0]}_{args.target_name}_{args.suffix}'
    }
def modification_to_wt_aa(modifications, modifications_wt):
    """Convert modifications to WT AA. Accepts either the legacy CSV strings or
    the nargs="+" lists -- _split_csv normalizes both."""
    mods = _split_csv(modifications)
    wts = _split_csv(modifications_wt)
    if not mods:
        return None, None
    return {mod: wt for mod, wt in zip(mods, wts)}

def run_pipeline_steps(args, config, boltz_model, yaml_dir, output_dir):
    """Run the pipeline steps based on arguments"""
    results = {'ligandmpnn_dir': f"{output_dir['main_dir']}/{output_dir['version']}/ligandmpnn_cutoff_{args.cutoff}", 'af_output_dir': None, 'af_output_apo_dir': None, 'af_pdb_dir': None, 'af_pdb_dir_apo': None}
    
    if args.run_boltz_design:
        run_boltz_design_step(args, config, boltz_model, yaml_dir, 
                            output_dir['main_dir'], output_dir['version'])

    if args.run_ligandmpnn:
        run_ligandmpnn_step(
            args, output_dir['main_dir'], output_dir['version'], 
            results['ligandmpnn_dir'], yaml_dir, args.work_dir or os.getcwd()
        )
    if args.run_alphafold:
        mod_to_wt_aa = modification_to_wt_aa(args.modifications, args.modifications_wt)
        results['af_output_dir'], results['af_output_apo_dir'], results['af_pdb_dir'], results['af_pdb_dir_apo'] = run_alphafold_step(
            args, results['ligandmpnn_dir'], args.work_dir or os.getcwd(), mod_to_wt_aa
        )
    
    if args.run_rosetta:
        run_rosetta_step(args, results['ligandmpnn_dir'], 
                        results['af_output_dir'], results['af_output_apo_dir'], results['af_pdb_dir'], results['af_pdb_dir_apo'])
    
    return results

def main():
    """Main function for running the BoltzDesign pipeline"""
    args = setup_environment()
    boltz_model, config_obj = initialize_pipeline(args)
    yaml_dict, yaml_dir = generate_yaml_config(args, config_obj)

    print("Generated YAML configuration:")
    for key, value in yaml_dict.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")
    
    # Setup pipeline configuration
    config = setup_pipeline_config(args)
    output_dir = setup_output_directories(args)
    
    # Run pipeline steps
    print("config:")
    items = list(config.items())
    max_key_len = max(len(key) for key, _ in items)
    max_val_len = max(len(str(val)) for _, val in items)
    
    # Print header
    print("  " + "=" * (max_key_len + max_val_len + 5))
    
    # Print items in two columns
    for i in range(0, len(items), 2):
        key1, value1 = items[i]
        if i+1 < len(items):
            key2, value2 = items[i+1]
            print(f"  {key1:<{max_key_len}}: {str(value1):<{max_val_len}}    "
                  f"{key2:<{max_key_len}}: {value2}")
        else:
            print(f"  {key1:<{max_key_len}}: {value1}")
    
    print("  " + "=" * (max_key_len + max_val_len + 5))
    results = run_pipeline_steps(args, config, boltz_model, yaml_dir, output_dir)
    
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
