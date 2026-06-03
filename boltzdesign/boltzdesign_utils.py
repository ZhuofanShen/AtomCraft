import os
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional
import copy
import random
from boltz.data import const
from boltz.data.types import MSA, Connection, Input, Structure, Interface
from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
import yaml
import shutil
from Bio.PDB import PDBParser, MMCIFParser  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import csv
import gc
import json
import logging

logging.basicConfig(level=logging.WARNING)

def save_confidence_scores(folder_dir, output, structure,name, model_idx=0):
    output_dir = os.path.join(folder_dir, f"boltz_results_{name}", "predictions", name)

    os.makedirs(output_dir, exist_ok=True)
    atoms = structure.atoms
    atoms['coords'] = output['coords'][0].detach().cpu().numpy()[:atoms['coords'].shape[0],:]
    atoms["is_present"] = True
    residues = structure.residues
    residues["is_present"] = True
    interfaces = np.array([], dtype=Interface)
    new_structure: Structure = replace(
        structure,
        atoms=atoms,
        residues=residues,
        interfaces=interfaces,
    )
    plddts= output['plddt'].detach().cpu().numpy()[0]        
    path = Path(output_dir) / f"{name}_model_{model_idx}.cif"
    with path.open("w") as f:
        f.write(to_mmcif(new_structure, plddts=plddts))

    # Save confidence summary
    if "plddt" in output:
        confidence_summary_dict = {}
        for key in [
            "confidence_score",
            "ptm", 
            "iptm",
            "ligand_iptm",
            "protein_iptm",
            "complex_plddt",
            "complex_iplddt", 
            "complex_pde",
            "complex_ipde",
        ]:
            if key in output:
                confidence_summary_dict[key] = output[key].item()
        
        if "pair_chains_iptm" in output:
            confidence_summary_dict["chains_ptm"] = {
                idx: output["pair_chains_iptm"][idx][idx].item()
                for idx in output["pair_chains_iptm"]
            }
            confidence_summary_dict["pair_chains_iptm"] = {
                idx1: {
                    idx2: output["pair_chains_iptm"][idx1][idx2].item()
                    for idx2 in output["pair_chains_iptm"][idx1]
                }
                for idx1 in output["pair_chains_iptm"]
            }

        json_path = os.path.join(output_dir, f"confidence_{name}_model_{model_idx}.json")
        with open(json_path, 'w') as f:
            json.dump(confidence_summary_dict, f, indent=4)
        # Save plddt
        plddt = output["plddt"]
        plddt_path = os.path.join(output_dir, f"plddt_{name}_model_{model_idx}.npz")
        np.savez_compressed(plddt_path, plddt=plddt.cpu().detach().numpy())

    if "pae" in output:
        pae = output["pae"]
        pae_path = os.path.join(output_dir, f"pae_{name}_model_{model_idx}.npz")
        np.savez_compressed(pae_path, pae=pae.cpu().detach().numpy())


tokens = [
    "<pad>",
    "-",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",  # unknown protein token
    "A",
    "G",
    "C",
    "U",
    "N",  # unknown rna token
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",  # unknown dna token
]


chain_to_number = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
}
# Loss-history categorization for the per-iteration plots. Each tuple is
# (series_key in loss_component_history, plot label, color). Series with no
# data this run are silently skipped so a single-target / fast-mode run
# doesn't waste panel space on inert losses.
_LOSS_GROUPS = {
    'distogram': [
        ('con_loss',                 'Intra-Contact Loss',        '#ff3366'),
        ('i_con_loss',               'Inter-Contact Loss',        '#3366ff'),
        ('helix_loss',               'Helix Loss',                '#ffaa00'),
        ('motif_distogram_loss',     'Motif Distogram Loss',      '#aa66ff'),
        ('atom_pair_distogram_loss', 'Atom-Pair Distogram Loss',  '#66ffcc'),
    ],
    'confidence': [
        ('plddt_loss',        'pLDDT Loss',         '#00ff99'),
        ('pae_loss',          'PAE Loss',           '#3366ff'),
        ('i_pae_loss',        'Interface PAE Loss', '#ff3366'),
        ('target_plddt_loss', 'Target pLDDT Loss',  '#66ffcc'),
    ],
    'coords': [
        ('rg_loss',               'Rg Loss',                '#cc66ff'),
        ('com_loss',              'COM Loss',               '#ff9933'),
        ('motif_coords_loss',     'Motif Coords RMSD',      '#aa66ff'),
        ('motif_bb_rmsd',         'Motif BB RMSD',          '#66aaff'),
        ('motif_lig_rmsd',        'Motif Ligand RMSD',      '#ffaa66'),
        ('motif_fape_loss',       'Motif FAPE Loss',        '#cc99ff'),
        ('atom_pair_coords_loss', 'Atom-Pair Coords Loss',  '#66ffcc'),
    ],
}


def _plot_loss_group(history, group_specs, png_path, csv_path, suptitle,
                     csv_col_overrides=None):
    """Plot one category as a horizontal row of subplots (per-loss y-axis, NOT
    shared) into a single figure, and write a matching CSV.

    history          : dict mapping series_key -> list of floats (per logged step)
    group_specs      : list of (series_key, label, color); skipped if no data.
    csv_col_overrides: optional {series_key: csv_column_name} (used to rename
                       atom_pair_distogram_loss to include the active method).
    """
    import matplotlib.pyplot as _plt
    present = [(k, lbl, c) for k, lbl, c in group_specs if history.get(k)]
    if not present:
        return False
    _plt.style.use('dark_background')
    n = len(present)
    fig, axes = _plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor('#1C1C1C')
    for ax, (k, lbl, c) in zip(axes, present):
        ax.plot(history[k], color=c, linewidth=2)
        ax.set_xlabel('Logged Steps', fontsize=12)
        ax.set_ylabel(lbl, fontsize=12)
        ax.set_title(lbl, fontsize=12, pad=10)
        ax.grid(True, linestyle='--', alpha=0.3)
    fig.suptitle(suptitle, fontsize=14, color='white')
    _plt.tight_layout(pad=3.0, rect=(0, 0, 1, 0.96))
    _plt.savefig(png_path, facecolor='#1C1C1C', edgecolor='none',
                 bbox_inches='tight', dpi=300)
    _plt.close(fig)
    # Matching CSV (one row per logged step; shorter series blank-padded).
    overrides = csv_col_overrides or {}
    n_steps = max(len(history[k]) for k, _, _ in present)
    with open(csv_path, 'w', newline='') as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(['step'] + [overrides.get(k, k) for k, _, _ in present])
        for t in range(n_steps):
            row = [t]
            for k, _, _ in present:
                s = history[k]
                row.append(f'{s[t]:.4f}' if t < len(s) else '')
            w.writerow(row)
    return True


def visualize_training_history(best_batch, loss_history, sequence_history, distogram_history, length, binder_chain='A', save_dir=None, save_filename=None, plddt_history=None, pae_history=None):
    """
    Visualize training history including distogram, sequence, pLDDT, and PAE
    evolution animations.
    Args:
        loss_history (list): List of loss values over training
        sequence_history (list): List of sequence probability matrices over training
        distogram_history (list): List of distogram matrices over training
        length (int): Length of sequence to visualize
        save_dir (str): Directory to save visualizations
        plddt_history (list|None): Per-epoch per-token pLDDT vectors (full mode
            only; None/empty -> pLDDT animation skipped).
        pae_history (list|None): Per-epoch symmetric PAE matrices in A (full
            mode only; None/empty -> PAE animation skipped).
    """

    mask = (best_batch['entity_id']==chain_to_number[binder_chain]).squeeze(0).detach().cpu().numpy()
    sequence_history = [seq[mask] for seq in sequence_history]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)


    def create_distogram_animation():
        plt.style.use('default')  # Use default white background style
        fig, ax = plt.subplots(figsize=(6,6))
        distogram_2d = distogram_history[0]
        im = ax.imshow(distogram_2d)
    
        plt.colorbar(im, ax=ax)
        ax.set_title('Distogram Evolution')

        def update(frame):
            distogram_2d = distogram_history[frame]
            im.set_data(distogram_2d)
            ax.set_title(f'Distogram Epoch {frame + 1}')
            return im,

        ani = FuncAnimation(fig, update, frames=len(distogram_history), interval=200)
        if save_dir:
            ani.save(os.path.join(save_dir, f'{save_filename}_distogram_evolution.gif'), writer='pillow')
        plt.close()
        return ani

    # Create sequence evolution animation
    def create_sequence_animation():
        plt.style.use('default')  # Use default white background style
        fig, ax = plt.subplots(figsize=(12,3.5))
        im = ax.imshow(sequence_history[0].T, vmin=0, vmax=1, cmap='Blues', aspect='auto', alpha=0.8)
        plt.colorbar(im, ax=ax)
        ax.set_yticks(np.arange(20))
        ax.set_yticklabels(list('ARNDCQEGHILKMFPSTWYV'))
        ax.set_title('Sequence Evolution')

        def update(frame):
            im.set_data(sequence_history[frame].T)
            ax.set_title(f'Sequence Epoch {frame + 1}')
            return im,

        ani = FuncAnimation(fig, update, frames=len(sequence_history), interval=200)
        if save_dir:
            ani.save(os.path.join(save_dir, f'{save_filename}_sequence_evolution.gif'), writer='pillow')
        plt.close()
        return ani

    # Token-level chain boundaries (indices where entity_id changes), used to
    # draw chain borders on the PAE map and separate binder/target on pLDDT,
    # mirroring the AF3-server PAE plot.
    entity_ids = best_batch['entity_id'].squeeze(0).detach().cpu().numpy()
    chain_boundaries = [i for i in range(1, len(entity_ids))
                        if entity_ids[i] != entity_ids[i - 1]]

    # pLDDT evolution: line chart of per-token pLDDT vs token (residue) index.
    def create_plddt_animation():
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 3.5))
        n = len(plddt_history[0])
        x = np.arange(n)
        (line,) = ax.plot(x, plddt_history[0], color='#1f77b4', linewidth=1.5)
        ax.set_xlim(0, max(n - 1, 1))
        ax.set_ylim(0, 1)
        ax.set_xlabel('Token (residue) index')
        ax.set_ylabel('pLDDT')
        for b in chain_boundaries:
            ax.axvline(b - 0.5, color='black', lw=1, linestyle='--', alpha=0.6)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title('pLDDT Evolution')

        def update(frame):
            line.set_ydata(plddt_history[frame])
            ax.set_title(f'pLDDT Epoch {frame + 1}')
            return line,

        ani = FuncAnimation(fig, update, frames=len(plddt_history), interval=200)
        if save_dir:
            ani.save(os.path.join(save_dir, f'{save_filename}_plddt_evolution.gif'), writer='pillow')
        plt.close()
        return ani

    # PAE evolution: AF3-server-style heatmap (Greens_r, 0-30 A, chain borders).
    # The full symmetric PAE already contains both the intra (PAE) and
    # interface (iPAE) blocks, so a single map shows both.
    def create_pae_animation():
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(pae_history[0], cmap='Greens_r', vmin=0, vmax=30, origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='PAE (Å)')
        for b in chain_boundaries:
            ax.axhline(b - 0.5, color='black', lw=1)
            ax.axvline(b - 0.5, color='black', lw=1)
        ax.set_xlabel('Aligned residue')
        ax.set_ylabel('Aligned residue')
        ax.set_title('Predicted Aligned Error (PAE)')

        def update(frame):
            im.set_data(pae_history[frame])
            ax.set_title(f'PAE Epoch {frame + 1}')
            return im,

        ani = FuncAnimation(fig, update, frames=len(pae_history), interval=200)
        if save_dir:
            ani.save(os.path.join(save_dir, f'{save_filename}_pae_evolution.gif'), writer='pillow')
        plt.close()
        return ani

    # Create and save animations
    distogram_ani = create_distogram_animation()
    sequence_ani = create_sequence_animation()
    # pLDDT/PAE are full-mode-only -> skip cleanly if no frames were captured.
    plddt_ani = create_plddt_animation() if plddt_history else None
    pae_ani = create_pae_animation() if pae_history else None

    return distogram_ani, sequence_ani, plddt_ani, pae_ani

def get_mid_points(pdistogram):
    boundaries = torch.linspace(2, 22.0, 63)
    lower = torch.tensor([1.0])
    upper = torch.tensor([22.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(
        pdistogram.device
    )

    return mid_points


def get_CA_and_sequence(structure_file, chain_id='A'):
    # Determine file type and use appropriate parser
    if structure_file.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    elif structure_file.endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("File must be either .cif or .pdb format")
        
    structure = parser.get_structure("structure", structure_file)
    xyz = []
    sequence = []
    aa_map = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    model = structure[0]  # Get first model (default for most structures)
    
    if chain_id in model:
        chain = model[chain_id]
        for residue in chain:
            if "CA" in residue:
                xyz.append(residue["CA"].coord)
                sequence.append(aa_map.get(residue.resname, 'X'))
    else:
        raise ValueError(f"Chain {chain_id} not found in {structure_file}")
    
    return xyz, sequence


# ---------------------------------------------------------------------------
# Motif scaffolding (port of ColabDesign AFDesign `partial` protocol)
#
# ColabDesign's partial-hallucination supervised loss is `dgram_cce`
# (af/loss.py::get_dgram_loss / _loss_partial): take the reference motif's
# pseudo-Cβ coordinates, discretize the pairwise distance matrix into the
# folding model's distogram bins, and apply categorical cross-entropy between
# that one-hot target and the *predicted* distogram restricted to the motif
# positions. Boltz1's distogram head uses 64 bins with edges
# `torch.linspace(2, 22, 63)` and binning `(d > edges).sum(-1)` (see
# boltz/.../confidence.py:81,295 and get_mid_points above), so the port is
# exact. The whole loss lives on `pdistogram`, so it works in BoltzDesign1's
# default fast mode (distogram_only=True) — no diffusion/coords needed, exactly
# like the contact losses. This is what makes "design a binder that also
# retains a catalytic/cofactor-binding motif" feasible in the cheap path.
# ---------------------------------------------------------------------------

_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# res_type column for each one-letter AA, matching the alphabet used
# throughout boltz_hallucination: list('XXARNDCQEGHILKMFPSTWYV-').
_ALPHABET = list('XXARNDCQEGHILKMFPSTWYV-')
_AA_TO_COL = {aa: i for i, aa in enumerate(_ALPHABET) if i >= 2 and aa not in ('X', '-')}


def parse_residue_spec(spec):
    """Parse a ColabDesign-style residue selection string.

    Accepts plain author/PDB residue numbers, comma-separated, with optional
    inclusive ranges, e.g. "10-14,57,102". Returns the ordered list of ints
    (order preserved as written so motif placement is deterministic). Used by
    ``--motif_binder_positions`` (one chain by construction -- the binder).
    """
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    residues = []
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok:
            continue
        if '-' in tok and not tok.startswith('-'):
            lo, hi = tok.split('-')
            residues.extend(range(int(lo), int(hi) + 1))
        else:
            residues.append(int(tok))
    return residues


def parse_motif_residue_spec(spec):
    """Parse ``--motif_residues`` (chain-prefixed motif protein residue selection).

    Every comma-separated token must carry a ``CHAIN`` prefix (leading letters)
    followed by an author/PDB residue number or inclusive range, so a single
    ``--motif_residues`` string can address motif residues across multiple
    chains in the motif PDB (e.g. a catalytic triad spread over chains A and B).

    Grammar (comma-separated, inclusive ranges with '-'):
      ``"A57,A102,A195"``  -> all chain A
      ``"B57,A102,C195"``  -> mixed chains
      ``"A10-14,B57"``     -> ranges expanded inside each chain prefix

    Returns a list of ``(chain_id, residue_number)`` tuples in declaration
    order -- the same order ``--motif_binder_positions`` matches against
    (1-to-1 with binder positions; ligand carry-along lives in the separate
    ``--motif_ligand_residues`` flag).
    """
    if spec is None or spec == '':
        return []
    out = []
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok:
            continue
        i = 0
        while i < len(tok) and tok[i].isalpha():
            i += 1
        if i == 0:
            raise ValueError(
                f"--motif_residues token '{tok}' must be CHAIN+RESNUM "
                f"(e.g. 'A57' or 'A10-14'); bare residue numbers are not "
                f"accepted -- prefix the chain explicitly.")
        chain, body = tok[:i], tok[i:]
        if not body:
            raise ValueError(
                f"--motif_residues token '{tok}' has a chain but no residue number")
        if '-' in body and not body.startswith('-'):
            lo, hi = body.split('-')
            for r in range(int(lo), int(hi) + 1):
                out.append((chain, r))
        else:
            out.append((chain, int(body)))
    return out


def extract_motif_coords(structure_file, chain_residues):
    """Extract pseudo-Cβ coordinates + sequence for the requested motif.

    Mirrors ColabDesign's pseudo_beta_fn: use CB when present, fall back to CA
    (glycine / missing CB). ``chain_residues`` is a list of
    ``(chain_id, author_resnum)`` tuples (the parsed ``--motif_residues``
    output); entries are looked up per-chain and returned in declaration
    order, so a single motif can span multiple chains in the motif PDB.
    Raises a clear error if any requested residue is absent.
    """
    if structure_file.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    elif structure_file.endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("Motif structure must be .cif or .pdb")

    model = parser.get_structure("motif", structure_file)[0]
    # Cache per-chain residue maps so a mixed-chain spec doesn't re-scan.
    chain_byres = {}
    def _get(chain_id):
        if chain_id in chain_byres:
            return chain_byres[chain_id]
        if chain_id not in model:
            raise ValueError(
                f"Motif chain {chain_id!r} not found in {structure_file}")
        m = {res.id[1]: res for res in model[chain_id]
             if res.id[0].strip() == ''}
        chain_byres[chain_id] = m
        return m

    xyz, seq = [], []
    for chain_id, rnum in chain_residues:
        by_resseq = _get(chain_id)
        if rnum not in by_resseq:
            raise ValueError(
                f"Motif residue {rnum} not found in chain {chain_id} of "
                f"{structure_file} (available: "
                f"{min(by_resseq)}-{max(by_resseq)})")
        res = by_resseq[rnum]
        atom = res['CB'] if 'CB' in res else (res['CA'] if 'CA' in res else None)
        if atom is None:
            raise ValueError(
                f"Motif residue {chain_id}{rnum} ({res.resname}) has no CB or CA atom")
        xyz.append(atom.coord)
        seq.append(_THREE_TO_ONE.get(res.resname, 'X'))
    return np.asarray(xyz, dtype=np.float32), ''.join(seq)


def extract_motif_residue_atoms(structure_file, chain_residues):
    """Per-residue heavy-atom coordinates for the motif: list of dicts
    ``{'chain': str, 'resnum': int, 'resname': str,
       'atoms': {atom_name: np.array([x,y,z])}}`` in declaration order.
    Hydrogens are skipped.

    Feeds both the backbone (N, CA, C, CB) Kabsch-RMSD target and the FAPE
    sidechain target. ``chain_residues`` is a list of
    ``(chain_id, author_resnum)`` tuples (the parsed ``--motif_residues``
    output); per-chain lookups are cached so a mixed-chain spec doesn't
    re-walk the structure.
    """
    if structure_file.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    elif structure_file.endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("Motif structure must be .cif or .pdb")

    model = parser.get_structure("motif", structure_file)[0]
    chain_byres = {}
    def _get(chain_id):
        if chain_id in chain_byres:
            return chain_byres[chain_id]
        if chain_id not in model:
            raise ValueError(
                f"Motif chain {chain_id!r} not found in {structure_file}")
        m = {res.id[1]: res for res in model[chain_id]
             if res.id[0].strip() == ''}
        chain_byres[chain_id] = m
        return m

    out = []
    for chain_id, rnum in chain_residues:
        by_resseq = _get(chain_id)
        if rnum not in by_resseq:
            raise ValueError(
                f"Motif residue {chain_id}{rnum} not found in chain "
                f"{chain_id} of {structure_file}")
        res = by_resseq[rnum]
        atoms = {a.get_name(): np.asarray(a.coord, dtype=np.float32)
                 for a in res if not a.get_name().startswith('H')}
        if 'CA' not in atoms:
            raise ValueError(
                f"Motif residue {chain_id}{rnum} ({res.resname}) has no CA atom")
        out.append({'chain': chain_id, 'resnum': rnum,
                    'resname': res.resname, 'atoms': atoms})
    return out


def parse_motif_ligand_spec(spec):
    """Parse the ``--motif_ligand_residues`` selection string.

    Comma-separated chain-prefixed residue tokens, e.g. ``"B1"`` or
    ``"B1,C401"``. Each token is one or more chain letters followed by the
    author/PDB residue number in the motif PDB. Returns a list of
    ``(chain_id, resnum)`` tuples in declaration order.

    Unlike ``--motif_residues`` these entries do **not** consume slots in
    ``--motif_binder_positions`` -- they are *carried along* by the rigid
    transform Kabsch-fit on the protein motif backbone (decoupled alignment),
    so the count is independent.
    """
    if spec is None or spec == '':
        return []
    out = []
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok:
            continue
        i = 0
        while i < len(tok) and tok[i].isalpha():
            i += 1
        if i == 0:
            raise ValueError(
                f"--motif_ligand_residues token '{tok}' must be CHAIN+RESNUM "
                f"(e.g. 'B1' = chain B, residue 1 in the motif PDB)")
        chain = tok[:i]
        body = tok[i:]
        if not body or not body.lstrip('-').isdigit():
            raise ValueError(
                f"--motif_ligand_residues token '{tok}': residue number must "
                f"be an integer")
        out.append((chain, int(body)))
    return out


def extract_motif_ligand_atoms(structure_file, chain_residues):
    """Per-entry heavy-atom coordinates for ligand residues in a motif PDB.

    ``chain_residues``: list of ``(chain_id, author_resnum)`` tuples. Returns
    a list of dicts ``{'chain', 'resnum', 'resname', 'atoms': {name: xyz}}``
    in input order. Hydrogens skipped; HETATM allowed (no CA required, unlike
    ``extract_motif_residue_atoms``). Used by the optional ligand carry-along
    path of ``get_motif_coords_loss``.
    """
    if structure_file.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    elif structure_file.endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("Motif structure must be .cif or .pdb")
    model = parser.get_structure("motif", structure_file)[0]
    out = []
    for chain_id, rnum in chain_residues:
        if chain_id not in model:
            raise ValueError(
                f"--motif_ligand_residues: chain {chain_id!r} not in {structure_file}")
        chain = model[chain_id]
        # Match by author residue number; allow either standard or HETATM
        # (ligands typically come in as HETATM with hetfield 'H_...').
        match = next((r for r in chain if r.id[1] == rnum), None)
        if match is None:
            raise ValueError(
                f"--motif_ligand_residues: residue {rnum} not in chain "
                f"{chain_id} of {structure_file}")
        atoms = {a.get_name(): np.asarray(a.coord, dtype=np.float32)
                 for a in match if not a.get_name().startswith('H')}
        if not atoms:
            raise ValueError(
                f"--motif_ligand_residues: residue {chain_id}{rnum} "
                f"({match.resname}) has no heavy atoms")
        out.append({'chain': chain_id, 'resnum': rnum,
                    'resname': match.resname.strip(), 'atoms': atoms})
    return out


# ---- Sliding-window motif placements (--motif_unindex_residues) -------------
# A motif whose **binder positions are not pre-specified**: only the per-residue
# chain+resnum in the motif PDB and intra-island gap structure are fixed; the
# absolute binder positions are sampled by MCMC + simulated annealing during
# design (each epoch updates the placement state via Metropolis on the backbone
# Kabsch RMSD). Coexists with --motif_residues: those occupy a fixed disjoint
# set of binder positions; sliding islands are constrained to avoid them.
def parse_motif_islands_spec(spec):
    """Parse ``--motif_unindex_residues`` into a list of islands.

    Grammar (comma-separated tokens):
      * ``CHAIN+RESNUM`` (chain prefix + author/PDB residue number): a motif
        residue.
      * **bare integer**: spacer length *within the current island* (= number
        of free binder positions between the two flanking residues). So
        ``A38,3,A42`` means A38 at intra-offset 0 and A42 at intra-offset 4
        (positions p+1, p+2, p+3 are free, A42 is at p+4).
      * Two adjacent residue tokens (no integer between them) **start a new
        island**: ``A38,3,A42,A170`` -> Island 1 [A38, A42], Island 2 [A170].
        To put residues consecutively in *one* island, write the spacer
        explicitly: ``A38,0,A39,0,A40``.

    Returns a list of islands; each island is a list of dicts
    ``{'chain': str, 'resnum': int, 'intra_offset': int}`` ordered so the
    first residue has ``intra_offset == 0``. Intra-island order is preserved;
    inter-island order/gaps are unspecified and will be sampled by the MCMC.
    """
    if spec is None or spec == '':
        return []

    def _parse_residue(tok):
        i = 0
        while i < len(tok) and tok[i].isalpha():
            i += 1
        if i == 0 or i == len(tok):
            return None
        chain, body = tok[:i], tok[i:]
        try:
            return chain, int(body)
        except ValueError:
            return None

    def _is_spacer(tok):
        if _parse_residue(tok) is not None:
            return False
        try:
            int(tok)
            return True
        except ValueError:
            return False

    raw = [t.strip() for t in str(spec).split(',') if t.strip()]
    islands = []
    pending_spacer = None
    for tok in raw:
        if _is_spacer(tok):
            if not islands:
                raise ValueError(
                    f"--motif_unindex_residues: leading spacer '{tok}' has no "
                    f"preceding residue")
            if pending_spacer is not None:
                raise ValueError(
                    f"--motif_unindex_residues: two consecutive spacers near '{tok}'")
            spacer = int(tok)
            if spacer < 0:
                raise ValueError(
                    f"--motif_unindex_residues: negative spacer '{tok}'")
            pending_spacer = spacer
            continue

        res = _parse_residue(tok)
        if res is None:
            raise ValueError(
                f"--motif_unindex_residues: token '{tok}' is neither "
                f"CHAIN+RESNUM nor a non-negative integer spacer")
        chain, resnum = res
        if pending_spacer is not None:
            last = islands[-1][-1]
            new_offset = last['intra_offset'] + pending_spacer + 1
            islands[-1].append({'chain': chain, 'resnum': resnum,
                                'intra_offset': new_offset})
            pending_spacer = None
        else:
            islands.append([{'chain': chain, 'resnum': resnum,
                             'intra_offset': 0}])

    if pending_spacer is not None:
        raise ValueError("--motif_unindex_residues: trailing spacer with no residue")
    # Sanity: no duplicate residues across islands.
    seen = set()
    for isl in islands:
        for r in isl:
            key = (r['chain'], r['resnum'])
            if key in seen:
                raise ValueError(
                    f"--motif_unindex_residues: residue {r['chain']}{r['resnum']} "
                    f"appears more than once")
            seen.add(key)
    return islands


def island_length(island):
    """Total span (max intra-offset + 1) of a parsed island."""
    return max(r['intra_offset'] for r in island) + 1


def _placement_to_positions(slide_state, islands):
    """Expand per-island starts (slide_state[i]) into binder positions per
    sliding motif residue, ordered by island-then-residue declaration order
    (matches the reference-atom tensor layout)."""
    positions = []
    for i, island in enumerate(islands):
        for r in island:
            positions.append(slide_state[i] + r['intra_offset'])
    return positions


def is_placement_valid(slide_state, island_lengths, binder_length,
                       fixed_positions):
    """A placement is valid iff every island fits inside [0, binder_length),
    no two sliding-motif positions collide, and none lands on
    ``fixed_positions`` (the binder positions claimed by --motif_residues)."""
    occupied = set(fixed_positions)
    for o, L in zip(slide_state, island_lengths):
        if o < 0 or o + L > binder_length:
            return False
        for p in range(o, o + L):
            if p in occupied:
                return False
            occupied.add(p)
    return True


def random_valid_placement(island_lengths, binder_length, fixed_positions,
                           rng, max_attempts=200):
    """Greedy random init: random permutation of islands, then for each pick a
    random valid starting position given already-placed islands and the
    fixed-position exclusion set. Retries with fresh permutations on failure."""
    n = len(island_lengths)
    for _ in range(max_attempts):
        perm = list(range(n))
        rng.shuffle(perm)
        positions = [None] * n
        occupied = set(fixed_positions)
        ok = True
        for isl_idx in perm:
            L = island_lengths[isl_idx]
            valid_starts = [
                s for s in range(binder_length - L + 1)
                if all((s + k) not in occupied for k in range(L))
            ]
            if not valid_starts:
                ok = False
                break
            s = rng.choice(valid_starts)
            positions[isl_idx] = s
            for k in range(L):
                occupied.add(s + k)
        if ok:
            return positions
    raise RuntimeError(
        f"random_valid_placement: could not place all islands "
        f"(lengths={island_lengths}) within binder of length {binder_length} "
        f"excluding {len(fixed_positions)} fixed position(s)")


def propose_mcmc_move(slide_state, island_lengths, binder_length,
                      fixed_positions, shift_max, swap_prob, rng,
                      max_attempts=50):
    """Sample one MCMC proposal that is *valid* by construction.

    With probability ``swap_prob`` (and only when 2+ islands exist), pick two
    islands and swap their starting positions; otherwise shift one island's
    start by a non-zero delta in [-shift_max, +shift_max]. Retry up to
    ``max_attempts`` times for a valid proposal; return ``None`` if no valid
    move was found (caller should treat as "stay")."""
    n = len(slide_state)
    if n == 0:
        return None
    for _ in range(max_attempts):
        new_state = list(slide_state)
        if n >= 2 and rng.random() < swap_prob:
            i, j = rng.sample(range(n), 2)
            new_state[i], new_state[j] = slide_state[j], slide_state[i]
        else:
            i = rng.randrange(n)
            delta = rng.randint(-shift_max, shift_max)
            if delta == 0:
                continue
            new_state[i] = slide_state[i] + delta
        if is_placement_valid(new_state, island_lengths, binder_length,
                              fixed_positions):
            return new_state
    return None


def get_motif_target_distances(ref_xyz, device=None):
    """Raw reference pairwise distances from motif coords (A), [M, M].

    No bin discretization -- callers that need bin-aware loss (e.g.
    ``get_motif_distogram_loss``) consume these together with the bin
    midpoints to compute bin-center MSE.
    """
    x = torch.as_tensor(ref_xyz, dtype=torch.float32)
    if device is not None:
        x = x.to(device)
    return torch.cdist(x, x)                              # [M, M]


def get_motif_distogram_loss(pdistogram, motif_token_idx, target_dist, mid_pts):
    """Bin-center MSE on the trunk distogram, restricted to motif token pairs.

    For each motif pair (i, j), compute the expected squared error in bin-center
    space against the reference pairwise distance:

        L_ij = sum_k p_ijk * (mid_pts_k - target_dist[i, j])^2          [A^2]

    averaged over distinct motif pairs (i != j).

    MSE-style: decomposes as Var(d_k | i,j) + (E[d_k | i,j] - target_dist_ij)^2,
    so it drives both the expected distance to the reference AND the prediction
    toward a sharp delta there. Bin-center-aware: putting mass on a near-but-
    wrong bin is cheap, on a far-wrong bin is expensive (penalty grows with
    squared bin-center distance from reference). Replaces the previous one-hot
    categorical CE form, which was bin-membership-only (every wrong bin
    contributed -log P regardless of how far it was from the right bin) and
    was sensitive to which bin straddle the reference distance fell into.

    ``pdistogram`` is the raw predicted logits [1, N, N, num_bins]; we gather
    the sub-block at the motif token indices so this works with the trunk-only
    fast path (distogram_only=True).
    """
    p = pdistogram[0].index_select(0, motif_token_idx).index_select(1, motif_token_idx)
    prob = torch.softmax(p, dim=-1)                            # [M, M, num_bins]
    err2 = (mid_pts.view(1, 1, -1) - target_dist.unsqueeze(-1)) ** 2  # [M, M, num_bins]
    per_pair = (prob * err2).sum(dim=-1)                       # [M, M], A^2
    M = motif_token_idx.shape[0]
    mask = ~torch.eye(M, dtype=torch.bool, device=per_pair.device)
    return per_pair[mask].mean()


def np_kabsch(a, b, return_v=False):
    '''Get alignment matrix for two sets of coordinates using numpy
    
    Args:
        a: First set of coordinates
        b: Second set of coordinates
        return_v: If True, return U matrix from SVD. If False, return rotation matrix
        
    Returns:
        Rotation matrix (or U matrix if return_v=True) to align coordinates
    '''
    # Calculate covariance matrix
    ab = np.swapaxes(a, -1, -2) @ b
    
    # Singular value decomposition
    u, s, vh = np.linalg.svd(ab, full_matrices=False)
    
    # Handle reflection case
    flip = np.linalg.det(u @ vh) < 0
    if flip:
        u[...,-1] = -u[...,-1]
    
    return u if return_v else (u @ vh)


def align_points(a, b):
    a_centroid = a.mean(axis=0)
    b_centroid = b.mean(axis=0)

    a_centered = a - a_centroid
    b_centered = b - b_centroid

    R = np_kabsch(a_centered, b_centered)
    a_aligned = a_centered @ R + b_centroid
    return a_aligned


def np_rmsd(true, pred):
    '''Compute RMSD of coordinates after alignment using numpy
    
    Args:
        true: Reference coordinates
        pred: Predicted coordinates to align
        
    Returns:
        Root mean square deviation after optimal alignment
    '''
    # Center coordinates
    p = true - np.mean(true, axis=-2, keepdims=True)
    q = pred - np.mean(pred, axis=-2, keepdims=True)
    
    # Get optimal rotation matrix and apply it
    p = p @ np_kabsch(p, q)
    
    # Calculate RMSD
    return np.sqrt(np.mean(np.sum(np.square(p-q), axis=-1)) + 1e-8)

    
def min_k(x, k=1, mask=None):
    # Convert mask to boolean if it's not None
    if mask is not None:
        mask = mask.bool()  # Convert to boolean tensor
    
    # Sort the tensor, replacing masked values with Nan
    y = torch.sort(x if mask is None else torch.where(mask, x, float('nan')))[0]

    # Create a mask for the top k value
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    # Compute the mean of the top k values
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def get_con_loss(dgram, dgram_bins, num=None, seqsep=None, num_pos = float("inf"), cutoff=None, binary=False, mask_1d=None, mask_1b=None):
    con_loss = _get_con_loss(dgram, dgram_bins, cutoff, binary)
    idx = torch.arange(dgram.shape[1])
    offset = idx[:,None] - idx[None,:]
    # Add mask for position separation > 3
    m =(torch.abs(offset)>=seqsep).to(dgram.device)
    if mask_1d is None: mask_1d = torch.ones(m.shape[0])
    if mask_1b is None: mask_1b = torch.ones(m.shape[0])

    m = torch.logical_and(m, mask_1b)
    p = min_k(con_loss, num, m).to(dgram.device)
    p = min_k(p, num_pos, mask_1d).to(dgram.device)
    return p


def _get_con_loss(dgram, dgram_bins, cutoff=None, binary=False):
    '''dgram to contacts'''
    if cutoff is None: cutoff = dgram_bins[-1]
    bins = dgram_bins < cutoff  
    px = torch.softmax(dgram, dim=-1)
    px_ = torch.softmax(dgram - 1e7 * (~ bins), dim=-1)        
    # binary/categorical cross-entropy
    con_loss_cat_ent = -(px_ * torch.log_softmax(dgram, dim=-1)).sum(-1)
    con_loss_bin_ent = -torch.log((bins * px + 1e-8).sum(-1))

    return binary * con_loss_bin_ent + (1 - binary) * con_loss_cat_ent


def mask_loss(x, mask=None, mask_grad=False):
    if mask is None:
        return x.mean()
    else:
        x_masked = (x * mask).sum() / (1e-8 + mask.sum())
        if mask_grad:
            return (x.mean() - x_masked).detach() + x_masked
        else:
            return x_masked


def get_plddt_loss(plddt, mask_1d=None):
    p = 1 - plddt
    return mask_loss(p, mask_1d)


def get_pae_loss(pae, mask_1d=None, mask_1b=None, mask_2d=None):
  pae = pae/31.0
  L = pae.shape[1]
  if mask_1d is None: mask_1d = torch.ones(L).to(pae.device)
  if mask_1b is None: mask_1b = torch.ones(L).to(pae.device)
  if mask_2d is None: mask_2d = torch.ones((L, L)).to(pae.device)
  mask_2d = mask_2d * mask_1d[:, :, None] * mask_1b[:, None, :]
  return mask_loss(pae, mask_2d)


def _get_helix_loss(dgram, dgram_bins, offset=None, mask_2d=None, binary=False, **kwargs):
    '''helix bias loss'''
    x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=binary)
    if offset is None:
        if mask_2d is None:
            return x.diagonal(offset=3).mean()
        else:
            mask_2d = mask_2d.float() 
            return (x * mask_2d).diagonal(offset=3, dim1=-2, dim2=-1).sum() / (torch.diagonal(mask_2d, offset=3, dim1=-2, dim2=-1).sum() + 1e-8)

    else:
        mask = (offset == 3).float()
        if mask_2d is not None:
            mask = mask * mask_2d.float()
        return (x * mask).sum() / (mask.sum() + 1e-8)


def get_ca_coords(sample_atom_coords, batch, binder_chain='A'):
    atom_to_token = batch['atom_to_token'] * (batch['entity_id']==chain_to_number[binder_chain])
    atom_order = torch.cumsum(atom_to_token, dim=1)
    ca_mask = torch.sum((atom_order == 2).to(atom_to_token.dtype), dim=-1)[0]
    ca_coords = sample_atom_coords[:,ca_mask==1,:]
    return ca_coords


def add_rg_loss(sample_atom_coords, batch, length, binder_chain='A'):
    ca_coords = get_ca_coords(sample_atom_coords, batch, binder_chain)
    center_of_mass = ca_coords.mean(1, keepdim=True)  # keepdim for proper broadcasting
    squared_distances = torch.sum(torch.square(ca_coords - center_of_mass), dim=-1)
    rg = torch.sqrt(squared_distances.mean() + 1e-8)
    rg_th = 2.38 * ca_coords.shape[1] ** 0.365
    loss = torch.nn.functional.elu(rg - rg_th)
    return loss, rg


def parse_pdb_ori_atoms(pdb_path):
    """Return (N,3) numpy array of HETATM ORI xyz coords found in a PDB file.

    Matches by atom name == 'ORI' in the standard PDB column layout. The user
    inserts these into their --pdb_path / --motif_pdb manually (custom HETATM
    records, e.g. ``HETATM ... ORI  ORI z   1   x  y  z``) to anchor a desired
    binder centroid. Returns an empty array if the file is missing or contains
    no ORI atoms (caller treats this as "no source from this PDB").
    """
    if not pdb_path or not os.path.exists(pdb_path):
        return np.zeros((0, 3), dtype=np.float32)
    out = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('HETATM'):
                continue
            # PDB column layout: 13-16 atom name, 31-38 x, 39-46 y, 47-54 z.
            if line[12:16].strip() == 'ORI':
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                out.append([x, y, z])
    return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 3), dtype=np.float32)


def _kabsch_transform(src, dst):
    """Shared Kabsch SVD core: returns ``(R, src_mean, dst_mean)`` such that
    ``aligned = (src - src_mean) @ R.T + dst_mean`` is the least-squares fit
    of src onto dst (proper rotation, no chirality flip via the det term).

    Used by both ``kabsch_align`` and ``add_com_loss`` (detached).
    """
    src_mean = src.mean(dim=0, keepdim=True)
    dst_mean = dst.mean(dim=0, keepdim=True)
    H = (src - src_mean).transpose(-1, -2) @ (dst - dst_mean)
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    d = torch.sign(torch.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2)))
    D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    R = Vh.transpose(-1, -2) @ D @ U.transpose(-1, -2)
    return R, src_mean.squeeze(0), dst_mean.squeeze(0)


def add_com_loss(sample_atom_coords, batch, com_sources, binder_chain='A'):
    """COM loss: binder centroid pulled toward the mean of user-specified ORI
    HETATM atoms (parsed once from --pdb_path / --motif_pdb), each transformed
    from its input PDB frame into the current co-fold frame by a per-source
    Kabsch fit on that source's anchor atoms.

    The fit is computed under ``torch.no_grad()`` and the transformed ORI is
    detached: the gradient flows ONLY through ``binder_com`` (i.e. through the
    binder's predicted CA positions). This is both physically right -- we want
    the binder to MOVE toward where the ORI is in the current target frame,
    not to "nudge the target's prediction so the ORI moves toward the binder"
    -- and numerically right (SVD backward is unstable near degenerate
    singular values; same instability that ``--deterministic_sampler`` avoids
    in the diffusion sampler). Mirrors ``kabsch_align``'s SVD core via the
    shared ``_kabsch_transform`` helper.

    com_sources : list of dicts, each
        {
          'anchor_idx': LongTensor [M]  -- atom indices into sample_atom_coords
                                          for this source's anchor atoms in
                                          the co-fold (predicted) frame.
          'anchor_ref': FloatTensor [M,3]  -- input-frame anchor xyz (constant).
          'ori_ref':    FloatTensor [K,3]  -- input-frame ORI xyz from this PDB.
          'label':      str (for diagnostics).
        }
    Per source: fit (anchor_ref -> anchor_pred), apply to ori_ref to get
    predicted ORI xyz in the co-fold frame; concatenate across sources and
    average. Loss = ||binder_COM - mean_ORI||_2 in Angstrom.

    Returns (loss, binder_com, mean_ori, transformed_oris_concat) -- the extra
    return values let the caller log the achieved distance for diagnostics.
    """
    coords = sample_atom_coords[0]                                       # [N, 3]
    transformed = []
    with torch.no_grad():
        for src in com_sources:
            anchor_pred = coords[src['anchor_idx'], :].detach()          # [M, 3]
            R, s_mean, d_mean = _kabsch_transform(src['anchor_ref'], anchor_pred)
            ori_pred = (src['ori_ref'] - s_mean) @ R.transpose(-1, -2) + d_mean  # [K, 3]
            transformed.append(ori_pred)
    all_oris = torch.cat(transformed, dim=0).detach()                    # [sum K, 3]
    mean_ori = all_oris.mean(0)                                          # [3], constant
    ca = get_ca_coords(sample_atom_coords, batch, binder_chain)[0]       # [N_binder, 3]
    binder_com = ca.mean(0)                                              # [3], differentiable
    loss = torch.norm(binder_com - mean_ori, p=2)
    return loss, binder_com, mean_ori, all_oris


def kabsch_align(pred, ref):
    """Kabsch alignment of pred onto ref (least-squares, proper rotation).

    SVD is run under ``torch.no_grad()`` so R/p_mean/r_mean are treated as
    constants; the linear apply ``(pred - p_mean) @ R.T + r_mean`` is in-graph,
    so ``pred``'s gradient still flows through the affine transform. By the
    envelope theorem this gives the same gradient w.r.t. ``pred`` as a fully
    in-graph SVD would, without the SVD-backward instability (pathological
    near degenerate singular values, same risk that ``--deterministic_sampler``
    works around in the diffusion sampler).
    """
    with torch.no_grad():
        R, p_mean, r_mean = _kabsch_transform(pred, ref)
    return (pred - p_mean) @ R.transpose(-1, -2) + r_mean


def get_motif_coords_loss(sample_atom_coords, bb_pred_idx, bb_ref_xyz,
                          lig_pred_idx=None, lig_ref_xyz=None,
                          return_stats=False):
    """Decoupled-alignment motif RMSD in Å.

    Kabsch alignment is fit from the **protein motif backbone** (N, CA, C, CB)
    alone — typically ~4·M atoms, a well-defined rigid frame. The same rigid
    transform is then applied to predicted **ligand atoms** (if provided), and
    a combined RMSD over backbone + ligand is returned. The decoupled scheme
    is deliberate for enzyme/cofactor scaffolding (e.g. hemoprotein): if a
    ~40-atom heme were folded into the Kabsch fit it would dominate the
    alignment and absorb its own placement error, defeating the point of the
    loss. Holding the transform to the backbone instead makes the ligand RMSD
    report ligand placement *relative to the motif framework*.

    Backward-compat: pass ``lig_pred_idx=lig_ref_xyz=None`` for the
    backbone-only path (byte-identical to the previous loss). Carries a
    gradient through ``sample_atom_coords`` only with ``--attach_coords True``
    (mirrors rg_loss); otherwise inert but still reported.

    sample_atom_coords: [1, n_atoms, 3] from the diffusion sampler.
    bb_pred_idx:        [K] global atom indices (motif backbone N,CA,C,CB).
    bb_ref_xyz:         [K, 3] reference backbone coords (Å), same order.
    lig_pred_idx:       [L] global atom indices (motif ligand heavy atoms).
    lig_ref_xyz:        [L, 3] reference ligand coords (Å), same order.
    return_stats:       if True, also return a dict with per-component RMSDs
                        (``bb_rmsd``, ``lig_rmsd``) for diagnostics.
    """
    pred_bb = sample_atom_coords[0].index_select(0, bb_pred_idx)           # [K,3]
    p_mean = pred_bb.mean(dim=0, keepdim=True)
    r_mean = bb_ref_xyz.mean(dim=0, keepdim=True)
    pc = pred_bb - p_mean
    rc = bb_ref_xyz - r_mean
    H = pc.transpose(-1, -2) @ rc                                          # [3,3]
    U, _S, Vh = torch.linalg.svd(H, full_matrices=False)
    d = torch.sign(torch.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2)))
    D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    R = Vh.transpose(-1, -2) @ D @ U.transpose(-1, -2)                     # [3,3]
    pred_bb_al = pc @ R.transpose(-1, -2) + r_mean                         # [K,3]
    bb_err2 = ((pred_bb_al - bb_ref_xyz) ** 2).sum(dim=-1)                 # [K]

    if lig_pred_idx is None or lig_ref_xyz is None:
        loss = torch.sqrt(bb_err2.mean() + 1e-8)
        if return_stats:
            return loss, {'bb_rmsd': loss.detach().item(), 'lig_rmsd': None}
        return loss

    pred_lig = sample_atom_coords[0].index_select(0, lig_pred_idx)         # [L,3]
    pred_lig_al = (pred_lig - p_mean) @ R.transpose(-1, -2) + r_mean       # [L,3]
    lig_err2 = ((pred_lig_al - lig_ref_xyz) ** 2).sum(dim=-1)              # [L]
    all_err2 = torch.cat([bb_err2, lig_err2], dim=0)
    loss = torch.sqrt(all_err2.mean() + 1e-8)
    if return_stats:
        return loss, {
            'bb_rmsd': torch.sqrt(bb_err2.mean() + 1e-8).detach().item(),
            'lig_rmsd': torch.sqrt(lig_err2.mean() + 1e-8).detach().item(),
        }
    return loss


def _rigid_frames(n_xyz, ca_xyz, c_xyz, eps=1e-8):
    """AlphaFold-style rigid frames (Alg. 21) from backbone N, CA, C.

    Returns R [F, 3, 3] with columns (e1, e2, e3) and origin t = CA [F, 3]; the
    local coordinate of a point x in frame f is R_f^T (x - t_f).
    """
    v1 = c_xyz - ca_xyz
    v2 = n_xyz - ca_xyz
    e1 = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
    u2 = v2 - e1 * (e1 * v2).sum(dim=-1, keepdim=True)
    e2 = u2 / (u2.norm(dim=-1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)                                 # [F,3,3]
    return R, ca_xyz


def get_motif_fape_loss(sample_atom_coords, frame_pred_idx, sc_pred_idx, loc_ref,
                        clamp=10.0):
    """Frame-based (FAPE-style) sidechain error for the motif, in Å.

    A backbone frame (N, CA, C) is built for each motif residue; every motif
    sidechain heavy atom is expressed in *every* motif frame and compared, in
    local-frame coordinates, to the reference (`loc_ref`, precomputed once since
    the reference is constant). This is the AlphaFold FAPE restricted to motif
    residues: invariant to the global pose, sensitive to relative sidechain
    placement (e.g. catalytic-triad geometry), clamped at `clamp` Å, averaged
    over (frame, atom) pairs.

    Note: during the design loop the binder is built as UNK, so the only shared
    sidechain heavy atoms are typically CB and CG; full rotamers are not present
    unless the binder is built with the reference residue types. Gradient through
    coords only with `--attach_coords True`.

    frame_pred_idx: [F, 3] global atom indices (N, CA, C) per motif frame.
    sc_pred_idx:    [S]    global atom indices of motif sidechain atoms.
    loc_ref:        [F, S, 3] reference local-frame coords (constant).
    """
    coords = sample_atom_coords[0]
    R, t = _rigid_frames(coords[frame_pred_idx[:, 0]],
                         coords[frame_pred_idx[:, 1]],
                         coords[frame_pred_idx[:, 2]])                     # [F,3,3],[F,3]
    x = coords.index_select(0, sc_pred_idx)                               # [S,3]
    diff = x[None, :, :] - t[:, None, :]                                  # [F,S,3]
    loc_pred = torch.einsum('fij,fsj->fsi', R.transpose(-1, -2), diff)    # [F,S,3]
    d = torch.sqrt(((loc_pred - loc_ref) ** 2).sum(dim=-1) + 1e-8)        # [F,S]
    return torch.clamp(d, max=clamp).mean()


# ---- Explicit atom/residue distance restraints (--atom_pairs) ---------------
# A loss term, independent of the general binder-target contact loss, that
# pushes the distance between two *specified* points toward a window [lo,hi].
# The pair may span ANY two chains -- including target<->target (e.g. holding a
# cofactor near a particular target-protein residue). Two flavors, wired like
# the motif losses:
#   * distogram (token-level, fast-mode safe): -log P(dist in [lo,hi]) from the
#     trunk distogram. Ligand atoms address an exact token; polymer residues
#     resolve to their Cb token.
#   * coords (atom-level, full mode): flat-bottom restraint on sample_atom_coords
#     for any named atom on either side. Gradient requires --attach_coords True.
def _decode_atom_name(name_arr):
    """Decode a boltz Atom['name'] (4x int8, char == value + 32) to a string."""
    return ''.join(chr(int(x) + 32) for x in name_arr if int(x) != 0).strip()


def _parse_endpoint(ep):
    """'C:FE' | 'B:145' | 'B:145@NE2' -> (chain, selector, atom_override)."""
    if ':' not in ep:
        raise ValueError(f"atom-pair endpoint '{ep}' must be 'CHAIN:SELECTOR'")
    chain, sel = ep.split(':', 1)
    atom_override = None
    if '@' in sel:
        sel, atom_override = sel.split('@', 1)
        atom_override = atom_override.strip()
    return chain.strip(), sel.strip(), atom_override


def parse_atom_pairs_spec(spec):
    """Parse the --atom_pairs string into raw endpoint dicts (no batch needed).

    Grammar (per ';'-separated chunk):
       4 fields -- "<epA>, <epB>, <lo>, <hi>"   window restraint
       3 fields -- "<epA>, <epB>, <d_ref>"      point-target restraint (lo=hi=d_ref)

    Each endpoint is CHAIN:SEL[@ATOM]; SEL is a 1-indexed residue position for
    polymer chains or an atom name for ligand chains, and @ATOM optionally pins
    a named atom (used by the coord loss; the distogram loss is token/Cb-level).
    Distances are in Angstrom.

    The point-target form is what the 'expected' distogram method turns into the
    literal SwitchCraft motif loss sum_k p_k (d_k - d_ref)^2 (which decomposes
    as variance + bias^2 about d_ref, so it both pulls the expected distance to
    d_ref AND drives the prediction toward a sharp delta there); the window
    form stays a flat-bottom on the expected distance E[d].
    """
    pairs = []
    for chunk in spec.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        fields = [f.strip() for f in chunk.split(',')]
        if len(fields) == 4:
            epA, epB, lo, hi = fields
            lo, hi = float(lo), float(hi)
            if hi < lo:
                raise ValueError(f"atom-pair '{chunk}': hi ({hi}) < lo ({lo})")
        elif len(fields) == 3:
            epA, epB, d_ref = fields
            lo = hi = float(d_ref)
        else:
            raise ValueError(
                f"atom-pair '{chunk}' must have 3 fields 'epA, epB, d_ref' "
                f"(point target) or 4 fields 'epA, epB, lo, hi' (window)")
        cA, sA, aA = _parse_endpoint(epA)
        cB, sB, aB = _parse_endpoint(epB)
        pairs.append({'chainA': cA, 'selA': sA, 'atomA': aA,
                      'chainB': cB, 'selB': sB, 'atomB': aB,
                      'lo': lo, 'hi': hi})
    if not pairs:
        raise ValueError("--atom_pairs is set but no valid pairs were parsed")
    return pairs


def _find_named_atom(structure, a0, anum, name, chain, sel):
    for i in range(anum):
        if _decode_atom_name(structure.atoms['name'][a0 + i]) == name:
            return a0 + i
    where = f"chain {chain}" + (f" residue {sel}" if sel else "")
    raise ValueError(f"atom '{name}' not found in {where}")


def _resolve_endpoint_indices(chain, sel, atom_override, entity_id, structure):
    """Map one endpoint to (token_idx, atom_idx, label_name).

    token_idx indexes the distogram axis; atom_idx indexes sample_atom_coords
    (== global structure.atoms order). Polymer SEL = 1-indexed residue position
    (token = that residue; distogram atom = Cb via atom_disto; atom_override
    picks a specific atom for the coord loss). Ligand SEL = atom name (token ==
    that atom; ligands are tokenized one-token-per-atom, in order).
    """
    if chain not in chain_to_number:
        raise ValueError(f"unknown chain '{chain}' in atom-pair spec")
    chain_tokens = torch.where(entity_id == chain_to_number[chain])[0]
    if chain_tokens.numel() == 0:
        raise ValueError(f"chain '{chain}' has no tokens in the batch")
    crec = next((c for c in structure.chains if c['name'] == chain), None)
    if crec is None:
        raise ValueError(f"chain '{chain}' not found in structure")
    a0, anum = int(crec['atom_idx']), int(crec['atom_num'])

    if sel.isdigit():                                    # polymer residue
        pos = int(sel) - 1
        if not (0 <= pos < chain_tokens.numel()):
            raise ValueError(
                f"chain {chain} residue position {sel} out of range "
                f"(1..{chain_tokens.numel()})")
        tok = int(chain_tokens[pos])
        res = structure.residues[int(crec['res_idx']) + pos]
        if atom_override:
            atom_idx = _find_named_atom(structure, int(res['atom_idx']),
                                        int(res['atom_num']), atom_override, chain, sel)
        else:
            atom_idx = int(res['atom_disto'])            # Cb (matches distogram)
        return tok, atom_idx, str(res['name'])
    # ligand atom name -> its token (one token per atom, in atom order)
    atom_idx = _find_named_atom(structure, a0, anum, sel, chain, None)
    local = atom_idx - a0
    if not (0 <= local < chain_tokens.numel()):
        raise ValueError(f"atom {sel} in chain {chain} maps outside token range")
    return int(chain_tokens[local]), atom_idx, sel


def resolve_atom_pairs(spec, batch, structure):
    """Parse + resolve --atom_pairs against a built batch/structure.

    Returns dicts with integer indices ready for the loss: tok_a/tok_b on the
    distogram token axis, atom_a/atom_b on the sample_atom_coords atom axis,
    plus lo/hi (A) and a human-readable label. Resolved once per run -- the
    binder length is fixed, so the target token/atom indices are stable.
    """
    entity_id = batch['entity_id']
    if entity_id.dim() > 1:
        entity_id = entity_id[0]
    resolved = []
    for p in parse_atom_pairs_spec(spec):
        ta, aa, na = _resolve_endpoint_indices(p['chainA'], p['selA'], p['atomA'],
                                               entity_id, structure)
        tb, ab, nb = _resolve_endpoint_indices(p['chainB'], p['selB'], p['atomB'],
                                               entity_id, structure)
        labA = f"{p['chainA']}:{p['selA']}{('@'+p['atomA']) if p['atomA'] else ''}({na})"
        labB = f"{p['chainB']}:{p['selB']}{('@'+p['atomB']) if p['atomB'] else ''}({nb})"
        resolved.append({'tok_a': ta, 'tok_b': tb, 'atom_a': aa, 'atom_b': ab,
                         'lo': p['lo'], 'hi': p['hi'],
                         'label': f"{labA} - {labB} [{p['lo']},{p['hi']}]A"})
    return resolved


def get_atom_pair_distogram_loss(pdistogram, pair_specs, mid_pts,
                                 method='expected', return_stats=False):
    """Distogram-based distance restraint over the --atom_pairs token pairs.
    Three selectable formulations (`method=`):

      'expected' (default) -- bin-center MSE on the predicted distribution:
                    POINT TARGET (lo == hi == d_ref, set by the 3-field spec
                    "epA, epB, d_ref"):  sum_k p_k * (mid_pts_k - d_ref)^2
                       -- the literal SwitchCraft motif loss (Eq 1). Equals
                       Var(d_k) + (E[d_k] - d_ref)^2, so it pulls the expected
                       distance to d_ref AND drives the prediction toward a sharp
                       delta at d_ref. Always > 0 unless the prediction is a
                       perfect delta at d_ref.
                    WINDOW (lo < hi, 4-field spec "epA, epB, lo, hi"):
                       relu(lo - E[d])^2 + relu(E[d] - hi)^2 on the expected
                       distance E[d] = sum(prob * mid_pts). Flat-bottom: zero
                       once E[d] in [lo, hi]; only the mean is penalized
                       (variance is invisible). Force grows with distance inside
                       the resolved range, but the gradient saturates at the
                       far end (see ceiling note).
      'prob'     -- -log P(d in [lo,hi]): the in-window probability MASS read
                    from the symmetrized token-pair distribution. Can be driven
                    down by parking a little mass in-window while the bulk of
                    the distribution -- and hence the sampled structure -- stays
                    far away; its gradient also collapses by ~20 A (the eps clamp
                    + softmax saturation), so it stops pulling exactly when you
                    need it to.
      'contact'  -- the same objective BoltzDesign1 uses to FORM binder-target
                    contacts: ColabDesign's categorical contact cross-entropy
                    (_get_con_loss, binary=False) with the contact cutoff = hi.
                    A -log-type loss whose gradient is large when the pair is far
                    and tapers as probability moves within hi (i.e. large far,
                    small near -- a robust monotonic "pull them together" force).
                    One-sided: lo is ignored. Use when the pair starts far apart
                    and you want it driven into contact.

    Averaged over pairs. Fast-mode safe. Token resolution: ligand atoms exact,
    polymer residues at Cb.

    CEILING (applies to ALL three methods): the distogram spans ~2-22 A with one
    catch-all last bin (center ~24.5 A), so it cannot distinguish e.g. 30 A from
    100 A. Once the trunk is confident the pair is past ~24 A the softmax is a
    near-delta on that bin and the gradient of any distogram loss vanishes there.
    To pull a pair that is *genuinely* far with a force that keeps growing with
    distance, use the COORD restraint (get_atom_pair_coords_loss) in full mode
    (--distogram_only False --attach_coords True): ||x_a - x_b|| is unbounded.

    return_stats=True also returns per-pair diagnostics (detached): exp_dist,
    mode_dist, p_window and the per-pair loss -- so callers can log the exact
    distance regardless of which method drives the gradient."""
    p = pdistogram[0]                                     # [L, L, 64]
    terms, stats = [], []
    for s in pair_specs:
        logits = (p[s['tok_a'], s['tok_b']] + p[s['tok_b'], s['tok_a']]) / 2  # symmetric
        prob = torch.softmax(logits, dim=-1)
        exp_d = (prob * mid_pts).sum()                    # expected distance (A)
        win = ((mid_pts >= s['lo']) & (mid_pts <= s['hi'])).to(prob.dtype)
        p_window = (prob * win).sum()
        if method == 'expected':
            if s['lo'] == s['hi']:                        # point target -> SwitchCraft Eq 1
                # sum_k p_k * (mid_pts_k - d_ref)^2 = Var(d_k) + (E[d_k] - d_ref)^2
                # (penalizes both prediction spread AND mean-distance bias).
                d_ref = s['lo']
                term = (prob * (mid_pts - d_ref).pow(2)).sum()
            else:                                         # window -> flat-bottom on E[d]
                term = torch.relu(s['lo'] - exp_d) ** 2 + torch.relu(exp_d - s['hi']) ** 2
        elif method == 'prob':
            term = -torch.log(p_window + 1e-8)
        elif method == 'contact':
            # Reuse BoltzDesign1's proven contact loss (ColabDesign categorical
            # cross-entropy) on this single token pair, contact cutoff = hi. Gives
            # a robust attractive gradient that is large when the pair is far and
            # tapers as it approaches the window (lo ignored). Same numerically
            # stable log_softmax path used for binder-target contacts; still
            # subject to the distogram ceiling (see docstring).
            term = _get_con_loss(logits, mid_pts, cutoff=s['hi'], binary=False)
        else:
            raise ValueError(
                f"unknown atom_pair_distogram_loss_type '{method}' "
                "(expected 'prob', 'expected', or 'contact')")
        terms.append(term)
        if return_stats:
            with torch.no_grad():
                stats.append({
                    'label': s['label'],
                    'exp_dist': exp_d.item(),
                    'mode_dist': mid_pts[torch.argmax(prob)].item(),
                    'p_window': p_window.item(),
                    'loss': term.item(),
                    'lo': float(s['lo']), 'hi': float(s['hi']),
                })
    loss = torch.stack(terms).mean()
    return (loss, stats) if return_stats else loss


def get_atom_pair_coords_loss(sample_atom_coords, pair_specs, return_stats=False):
    """Flat-bottom distance restraint on the sampled coords: zero inside [lo,hi],
    squared violation outside, on the true Euclidean distance ||x_a - x_b||
    (the only sensible computation for real coords -- no 'method' choice). Atom-
    level on both sides (any named atom). Full mode only; carries a gradient only
    with --attach_coords True (mirrors rg_loss / motif_coords_loss).

    return_stats=True also returns per-pair diagnostics (detached) reusing the
    computed distance, so the logged coord distance is exactly the optimized one."""
    coords = sample_atom_coords[0]                        # [num_atoms, 3]
    terms, stats = [], []
    for s in pair_specs:
        d = torch.linalg.norm(coords[s['atom_a']] - coords[s['atom_b']])
        term = torch.relu(s['lo'] - d) ** 2 + torch.relu(d - s['hi']) ** 2
        terms.append(term)
        if return_stats:
            stats.append({
                'label': s['label'],
                'dist': d.item(),
                'loss': term.item(),
                'lo': float(s['lo']), 'hi': float(s['hi']),
            })
    loss = torch.stack(terms).mean()
    return (loss, stats) if return_stats else loss


def get_boltz_model(checkpoint: Optional[str] = None, predict_args=None, device: Optional[str] = None, use_heun: bool = False, attach_coords: bool = False, step_scale: float = 1.638, deterministic_sampler: bool = False, gamma_0: float = 0.605) -> Boltz1:
    torch.set_grad_enabled(True)
    torch.set_float32_matmul_precision("highest")
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale  # Sampler over-relaxation / step length (eta); 1.0 = velocity-consistent ODE step
    diffusion_params.gamma_0 = gamma_0  # EDM churn; 0.0 = noise-free ODE sampler for stable few-step inference
    diffusion_params.use_heun = use_heun
    diffusion_params.deterministic_sampler = deterministic_sampler  # Aug/SVD/churn-free sampler for backprop
    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=False,
        predict_args=predict_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        structure_prediction_training=True,
        no_msa=False,
        no_atom_encoder=False,
    )
    # Toggle off the sampler's no_grad wrapper. Paired with disconnect_coords=False
    # in confidence_args so gradients reach the confidence head through coords.
    model_module.structure_module.attach_coords = attach_coords
    return model_module



def boltz_hallucination(
    # Required arguments
    boltz_model,
    yaml_path,
    ccd_lib,
    length=100,
    binder_chain='A',
    design_algorithm="3stages",
    recycling_steps=0,
    sampling_steps=200,
    pre_iteration=20,
    soft_iteration=50, 
    soft_iteration_1=50,
    soft_iteration_2=25,
    temp_iteration=50,
    hard_iteration=10,
    semi_greedy_steps=0,
    learning_rate=0.1,
    learning_rate_pre=0.1,
    # Per-target lists (one entry per non-binder chain, in target_chain_ids
    # order). Each defaults to the legacy single-scalar behavior when only one
    # value is provided. The design loop builds per-target masks and applies
    # these settings independently per target inside get_model_loss.
    inter_chain_cutoff=(21.0,),
    intra_chain_cutoff=14.0,
    num_inter_contacts=(2,),
    num_intra_contacts=4,
    # Per-target inter-chain loss weights. Default 1.0/0.1 each reproduces the
    # prior global loss_scales values. Setting a per-target weight to 0
    # disables that target's term cheaply (just a multiply, no skipped masks).
    i_con_loss_weights=(1.0,),
    i_pae_loss_weights=(0.1,),
    # Per-target pLDDT loss weights. Default 0.0 = off (replaces the removed
    # target_plddt_chains gate). Setting a per-target weight to non-zero opts
    # that target in to a separately-weighted pLDDT term over its tokens.
    target_plddt_loss_weights=(0.0,),
    # Per-target chain IDs in target_types order (skipping binder_chain). Used
    # to build per-target token masks against batch['entity_id']. Defaults to a
    # single-target shim that auto-resolves from entity_id at setup time.
    target_chain_ids=None,
    e_soft=0.8,
    e_soft_1=0.8,
    e_soft_2=1.0,
    alpha=2.0,
    pre_run=False,
    set_train=True,
    use_temp=False,
    disconnect_feats=False,
    disconnect_pairformer=False,
    attach_coords=False,
    # Per-target boolean: when True (for a given target), that target's tokens
    # are masked during the warm-up (pre_iteration) stage so the binder evolves
    # topology before seeing it. Renamed from mask_ligand. Default (False,)
    # reproduces the legacy "don't mask anything" behavior.
    mask_target_prerun=(False,),
    distogram_only=False,
    input_res_type=False,
    non_protein_target=False,
    increasing_contact_over_itr=False,
    loss_scales=None,
    # Per-target boolean: when True, every binder position must reach
    # num_inter_contacts to that target; when False the aggregate count is used.
    # Default (False,) reproduces the legacy global behavior.
    optimize_contact_per_binder_pos=(False,),
    pocket_conditioning=False,
    chain_to_number=None,
    msa_max_seqs=4096,
    optimizer_type='SGD',
    save_trajectory=False,
    noise_scaling=0.1,
    # Motif scaffolding (ColabDesign `partial` protocol port). Inactive unless
    # motif_pdb is set, so non-motif runs are byte-for-byte unchanged.
    # motif_residues uses chain-prefixed tokens (e.g. "A57,A102,B195"), so a
    # single motif can span multiple chains in the motif PDB.
    motif_pdb=None,
    motif_residues=None,
    motif_binder_positions=None,
    fix_motif_seq=True,
    # Sliding-window motif scaffolding (--motif_unindex_residues). Islands of
    # fixed intra-island gap structure (e.g. "A38,3,A42,A170" = island1 [A38,
    # A42 separated by 3 free residues], island2 [A170]) whose absolute binder
    # positions are sampled per-epoch by MCMC with simulated annealing. Energy
    # = backbone Kabsch RMSD (joint with fixed motif residues if any); accept
    # by Metropolis with T decaying T_init -> T_final exponentially over the
    # whole design loop. Coexists with --motif_residues: fixed binder positions
    # are excluded from sliding placements. fix_motif_seq still applies -- the
    # pin moves with the MCMC state.
    motif_unindex_residues='',
    motif_slide_steps_per_epoch=10,
    motif_slide_T_init=5.0,
    motif_slide_T_final=0.1,
    motif_slide_swap_prob=0.2,
    motif_slide_shift_max=10,
    # Optional ligand carry-along: ligand residues in the motif PDB (chain-
    # prefixed: 'B1' = chain B residue 1, 'B1,C401' for multiple) whose heavy
    # atoms are pulled along by the rigid transform Kabsch-fit on the protein
    # motif backbone. The combined backbone + ligand RMSD is reported under
    # `motif_coords_loss`. Independent of --motif_binder_positions: ligand
    # residues do NOT consume binder slots -- they live in their own target
    # chain in the designed system (typically added via --target_mols).
    # Default '' = no ligand carry-along; backward-compat is byte-identical.
    motif_ligand_residues='',
    # Explicit atom/residue distance restraints (--atom_pairs). Inactive unless
    # set, so existing runs are byte-for-byte unchanged. Weights come via
    # loss_scales['atom_pair_distogram_loss'/'atom_pair_coords_loss'].
    atom_pairs='',
    # How the --atom_pairs distogram restraint is computed: 'expected' (default;
    # bin-center MSE -- SwitchCraft Eq 1 for a point target, flat-bottom on E[d]
    # for a window), 'prob' (-log P(d in window)), or 'contact' (BoltzDesign1's
    # categorical contact loss; robust attractive gradient, large when far /
    # tapering near). See get_atom_pair_distogram_loss.
    atom_pair_distogram_loss_type='expected',
    # COM loss inputs. Inactive unless an HETATM ORI atom is found in --pdb_path
    # or --motif_pdb. pdb_path: original target PDB whose target chains seed
    # the Kabsch anchor (all non-binder atoms in `structure`). motif_pdb is
    # already received above; its ORI uses the motif backbone as anchor.
    # com_loss_weight: scalar, 0 disables.
    pdb_path='',
    com_loss_weight=0.0,
    # Per-epoch intermediate folded-structure dump. Only fires on full-mode
    # forwards (the diffusion sampler must run, i.e. distogram_only=False or
    # save_trajectory=True). intermediate_dir is set by the caller.
    save_intermediate_structures=False,
    intermediate_dir=None,
):

    predict_args = {
        "recycling_steps": recycling_steps,  # Default value
        "sampling_steps": sampling_steps,  # Total diffusion step count
        "diffusion_samples": 1,  # Default value
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    boltz_model.predict_args = predict_args

    with yaml_path.open("r") as file:
        data = yaml.safe_load(file)

    data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = 'X'*length
    name = yaml_path.stem
    target = parse_boltz_schema(name, data, ccd_lib)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    boltz_model.train() if set_train else boltz_model.eval()
    print(f"set in {'train' if set_train else 'eval'} mode")

    def get_batch(target, max_seqs=0, length=100, pocket_conditioning=False, keep_record=False):
        target_id = target.record.id
        structure = target.structure

        structure = Structure(
                atoms=structure.atoms,
                bonds=structure.bonds,
                residues=structure.residues,
                chains=structure.chains,
                connections=structure.connections.astype(Connection),
                interfaces=structure.interfaces,
                mask=structure.mask,
            )

        msas = {}
        for chain in target.record.chains:
            msa_id = chain.msa_id
            if msa_id != -1:
                msa = np.load(msa_id)
                msas[chain.chain_id] = MSA(**msa)

        input = Input(structure, msas) 
        
        tokenizer = BoltzTokenizer()
        tokenized = tokenizer.tokenize(input)
        featurizer = BoltzFeaturizer()

        if pocket_conditioning:
            options = target.record.inference_options
            binders, pocket = options.binders, options.pocket  
            batch = featurizer.process(
                        tokenized,
                        training=False,
                        max_atoms=None,
                        max_tokens=None,
                        max_seqs=max_seqs,
                        pad_to_max_seqs=False,
                        symmetries={},
                        compute_symmetries=False,
                        inference_binder=binders,
                        inference_pocket=pocket,
                    )
        else:
            batch = featurizer.process(
                        tokenized,
                        training=False,
                        max_atoms=None,
                        max_tokens=None,
                        max_seqs=max_seqs,
                        pad_to_max_seqs=False,
                        symmetries={},
                        compute_symmetries=False,
                        inference_binder=None,
                        inference_pocket=None,
                    )

        if keep_record:
            batch['record'] = target.record

        return batch, structure
    
    batch, structure = get_batch(target, max_seqs=msa_max_seqs, length=length, pocket_conditioning=pocket_conditioning)
    batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}
    
    ## initialize res_type_logits
    if pre_run:
        batch['res_type_logits'] = batch['res_type'].clone().detach().to(device).float()
        batch['res_type_logits'][batch['entity_id']==chain_to_number[binder_chain],:] = noise_scaling*torch.softmax(torch.distributions.Gumbel(0, 1).sample(batch['res_type'][batch['entity_id']==chain_to_number[binder_chain],:].shape).to(device) - torch.sum(torch.eye(batch['res_type'].shape[-1])[[0,1,6,22,23,24,25,26,27,28,29,30,31,32]],dim=0).to(device)*(1e10), dim=-1)
    else:
        batch['res_type_logits'] = torch.from_numpy(input_res_type).to(device)

    if  non_protein_target:
        batch['msa'] = batch['res_type_logits'].unsqueeze(0).to(device)
        batch['msa_paired'] = torch.ones(batch['res_type'].shape[0], 1, batch['res_type'].shape[1]).to(device)
        batch['deletion_value'] = torch.zeros(batch['res_type'].shape[0], 1, batch['res_type'].shape[1]).to(device)
        batch['has_deletion'] = torch.full((batch['res_type'].shape[0], 1, batch['res_type'].shape[1]), False).to(device)  
        batch['msa_mask'] = torch.ones(batch['res_type'].shape[0], 1, batch['res_type'].shape[1]).to(device)
        batch['profile'] = batch['msa'].float().mean(dim=0).to(device)
        batch['deletion_mean'] = torch.zeros(batch['deletion_mean'].shape).to(device)
        batch['res_type'] = batch['res_type'].float()

    batch['res_type_logits'].requires_grad = True
    optimizer = torch.optim.AdamW([batch['res_type_logits']], lr=learning_rate_pre if pre_run else learning_rate) if optimizer_type == 'AdamW' else torch.optim.SGD([batch['res_type_logits']], lr=learning_rate_pre if pre_run else learning_rate)

    def norm_seq_grad(grad, chain_mask):
        chain_mask = chain_mask.bool()
        masked_grad = grad[:, chain_mask.squeeze(0), :] 
        eff_L = (masked_grad.pow(2).sum(-1, keepdim=True) > 0).sum(-2, keepdim=True)
        gn = masked_grad.norm(dim=(-1, -2), keepdim=True) 
        return grad * torch.sqrt(torch.tensor(eff_L)) / (gn + 1e-7)

    alphabet = list('XXARNDCQEGHILKMFPSTWYV-')
    best_loss = float('inf')  
    min_loss = float('inf') 
    best_batch = None     
    first_step_best_batch=None

    plots = []
    distogram_history = []
    sequence_history = []
    loss_history = []
    lr_history = []
    con_loss_history = []
    i_con_loss_history = []
    plddt_loss_history = []
    # Per-loss-component history, keyed by the name in `losses` (helix_loss,
    # plddt_loss, i_pae_loss, pae_loss, rg_loss, ...). Accessed via closure
    # inside get_model_loss (mutated only, never rebound -> no `nonlocal`
    # needed) so we don't have to thread a new arg through every design(...)
    # call. Full-mode-only keys (plddt/pae/i_pae/rg) are appended only on
    # full-mode epochs, so their series can be shorter than helix_loss/con_loss.
    loss_component_history = {}
    # Per-epoch full pLDDT vector ([N tokens]) and symmetric PAE matrix
    # ([N, N], A) from the confidence head. Same closure pattern as above;
    # appended only on full-mode epochs (distogram_only=False), so these are
    # empty for pure fast-mode runs and the corresponding animations are skipped.
    plddt_history = []
    pae_history = []

    mask = torch.ones_like(batch['res_type_logits'])
    mask[batch['entity_id']!=chain_to_number[binder_chain], :] = 0
    chain_mask = (batch['entity_id'] == chain_to_number[binder_chain]).int()
    mid_points = torch.linspace(2, 22, 64).to(device)

    # ---- Per-target masks and per-target settings -------------------------
    # target_chain_ids is supplied by the caller (boltzdesign.py) in
    # --target_types order; if absent (legacy single-target call), derive it
    # from the chains present in entity_id that aren't the binder.
    if target_chain_ids is None:
        present_idx = sorted(int(x) for x in torch.unique(batch['entity_id']).tolist()
                             if int(x) != chain_to_number[binder_chain])
        _num_to_chain = {v: k for k, v in chain_to_number.items()}
        target_chain_ids = [_num_to_chain[i] for i in present_idx]
    n_targets = len(target_chain_ids)

    def _as_per_target(values, name, n):
        """Validate / broadcast a length-1 or length-n list to length n."""
        v = list(values) if values is not None else []
        if len(v) == 1 and n > 1:
            return v * n
        if len(v) != n:
            raise ValueError(
                f"{name} has {len(v)} entries but {n} target chain(s) are "
                f"present ({target_chain_ids}); provide either 1 or {n} values.")
        return v

    inter_chain_cutoff_pt = _as_per_target(inter_chain_cutoff, 'inter_chain_cutoff', n_targets)
    num_inter_contacts_pt = _as_per_target(num_inter_contacts, 'num_inter_contacts', n_targets)
    optimize_contact_per_binder_pos_pt = _as_per_target(
        optimize_contact_per_binder_pos, 'optimize_contact_per_binder_pos', n_targets)
    mask_target_prerun_pt = _as_per_target(mask_target_prerun, 'mask_target_prerun', n_targets)
    i_con_w = _as_per_target(i_con_loss_weights, 'i_con_loss', n_targets)
    i_pae_w = _as_per_target(i_pae_loss_weights, 'i_pae_loss', n_targets)
    tplddt_w = _as_per_target(target_plddt_loss_weights, 'target_plddt_loss', n_targets)

    # Per-target token masks, aligned to target_chain_ids.
    target_masks = []
    for c in target_chain_ids:
        if c not in chain_to_number:
            raise ValueError(f"unknown chain '{c}' in target_chain_ids")
        target_masks.append((batch['entity_id'] == chain_to_number[c]).int())
    # Pooled non-binder mask, used for binder-internal PAE term (binder ↔ binder).
    pooled_target_mask = (1 - chain_mask)

    # Useful diagnostic banner.
    print(f"[per-target] {n_targets} target chain(s): {target_chain_ids}")
    print(f"[per-target] i_con_loss weights : {i_con_w}")
    print(f"[per-target] i_pae_loss weights : {i_pae_w}")
    print(f"[per-target] target_plddt weights: {tplddt_w}")
    if any(w > 0 for w in tplddt_w) and distogram_only:
        print("[per-target] WARNING: target_plddt_loss needs the confidence head; "
              "it is INERT in fast mode. Pass --distogram_only False for it to "
              "take effect.")

    # ---- Motif scaffolding setup (ColabDesign `partial` protocol) ----------
    # Builds the static + dynamic state for two coexisting motif modes:
    #   * fixed  (--motif_residues + --motif_binder_positions): static binder
    #     positions known up-front
    #   * sliding (--motif_unindex_residues): per-island starts sampled by
    #     MCMC + simulated annealing each epoch
    # Both contribute to the SAME `motif_token_idx`/`motif_bb_pred_idx`/
    # `motif_target_dist`/sequence-pin tensors (fixed entries first, then
    # sliding in declaration order). Sliding entries are *dynamic* -- a helper
    # rebuilds them from `slide_state` after every MCMC step.
    motif_active = bool(motif_pdb)
    motif_fixed_active = bool(motif_residues)
    motif_slide_active = bool(motif_unindex_residues)
    if motif_active and not (motif_fixed_active or motif_slide_active):
        raise ValueError(
            "--motif_pdb is set but neither --motif_residues nor "
            "--motif_unindex_residues was provided")
    if (motif_fixed_active or motif_slide_active) and not motif_active:
        raise ValueError(
            "--motif_residues / --motif_unindex_residues require --motif_pdb")
    motif_token_idx = None        # LongTensor of token-axis indices (the binder
                                  # residues that hold the motif). Rebuilt per
                                  # epoch when sliding is active.
    motif_target_dist = None      # [M, M] raw reference pairwise distances (A)
    motif_onehot = None           # [M, V] hard res_type for sequence pinning
    if motif_active:
        # ----- Shared binder lookup helpers (used by both modes) -----
        binder_tok = torch.where(
            batch['entity_id'][0] == chain_to_number[binder_chain])[0]
        _bcrec = next((c for c in structure.chains if c['name'] == binder_chain), None)
        if _bcrec is None:
            raise ValueError(f"binder chain {binder_chain!r} not found in structure")
        _bres0 = int(_bcrec['res_idx'])

        def _pred_atom_idx(local_pos, name):
            pres = structure.residues[_bres0 + int(local_pos)]
            try:
                return _find_named_atom(structure, int(pres['atom_idx']),
                                        int(pres['atom_num']), name,
                                        binder_chain, local_pos + 1)
            except ValueError:
                return None

        _BB = ['N', 'CA', 'C', 'CB']
        _SC_SKIP = {'N', 'CA', 'C', 'O', 'OXT'}
        V = batch['res_type'].shape[-1]
        N = batch['res_type'].shape[1]

        # ----- Fixed-mode setup (existing behavior) -----
        m_chain_residues = []
        fixed_motif_seq = ''
        fixed_ref_xyz = np.zeros((0, 3), dtype=np.float32)
        fixed_bb_pred_list = []
        fixed_bb_ref_list = []
        fixed_token_idx_list = []
        fixed_positions = set()
        binder_pos = []  # legacy name -- exposed for print + ligand check
        M_fix = 0
        # FAPE tensors -- only active in fixed-only mode (sliding mode forces
        # FAPE off because frames per residue per epoch is a different
        # vectorization story; the user can run a final fix-position phase).
        motif_fape_active = False
        motif_frame_pred_idx = motif_fape_sc_pred_idx = motif_fape_loc_ref = None
        if motif_fixed_active:
            m_chain_residues = parse_motif_residue_spec(motif_residues)
            if not m_chain_residues:
                raise ValueError("--motif_residues set but parsed to no entries")
            fixed_ref_xyz, fixed_motif_seq = extract_motif_coords(motif_pdb, m_chain_residues)
            M_fix = fixed_ref_xyz.shape[0]
            if motif_binder_positions:
                binder_pos = parse_residue_spec(motif_binder_positions)
                binder_pos = [p - 1 for p in binder_pos]
                if len(binder_pos) != M_fix:
                    raise ValueError(
                        f"--motif_binder_positions has {len(binder_pos)} entries "
                        f"but --motif_residues resolves to {M_fix} residues")
            else:
                binder_pos = list(range(M_fix))
            if max(binder_pos) >= length:
                raise ValueError(
                    f"motif binder position {max(binder_pos)+1} exceeds binder "
                    f"length {length}; raise --length_min/--length_max")
            fixed_positions = set(binder_pos)
            fixed_token_idx_list = [int(binder_tok[p]) for p in binder_pos]

            _ref_res_atoms = extract_motif_residue_atoms(motif_pdb, m_chain_residues)
            _fr_pred, _fr_ref, _sc_pred, _sc_ref = [], [], [], []
            for _k, _pos in enumerate(binder_pos):
                _ratoms = _ref_res_atoms[_k]['atoms']
                for _nm in _BB:
                    _pi = _pred_atom_idx(_pos, _nm)
                    if _pi is not None and _nm in _ratoms:
                        fixed_bb_pred_list.append(_pi)
                        fixed_bb_ref_list.append(_ratoms[_nm])
                if not motif_slide_active:
                    _pN, _pCA, _pC = (_pred_atom_idx(_pos, _n) for _n in ('N', 'CA', 'C'))
                    if (None not in (_pN, _pCA, _pC)
                            and all(_n in _ratoms for _n in ('N', 'CA', 'C'))):
                        _fr_pred.append([_pN, _pCA, _pC])
                        _fr_ref.append([_ratoms['N'], _ratoms['CA'], _ratoms['C']])
                    for _nm, _rc in _ratoms.items():
                        if _nm in _SC_SKIP:
                            continue
                        _pi = _pred_atom_idx(_pos, _nm)
                        if _pi is not None:
                            _sc_pred.append(_pi)
                            _sc_ref.append(_rc)
            if not fixed_bb_pred_list:
                raise ValueError("motif: no fixed-mode backbone atoms (N/CA/C/CB) could be resolved")
            if not motif_slide_active:
                motif_fape_active = bool(_fr_pred) and bool(_sc_pred)
                if motif_fape_active:
                    motif_frame_pred_idx = torch.as_tensor(_fr_pred, dtype=torch.long, device=device)
                    motif_fape_sc_pred_idx = torch.as_tensor(_sc_pred, dtype=torch.long, device=device)
                    _fr_ref_t = torch.as_tensor(np.asarray(_fr_ref), dtype=torch.float32, device=device)
                    _sc_ref_t = torch.as_tensor(np.asarray(_sc_ref), dtype=torch.float32, device=device)
                    with torch.no_grad():
                        _Rr, _tr = _rigid_frames(_fr_ref_t[:, 0], _fr_ref_t[:, 1], _fr_ref_t[:, 2])
                        _diff = _sc_ref_t[None, :, :] - _tr[:, None, :]
                        motif_fape_loc_ref = torch.einsum('fij,fsj->fsi',
                                                          _Rr.transpose(-1, -2), _diff)

        # ----- Sliding-mode setup (--motif_unindex_residues) -----
        slide_islands = []
        slide_state = None
        slide_island_lengths = []
        slide_motif_seq = ''
        slide_ref_xyz = np.zeros((0, 3), dtype=np.float32)
        slide_atom_res_idx = slide_atom_within_idx = slide_atom_ref_xyz = None
        binder_bb_atom_idx = None
        slide_runtime = None
        M_slide = 0
        if motif_slide_active:
            slide_islands = parse_motif_islands_spec(motif_unindex_residues)
            if not slide_islands:
                raise ValueError("--motif_unindex_residues set but parsed to no entries")
            slide_island_lengths = [island_length(isl) for isl in slide_islands]
            slide_chain_residues = [(r['chain'], r['resnum'])
                                    for isl in slide_islands for r in isl]
            M_slide = len(slide_chain_residues)
            if motif_fixed_active:
                _fixed_keys = set(m_chain_residues)
                for _c, _r in slide_chain_residues:
                    if (_c, _r) in _fixed_keys:
                        raise ValueError(
                            f"--motif_unindex_residues residue {_c}{_r} also in "
                            f"--motif_residues (residues cannot be in both)")
            if sum(slide_island_lengths) + len(fixed_positions) > length:
                raise ValueError(
                    f"sliding islands need {sum(slide_island_lengths)} binder "
                    f"positions + fixed motif occupies {len(fixed_positions)}; "
                    f"binder length {length} insufficient")
            slide_ref_xyz, slide_motif_seq = extract_motif_coords(motif_pdb, slide_chain_residues)
            _slide_ref_res_atoms = extract_motif_residue_atoms(motif_pdb, slide_chain_residues)

            # Precompute binder backbone atom index map [length, 4]: maps each
            # binder position + (N/CA/C/CB) -> global atom index. Stable across
            # all MCMC moves (only depends on the binder structure layout).
            binder_bb_atom_idx = torch.full((length, 4), -1, dtype=torch.long, device=device)
            for _p in range(length):
                for _j, _nm in enumerate(_BB):
                    _idx = _pred_atom_idx(_p, _nm)
                    if _idx is not None:
                        binder_bb_atom_idx[_p, _j] = _idx

            # Per-(residue, atom) specs for sliding side: only atoms present on
            # both motif PDB and the UNK binder.
            _slide_specs = []
            for _ri, ent in enumerate(_slide_ref_res_atoms):
                for _ai, _nm in enumerate(_BB):
                    if _nm in ent['atoms']:
                        _slide_specs.append((_ri, _ai, ent['atoms'][_nm]))
            if not _slide_specs:
                raise ValueError("sliding motif: no usable backbone atoms")
            slide_atom_res_idx = torch.as_tensor([s[0] for s in _slide_specs],
                                                  dtype=torch.long, device=device)
            slide_atom_within_idx = torch.as_tensor([s[1] for s in _slide_specs],
                                                     dtype=torch.long, device=device)
            slide_atom_ref_xyz = torch.as_tensor(np.stack([s[2] for s in _slide_specs]),
                                                  dtype=torch.float32, device=device)

            import random as _random
            _slide_rng = _random.Random(123)
            slide_state = random_valid_placement(slide_island_lengths, length,
                                                  fixed_positions, _slide_rng)
            # Total expected design iterations for the T schedule. Mirrors the
            # design() call-count pattern in `run_boltz_design`.
            if design_algorithm == "3stages_extra":
                _exp_total = (pre_iteration + soft_iteration + soft_iteration_1
                              + soft_iteration_2 + temp_iteration + hard_iteration)
            elif design_algorithm == "hard_only":
                _exp_total = pre_iteration + temp_iteration + hard_iteration
            else:
                _exp_total = (pre_iteration + soft_iteration + temp_iteration
                              + hard_iteration)
            slide_runtime = {
                'rng': _slide_rng,
                'call_count': 0,
                'total_calls': max(_exp_total, 1),
                'accept': 0,
                'attempts': 0,
                'current_T': float(motif_slide_T_init),
                'current_E': None,
            }

        # ----- Combined reference + target distogram + per-residue sequence one-hot -----
        if motif_fixed_active and motif_slide_active:
            combined_ref_xyz = np.concatenate([fixed_ref_xyz, slide_ref_xyz], axis=0)
            motif_seq = fixed_motif_seq + slide_motif_seq
        elif motif_fixed_active:
            combined_ref_xyz = fixed_ref_xyz
            motif_seq = fixed_motif_seq
        else:
            combined_ref_xyz = slide_ref_xyz
            motif_seq = slide_motif_seq
        M = len(motif_seq)
        motif_target_dist = get_motif_target_distances(
            combined_ref_xyz, device=device)
        motif_onehot = torch.zeros(M, V, device=device, dtype=batch['res_type'].dtype)
        for j, aa in enumerate(motif_seq):
            col = _AA_TO_COL.get(aa)
            if col is None:
                continue
            motif_onehot[j, col] = 1.0

        # ----- Combined backbone reference (constant) -----
        _bb_ref_fixed_arr = (np.asarray(fixed_bb_ref_list, dtype=np.float32)
                             if fixed_bb_ref_list else np.zeros((0, 3), dtype=np.float32))
        _bb_ref_fixed_t = torch.as_tensor(_bb_ref_fixed_arr, dtype=torch.float32, device=device)
        if motif_slide_active:
            motif_bb_ref_xyz = torch.cat([_bb_ref_fixed_t, slide_atom_ref_xyz], dim=0)
        else:
            motif_bb_ref_xyz = _bb_ref_fixed_t
        _K_fix = _bb_ref_fixed_t.shape[0]
        _K_total = motif_bb_ref_xyz.shape[0]
        # Static fixed backbone idx (used by the dynamic rebuilder + MCMC energy)
        motif_bb_pred_idx_fixed_static = (
            torch.as_tensor(fixed_bb_pred_list, dtype=torch.long, device=device)
            if fixed_bb_pred_list else torch.zeros(0, dtype=torch.long, device=device))

        # ----- Mutable dynamic tensors (refreshed per MCMC step) -----
        motif_token_idx = torch.zeros(M, dtype=torch.long, device=device)
        motif_bb_pred_idx = torch.zeros(_K_total, dtype=torch.long, device=device)
        motif_row_mask = torch.zeros(1, N, 1, device=device, dtype=batch['res_type'].dtype)
        motif_res_type_full = torch.zeros(1, N, V, device=device, dtype=batch['res_type'].dtype)

        def _rebuild_motif_dynamic():
            """Repopulate ``motif_token_idx`` / ``motif_bb_pred_idx`` /
            ``motif_row_mask`` / ``motif_res_type_full`` from the current
            ``slide_state``. Fixed entries are constant; sliding entries are
            indexed via the precomputed binder atom map. Called once after
            setup and after every accepted MCMC step (cheap, all torch ops)."""
            with torch.no_grad():
                if motif_fixed_active:
                    motif_token_idx[:M_fix] = torch.as_tensor(
                        fixed_token_idx_list, device=device, dtype=torch.long)
                    motif_bb_pred_idx[:_K_fix] = motif_bb_pred_idx_fixed_static
                if motif_slide_active:
                    positions = _placement_to_positions(slide_state, slide_islands)
                    positions_t = torch.as_tensor(positions, dtype=torch.long, device=device)
                    motif_token_idx[M_fix:] = binder_tok[positions_t]
                    positions_per_atom = positions_t[slide_atom_res_idx]
                    motif_bb_pred_idx[_K_fix:] = binder_bb_atom_idx[
                        positions_per_atom, slide_atom_within_idx]
                # Sequence pin masks: re-stamp from scratch each call (sliding
                # rows move; previously-pinned rows must be cleared).
                motif_row_mask.zero_()
                motif_res_type_full.zero_()
                for j in range(M):
                    if motif_onehot[j].sum() > 0:
                        tok = int(motif_token_idx[j])
                        motif_row_mask[0, tok, 0] = 1.0
                        motif_res_type_full[0, tok, :] = motif_onehot[j]

        def _placement_energy_state(state, coords_det):
            """Backbone Kabsch RMSD under a candidate placement (no grad).
            Concatenates the fixed bb idx (constant) with the sliding bb idx
            computed from ``state`` and uses the existing coord-loss kernel."""
            with torch.no_grad():
                positions = _placement_to_positions(state, slide_islands)
                positions_t = torch.as_tensor(positions, dtype=torch.long, device=device)
                positions_per_atom = positions_t[slide_atom_res_idx]
                slide_bb_pred = binder_bb_atom_idx[positions_per_atom, slide_atom_within_idx]
                if motif_fixed_active:
                    bb_pred_idx_state = torch.cat(
                        [motif_bb_pred_idx_fixed_static, slide_bb_pred], dim=0)
                else:
                    bb_pred_idx_state = slide_bb_pred
                # coords_det is [1, n_atoms, 3]
                return float(get_motif_coords_loss(
                    coords_det, bb_pred_idx_state, motif_bb_ref_xyz).item())

        def _mcmc_step_with_coords(coords_det):
            """Run motif_slide_steps_per_epoch MCMC moves on slide_state using
            the current (detached) predicted coords as the energy surface. T
            anneals from T_init to T_final exponentially over the expected
            total iteration count. Refreshes the dynamic tensors at the end."""
            if not motif_slide_active or coords_det is None:
                return
            t = min(slide_runtime['call_count']
                    / float(slide_runtime['total_calls']), 1.0)
            T = motif_slide_T_init * (motif_slide_T_final
                                       / motif_slide_T_init) ** t
            E_cur = _placement_energy_state(slide_state, coords_det)
            for _ in range(int(motif_slide_steps_per_epoch)):
                slide_runtime['attempts'] += 1
                prop = propose_mcmc_move(slide_state, slide_island_lengths,
                                          length, fixed_positions,
                                          motif_slide_shift_max,
                                          motif_slide_swap_prob,
                                          slide_runtime['rng'])
                if prop is None:
                    continue
                E_prop = _placement_energy_state(prop, coords_det)
                dE = E_prop - E_cur
                if dE < 0 or slide_runtime['rng'].random() < math.exp(-dE / T):
                    for i in range(len(slide_state)):
                        slide_state[i] = prop[i]
                    E_cur = E_prop
                    slide_runtime['accept'] += 1
            slide_runtime['call_count'] += 1
            slide_runtime['current_T'] = T
            slide_runtime['current_E'] = E_cur
            _rebuild_motif_dynamic()

        # Initialize dynamic tensors from the initial slide_state.
        _rebuild_motif_dynamic()

        # Optional ligand carry-along (--motif_ligand_residues). The Kabsch
        # transform is fit from the motif backbone above (joint over fixed +
        # sliding); the ligand atoms are simply *carried* by that transform.
        motif_lig_active = bool(motif_ligand_residues)
        motif_lig_pred_idx = motif_lig_ref_xyz = None
        if motif_lig_active:
            _lig_specs = parse_motif_ligand_spec(motif_ligand_residues)
            if not _lig_specs:
                raise ValueError("--motif_ligand_residues set but parsed to no entries")
            _ref_lig = extract_motif_ligand_atoms(motif_pdb, _lig_specs)
            _lig_pred, _lig_ref, _lig_summary = [], [], []
            for _ent in _ref_lig:
                _match_chain = _match_res = None
                for _c in structure.chains:
                    _r0 = int(_c['res_idx']); _rn = int(_c['res_num'])
                    for _ri in range(_rn):
                        _rr = structure.residues[_r0 + _ri]
                        if str(_rr['name']).strip() == _ent['resname']:
                            _match_chain, _match_res = _c, _rr
                            break
                    if _match_res is not None:
                        break
                if _match_res is None:
                    raise ValueError(
                        f"--motif_ligand_residues: resname {_ent['resname']!r} "
                        f"(motif PDB {_ent['chain']}{_ent['resnum']}) not "
                        f"present in the designed system. Make sure the "
                        f"ligand is in the YAML (e.g. --target_mols "
                        f"{_ent['resname']}).")
                _n_match = 0
                for _nm, _xyz in _ent['atoms'].items():
                    try:
                        _pi = _find_named_atom(structure,
                                               int(_match_res['atom_idx']),
                                               int(_match_res['atom_num']),
                                               _nm, str(_match_chain['name']), None)
                    except ValueError:
                        continue
                    _lig_pred.append(_pi)
                    _lig_ref.append(_xyz)
                    _n_match += 1
                _lig_summary.append(
                    f"{_ent['chain']}{_ent['resnum']}({_ent['resname']})->"
                    f"{str(_match_chain['name'])}:"
                    f"{_n_match}/{len(_ent['atoms'])}atoms")
            if not _lig_pred:
                raise ValueError(
                    "--motif_ligand_residues set but no ligand atoms resolved "
                    "in the designed system (atom names may not match between "
                    "motif PDB and CCD definition)")
            motif_lig_pred_idx = torch.as_tensor(_lig_pred, dtype=torch.long, device=device)
            motif_lig_ref_xyz = torch.as_tensor(np.asarray(_lig_ref),
                                                dtype=torch.float32, device=device)
            print(f"[motif] ligand carry-along over {len(_lig_pred)} atoms: "
                  + ", ".join(_lig_summary))

        # ----- Startup summary -----
        if motif_fixed_active:
            _motif_chains = sorted({c for c, _ in m_chain_residues})
            print(f"[motif/fixed] {M_fix} residues from {motif_pdb} "
                  f"chain(s) {','.join(_motif_chains)} "
                  f"({','.join(f'{c}{r}' for c, r in m_chain_residues)}) "
                  f"-> binder positions {[p+1 for p in binder_pos]}; "
                  f"seq={fixed_motif_seq}; fix_seq={bool(fix_motif_seq)}")
        if motif_fape_active:
            _n_fr = int(motif_frame_pred_idx.shape[0])
            _n_sc = int(motif_fape_sc_pred_idx.shape[0])
            print(f"[motif/fixed] FAPE over {_n_fr} frame(s) x {_n_sc} sidechain atom(s)")
        elif motif_fixed_active and motif_slide_active:
            print(f"[motif/fixed] FAPE DISABLED (sliding mode active; "
                  f"per-epoch frame rebuild not implemented in v1)")
        if motif_slide_active:
            _slide_chains = sorted({r['chain'] for isl in slide_islands for r in isl})
            print(f"[motif/slide] {len(slide_islands)} island(s), {M_slide} "
                  f"residues from chain(s) {','.join(_slide_chains)}; "
                  f"island_lengths={slide_island_lengths}; "
                  f"init state={slide_state}; T schedule "
                  f"{motif_slide_T_init} -> {motif_slide_T_final} over "
                  f"{slide_runtime['total_calls']} calls; "
                  f"steps/call={motif_slide_steps_per_epoch}; "
                  f"swap_prob={motif_slide_swap_prob}, "
                  f"shift_max={motif_slide_shift_max}; "
                  f"fix_seq={bool(fix_motif_seq)} (pin moves with state)")

    # ---- Explicit atom-pair distance restraints (--atom_pairs) -------------
    # Resolved once here (binder length fixed -> stable target indices); the
    # loss closures read `atom_pair_specs` from this enclosing scope.
    atom_pairs_active = bool(atom_pairs)
    atom_pair_specs = None
    if atom_pairs_active:
        atom_pair_specs = resolve_atom_pairs(atom_pairs, batch, structure)
        print(f"[atom_pairs] {len(atom_pair_specs)} distance restraint(s):")
        for _s in atom_pair_specs:
            print(f"  {_s['label']}  tok({_s['tok_a']},{_s['tok_b']}) "
                  f"atom({_s['atom_a']},{_s['atom_b']})")

    # ---- COM loss sources --------------------------------------------------
    # Parse ORI HETATMs from --pdb_path (anchor = all non-binder atoms in the
    # YAML-built `structure`) and from --motif_pdb (anchor = the protein motif
    # backbone atoms set up above). Per-source Kabsch fit per epoch transforms
    # the fixed ORI into the moving co-fold frame; multiple sources average.
    # Inert unless com_loss_weight > 0 AND at least one ORI was found.
    com_sources = []
    if com_loss_weight > 0:
        binder_eid = chain_to_number[binder_chain]
        # Source 1: target PDB (--pdb_path). Anchor = all atoms whose token has
        # entity_id != binder. Uses input-frame coords from `structure.atoms`.
        ori_pdb = parse_pdb_ori_atoms(pdb_path)
        if ori_pdb.shape[0] > 0:
            # token->atom expansion: pick atoms whose owning token is in a
            # non-binder chain. `atom_to_token` is [B, n_atoms, n_tokens] in
            # one-hot form; argmax over tokens gives each atom's token index.
            a2t = batch['atom_to_token'][0]                       # [n_atoms, n_tokens]
            atom_tok = a2t.argmax(dim=-1)                         # [n_atoms]
            ent_per_token = batch['entity_id']
            if ent_per_token.dim() > 1:
                ent_per_token = ent_per_token[0]
            atom_ent = ent_per_token[atom_tok]                    # [n_atoms]
            non_binder_atom_idx = torch.nonzero(
                (atom_ent != binder_eid) & (atom_tok < ent_per_token.numel()),
                as_tuple=True,
            )[0]
            # Constant input-frame anchor xyz lives in the YAML-built structure.
            ref_xyz_np = structure.atoms['coords'][
                non_binder_atom_idx.detach().cpu().numpy(), :]
            if ref_xyz_np.shape[0] >= 3:                          # need >=3 for Kabsch
                com_sources.append({
                    'label': '--pdb_path',
                    'anchor_idx': non_binder_atom_idx.to(device).long(),
                    'anchor_ref': torch.as_tensor(
                        ref_xyz_np, dtype=torch.float32, device=device),
                    'ori_ref': torch.as_tensor(
                        ori_pdb, dtype=torch.float32, device=device),
                })
                print(f"[com_loss] --pdb_path source: {ori_pdb.shape[0]} ORI "
                      f"atom(s), {ref_xyz_np.shape[0]} anchor atoms")
            else:
                print(f"[com_loss] WARNING: --pdb_path has ORI but only "
                      f"{ref_xyz_np.shape[0]} non-binder anchor atoms (<3); skipping")
        # Source 2: motif PDB (--motif_pdb). Anchor = protein motif backbone
        # already plumbed (motif_bb_pred_idx + motif_bb_ref_xyz). Reuses them.
        ori_motif = parse_pdb_ori_atoms(motif_pdb) if motif_pdb else np.zeros((0, 3))
        if ori_motif.shape[0] > 0 and motif_active:
            try:
                _anchor_idx_m = motif_bb_pred_idx_fixed_static
                _anchor_ref_m = _bb_ref_fixed_t  # set up above in motif block
            except NameError:
                _anchor_idx_m = None
            if _anchor_idx_m is not None and _anchor_idx_m.numel() >= 3:
                com_sources.append({
                    'label': '--motif_pdb',
                    'anchor_idx': _anchor_idx_m.to(device).long(),
                    'anchor_ref': _anchor_ref_m.to(device).float(),
                    'ori_ref': torch.as_tensor(
                        ori_motif, dtype=torch.float32, device=device),
                })
                print(f"[com_loss] --motif_pdb source: {ori_motif.shape[0]} ORI "
                      f"atom(s), {_anchor_idx_m.numel()} motif-backbone anchor atoms")
            else:
                print("[com_loss] WARNING: --motif_pdb has ORI but motif backbone "
                      "anchor unavailable; skipping")
        if not com_sources:
            print("[com_loss] WARNING: com_loss_weight > 0 but no ORI atoms "
                  "found in --pdb_path or --motif_pdb; loss will be inert")
        elif distogram_only:
            print("[com_loss] WARNING: COM loss needs sample_atom_coords; it is "
                  "INERT in fast mode. Pass --distogram_only False (and "
                  "--attach_coords True for a live gradient).")
    com_active = bool(com_sources) and com_loss_weight > 0

    def design(batch,
               iters = None,
                soft=0.0, e_soft=None,
                step=1.0, e_step=None,
                temp=1.0, e_temp=None,
                hard=0.0, e_hard=None,
                num_optimizing_binder_pos=1, e_num_optimizing_binder_pos=1,
                learning_rate=1.0,
                intra_chain_cutoff=14.0,
                mask=None,
                chain_mask=None,
                length=100,
                plots=None,
                loss_history=None,
                i_con_loss_history=None,
                con_loss_history=None,
                plddt_loss_history=None,
                distogram_history=None,
                sequence_history=None,
                pre_run=False,
                distogram_only=False,
                predict_args=None,
                alpha=2.0,
                loss_scales=None,
                binder_chain='A',
                non_protein_target=False,
                increasing_contact_over_itr=False,
                num_intra_contacts=4,
                save_trajectory=False,
                stage_name='unknown',
                ):

        prev_sequence=""
        # Per-target lists (inter_chain_cutoff_pt, num_inter_contacts_pt,
        # optimize_contact_per_binder_pos_pt, mask_target_prerun_pt, i_con_w,
        # i_pae_w, tplddt_w, target_masks, target_chain_ids) are captured from
        # the enclosing boltz_hallucination scope; get_model_loss reads them
        # directly so the long arg list doesn't need to thread per-target lists.
        def get_model_loss(batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, pre_run=False, distogram_only=False, predict_args=None, loss_scales=None, binder_chain='A', increasing_contact_over_itr=False, num_intra_contacts=4,  num_optimizing_binder_pos =1, intra_chain_cutoff=14.0, save_trajectory=False, epoch_idx=0, stage_name='unknown'):
            traj_coords = None
            traj_plddt = None

            # Per-target pre-run masking: for each target whose
            # mask_target_prerun is True, zero its tokens (and their
            # rep-atoms) so the binder evolves topology without seeing it.
            if pre_run:
                for ti, c_id in enumerate(target_chain_ids):
                    if not mask_target_prerun_pt[ti]:
                        continue
                    t_id = chain_to_number[c_id]
                    t_tok = (batch['entity_id'] == t_id)
                    batch['token_pad_mask'][t_tok] = 0
                    keep = torch.zeros_like(batch['token_to_rep_atom'])
                    keep[t_tok, :] = 1
                    atoms_idx = torch.nonzero(
                        batch['token_to_rep_atom'] * keep, as_tuple=True)[2]
                    batch['atom_pad_mask'][:, atoms_idx] = 0

            # Common arguments for get_distogram_confidence
            confidence_args = {
                'recycling_steps': predict_args["recycling_steps"],
                'num_sampling_steps': predict_args["sampling_steps"],
                'multiplicity_diffusion_train': 1,
                'diffusion_samples': predict_args["diffusion_samples"],
                'run_confidence_sequentially': True,
                'disconnect_feats': disconnect_feats,
                'disconnect_pairformer': disconnect_pairformer,
                'disconnect_coords': not attach_coords,
            }

            if save_trajectory:
                # Get model output with trajectory info
                dict_out = boltz_model.get_distogram_confidence(batch, **confidence_args)
                traj_coords = dict_out['sample_atom_coords'][0].detach().cpu().numpy()
                traj_plddt = dict_out['plddt'][0].detach().cpu().numpy()
            else:
                # Get model output without trajectory
                if pre_run or distogram_only:
                    dict_out, s, z, s_inputs = boltz_model.get_distogram(batch)
                else:
                    dict_out = boltz_model.get_distogram_confidence(batch, **confidence_args)

            # Per-epoch intermediate structure dump. Fires only when the
            # diffusion sampler actually ran this step (full-mode path, i.e.
            # `sample_atom_coords` was produced). `attach_coords` is irrelevant
            # here — the coords are materialized either way; that flag only
            # controls whether gradients flow through them.
            if (save_intermediate_structures and intermediate_dir is not None
                    and 'sample_atom_coords' in dict_out):
                try:
                    coords_np = dict_out['sample_atom_coords'][0].detach().cpu().numpy()
                    plddt_np = (dict_out['plddt'][0].detach().cpu().numpy()
                                if 'plddt' in dict_out else None)
                    n_atoms = structure.atoms['coords'].shape[0]
                    coords_np = coords_np[:n_atoms, :]
                    saved_coords = structure.atoms['coords'].copy()
                    saved_present = structure.atoms['is_present'].copy()
                    try:
                        structure.atoms['coords'] = coords_np
                        structure.atoms['is_present'] = True
                        out_path = os.path.join(
                            intermediate_dir,
                            f"{stage_name}_epoch{epoch_idx:04d}.pdb",
                        )
                        pdb_str = to_pdb(structure, plddts=plddt_np)
                        with open(out_path, 'w') as _f:
                            _f.write(pdb_str)
                    finally:
                        structure.atoms['coords'] = saved_coords
                        structure.atoms['is_present'] = saved_present
                except Exception as _e:
                    print(f"[intermediate] save failed at "
                          f"{stage_name}_epoch{epoch_idx:04d}: "
                          f"{type(_e).__name__}: {_e}")

            pdist = dict_out['pdistogram']
            mid_pts = get_mid_points(pdist).to(device)

            # Binder-internal (intra) contact loss -- singular.
            con_loss = get_con_loss(pdist, mid_pts,
                                num=num_intra_contacts, seqsep=9, cutoff=intra_chain_cutoff,
                                binary=False,
                                mask_1d=chain_mask, mask_1b=chain_mask)

            # Per-target inter-chain contact loss. Each target contributes its
            # own get_con_loss(..., cutoff, num) term, weighted by i_con_w[ti].
            # Targets that are masked this step (pre_run + mask_target_prerun)
            # are skipped, matching the previous "if pre_run and mask_ligand:
            # drop i_con_loss" behavior but now per-target.
            i_con_loss = pdist.new_zeros(())
            i_con_any = False
            for ti in range(n_targets):
                if pre_run and mask_target_prerun_pt[ti]:
                    continue
                w = i_con_w[ti]
                if w == 0:
                    continue
                t_mask = target_masks[ti]
                if optimize_contact_per_binder_pos_pt[ti]:
                    n_pos = (0 if (pre_run and increasing_contact_over_itr)
                             else num_optimizing_binder_pos)
                    li = get_con_loss(pdist, mid_pts,
                                      num=num_inter_contacts_pt[ti], seqsep=0,
                                      num_pos=n_pos,
                                      cutoff=inter_chain_cutoff_pt[ti], binary=False,
                                      mask_1d=chain_mask, mask_1b=t_mask)
                else:
                    li = get_con_loss(pdist, mid_pts,
                                      num=num_inter_contacts_pt[ti], seqsep=0,
                                      cutoff=inter_chain_cutoff_pt[ti], binary=False,
                                      mask_1d=t_mask, mask_1b=chain_mask)
                i_con_loss = i_con_loss + w * li
                i_con_any = True

            mask_2d = chain_mask[:, :, None] * chain_mask[:, None, :]
            helix_loss = _get_helix_loss(pdist, mid_pts,
                                    offset=None, mask_2d=mask_2d, binary=True)

            losses = {'con_loss': con_loss, 'helix_loss': helix_loss}
            if i_con_any:
                losses['i_con_loss'] = i_con_loss

            if not pre_run and not distogram_only:
                plddt_loss = get_plddt_loss(dict_out['plddt'], mask_1d=chain_mask)
                pae = (dict_out['pae'] + dict_out['pae'].transpose(-2,-1))/2
                # Binder-internal PAE (binder ↔ binder) -- singular.
                pae_loss = get_pae_loss(pae, mask_1d=chain_mask, mask_1b=chain_mask)
                rg_loss, rg = add_rg_loss(dict_out['sample_atom_coords'], batch, length, binder_chain=binder_chain)

                # Per-target inter-chain PAE. Same skip rules as i_con_loss.
                i_pae_loss = pdist.new_zeros(())
                i_pae_any = False
                for ti in range(n_targets):
                    if pre_run and mask_target_prerun_pt[ti]:
                        continue
                    w = i_pae_w[ti]
                    if w == 0:
                        continue
                    li = get_pae_loss(pae, mask_1d=target_masks[ti], mask_1b=chain_mask)
                    i_pae_loss = i_pae_loss + w * li
                    i_pae_any = True

                losses.update({
                    'plddt_loss': plddt_loss,
                    'pae_loss': pae_loss,
                    'rg_loss': rg_loss,
                })
                if i_pae_any:
                    losses['i_pae_loss'] = i_pae_loss

                # COM loss: binder centroid pulled toward Kabsch-transformed
                # ORI atoms parsed from --pdb_path / --motif_pdb. Full-mode
                # only (needs sample_atom_coords); gradient with --attach_coords
                # True (mirrors rg_loss).
                if com_active:
                    com_loss, _bc, _mo, _ao = add_com_loss(
                        dict_out['sample_atom_coords'], batch, com_sources,
                        binder_chain=binder_chain)
                    losses['com_loss'] = com_loss
                    # Diagnostic: log the actual binder COM distance to mean ORI
                    # (the loss VALUE itself, in Å).
                    loss_component_history.setdefault('com_dist', []).append(
                        float(com_loss.detach().item()))

                # Per-target pLDDT loss: separately weighted per target. Replaces
                # the old --target_plddt_chains gate -- a per-target weight of 0
                # cleanly excludes that target with negligible cost.
                tplddt_total = dict_out['plddt'].new_zeros(())
                tplddt_any = False
                for ti in range(n_targets):
                    w = tplddt_w[ti]
                    if w <= 0:
                        continue
                    li = get_plddt_loss(dict_out['plddt'], mask_1d=target_masks[ti])
                    tplddt_total = tplddt_total + w * li
                    tplddt_any = True
                if tplddt_any:
                    losses['target_plddt_loss'] = tplddt_total

                plddt_loss_history.append(plddt_loss.item())
                # Snapshot the full per-token pLDDT vector and symmetric PAE
                # matrix (A) for the evolution animations (closure mutate).
                plddt_history.append(dict_out['plddt'][0].detach().cpu().numpy())
                pae_history.append(pae[0].detach().cpu().numpy())

            # Motif scaffolding supervised losses (ColabDesign `partial`).
            #   - `motif_distogram_loss`: bin-center MSE on the trunk distogram
            #     restricted to motif token pairs (fast-mode safe).
            #   - `motif_coords_loss`:    Kabsch (backbone-only) + RMSD over
            #     backbone (and ligand if --motif_ligand_residues set), full-
            #     mode only; gradient needs --attach_coords True like rg_loss.
            #   - `motif_fape_loss`:      AF-style frame-aligned sidechain
            #     error (full-mode only; same gating).
            if motif_active:
                losses['motif_distogram_loss'] = get_motif_distogram_loss(
                    pdist, motif_token_idx, motif_target_dist, mid_pts)
                if (not pre_run and not distogram_only
                        and 'sample_atom_coords' in dict_out):
                    _mloss, _mstats = get_motif_coords_loss(
                        dict_out['sample_atom_coords'], motif_bb_pred_idx,
                        motif_bb_ref_xyz, motif_lig_pred_idx, motif_lig_ref_xyz,
                        return_stats=True)
                    losses['motif_coords_loss'] = _mloss
                    # Per-component diagnostics for plot_final_aux_losses /
                    # animation-folder CSV. Backbone always present; ligand
                    # only when --motif_ligand_residues is set.
                    loss_component_history.setdefault(
                        'motif_bb_rmsd', []).append(_mstats['bb_rmsd'])
                    if _mstats['lig_rmsd'] is not None:
                        loss_component_history.setdefault(
                            'motif_lig_rmsd', []).append(_mstats['lig_rmsd'])
                    if motif_fape_active:
                        losses['motif_fape_loss'] = get_motif_fape_loss(
                            dict_out['sample_atom_coords'], motif_frame_pred_idx,
                            motif_fape_sc_pred_idx, motif_fape_loc_ref)

            # Explicit atom-pair distance restraints (--atom_pairs). Distogram
            # term is fast-mode safe; coord term needs the sampler (full mode)
            # and carries a gradient only with --attach_coords True.
            if atom_pairs_active:
                ap_dgram_loss, ap_dgram_stats = get_atom_pair_distogram_loss(
                    pdist, atom_pair_specs, mid_pts,
                    method=atom_pair_distogram_loss_type, return_stats=True)
                losses['atom_pair_distogram_loss'] = ap_dgram_loss
                # Per-pair diagnostics for the animation-folder distance/loss
                # curves + CSV. The logged distance is exactly what the loss
                # reads (single source of truth); exp_dist/mode/p_window are
                # logged regardless of method so the restraint can be debugged.
                for _st in ap_dgram_stats:
                    _lab = _st['label']
                    loss_component_history.setdefault(f"atom_pair|dist|{_lab}", []).append(_st['exp_dist'])
                    loss_component_history.setdefault(f"atom_pair|mode|{_lab}", []).append(_st['mode_dist'])
                    loss_component_history.setdefault(f"atom_pair|pwin|{_lab}", []).append(_st['p_window'])
                    loss_component_history.setdefault(f"atom_pair|loss|{_lab}", []).append(_st['loss'])
                if (not pre_run and not distogram_only
                        and 'sample_atom_coords' in dict_out):
                    ap_coords_loss, ap_coords_stats = get_atom_pair_coords_loss(
                        dict_out['sample_atom_coords'], atom_pair_specs,
                        return_stats=True)
                    losses['atom_pair_coords_loss'] = ap_coords_loss
                    for _st in ap_coords_stats:
                        _lab = _st['label']
                        loss_component_history.setdefault(f"atom_pair|cdist|{_lab}", []).append(_st['dist'])
                        loss_component_history.setdefault(f"atom_pair|closs|{_lab}", []).append(_st['loss'])

            bins = mid_points < 8.0
            px = torch.sum(torch.softmax(dict_out['pdistogram'], dim=-1)[:,:,:,bins], dim=-1)

            if loss_scales is None:
                loss_scales = {
                    'con_loss': 1.0,
                    'i_con_loss': 1.0,
                    'helix_loss': random.uniform(-0.4, 0.0),
                    'plddt_loss': 0.1,
                    'pae_loss': 0.4,
                    'i_pae_loss': 0.1,
                    'rg_loss': 0.3,
                    'target_plddt_loss': 0.1,
                    'motif_distogram_loss': 1.0,
                    'motif_coords_loss': 1.0,
                    'motif_fape_loss': 1.0,
                    'atom_pair_distogram_loss': 1.0,
                    'atom_pair_coords_loss': 1.0,
                }

            # Defensive: these keys may be present in `losses` while a
            # caller-supplied loss_scales (configs/CLI) predates the feature.
            for _k in ('target_plddt_loss', 'motif_distogram_loss', 'motif_coords_loss',
                       'motif_fape_loss',
                       'atom_pair_distogram_loss', 'atom_pair_coords_loss', 'com_loss'):
                if _k in losses and _k not in loss_scales:
                    loss_scales = {**loss_scales, _k: 1.0}

            # Calculate total loss and print individual losses
            total_loss = sum(loss * loss_scales[name] for name, loss in losses.items())
            # Single consolidated per-epoch readout: build the mode-adaptive
            # per-loss breakdown here (only keys present in `losses` are shown --
            # plddt/pae/i_pae/rg_loss exist only in full mode) and let the
            # `Epoch i:` print at the call site emit it as ONE line. No separate
            # print here, so each epoch logs exactly one line.
            loss_str = " | ".join(f"{k}:{v.item():.3f}" for k, v in losses.items())
            if 'rg_loss' in losses:
                # rg_loss is the elu() output; rg is the raw radius of gyration.
                # rg is the diagnostic value (a folded ~115-res binder ~= 13 A;
                # values in the thousands mean the sampler has not converged).
                loss_str += f" (rg={rg.item():.2f} A)"
            plots.append(px[0].detach().cpu().numpy())
            # Unified per-epoch history dict: every loss component goes through
            # `loss_component_history`. Legacy lists mirror specific keys so
            # the existing return-tuple consumers (visualize_training_history,
            # downstream callers) keep working unchanged.
            loss_component_history.setdefault('total_loss', []).append(total_loss.item())
            for _name, _val in losses.items():
                loss_component_history.setdefault(_name, []).append(_val.item())
            loss_history.append(total_loss.item())
            i_con_loss_history.append(i_con_loss.item())
            con_loss_history.append(con_loss.item())
            # distogram_history.append(torch.softmax(dict_out['pdistogram'], dim=-1)[0].detach().cpu().numpy())
            distogram_history.append(px[0].detach().cpu().numpy())
            sequence_history.append(batch['res_type'][0, :, 2:22].detach().cpu().numpy())

            return total_loss, plots, loss_history, i_con_loss_history, con_loss_history, distogram_history, sequence_history, plddt_loss_history, loss_str, traj_coords, traj_plddt
        
        def update_sequence(opt, batch, mask, alpha=2.0, non_protein_target=False, binder_chain='A'):
            batch["logits"] = alpha*batch['res_type_logits']
            X =  batch['logits']- torch.sum(torch.eye(batch['logits'].shape[-1])[[0,1,6,22,23,24,25,26,27,28,29,30,31,32]],dim=0).to(device)*(1e10)
            batch['soft'] = torch.softmax(X/opt["temp"],dim=-1)
            batch['hard'] =  torch.zeros_like(batch['soft']).scatter_(-1, batch['soft'].max(dim=-1, keepdim=True)[1], 1.0)
            batch['hard'] =  (batch['hard'] - batch['soft']).detach() + batch['soft']
            batch['pseudo'] =  opt["soft"] * batch["soft"] + (1-opt["soft"]) * batch["res_type_logits"]
            batch['pseudo'] = opt["hard"] * batch["hard"] + (1-opt["hard"]) * batch["pseudo"]
            batch['res_type'] = batch['pseudo']*mask + batch['res_type_logits']*(1-mask)

            # Motif sequence pin: overwrite the motif rows with the reference
            # one-hot every step (constant write -> those rows carry no grad,
            # so the motif AA identity is retained regardless of the soft/hard
            # /omit machinery, including residues like Cys that BoltzDesign1
            # otherwise excludes from design).
            if motif_active and fix_motif_seq:
                batch['res_type'] = (batch['res_type'] * (1 - motif_row_mask)
                                     + motif_res_type_full)

            if non_protein_target:
                batch['msa'] = batch['res_type'].unsqueeze(0).to(device).detach()
                batch['profile'] = batch['msa'].float().mean(dim=0).to(device).detach()
            else:
                batch['msa'][:,0,:,:] = batch['res_type'].to(device).detach()
                batch['profile'][batch['entity_id']==chain_to_number[binder_chain],:] = batch['msa'][:, 0, (batch['entity_id']==chain_to_number[binder_chain])[0],:].float().mean(dim=1).to(device).detach()

            return batch
        
        m = {"soft":[soft,e_soft],"temp":[temp,e_temp],"hard":[hard,e_hard], "step":[step,e_step], 'num_optimizing_binder_pos':[num_optimizing_binder_pos, e_num_optimizing_binder_pos]}
        m = {k:[s,(s if e is None else e)] for k,(s,e) in m.items()}

        opt = {}
        traj_coords_list = []
        traj_plddt_list = []
        for i in range(iters):
            for k,(s,e) in m.items():
                if k == "temp":
                    opt[k] = (e+(s-e)*(1-(i)/iters)**2)
                else:
                    v = (s+(e-s)*((i)/iters))
                    if k == "step": step = v
                    opt[k] = v
                
            lr_scale = step * ((1 - opt["soft"]) + (opt["soft"] * opt["temp"]))
            num_optimizing_binder_pos = int(opt["num_optimizing_binder_pos"])

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_scale

            opt["lr_rate"] = learning_rate * lr_scale
                
            batch = update_sequence(opt, batch, mask, non_protein_target=non_protein_target, binder_chain=binder_chain)
            total_loss, plots, loss_history, i_con_loss_history, con_loss_history, distogram_history, sequence_history, plddt_loss_history, loss_str, traj_coords, traj_plddt = get_model_loss(batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, pre_run, distogram_only, predict_args, loss_scales, binder_chain, increasing_contact_over_itr, num_intra_contacts=num_intra_contacts, num_optimizing_binder_pos=num_optimizing_binder_pos, intra_chain_cutoff=intra_chain_cutoff, save_trajectory = save_trajectory, epoch_idx=i, stage_name=stage_name)
            traj_coords_list.append(traj_coords)
            traj_plddt_list.append(traj_plddt)
            current_sequence = ''.join([alphabet[i] for i in torch.argmax(batch['res_type'][batch['entity_id']==chain_to_number[binder_chain],:], dim=-1).detach().cpu().numpy()])
            if prev_sequence is not None:
                diff_count = sum(1 for a, b in zip(current_sequence, prev_sequence) if a != b)
                diff_percentage = (diff_count / length) * 100
            prev_sequence = current_sequence
            total_loss.backward()
            if batch['res_type_logits'].grad is not None:
                batch['res_type_logits'].grad[batch['entity_id']!=chain_to_number[binder_chain],:] = 0
                batch['res_type_logits'].grad[..., [0,1,6,22,23,24,25,26,27,28,29,30,31,32]] = 0
                # Freeze pinned motif positions: zero their grad before the
                # norm so fixed residues neither move nor skew the gradient
                # normalization of the free (designed) positions.
                if motif_active and fix_motif_seq:
                    batch['res_type_logits'].grad[0, motif_token_idx, :] = 0
                batch['res_type_logits'].grad = norm_seq_grad(batch['res_type_logits'].grad, chain_mask)
                optimizer.step()
                optimizer.zero_grad()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {i}: lr: {current_lr:.3f}, soft: {opt['soft']:.2f}, hard: {opt['hard']:.2f}, temp: {opt['temp']:.2f}, total loss: {total_loss.item():.2f}, {loss_str}")
        
        return batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list, traj_plddt_list

    if pre_run:
        batch, plots, loss_history, i_con_loss_history, con_loss_history,plddt_loss_history, distogram_history, sequence_history, traj_coords_list, traj_plddt_list = design(batch, iters=pre_iteration, stage_name='pre', soft=1.0, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate_pre, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, non_protein_target=non_protein_target, intra_chain_cutoff=intra_chain_cutoff, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)
    else:
        if design_algorithm == "3stages":
            print('-'*100)
            print(f"logits to softmax(T={e_soft})")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list1, traj_plddt_list1 = design(batch, iters=soft_iteration, stage_name='soft', e_soft=e_soft, num_optimizing_binder_pos=1, e_num_optimizing_binder_pos=8, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, non_protein_target=non_protein_target, intra_chain_cutoff=intra_chain_cutoff, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)
            print('-'*100)
            print("softmax(T=1) to softmax(T=0.01)")
            print('-'*100)
            print("set res_type_logits to logits")
            new_logits = (alpha * batch["res_type_logits"]).clone().detach().requires_grad_(True)
            batch['res_type_logits'] = new_logits
            optimizer = torch.optim.SGD([batch['res_type_logits']], lr=learning_rate)
            batch, plots, loss_history, i_con_loss_history, con_loss_history,plddt_loss_history, distogram_history, sequence_history, traj_coords_list2, traj_plddt_list2 = design(batch, iters=temp_iteration, stage_name='temp', soft=1.0, temp = 1.0,e_temp=0.01, num_optimizing_binder_pos=8, e_num_optimizing_binder_pos=12,  mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, non_protein_target=non_protein_target, intra_chain_cutoff=intra_chain_cutoff, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)
            print('-'*100)
            print("hard")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list3, traj_plddt_list3 = design(batch, iters=hard_iteration, stage_name='hard', soft=1.0, hard = 1.0,temp=0.01, num_optimizing_binder_pos=12, e_num_optimizing_binder_pos=16, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, non_protein_target=non_protein_target, intra_chain_cutoff=intra_chain_cutoff, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)
            traj_coords_list = traj_coords_list1 + traj_coords_list2 + traj_coords_list3 if save_trajectory else []
            traj_plddt_list = traj_plddt_list1 + traj_plddt_list2 + traj_plddt_list3 if save_trajectory else []

        elif design_algorithm == "3stages_extra":
            print('-'*100)
            print(f"logits to softmax(T={e_soft_1})")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list1, traj_plddt_list1 = design(batch, iters=soft_iteration_1, stage_name='soft1', e_soft=e_soft_1, num_optimizing_binder_pos=1, e_num_optimizing_binder_pos=8, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, non_protein_target=non_protein_target, intra_chain_cutoff=intra_chain_cutoff, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory) 
            print('-'*100)
            print(f"logits to softmax(T={e_soft_2})")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list2, traj_plddt_list2 = design(batch, iters=soft_iteration_2, stage_name='soft2', e_soft=e_soft_2, num_optimizing_binder_pos=1, e_num_optimizing_binder_pos=8, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, non_protein_target=non_protein_target, intra_chain_cutoff=intra_chain_cutoff, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)
            print('-'*100)
            print("softmax(T=1) to softmax(T=0.01)")
            print('-'*100)
            print("set res_type_logits to logits")
            new_logits = (alpha * batch["res_type_logits"]).clone().detach().requires_grad_(True)
            batch['res_type_logits'] = new_logits
            optimizer = torch.optim.SGD([batch['res_type_logits']], lr=learning_rate)
            batch, plots, loss_history, i_con_loss_history, con_loss_history,plddt_loss_history, distogram_history, sequence_history, traj_coords_list3, traj_plddt_list3 = design(batch, iters=temp_iteration, stage_name='temp', soft=1.0, temp = 1.0,e_temp=0.01, num_optimizing_binder_pos=8, e_num_optimizing_binder_pos=12,  mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, non_protein_target=non_protein_target, intra_chain_cutoff=intra_chain_cutoff, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)
            print('-'*100)
            print("hard")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list4, traj_plddt_list4 = design(batch, iters=hard_iteration, stage_name='hard', soft=1.0, hard = 1.0,temp=0.01, num_optimizing_binder_pos=12, e_num_optimizing_binder_pos=16, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, non_protein_target=non_protein_target, intra_chain_cutoff=intra_chain_cutoff, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)

            traj_coords_list = traj_coords_list1 + traj_coords_list2 + traj_coords_list3 + traj_coords_list4 if save_trajectory else []
            traj_plddt_list = traj_plddt_list1 + traj_plddt_list2 + traj_plddt_list3 + traj_plddt_list4 if save_trajectory else []
                
        elif design_algorithm == "logits":
            print('-'*100)
            print("logits")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list, traj_plddt_list= design(batch, iters=soft_iteration, stage_name='hard_only', soft = 0.0, e_soft=0.0, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, non_protein_target=non_protein_target, intra_chain_cutoff=intra_chain_cutoff, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)

    def _run_model(boltz_model, batch, predict_args):
        # Best-effort full-mode prediction. Returns None on OOM / CUBLAS /
        # the `raise {"exception": True}` bug in predict_step so callers can
        # still emit loss curves and trajectory animations.
        boltz_model.predict_args = predict_args
        try:
            with torch.no_grad():
                output = boltz_model.predict_step(batch, batch_idx=0, dataloader_idx=0)
        except (RuntimeError, TypeError) as e:
            print(f"[_run_model] final-fold failed ({type(e).__name__}): {e}. "
                  f"Returning None so plotting/animation can still run.")
            torch.cuda.empty_cache()
            gc.collect()
            return None
        if isinstance(output, dict) and output.get("exception") is True:
            print("[_run_model] predict_step reported exception (likely OOM). "
                  "Returning None so plotting/animation can still run.")
            torch.cuda.empty_cache()
            gc.collect()
            return None
        torch.cuda.empty_cache()
        return output

    def visualize_results(plots):
        # Plot distogram predictions
        if plots:
            num_plots = len(plots)
            num_rows = (num_plots + 5) // 6
            fig, axs = plt.subplots(num_rows, 6, figsize=(15, num_rows * 2.5))
            
            if num_rows == 1:
                axs = axs.reshape(1, -1)

            for i, plot_data in enumerate(plots):
                row, col = i // 6, i % 6
                axs[row, col].imshow(plot_data)
                axs[row, col].set_title(f'Epoch {i + 1}')
                axs[row, col].axis('off')

            # Hide unused subplots
            for j in range(num_plots, num_rows * 6):
                axs[j // 6, j % 6].axis('off')

            plt.tight_layout()
            plt.show()
            plots.clear()

    # visualize_results(plots)

    if pre_run:
        predict_args = {
        "recycling_steps": 3,  # Default value
        "sampling_steps": sampling_steps,  # Total diffusion step count
        "diffusion_samples": 1,  # Default value
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": False,
        }

        best_logits = batch['res_type_logits']
        best_seq = ''.join([alphabet[i] for i in torch.argmax(batch['res_type'][batch['entity_id']==chain_to_number[binder_chain],:], dim=-1).detach().cpu().numpy()])
        data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = best_seq
        return batch['res_type'].detach().cpu().numpy(), plots, loss_history, distogram_history, sequence_history, traj_coords_list, traj_plddt_list

    boltz_model.eval()

    if best_batch is None:
        if first_step_best_batch is not None:
            best_batch = first_step_best_batch
        else:
            best_batch = batch  

    # Per-epoch optimization-loop predict args (consumed by confidence_args in
    # get_model_loss). Cheap on purpose so the inner loop fits in memory.
    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    def _mutate(sequence, best_logits, i_prob):
        mutated_sequence = list(sequence) # Create a copy of the input tensor
        i = np.random.choice(np.arange(length),p=i_prob/i_prob.sum())
        i_logits = best_logits[:, i]
        i_logits = i_logits - torch.max(i_logits)
        i_X = i_logits- (torch.sum(torch.eye(i_logits.shape[-1])[[0,1,6,22,23,24,25,26,27,28,29,30,31,32]],dim=0)*(1e10)).to(device)
        i_aa = torch.multinomial(torch.softmax(i_X, dim=-1), 1).item()
        mutated_sequence[i] = alphabet[i_aa]
        return ''.join(mutated_sequence)

    best_logits = best_batch['res_type_logits']
    best_seq = ''.join([alphabet[i] for i in torch.argmax(best_batch['res_type'][best_batch['entity_id']==chain_to_number[binder_chain],:], dim=-1).detach().cpu().numpy()])
    data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = best_seq

    data_apo = copy.deepcopy(data)  # This handles all types of values correctly
    data_apo.pop('constraints', None)  # Remove constraints if they exist
    data_apo['sequences'] = [data_apo['sequences'][chain_to_number[binder_chain]]]  # Keep only chain B

    def _update_batches(data, data_apo):
        target = parse_boltz_schema(name, data, ccd_lib)
        target_apo = parse_boltz_schema(name, data_apo, ccd_lib)
        best_batch, best_structure = get_batch(target, msa_max_seqs, length, keep_record=True)
        best_batch_apo, best_structure_apo = get_batch(target_apo, msa_max_seqs, length, keep_record=True)
        best_batch = {key: value.unsqueeze(0).to(device) if key != 'record' else value for key, value in best_batch.items()}
        best_batch_apo = {key: value.unsqueeze(0).to(device) if key != 'record' else value for key, value in best_batch_apo.items()}
        return best_batch, best_batch_apo, best_structure, best_structure_apo

    # Stand-alone final structural validation. Deliberately decoupled from the
    # per-epoch optimization knobs so the final fold is a fixed,
    # well-conditioned prediction regardless of how cheap/heavy the
    # optimization-loop forwards were.
    final_predict_args = {
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": False,
    }

    best_batch, best_batch_apo, best_structure, best_structure_apo = _update_batches(data, data_apo)
    output = _run_model(boltz_model, best_batch, final_predict_args)
    output_apo = _run_model(boltz_model, best_batch_apo, final_predict_args)

    # If the final fold OOM'd, skip semi-greedy and return histories so the
    # caller can still write loss plots and the training animation.
    if output is None or output_apo is None:
        print("[boltz_hallucination] final-fold output unavailable; skipping "
              "semi-greedy and returning collected histories for plotting.")
        return output, output_apo, best_batch, best_batch_apo, best_structure, best_structure_apo, distogram_history, sequence_history, loss_history, con_loss_history, i_con_loss_history, plddt_loss_history, traj_coords_list, traj_plddt_list, structure, loss_component_history, plddt_history, pae_history

    prev_sequence = ''.join([alphabet[i] for i in torch.argmax(best_batch['res_type'][best_batch['entity_id']==chain_to_number[binder_chain],:], dim=-1).detach().cpu().numpy()])
    prev_iptm = output['iptm'].detach().cpu().numpy()
    print("best design iptm", prev_iptm)
    print("Semi-greedy steps", semi_greedy_steps)
    for step in range(semi_greedy_steps):
        confidence_score = []
        mutated_sequence_ls = []
        for t in range(10):
            plddt = output['plddt'][best_batch['entity_id']==chain_to_number[binder_chain]]
            i_prob = np.ones(length) if plddt is None else torch.maximum(1-plddt,torch.tensor(0))
            i_prob = i_prob.detach().cpu().numpy() if torch.is_tensor(i_prob) else i_prob
            mutated_sequence = _mutate(prev_sequence, best_logits, i_prob)
            data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = mutated_sequence
            best_batch, _, _, _ = _update_batches(data, data_apo)
            output = _run_model(boltz_model, best_batch, predict_args)
            if output is None:
                print(f"[semi-greedy] step {step} epoch {t}: fold failed; "
                      f"aborting semi-greedy with histories collected.")
                return output, output_apo, best_batch, best_batch_apo, best_structure, best_structure_apo, distogram_history, sequence_history, loss_history, con_loss_history, i_con_loss_history, plddt_loss_history, traj_coords_list, traj_plddt_list, structure, loss_component_history, plddt_history, pae_history

            iptm = output['iptm'].detach().cpu().numpy()
            confidence_score.append(iptm)
            mutated_sequence_ls.append(mutated_sequence)
            print(f"Step {step}, Epoch {t}, iptm {iptm[0]:.3f}")

        best_id = np.argmax(confidence_score)
        best_iptm = confidence_score[best_id]
        
        if best_iptm > prev_iptm:
            best_seq = mutated_sequence_ls[best_id]
            for seq_data in [data, data_apo]:
                seq_data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = best_seq
            print(f"Step {step}, Epoch {best_id}, Update sequence, iptm {best_iptm}, previous iptm {prev_iptm}")
            print(f"Update sequence {best_seq}")
            prev_iptm = best_iptm
            prev_sequence = best_seq
        else:
            for seq_data in [data, data_apo]:
                seq_data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = prev_sequence

        best_batch, best_batch_apo, best_structure, best_structure_apo = _update_batches(data, data_apo)

        if step == semi_greedy_steps - 1:
            # Final post-semi-greedy fold uses stand-alone validation settings,
            # not the cheaper per-mutation eval settings used inside the loop.
            output = _run_model(boltz_model, best_batch, final_predict_args)
            output_apo = _run_model(boltz_model, best_batch_apo, final_predict_args)

    return output, output_apo, best_batch, best_batch_apo, best_structure, best_structure_apo, distogram_history, sequence_history, loss_history, con_loss_history, i_con_loss_history, plddt_loss_history, traj_coords_list, traj_plddt_list, structure, loss_component_history, plddt_history, pae_history


def run_boltz_design(
    boltz_path,
    main_dir,
    yaml_dir,
    boltz_model,
    ccd_path,
    design_samples =1,
    version_name=None,
    config=None,
    loss_scales=None,
    num_workers=1,
    show_animation=False,
    save_trajectory=False,
    save_intermediate_structures=False,
    redo_boltz_predict=True,
):
    """
    Run Boltz protein design pipeline.
    
    Args:
        main_dir (str): Main directory path
        yaml_dir (str): Directory containing input yaml files
        version_name (str): Name for version subdirectory
        config (dict): Configuration parameters. If None, uses defaults.
    """
    if config is None:
        config = {
            'recycling_steps': 0,
            'pre_iteration': 30,
            'soft_iteration': 75, 
            'soft_iteration_1': 50,
            'soft_iteration_2': 25,
            'temp_iteration': 45,
            'hard_iteration': 5,
            'semi_greedy_steps': 0,
            'learning_rate_pre': 0.2,
            'learning_rate': 0.1,
            'inter_chain_cutoff': (21.0,),
            'intra_chain_cutoff': 14.0,
            'num_inter_contacts': (2,),
            'num_intra_contacts': 4,
            'e_soft': 0.8,
            'e_soft_1': 0.8,
            'e_soft_2': 1.0,
            'design_algorithm': '3stages',
            'set_train': True,
            'use_temp': True,
            'disconnect_feats': True,
            'disconnect_pairformer': False,
            'distogram_only': True,
            'binder_chain': 'A',
            'non_protein_target': False,
            'increasing_contact_over_itr': False,
            'mask_target_prerun': (False,),
            'optimize_contact_per_binder_pos': (False,),
            'i_con_loss_weights': (1.0,),
            'i_pae_loss_weights': (0.1,),
            'target_plddt_loss_weights': (0.0,),
            'target_chain_ids': None,
            'pdb_path': '',
            'com_loss_weight': 0.0,
            'pocket_conditioning': False,
            'msa_max_seqs':4096,
            'length_min': 95,
            'length_max': 160,
            'helix_loss_min': -0.6,
            'helix_loss_max': -0.2,
            'optimizer_type': 'SGD',
        }


    version_dir = os.path.join(main_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)

    with open(os.path.expanduser(ccd_path), 'rb') as f:
        ccd_lib = pickle.load(f)
    
    results_final_dir = os.path.join(version_dir, 'results_final')
    results_yaml_dir = os.path.join(version_dir, 'results_yaml')
    results_final_dir_apo = os.path.join(version_dir, 'results_final_apo')
    results_yaml_dir_apo = os.path.join(version_dir, 'results_yaml_apo')
    loss_dir = os.path.join(version_dir, 'loss')
    animation_save_dir = os.path.join(version_dir, 'animation')
    fasta_dir = os.path.join(version_dir, 'fasta')
    intermediate_structures_root = os.path.join(version_dir, 'intermediate_structures') if save_intermediate_structures else None

    _dirs_to_make = [results_yaml_dir, results_final_dir, results_yaml_dir_apo, results_final_dir_apo, loss_dir, animation_save_dir, fasta_dir]
    if intermediate_structures_root is not None:
        _dirs_to_make.append(intermediate_structures_root)
    for directory in _dirs_to_make:
        os.makedirs(directory, exist_ok=True)

    # Save config
    config_path = os.path.join(results_final_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    alphabet = list('XXARNDCQEGHILKMFPSTWYV-')
    rmsd_csv_path = os.path.join(results_final_dir, 'rmsd_results.csv')
    csv_exists = os.path.exists(rmsd_csv_path)
    filtered_config = {k: v for k, v in config.items() 
                if k not in ['helix_loss_min', 'helix_loss_max', 'length_min', 'length_max']}
    for yaml_path in Path(yaml_dir).glob('*.yaml'):
        if yaml_path.name.endswith('.yaml'):
                target_binder_input = yaml_path.stem
                for itr in range(design_samples):
                    config['length'] = random.randint(config['length_min'],config['length_max'])
                    filtered_config['length'] = config['length']
                    loss_scales['helix_loss'] = random.uniform(config['helix_loss_min'], config['helix_loss_max'])

                    # Per-iteration intermediate-structure subdir keeps the
                    # per-epoch PDB names tidy: <root>/itr<N>_length<L>/<stage>_epoch<NNNN>.pdb
                    intermediate_dir_itr = None
                    if intermediate_structures_root is not None:
                        intermediate_dir_itr = os.path.join(
                            intermediate_structures_root,
                            f"{target_binder_input}_itr{itr + 1}_length{config['length']}",
                        )
                        os.makedirs(intermediate_dir_itr, exist_ok=True)

                    print('pre-run warm up')
                    input_res_type, plots, loss_history, distogram_history, sequence_history, traj_coords_list, traj_plddt_list = boltz_hallucination(
                        boltz_model,
                        yaml_path,
                        ccd_lib,
                        **filtered_config,
                        pre_run=True,
                        input_res_type=False,
                        loss_scales=loss_scales,
                        chain_to_number=chain_to_number,
                        save_trajectory=save_trajectory,
                        save_intermediate_structures=save_intermediate_structures,
                        intermediate_dir=intermediate_dir_itr,
                    )
                    print('warm up done')
                    output, output_apo, best_batch, best_batch_apo, best_structure, best_structure_apo ,distogram_history_2, sequence_history_2, loss_history_2, con_loss_history, i_con_loss_history, plddt_loss_history, traj_coords_list_2, traj_plddt_list_2, structure, loss_component_history, plddt_history, pae_history = boltz_hallucination(
                        boltz_model,
                        yaml_path,
                        ccd_lib,
                        **filtered_config,
                        pre_run=False,
                        input_res_type=input_res_type,
                        loss_scales=loss_scales,
                        chain_to_number=chain_to_number,
                        save_trajectory=save_trajectory,
                        save_intermediate_structures=save_intermediate_structures,
                        intermediate_dir=intermediate_dir_itr,
                    )
                    loss_history.extend(loss_history_2)
                    distogram_history.extend(distogram_history_2) 
                    sequence_history.extend(sequence_history_2)
                    traj_coords_list.extend(traj_coords_list_2)
                    traj_plddt_list.extend(traj_plddt_list_2)

                    if save_trajectory:
                        from logmd import LogMD
                        logmd = LogMD() 
                        logmd.notebook()
                        print(logmd.url) 
                        atoms = structure.atoms
                        ref_coords = traj_coords_list[-1][:atoms['coords'].shape[0], :]
                        for i in range(len(traj_coords_list)):
                            current_coords = traj_coords_list[i][:atoms['coords'].shape[0], :]
                            aligned_coords = align_points(current_coords, ref_coords)
                            structure.atoms['coords'] = aligned_coords
                            structure.atoms["is_present"] = True
                            pdb_str = to_pdb(structure, plddts=traj_plddt_list[i])
                            pdb_str = "\n".join([line for line in pdb_str.split("\n") if line.startswith("ATOM") or line.startswith("HETATM")])
                            try:
                                logmd(pdb_str)
                            except Exception as e:
                                # A diverged frame (e.g. exploded coords overflowing
                                # PDB's fixed-width columns) makes logmd's parser
                                # raise; skip that frame instead of killing an
                                # otherwise-finished run.
                                print(f"[logmd] skipped trajectory frame {i}: {type(e).__name__}: {e}")

                    # Final-fold metrics: skip cleanly if OOM/exception
                    # zeroed out `output` so plots and animations still run.
                    if output is not None and output_apo is not None:
                        print('-' * 100)
                        print(f"Holo Protein PLDDT: {output['plddt'][:config['length']].mean():.3f}")
                        print(f"Apo Protein PLDDT: {output_apo['plddt'][:config['length']].mean():.3f}")
                        print('-' * 100)
                        print(f"Holo Complex PLDDT: {float(output['complex_plddt'].detach().cpu().numpy()):.3f}")
                        print(f"Apo Complex PLDDT: {float(output_apo['complex_plddt'].detach().cpu().numpy()):.3f}")
                        print('-' * 100)

                        ca_coords = get_ca_coords(output['coords'], best_batch, binder_chain=config['binder_chain']).detach().cpu().numpy()
                        ca_coords_apo = get_ca_coords(output_apo['coords'], best_batch_apo, binder_chain='A').detach().cpu().numpy()

                        rmsd = np_rmsd(ca_coords, ca_coords_apo)
                        print('-' * 100)
                        print("rmsd", rmsd)
                        print('-' * 100)
                    else:
                        print("[run_boltz_design] final-fold output missing; "
                              "continuing to loss/animation plots.")
                        rmsd = float('nan')

                    if loss_dir:
                        os.makedirs(loss_dir, exist_ok=True)
                    # Single-panel total loss + three category figures (distogram,
                    # confidence, coords). _LOSS_GROUPS + _plot_loss_group at the
                    # top of this module drive layout; only series with data this
                    # run are plotted. Each subplot has its OWN y-axis (no
                    # sharing) so widely-different magnitudes coexist cleanly.
                    try:
                        plt.style.use('dark_background')
                        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
                        fig.patch.set_facecolor('#1C1C1C')
                        ax.plot(loss_component_history.get('total_loss', loss_history),
                                color='#00ff99', linewidth=2)
                        ax.set_xlabel('Logged Steps', fontsize=12)
                        ax.set_ylabel('Total Loss', fontsize=12)
                        ax.set_title('Total Loss History', fontsize=14, pad=15)
                        ax.grid(True, linestyle='--', alpha=0.3)
                        plt.tight_layout(pad=3.0)
                        if loss_dir:
                            plt.savefig(
                                os.path.join(loss_dir, f'{target_binder_input}_loss_history_itr{itr + 1}_length{config["length"]}.png'),
                                facecolor='#1C1C1C', edgecolor='none', bbox_inches='tight', dpi=300)
                        plt.show()
                        plt.close(fig)

                        distogram_ani, sequence_ani, plddt_ani, pae_ani = visualize_training_history(best_batch,loss_history, sequence_history, distogram_history, config["length"], binder_chain =config['binder_chain'], save_dir=animation_save_dir, save_filename=f"{target_binder_input}_itr{itr + 1}_length{config['length']}", plddt_history=plddt_history, pae_history=pae_history)
                        if show_animation:
                            panels = f"<div style='flex:0.4'>{distogram_ani.to_jshtml()}</div><div style='flex:0.6'>{sequence_ani.to_jshtml()}</div>"
                            if plddt_ani is not None:
                                panels += f"<div style='flex:0.6'>{plddt_ani.to_jshtml()}</div>"
                            if pae_ani is not None:
                                panels += f"<div style='flex:0.4'>{pae_ani.to_jshtml()}</div>"
                            display(HTML(f"<div style='display:flex;gap:10px;flex-wrap:wrap'>{panels}</div>"))
                    except Exception as e:
                        print(f"Error plotting total-loss history: {str(e)}")
                        continue

                    # Total-loss CSV (also captures con/icon for compatibility).
                    try:
                        if loss_dir:
                            main_series = {
                                'total_loss': loss_component_history.get('total_loss', loss_history),
                                'intra_contact_loss': con_loss_history,
                                'inter_contact_loss': i_con_loss_history,
                            }
                            main_csv = os.path.join(
                                loss_dir,
                                f'{target_binder_input}_loss_history_itr{itr + 1}_length{config["length"]}.csv')
                            main_cols = list(main_series.keys())
                            n_steps = max((len(s) for s in main_series.values()), default=0)
                            with open(main_csv, 'w', newline='') as f:
                                w = csv.writer(f)
                                w.writerow(['step'] + main_cols)
                                for t in range(n_steps):
                                    row = [t]
                                    for k in main_cols:
                                        s = main_series[k]
                                        row.append(f'{s[t]:.4f}' if t < len(s) else '')
                                    w.writerow(row)
                            print(f"[loss] total loss history -> {main_csv}")
                    except Exception as e:
                        print(f"Error writing loss-history CSV: {str(e)}")

                    # Three category figures: distogram / confidence / coords.
                    # Each has its own png + csv, atom_pair distogram column
                    # carries the method suffix so downstream tools can group it.
                    try:
                        _apd_method = config.get("atom_pair_distogram_loss_type", "expected")
                        _overrides_dist = {
                            'atom_pair_distogram_loss': f'atom_pair_distogram_loss_{_apd_method}',
                        }
                        _suffix = f'_itr{itr + 1}_length{config["length"]}'
                        for _gname, _specs, _title, _overrides in [
                            ('distogram',  _LOSS_GROUPS['distogram'],
                             'Distogram Losses', _overrides_dist),
                            ('confidence', _LOSS_GROUPS['confidence'],
                             'Confidence Losses', None),
                            ('coords',     _LOSS_GROUPS['coords'],
                             'Coordinate Losses', None),
                        ]:
                            if not loss_dir:
                                break
                            _png = os.path.join(
                                loss_dir,
                                f'{target_binder_input}_{_gname}_loss_history{_suffix}.png')
                            _csv = os.path.join(
                                loss_dir,
                                f'{target_binder_input}_{_gname}_loss_history{_suffix}.csv')
                            wrote = _plot_loss_group(
                                loss_component_history, _specs,
                                _png, _csv, _title,
                                csv_col_overrides=_overrides)
                            if wrote:
                                print(f"[loss] {_gname} loss history -> {_csv}")
                    except Exception as e:
                        print(f"Error plotting category loss histories: {str(e)}")

                    # ---- Atom-pair distance restraint diagnostics (--atom_pairs)
                    # Plot the per-pair distance the distogram loss optimizes
                    # (expected distance E[d]; argmax-bin "mode"; and, in full
                    # mode, the coord distance) plus the per-pair loss vs. logged
                    # step, and dump a CSV -- both into the animation folder so
                    # the restraint can be debugged. Own try/except (warn only).
                    try:
                        import re as _re
                        ap_keys = [k for k in loss_component_history
                                   if k.startswith('atom_pair|dist|')]
                        if ap_keys:
                            labels = [k.split('atom_pair|dist|', 1)[1] for k in ap_keys]
                            _apd_method = config.get("atom_pair_distogram_loss_type", "expected")

                            def _win(lab):
                                m = _re.search(r'\[([-\d.]+),\s*([-\d.]+)\]A\s*$', lab)
                                return (float(m.group(1)), float(m.group(2))) if m else (None, None)

                            def _h(kind, lab):
                                return loss_component_history.get(f'atom_pair|{kind}|{lab}', [])

                            plt.style.use('dark_background')
                            fig, (axd, axl) = plt.subplots(1, 2, figsize=(13, 4.5))
                            fig.patch.set_facecolor('#1C1C1C')
                            cmap = plt.get_cmap('tab10')
                            for i, lab in enumerate(labels):
                                c = cmap(i % 10)
                                short = lab.split(' [')[0]
                                axd.plot(_h('dist', lab), color=c, linewidth=2,
                                         label=f'{short}  E[d]')
                                mode = _h('mode', lab)
                                if mode:
                                    axd.plot(mode, color=c, linewidth=1, linestyle=':',
                                             alpha=0.7, label=f'{short}  mode')
                                cdist = _h('cdist', lab)
                                if cdist:
                                    axd.plot(cdist, color=c, linewidth=1.6, linestyle='--',
                                             label=f'{short}  coord d')
                                lo, hi = _win(lab)
                                if lo is not None:
                                    axd.axhspan(lo, hi, color=c, alpha=0.12)
                                axl.plot(_h('loss', lab), color=c, linewidth=2, label=short)
                            axd.set_xlabel('Logged Steps', fontsize=12)
                            axd.set_ylabel('Atom-pair distance (A)', fontsize=12)
                            axd.set_title('Atom-pair distance (shaded = target window)',
                                          fontsize=13, pad=12)
                            axd.grid(True, linestyle='--', alpha=0.3)
                            axd.legend(fontsize=7, loc='best')
                            axl.set_xlabel('Logged Steps', fontsize=12)
                            axl.set_ylabel('Atom-pair distogram loss', fontsize=12)
                            axl.set_title(f'Atom-pair distogram loss ({_apd_method})',
                                          fontsize=13, pad=12)
                            axl.grid(True, linestyle='--', alpha=0.3)
                            axl.legend(fontsize=7, loc='best')
                            plt.tight_layout(pad=2.5)
                            os.makedirs(animation_save_dir, exist_ok=True)
                            ap_png = os.path.join(
                                animation_save_dir,
                                f'{target_binder_input}_atom_pair_itr{itr + 1}_length{config["length"]}.png')
                            fig.savefig(ap_png, facecolor='#1C1C1C', edgecolor='none',
                                        bbox_inches='tight', dpi=200)
                            plt.show()
                            plt.close(fig)

                            # CSV: one row per (logged step, pair) with everything.
                            ap_csv = os.path.join(
                                animation_save_dir,
                                f'{target_binder_input}_atom_pair_itr{itr + 1}_length{config["length"]}.csv')
                            with open(ap_csv, 'w', newline='') as f:
                                w = csv.writer(f)
                                w.writerow(['step', 'pair', 'lo', 'hi', 'exp_dist',
                                            'mode_dist', 'p_window',
                                            f'distogram_loss_{_apd_method}',
                                            'coord_dist', 'coord_loss'])
                                for lab in labels:
                                    lo, hi = _win(lab)
                                    short = lab.split(' [')[0]
                                    dist = _h('dist', lab); mode = _h('mode', lab)
                                    pwin = _h('pwin', lab); dloss = _h('loss', lab)
                                    cdist = _h('cdist', lab); closs = _h('closs', lab)
                                    for t in range(len(dist)):
                                        g = lambda s, j: (f'{s[j]:.4f}' if j < len(s) else '')
                                        w.writerow([t, short, lo, hi, g(dist, t),
                                                    g(mode, t), g(pwin, t), g(dloss, t),
                                                    g(cdist, t), g(closs, t)])
                            print(f"[atom_pairs] distance/loss diagnostics -> {ap_png} , {ap_csv}")
                    except Exception as e:
                        print(f"Error plotting atom-pair diagnostics: {str(e)}")

                    # ---- Motif scaffolding loss diagnostics (--motif_*) --------
                    # Plot the supervised motif losses vs. logged step and dump a
                    # CSV into the loss folder. The distogram CE (fast-mode safe)
                    # is present whenever motif scaffolding is active; the Kabsch
                    # Cα-RMSD coord loss appears only in full mode. Keys live in
                    # loss_component_history exactly as the other components do.
                    # Own try/except (warn only).
                    try:
                        motif_specs = [
                            ('motif_distogram_loss', 'Motif Distogram Loss (bin-center MSE; A^2)', '#00ddff'),
                            ('motif_coords_loss', 'Motif Backbone RMSD (N,CA,C,CB; A)', '#ffaa00'),
                            ('motif_fape_loss', 'Motif Sidechain FAPE (A)', '#ff66aa'),
                        ]
                        motif_present = [(k, lbl, c) for k, lbl, c in motif_specs
                                         if loss_component_history.get(k)]
                        if motif_present and loss_dir:
                            plt.style.use('dark_background')
                            n = len(motif_present)
                            fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
                            if n == 1:
                                axes = [axes]
                            fig.patch.set_facecolor('#1C1C1C')
                            for ax, (k, lbl, c) in zip(axes, motif_present):
                                ax.plot(loss_component_history[k], color=c, linewidth=2)
                                ax.set_xlabel('Logged Steps', fontsize=12)
                                ax.set_ylabel(lbl, fontsize=12)
                                ax.set_title(f'{lbl} History', fontsize=13, pad=12)
                                ax.grid(True, linestyle='--', alpha=0.3)
                            plt.tight_layout(pad=2.5)
                            os.makedirs(loss_dir, exist_ok=True)
                            motif_png = os.path.join(
                                loss_dir,
                                f'{target_binder_input}_motif_loss_itr{itr + 1}_length{config["length"]}.png')
                            fig.savefig(motif_png, facecolor='#1C1C1C', edgecolor='none',
                                        bbox_inches='tight', dpi=200)
                            plt.show()
                            plt.close(fig)

                            # CSV: one row per logged step; full-mode-only coord
                            # loss is blank where its (shorter) series ends.
                            motif_csv = os.path.join(
                                loss_dir,
                                f'{target_binder_input}_motif_loss_itr{itr + 1}_length{config["length"]}.csv')
                            present_keys = [k for k, _, _ in motif_present]
                            n_steps = max(len(loss_component_history[k]) for k in present_keys)
                            with open(motif_csv, 'w', newline='') as f:
                                w = csv.writer(f)
                                w.writerow(['step'] + present_keys)
                                for t in range(n_steps):
                                    row = [t]
                                    for k in present_keys:
                                        s = loss_component_history[k]
                                        row.append(f'{s[t]:.4f}' if t < len(s) else '')
                                    w.writerow(row)
                            print(f"[motif] loss diagnostics -> {motif_png} , {motif_csv}")
                    except Exception as e:
                        print(f"Error plotting motif loss diagnostics: {str(e)}")

                    with open(rmsd_csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if not csv_exists:
                            writer.writerow(['target', 'length', 'iteration', 'apo_holo_rmsd', 'complex_plddt', 'iptm',  'helix_loss'])
                            csv_exists = True
                        if output is not None:
                            row_plddt = output['complex_plddt'].item()
                            row_iptm = output['iptm'].item()
                        else:
                            row_plddt = float('nan')
                            row_iptm = float('nan')
                        writer.writerow([target_binder_input, config['length'], itr + 1, rmsd, row_plddt, row_iptm, loss_scales['helix_loss']])

                    result_yaml = os.path.join(results_yaml_dir, f'{target_binder_input}_results_itr{itr + 1}_length{config["length"]}.yaml')
                    result_yaml_apo = os.path.join(results_yaml_dir_apo, f'{target_binder_input}_results_itr{itr + 1}_length{config["length"]}.yaml')
                    best_batch_cpu = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in best_batch.items()}
                    best_sequence = ''.join([alphabet[i] for i in np.argmax(best_batch_cpu['res_type'][best_batch_cpu['entity_id']==chain_to_number[config['binder_chain']],:], axis=-1)])
                    print("best_sequence", best_sequence)

                    # Always emit the designed binder sequence as FASTA, even
                    # when the final fold OOM'd — best_sequence is derived from
                    # best_batch and is independent of `output`.
                    fasta_tag = f"{target_binder_input}_itr{itr + 1}_length{config['length']}"
                    fasta_path = os.path.join(fasta_dir, f"{fasta_tag}.fasta")
                    fold_status = "ok" if output is not None else "fold_failed"
                    with open(fasta_path, 'w') as ff:
                        ff.write(f">{fasta_tag} chain={config['binder_chain']} length={len(best_sequence)} status={fold_status}\n")
                        for i in range(0, len(best_sequence), 60):
                            ff.write(best_sequence[i:i+60] + "\n")
                    print(f"Wrote FASTA: {fasta_path}")


                    shutil.copy2(yaml_path, result_yaml)
                    with open(result_yaml, 'r') as f:
                        data = yaml.safe_load(f)
                    chain_num = chain_to_number[config['binder_chain']]
                    data['sequences'][chain_num]['protein']['sequence'] = best_sequence
                    data.pop('constraints', None)

                    # Convert any MSA files from npz to a3m format
                    for seq in data['sequences']:
                        if 'protein' in seq and 'msa' in seq['protein'] and seq['protein']['msa']:
                            seq['protein']['msa'] = seq['protein']['msa'].replace('.npz', '.a3m')

                    with open(result_yaml, 'w') as f:
                        yaml.dump(data, f)

                    shutil.copy2(result_yaml, result_yaml_apo)
                    with open(result_yaml_apo, 'r') as f:
                        data_apo = yaml.safe_load(f)
                    data_apo['sequences'] = [data_apo['sequences'][chain_to_number[config['binder_chain']]]]
                    data_apo.pop('constraints', None)   

                    with open(result_yaml_apo, 'w') as f:
                        yaml.dump(data_apo, f)

                    if redo_boltz_predict:
                        subprocess.run([boltz_path, 'predict', str(result_yaml), '--out_dir', str(results_final_dir), '--write_full_pae'])
                        subprocess.run([boltz_path, 'predict', str(result_yaml_apo), '--out_dir', str(results_final_dir_apo), '--write_full_pae'])
                    elif output is not None and output_apo is not None:
                        save_confidence_scores(results_final_dir, output, best_structure, f"{target_binder_input}_results_itr{itr + 1}_length{config['length']}", 0)
                        save_confidence_scores(results_final_dir_apo, output_apo, best_structure_apo, f"{target_binder_input}_results_itr{itr + 1}_length{config['length']}", 0)
                    else:
                        print("[run_boltz_design] skipping save_confidence_scores: "
                              "final-fold output missing.")
                    gc.collect()
                    torch.cuda.empty_cache()