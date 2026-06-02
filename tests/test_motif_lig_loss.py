"""Synthetic unit tests for get_motif_coords_loss with ligand carry-along.

Exercises four properties without loading Boltz1:
  T1 backward-compat: backbone-only path is byte-identical to before
  T2 pose-invariance: rigid transform of full system -> loss ~ 0
  T3 ligand response: perturbing only the ligand increases the loss
  T4 gradient flow: gradient w.r.t. ligand coords is nonzero

Run:  conda activate boltz_design  &&  python tests/test_motif_lig_loss.py
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'boltzdesign'))
from boltzdesign_utils import (parse_motif_ligand_spec, parse_motif_residue_spec,
                               get_motif_coords_loss, kabsch_align)


def _rigid_transform(xyz, seed):
    """Random rotation + translation applied to a [N,3] tensor."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t = torch.randn(3, generator=g) * 5.0
    return xyz @ Q.T + t, Q, t


def synthetic_motif_atoms():
    """3 protein residues x 4 backbone atoms (N,CA,C,CB) + 5 ligand atoms.

    Returns: bb_ref [12,3], lig_ref [5,3], and the matching index tensors
    into a flat [n_atoms, 3] global atom array (bb first, lig after).
    """
    rng = np.random.default_rng(42)
    ca = np.array([[0., 0., 0.], [3.8, 0.5, 0.], [7.4, 0.0, 1.0]])    # 3 CAs
    bb = []
    for c in ca:
        n  = c + rng.normal(0, 0.3, 3)
        ca_ = c
        cc = c + rng.normal(0, 0.3, 3)
        cb = c + rng.normal(0, 0.5, 3)
        bb.extend([n, ca_, cc, cb])
    bb = np.asarray(bb, dtype=np.float32)                              # [12,3]
    # Ligand (5 atoms) sits ~4 A above the CAs
    lig = np.stack([ca[1] + np.array([dx, dy, 4.0])
                    for dx in (-2, 0, 2) for dy in (-1, 1)][:5]).astype(np.float32)
    return bb, lig                                                     # [12,3],[5,3]


def t1_backcompat():
    """Backbone-only path: equals the original kabsch_align RMSD."""
    bb, _ = synthetic_motif_atoms()
    pred_global, _, _ = _rigid_transform(torch.from_numpy(bb), seed=7)
    pred_global = pred_global + 0.1 * torch.randn_like(pred_global)
    coords = pred_global[None]                                         # [1,12,3]
    bb_pred_idx = torch.arange(bb.shape[0], dtype=torch.long)
    bb_ref = torch.from_numpy(bb)
    loss = get_motif_coords_loss(coords, bb_pred_idx, bb_ref)
    # Manual reference
    aligned = kabsch_align(coords[0], bb_ref)
    ref_loss = torch.sqrt(((aligned - bb_ref) ** 2).sum(-1).mean() + 1e-8)
    err = abs(loss.item() - ref_loss.item())
    print(f"T1 backcompat: loss={loss.item():.4e}  ref={ref_loss.item():.4e}  |Δ|={err:.2e}")
    assert err < 1e-6, "backbone-only path drifted from kabsch_align baseline"


def t2_pose_invariance():
    """Apply same rigid transform to BB + ligand => loss ~ 0."""
    bb, lig = synthetic_motif_atoms()
    full = np.concatenate([bb, lig], axis=0)
    full_t, _, _ = _rigid_transform(torch.from_numpy(full), seed=11)
    coords = full_t[None]
    n_bb = bb.shape[0]
    bb_pred_idx = torch.arange(n_bb, dtype=torch.long)
    lig_pred_idx = torch.arange(n_bb, n_bb + lig.shape[0], dtype=torch.long)
    bb_ref = torch.from_numpy(bb)
    lig_ref = torch.from_numpy(lig)
    loss = get_motif_coords_loss(coords, bb_pred_idx, bb_ref,
                                 lig_pred_idx, lig_ref).item()
    print(f"T2 pose-invariance: combined RMSD under rigid xform = {loss:.4e}")
    assert loss < 1e-3, f"loss {loss:.3e} not ~0 under rigid transform"


def t3_lig_response():
    """Identity placement of BB; shift the ligand by 1.5 A => ligand RMSD ~1.5."""
    bb, lig = synthetic_motif_atoms()
    full = np.concatenate([bb, lig.copy()], axis=0)
    coords = torch.from_numpy(full)[None].clone()
    n_bb = bb.shape[0]
    coords[0, n_bb:, 0] += 1.5                              # shift X by 1.5 A
    bb_pred_idx = torch.arange(n_bb, dtype=torch.long)
    lig_pred_idx = torch.arange(n_bb, n_bb + lig.shape[0], dtype=torch.long)
    bb_ref = torch.from_numpy(bb)
    lig_ref = torch.from_numpy(lig)
    loss, stats = get_motif_coords_loss(
        coords, bb_pred_idx, bb_ref, lig_pred_idx, lig_ref, return_stats=True)
    print(f"T3 ligand response: bb_rmsd={stats['bb_rmsd']:.4e}  "
          f"lig_rmsd={stats['lig_rmsd']:.4e}  combined={loss.item():.4e}")
    assert stats['bb_rmsd'] < 1e-3, "backbone RMSD should ~0 (identity placement)"
    assert abs(stats['lig_rmsd'] - 1.5) < 1e-2, "ligand RMSD should ~1.5 A"


def t4_gradient_flow():
    """Gradient w.r.t. ligand atom coords is nonzero."""
    bb, lig = synthetic_motif_atoms()
    full = np.concatenate([bb, lig], axis=0)
    full_t, _, _ = _rigid_transform(torch.from_numpy(full), seed=23)
    full_t[12:] += 0.4 * torch.randn(5, 3)                  # perturb ligand
    coords = full_t[None].clone().requires_grad_(True)
    n_bb = bb.shape[0]
    bb_pred_idx = torch.arange(n_bb, dtype=torch.long)
    lig_pred_idx = torch.arange(n_bb, n_bb + lig.shape[0], dtype=torch.long)
    bb_ref = torch.from_numpy(bb)
    lig_ref = torch.from_numpy(lig)
    loss = get_motif_coords_loss(coords, bb_pred_idx, bb_ref,
                                 lig_pred_idx, lig_ref)
    loss.backward()
    g = coords.grad[0]
    bb_g = g[:n_bb].norm().item()
    lig_g = g[n_bb:].norm().item()
    print(f"T4 gradient flow: loss={loss.item():.4e}  "
          f"|grad bb|={bb_g:.4e}  |grad lig|={lig_g:.4e}")
    assert lig_g > 1e-4, "no gradient reaching ligand coords"


def t5_parser():
    """Parser accepts CHAIN+RESNUM tokens, rejects bare numbers."""
    assert parse_motif_ligand_spec("") == []
    assert parse_motif_ligand_spec("B1") == [('B', 1)]
    assert parse_motif_ligand_spec("B1,C401") == [('B', 1), ('C', 401)]
    assert parse_motif_ligand_spec(" B1 , C401 ") == [('B', 1), ('C', 401)]
    try:
        parse_motif_ligand_spec("1")
        raise AssertionError("expected ValueError on bare residue number")
    except ValueError:
        pass
    try:
        parse_motif_ligand_spec("B")
        raise AssertionError("expected ValueError on chain without resnum")
    except ValueError:
        pass
    print("T5 parser: OK")


def t6_motif_residue_parser():
    """parse_motif_residue_spec: chain prefix required, ranges, multi-chain."""
    assert parse_motif_residue_spec("") == []
    assert parse_motif_residue_spec("A57") == [('A', 57)]
    assert parse_motif_residue_spec("A57,A102,A195") == [
        ('A', 57), ('A', 102), ('A', 195)]
    assert parse_motif_residue_spec("B57,A102,C195") == [
        ('B', 57), ('A', 102), ('C', 195)]
    # ranges within a chain
    assert parse_motif_residue_spec("A10-12,B57") == [
        ('A', 10), ('A', 11), ('A', 12), ('B', 57)]
    # multi-letter chain ID (e.g. CIF auth chain "AB")
    assert parse_motif_residue_spec("AB5,C10") == [('AB', 5), ('C', 10)]
    # bare numbers must be rejected (no --motif_chain default now)
    try:
        parse_motif_residue_spec("57,102")
        raise AssertionError("expected ValueError on bare residue numbers")
    except ValueError:
        pass
    try:
        parse_motif_residue_spec("A")
        raise AssertionError("expected ValueError on chain without resnum")
    except ValueError:
        pass
    print("T6 motif residue parser: OK")


def t7_motif_chain_flag_removed():
    """--motif_chain is no longer registered in the CLI."""
    import subprocess
    out = subprocess.run([sys.executable, 'boltzdesign.py', '--help'],
                         capture_output=True, text=True,
                         cwd=os.path.join(os.path.dirname(__file__), '..'))
    assert '--motif_chain' not in out.stdout, \
        "--motif_chain should be removed from CLI"
    assert '--motif_residues' in out.stdout
    assert '--motif_ligand_residues' in out.stdout
    print("T7 --motif_chain removed: OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    t5_parser()
    t6_motif_residue_parser()
    t7_motif_chain_flag_removed()
    t1_backcompat()
    t2_pose_invariance()
    t3_lig_response()
    t4_gradient_flow()
    print("\nall tests passed")
