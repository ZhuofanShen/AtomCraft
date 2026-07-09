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
                               get_motif_coords_loss, kabsch_align,
                               parse_motif_islands_spec, island_length,
                               get_motif_fape_loss, _rigid_frames,
                               _placement_to_positions)


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
    # :ATOMS suffix (with_atoms=True) -> (chain, resnum, atoms)
    assert parse_motif_ligand_spec("B1", with_atoms=True) == [('B', 1, None)]
    assert parse_motif_ligand_spec("C1:ALL", with_atoms=True) == [('C', 1, 'ALL')]
    assert parse_motif_ligand_spec("B1:FE,N1 C401", with_atoms=True) == [
        ('B', 1, ['FE', 'N1']), ('C', 401, None)]
    print("T5 parser: OK")


def t6_motif_residue_parser():
    """parse_motif_residue_spec: whitespace-separated tokens, chain prefix
    required, ranges, multi-chain. Comma-separated residue lists are rejected
    (commas only group atoms inside :ATOMS)."""
    assert parse_motif_residue_spec("") == []
    assert parse_motif_residue_spec("A57") == [('A', 57)]
    assert parse_motif_residue_spec("A57 A102 A195") == [
        ('A', 57), ('A', 102), ('A', 195)]
    assert parse_motif_residue_spec("B57 A102 C195") == [
        ('B', 57), ('A', 102), ('C', 195)]
    # ranges within a chain
    assert parse_motif_residue_spec("A10-12 B57") == [
        ('A', 10), ('A', 11), ('A', 12), ('B', 57)]
    # multi-letter chain ID (e.g. CIF auth chain "AB")
    assert parse_motif_residue_spec("AB5 C10") == [('AB', 5), ('C', 10)]
    # bare numbers must be rejected (no --motif_chain default now)
    try:
        parse_motif_residue_spec("57 102")
        raise AssertionError("expected ValueError on bare residue numbers")
    except ValueError:
        pass
    try:
        parse_motif_residue_spec("A")
        raise AssertionError("expected ValueError on chain without resnum")
    except ValueError:
        pass
    # legacy comma-separated residue lists are NO LONGER accepted
    for bad in ("A57,A102,A195", "A57,A102", "A10-12,B57"):
        try:
            parse_motif_residue_spec(bad)
            raise AssertionError(f"expected ValueError on comma list '{bad}'")
        except ValueError:
            pass
    # 2-tuple contract is unchanged even when :ATOMS is present
    assert parse_motif_residue_spec("A47:SG B53:CA,C") == [('A', 47), ('B', 53)]
    # with_atoms=True -> (chain, resnum, atoms); whitespace tokens, commas group
    # atoms (only inside :ATOMS), :ALL sentinel, range applies atoms to each
    assert parse_motif_residue_spec("A47:SG  B53:CA,C  A195:ALL", with_atoms=True) == [
        ('A', 47, ['SG']), ('B', 53, ['CA', 'C']), ('A', 195, 'ALL')]
    assert parse_motif_residue_spec("A10-12:CA", with_atoms=True) == [
        ('A', 10, ['CA']), ('A', 11, ['CA']), ('A', 12, ['CA'])]
    # a stray ':' inside the atom list (comma-joined colon tokens) is rejected
    try:
        parse_motif_residue_spec("A47:SG,B53:CA", with_atoms=True)
        raise AssertionError("expected ValueError on ':' inside atom name")
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


def t8_islands_reuse_residue_parser():
    """--motif_unindex_residues reuses the --motif_residues grammar; consecutive
    residues group into one rigid island, breaks start a new one, atoms carried."""
    isl = parse_motif_islands_spec("A38-40 A57:SG")
    assert [[(r['chain'], r['resnum'], r['intra_offset'], r['atoms']) for r in g]
            for g in isl] == [
        [('A', 38, 0, None), ('A', 39, 1, None), ('A', 40, 2, None)],
        [('A', 57, 0, ['SG'])]], isl
    assert [island_length(g) for g in isl] == [3, 1]
    assert len(parse_motif_islands_spec("A38 A42")) == 2          # independent
    # consecutive whitespace tokens group into one island
    assert [[(r['chain'], r['resnum']) for r in g]
            for g in parse_motif_islands_spec("A38 A39 B10")] == [
        [('A', 38), ('A', 39)], [('B', 10)]]
    # comma-separated residue lists are NO LONGER accepted (commas only group atoms)
    for bad in ("A38,A39,B10", "A38,3,A42"):
        try:
            parse_motif_islands_spec(bad)
            raise AssertionError(f"expected ValueError on comma list '{bad}'")
        except ValueError:
            pass
    try:
        parse_motif_islands_spec("A38 A38")
        raise AssertionError("expected ValueError on duplicate residue")
    except ValueError:
        pass
    # fixed-gap spacer: "A38 3 A42 A170" -> island [A38@0, A42@4] + island [A170]
    isl2 = parse_motif_islands_spec("A38:N,CA,C,O,CB 3 A42:N,CA,C,O,CB A170:N,CA,C,O,CB")
    assert [[(r['chain'], r['resnum'], r['intra_offset']) for r in g]
            for g in isl2] == [[('A', 38, 0), ('A', 42, 4)], [('A', 170, 0)]], isl2
    assert [island_length(g) for g in isl2] == [5, 1]
    assert isl2[0][0]['atoms'] == ['N', 'CA', 'C', 'O', 'CB']
    assert _placement_to_positions([10, 50], isl2) == [10, 14, 50]   # gap preserved
    # spacer 0 == adjacent
    assert [r['intra_offset'] for r in parse_motif_islands_spec("A38 0 A39")[0]] == [0, 1]
    # malformed spacers raise
    for bad in ("3 A38", "A38 3 4 A42", "A38 3"):
        try:
            parse_motif_islands_spec(bad)
            raise AssertionError(f"expected ValueError on spacer spec '{bad}'")
        except ValueError:
            pass
    print("T8 islands reuse residue parser + fixed-gap spacer: OK")


def t9_fape_ligand():
    """FAPE scores carried ligand atoms in the motif frames (the assembly's
    loc_ref construction): pose-invariant, responds to ligand perturbation,
    gradient reaches the ligand."""
    torch.manual_seed(1)
    R = 3
    base = torch.randn(R, 4, 3) * 3.0                 # per-residue N,CA,C,CB
    lig_ref = torch.randn(4, 3) * 4.0
    ref_atoms = torch.cat([base.reshape(-1, 3), lig_ref], dim=0)   # [16,3]
    fr_idx = torch.tensor([[4 * i, 4 * i + 1, 4 * i + 2] for i in range(R)])
    sc_idx = torch.tensor([4 * i + 3 for i in range(R)] + [12, 13, 14, 15])
    # loc_ref from REF coords, exactly as the boltzdesign_utils FAPE assembly does
    Rr, tr = _rigid_frames(ref_atoms[fr_idx[:, 0]], ref_atoms[fr_idx[:, 1]],
                           ref_atoms[fr_idx[:, 2]])
    diff = ref_atoms.index_select(0, sc_idx)[None, :, :] - tr[:, None, :]
    loc_ref = torch.einsum('fij,fsj->fsi', Rr.transpose(-1, -2), diff)
    # pred = rigid transform of the whole system -> FAPE ~ 0 (frame-invariant)
    pred, _Q, _t = _rigid_transform(ref_atoms, seed=7)
    f0 = get_motif_fape_loss(pred.unsqueeze(0), fr_idx, sc_idx, loc_ref).item()
    assert f0 < 1e-3, f"pose-invariance broken: {f0}"
    # perturb one LIGAND atom -> FAPE increases
    pred2 = pred.clone(); pred2[14] += torch.tensor([2.0, 0.0, 0.0])
    f1 = get_motif_fape_loss(pred2.unsqueeze(0), fr_idx, sc_idx, loc_ref).item()
    assert f1 > f0 + 0.1, f"ligand perturbation ignored: {f0}->{f1}"
    pc = pred2.clone().requires_grad_(True)
    get_motif_fape_loss(pc.unsqueeze(0), fr_idx, sc_idx, loc_ref).backward()
    lig_g = pc.grad[12:16].norm().item()
    assert lig_g > 1e-4, "no gradient reaching ligand coords"
    print(f"T9 FAPE+ligand: pose~{f0:.2e}  perturbed={f1:.3f}  |grad lig|={lig_g:.3e}  OK")


def t10_sliding_fape_indexing():
    """Replicates the sliding-FAPE rebuild index math (_fill_slide_fape): a
    sliding island placed at slide_state position s pulls the binder atoms at
    positions [s, s+1]; FAPE vs. the CONSTANT loc_ref is ~0 when the binder holds
    the (rigidly moved) motif geometry there, and large at a wrong placement."""
    torch.manual_seed(3)
    length = 10
    name_col = {'N': 0, 'CA': 1, 'C': 2, 'CB': 3}
    # binder global layout: position p holds atoms N,CA,C,CB at p*4 + {0,1,2,3}
    binder_map = torch.tensor([[p * 4 + c for c in range(4)] for p in range(length)])
    ref_res = torch.randn(2, 4, 3) * 3.0              # [res, (N,CA,C,CB), 3]
    # loc_ref: 2 frames x 2 scored CB (canonical residue order)
    Rr, tr = _rigid_frames(ref_res[:, 0], ref_res[:, 1], ref_res[:, 2])
    diff = ref_res[:, 3][None, :, :] - tr[:, None, :]
    loc_ref = torch.einsum('fij,fsj->fsi', Rr.transpose(-1, -2), diff)
    # binder coords: rigidly place the motif at positions [s, s+1]; rest arbitrary
    coords = torch.randn(length * 4, 3) * 10.0
    Q = torch.linalg.qr(torch.randn(3, 3))[0]
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    tt = torch.randn(3) * 5.0
    s = 4
    for ri, p in enumerate([s, s + 1]):
        for a in range(4):
            coords[p * 4 + a] = ref_res[ri, a] @ Q.T + tt
    fr_cols = torch.tensor([name_col['N'], name_col['CA'], name_col['C']])
    fr_slot = torch.tensor([0, 1]); sc_slot = torch.tensor([0, 1])
    sc_col = torch.tensor([name_col['CB'], name_col['CB']])
    # correct placement -> FAPE ~ 0
    pos = torch.tensor([s, s + 1])
    fr_pred = binder_map[pos[fr_slot]][:, fr_cols]      # mirrors _fill_slide_fape
    sc_pred = binder_map[pos[sc_slot], sc_col]
    f = get_motif_fape_loss(coords.unsqueeze(0), fr_pred, sc_pred, loc_ref).item()
    assert f < 1e-3, f"sliding-FAPE indexing/pose broke: {f}"
    # wrong placement (s=0, where the binder does NOT hold the motif) -> large
    pos2 = torch.tensor([0, 1])
    f2 = get_motif_fape_loss(coords.unsqueeze(0),
                             binder_map[pos2[fr_slot]][:, fr_cols],
                             binder_map[pos2[sc_slot], sc_col], loc_ref).item()
    assert f2 > 0.5, f"expected large FAPE at wrong placement: {f2}"
    print(f"T10 sliding-FAPE indexing: placed~{f:.2e}  misplaced={f2:.3f}  OK")


def t11_motif_mapping():
    """Motif residue -> FINAL binder position mapping (the dict boltz_hallucination
    populates): fixed residues map via binder_pos, sliding via the final
    slide_state through _placement_to_positions. Single binder chain -> all values
    carry that chain (e.g. 'A27')."""
    binder_chain = 'A'
    mapping = {}
    # fixed: A30,C28 -> binder positions (0-indexed) 26, 14
    for (mc, mr), bp in zip([('A', 30), ('C', 28)], [26, 14]):
        mapping[f"{mc}{mr}"] = f"{binder_chain}{bp + 1}"
    # sliding: one island [B5,B6] placed with start 40 -> positions [40,41]
    slide_islands = parse_motif_islands_spec("B5-6")
    slide_chain_residues = [(r['chain'], r['resnum']) for isl in slide_islands for r in isl]
    final_pos = _placement_to_positions([40], slide_islands)
    for k, (mc, mr) in enumerate(slide_chain_residues):
        mapping[f"{mc}{mr}"] = f"{binder_chain}{final_pos[k] + 1}"
    assert mapping == {"A30": "A27", "C28": "A15", "B5": "A41", "B6": "A42"}, mapping
    print("T11 motif mapping:", mapping, "OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    t5_parser()
    t6_motif_residue_parser()
    t7_motif_chain_flag_removed()
    t8_islands_reuse_residue_parser()
    t9_fape_ligand()
    t10_sliding_fape_indexing()
    t11_motif_mapping()
    t1_backcompat()
    t2_pose_invariance()
    t3_lig_response()
    t4_gradient_flow()
    print("\nall tests passed")
