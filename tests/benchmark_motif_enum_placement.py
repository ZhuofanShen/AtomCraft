"""Benchmark the exhaustive motif-placement engine on a REAL design.

Question: given a designed binder (folded coords) and the reference motif, can the
exhaustive engine (motif_enum) pick the *known-correct* placement as the global
RMSD/FAPE minimum -- i.e. distinguish it from all the suboptimal / infeasible
(overlapping) placements?

Design under test:
  outputs/outputs_heme_binder_motif_scaf/
    small_molecule_HEME_peroxidase_disto_coords_10dets1_ligplddt_rg1_motifligcoords1
    (HEME_results_itr1_length169) -- motif A38/A42/A170 placed at binder 38/42/120.

The engine sees ONLY backbone coordinates; the 38/42/120 answer is never given to
it. We try two island layouts (the search unit), since that is what `--motif_*`
parses to:
  (1) THREE independent single-residue islands  (A38, A42, A170 free separately)
      == `--motif_unindex_residues A38 A42 A170`
  (2) TWO islands: a RIGID [A38,A42] unit (offsets [0,4]) + [A170]
      == `--motif_unindex_residues "A38 3 A42" A170`

Run:  python tests/benchmark_motif_enum_placement.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'boltzdesign'))
from motif_enum import (  # noqa: E402
    MotifPlacementProblem, kabsch_rmsd_pair_scores, fape_pair_scores,
    best_placement, _OVERLAP,
)

ROOT = os.path.join(os.path.dirname(__file__), '..')
REF_PDB = os.path.join(ROOT, 'peroxidase_pos1-rot1.pdb')
CIF = os.path.join(
    ROOT, 'outputs', 'outputs_heme_binder_motif_scaf',
    'small_molecule_HEME_peroxidase_disto_coords_10dets1_ligplddt_rg1_motifligcoords1',
    'results_final', 'boltz_results_HEME_results_itr1_length169',
    'predictions', 'HEME_results_itr1_length169',
    'HEME_results_itr1_length169_model_0.cif')

MOTIF = [('A', 38), ('A', 42), ('A', 170)]      # reference motif residues
TRUTH_1IDX = [38, 42, 120]                       # known binder positions (1-indexed)
BINDER_CHAIN = 'A'
ATOMS = ['N', 'CA', 'C', 'CB']                   # auto-falls back to N,CA,C if needed


def parse_pdb_atoms(path, wanted):
    out, want = {k: {} for k in wanted}, set(wanted)
    for ln in open(path):
        if not ln.startswith('ATOM'):
            continue
        key = (ln[21], int(ln[22:26]))
        if key in want:
            out[key][ln[12:16].strip()] = np.array(
                [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])], dtype=np.float32)
    return out


def parse_cif_binder(path, chain):
    out, in_loop = {}, False
    for ln in open(path):
        if ln.startswith('_atom_site.'):
            in_loop = True
        elif in_loop and ln.startswith(('ATOM', 'HETATM')):
            p = ln.split()
            if p[15] == chain:
                out.setdefault(int(p[7]), {})[p[3]] = np.array(
                    [float(p[10]), float(p[11]), float(p[12])], dtype=np.float32)
        elif in_loop and ln.startswith('#'):
            in_loop = False
    return out


def run_config(label, parsed_label, refs, offsets, valid_starts, truth_starts,
               L, pred):
    """Build a problem, score with RMSD+FAPE, report match / rank / margin."""
    problem = MotifPlacementProblem(refs, L, contig_offsets=offsets,
                                    contig_valid_starts=valid_starts)
    M = problem.M
    print(f"=== {label} ===")
    print(f"    ({parsed_label})")
    for name, fn in (('RMSD', kabsch_rmsd_pair_scores),
                     ('FAPE', fape_pair_scores)):
        scores = fn(pred, problem)
        placement = best_placement(scores, problem, beta=10.0)
        got = [int(placement[c]) for c in range(M)]
        ok = (got == truth_starts)

        # Joint score grid (M in {2,3}) for rank + margin.
        if M == 2:
            H = scores[(0, 1)]
        else:
            H = (scores[(0, 1)][:, :, None] + scores[(1, 2)][None, :, :]
                 + scores[(0, 2)][:, None, :])
        starts = [problem.starts[c].tolist() for c in range(M)]
        tidx = tuple(starts[c].index(truth_starts[c]) for c in range(M))
        h_truth = float(H[tidx])
        h_min = float(H.min())
        n_overlap = int((H >= _OVERLAP).sum())
        n_feasible = H.numel() - n_overlap
        rank = int((H < h_truth).sum()) + 1

        # Best 'far' alternative: every island start >= 3 residues off the truth.
        far = torch.ones_like(H, dtype=torch.bool)
        for c in range(M):
            offc = (torch.tensor(starts[c]) - truth_starts[c]).abs() >= 3
            shape = [1] * M
            shape[c] = -1
            far &= offc.view(*shape)
        h_far = float(H[far].min()) if bool(far.any()) else float('nan')

        got_1 = [g + 1 for g in got]
        print(f"  [{name}] best -> {got_1} (1-idx)  "
              f"{'MATCHES truth' if ok else 'MISS (want %s)' % TRUTH_1IDX}")
        print(f"         truth score={h_truth:.3f}  global_min={h_min:.3f}  "
              f"truth_is_min={abs(h_truth - h_min) < 1e-4}  "
              f"rank={rank}/{n_feasible} feasible ({n_overlap} overlap-penalized)")
        print(f"         margin to best 'far' alt (>=3 off each): "
              f"{h_far:.3f}  (Delta={h_far - h_truth:+.3f})")
    print()


def main():
    ref_res = parse_pdb_atoms(REF_PDB, MOTIF)
    binder = parse_cif_binder(CIF, BINDER_CHAIN)
    L = max(binder)
    atoms = list(ATOMS)
    if not all(a in ref_res[k] for k in MOTIF for a in atoms):
        atoms = ['N', 'CA', 'C']
    A = len(atoms)

    ref_xyz = {k: np.stack([ref_res[k][a] for a in atoms]) for k in MOTIF}  # [A,3] each
    pred = torch.zeros(L, A, 3, dtype=torch.float32)
    valid = set()
    for p in range(L):
        res = binder.get(p + 1, {})
        if all(a in res for a in atoms):
            pred[p] = torch.tensor(np.stack([res[a] for a in atoms]), dtype=torch.float32)
            valid.add(p)
    truth = [t - 1 for t in TRUTH_1IDX]              # 0-indexed
    assert all(t in valid for t in truth), "a truth position was dropped"

    def starts_for(offs):
        span = max(offs) + 1
        return [s for s in range(L - span + 1)
                if all((s + o) in valid for o in offs)]

    print(f"binder L={L}; atoms={atoms} (A={A}); candidate positions={len(valid)} "
          f"(excluded {L - len(valid)} missing-atom/glycine)")
    print(f"motif {['%s%d' % m for m in MOTIF]} -> known binder {TRUTH_1IDX} (1-idx)\n")

    # (1) three independent single-residue islands.
    run_config(
        "Config 1: three independent single-residue islands",
        "--motif_unindex_residues A38 A42 A170",
        refs=[torch.tensor(ref_xyz[k], dtype=torch.float32).reshape(1, A, 3) for k in MOTIF],
        offsets=[[0], [0], [0]],
        valid_starts=[starts_for([0])] * 3,
        truth_starts=[truth[0], truth[1], truth[2]],
        L=L, pred=pred)

    # (2) rigid [A38,A42] island (offsets [0,4]) + [A170].
    isl0 = torch.tensor(np.stack([ref_xyz[('A', 38)], ref_xyz[('A', 42)]]),
                        dtype=torch.float32)            # [2, A, 3]
    isl1 = torch.tensor(ref_xyz[('A', 170)], dtype=torch.float32).reshape(1, A, 3)
    run_config(
        "Config 2: rigid [A38,A42] unit + [A170]",
        '--motif_unindex_residues "A38 3 A42" A170',
        refs=[isl0, isl1],
        offsets=[[0, 4], [0]],
        valid_starts=[starts_for([0, 4]), starts_for([0])],
        truth_starts=[truth[0], truth[2]],             # island starts: A38-site, A170-site
        L=L, pred=pred)


if __name__ == '__main__':
    main()
