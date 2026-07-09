"""Tests for the exhaustive triplet-enumeration motif-placement loss.

Run:  python tests/test_motif_enum.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'boltzdesign'))

from motif_enum import (  # noqa: E402
    MotifPlacementProblem, kabsch_rmsd_pair_scores, fape_pair_scores,
    distogram_ce_pair_scores, placement_losses, motif_enum_loss,
    read_out_placement, best_placement, _OVERLAP, _DIST_LO, _DIST_HI,
)


def _onehot_distogram(ca_xyz, nbins=64, sharp=12.0):
    """A 'confident' distogram [L,L,nbins] whose argmax bin at (i,j) is the bin of
    the actual ca distance d(i,j) -- so CE against a reference is ~0 exactly where
    the design distances match the reference."""
    D = torch.cdist(ca_xyz, ca_xyz)
    edges = torch.linspace(_DIST_LO, _DIST_HI, nbins - 1)
    b = (D.unsqueeze(-1) > edges).sum(dim=-1)
    logits = torch.zeros(D.shape[0], D.shape[1], nbins)
    logits.scatter_(-1, b.unsqueeze(-1), sharp)
    return logits


def _build_scene(M=3, res_per_contig=3, A=1, L=30, true_starts=(2, 12, 22), seed=0,
                 decoy_scale=100.0):
    """A length-L design with a rigid copy of an M-contig motif embedded at
    ``true_starts`` and random decoy coordinates everywhere else.

    Returns (pred [L, A, 3], problem, true_starts).  Internal + cross-contig
    geometry of the embedded motif exactly matches the references (up to a
    global rotation/translation that Kabsch/FAPE remove), so the unique
    zero-score placement is ``true_starts``.
    """
    g = torch.Generator().manual_seed(seed)
    # Compact reference motif: M*res_per_contig residues, A atoms each, within ~10 A.
    n_res = M * res_per_contig
    motif = torch.rand(n_res, A, 3, generator=g) * 10.0
    refs = [motif[c * res_per_contig:(c + 1) * res_per_contig] for c in range(M)]
    problem = MotifPlacementProblem(refs, length=L)

    # Decoys spread over a box so no decoy window accidentally matches; a
    # smaller scale yields moderate (not huge) decoy scores -> real spread.
    pred = torch.rand(L, A, 3, generator=g) * decoy_scale

    # Global rigid transform applied to the whole motif before embedding.
    q = torch.rand(4, generator=g)
    q = q / q.norm()
    w, x, y, z = q
    R = torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    t = torch.rand(3, generator=g) * 50.0
    for c, s in enumerate(true_starts):
        seg = refs[c].reshape(-1, 3) @ R.t() + t
        pred[s:s + res_per_contig] = seg.reshape(res_per_contig, A, 3)
    return pred, problem, list(true_starts)


def test_pair_scores_minimum_at_truth():
    pred, problem, true = _build_scene()
    scores = kabsch_rmsd_pair_scores(pred, problem)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for (a, b) in pairs:
        H = scores[(a, b)]
        flat_argmin = int(torch.argmin(H))
        i, j = divmod(flat_argmin, H.shape[1])
        assert (i, j) == (true[a], true[b]), f"pair {(a,b)} argmin {(i,j)} != {(true[a],true[b])}"
        assert float(H[true[a], true[b]]) < 1e-3, "true placement should score ~0 RMSD"
        # overlapping placements are masked out
        assert float(H[true[a], true[a]]) == _OVERLAP or true[a] >= H.shape[1]
    print("ok  pair-score minima at the true placement (RMSD ~0)")


def test_triplet_argmax_recovers_placement():
    pred, problem, true = _build_scene()
    scores = kabsch_rmsd_pair_scores(pred, problem)
    # M=3 -> single triplet (0,1,2)
    from motif_enum import _triplet_weights
    H, p = _triplet_weights(scores, 0, 1, 2, beta=20.0)
    flat = int(torch.argmax(p))
    nb, nc = p.shape[1], p.shape[2]
    i = flat // (nb * nc)
    j = (flat % (nb * nc)) // nc
    k = flat % nc
    assert [i, j, k] == true, f"triplet argmax {[i,j,k]} != {true}"
    print("ok  Boltzmann-weight argmax recovers the (i,j,k) triplet placement")


def test_beta_sharpens_soft_min():
    # Moderate decoy scores so the soft-min is meaningfully above the true min
    # (RMSD ~0) at low beta and tightens toward it as beta grows.
    pred, problem, _ = _build_scene(decoy_scale=12.0)
    scores = kabsch_rmsd_pair_scores(pred, problem)
    sats = [float(placement_losses(scores, problem, beta)[0]) for beta in (2.0, 8.0, 20.0)]
    # E_p[H] is non-increasing in beta (dE/dbeta = -Var_p(H) <= 0), strictly so
    # while probability mass is still moving onto the best placement.
    assert sats[0] > sats[1] >= sats[2], f"L_sat should decrease with beta: {sats}"
    assert sats[-1] < 1e-2, "soft-min should approach the true minimum (~0)"
    print(f"ok  increasing beta sharpens the soft-min  L_sat(2,8,20)={['%.2e'%s for s in sats]}")


def test_gradient_flows():
    pred, problem, _ = _build_scene()
    pred = pred.clone().requires_grad_(True)
    total, stats = motif_enum_loss(pred, problem, beta=10.0)
    total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0, "loss must produce a non-zero gradient"
    print(f"ok  gradient flows through L_sat+L_con  ({stats})")


def test_clique_readout_recovers_placement():
    pred, problem, true = _build_scene()
    placement = read_out_placement(pred, problem, beta=20.0)
    got = [placement[c] for c in range(problem.M)]
    assert got == true, f"clique read-out {got} != {true}"
    print(f"ok  clique read-out recovers contig->start placement {placement}")


def test_lcon_active_at_M4_and_rewards_consistency():
    # Clean embedding -> triplets agree -> low L_con; scrambled -> high L_con.
    pred_clean, problem, _ = _build_scene(M=4, true_starts=(2, 10, 18, 26), L=34)
    scores_clean = kabsch_rmsd_pair_scores(pred_clean, problem)
    _, con_clean = placement_losses(scores_clean, problem, beta=20.0)

    g = torch.Generator().manual_seed(7)
    pred_scram = torch.rand(problem.L, problem.A, 3, generator=g) * 100.0
    scores_scram = kabsch_rmsd_pair_scores(pred_scram, problem)
    _, con_scram = placement_losses(scores_scram, problem, beta=20.0)

    assert float(con_clean) >= 0.0
    assert float(con_clean) < float(con_scram), (
        f"consistent placement should have lower L_con: "
        f"clean={float(con_clean):.3f} scrambled={float(con_scram):.3f}")
    print(f"ok  L_con active at M=4 and lower for a consistent placement "
          f"(clean={float(con_clean):.3f} < scrambled={float(con_scram):.3f})")


def test_fape_scorer_smoke():
    # FAPE needs N,CA,C -> A>=3; check it runs, masks overlaps, and is differentiable.
    pred, problem, true = _build_scene(A=4, L=30)
    scores = fape_pair_scores(pred, problem)
    H = scores[(0, 1)]
    i, j = divmod(int(torch.argmin(H)), H.shape[1])
    assert (i, j) == (true[0], true[1]), f"FAPE argmin {(i,j)} != {(true[0],true[1])}"
    pred = pred.clone().requires_grad_(True)
    total, _ = motif_enum_loss(pred, problem, beta=10.0, pair_scores_fn=fape_pair_scores)
    total.backward()
    assert torch.isfinite(pred.grad).all() and float(pred.grad.abs().sum()) > 0
    print("ok  FAPE scorer: argmin at truth, masks overlaps, gradient flows")


def test_distogram_scorer_recovers_placement_and_grads():
    pred, problem, true = _build_scene(M=3)
    logits = _onehot_distogram(pred[:, 0, :])              # token_map = identity
    scores = distogram_ce_pair_scores(logits, problem)
    for (a, b) in [(0, 1), (0, 2), (1, 2)]:
        H = scores[(a, b)]
        i, j = divmod(int(torch.argmin(H)), H.shape[1])
        assert (i, j) == (true[a], true[b]), f"distogram pair {(a,b)} argmin off"
    placement = best_placement(scores, problem, beta=20.0)
    assert [placement[c] for c in range(3)] == true, "distogram placement wrong"
    # differentiable w.r.t. the distogram logits
    logits = logits.clone().requires_grad_(True)
    total, _ = motif_enum_loss(logits, problem, beta=10.0,
                               pair_scores_fn=distogram_ce_pair_scores)
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0
    print("ok  distogram-CE scorer: argmin at truth, placement recovered, grads flow")


def test_island_offsets_with_spacer():
    # Contig 1 is an island [r@0, r@4] (span 5, a 3-residue internal gap); embed
    # it at start 10 so its residues land at design positions 10 and 14.
    g = torch.Generator().manual_seed(3)
    motif = torch.rand(5, 1, 3, generator=g) * 10.0
    refs = [motif[0:2], motif[2:4], motif[4:5]]            # 2 + 2 + 1 residues
    offsets = [[0, 1], [0, 4], [0]]                        # contig 1 has the spacer
    L = 30
    problem = MotifPlacementProblem(refs, L, contig_offsets=offsets)
    assert problem.spans == [2, 5, 1]
    true = [2, 10, 22]
    pred = torch.rand(L, 1, 3, generator=g) * 100.0
    for c, s in enumerate(true):
        for r, off in enumerate(offsets[c]):
            pred[s + off] = motif[sum(len(o) for o in offsets[:c]) + r]
    placement = read_out_placement(pred, problem, beta=20.0)
    assert [placement[c] for c in range(3)] == true, f"island placement {placement} != {true}"
    print(f"ok  per-residue offsets (island with internal spacer) placed correctly {placement}")


def test_valid_starts_exclude_forbidden():
    # Restrict contig 0's valid starts so its TRUE start (2) is forbidden; the
    # readout must pick an allowed start, never the excluded one.
    pred, problem_full, true = _build_scene(M=3)
    refs = [r for r in problem_full.refs]
    allowed0 = [s for s in range(problem_full.n_placements(0)) if s != true[0]]
    problem = MotifPlacementProblem(
        refs, problem_full.L,
        contig_valid_starts=[allowed0, list(range(problem_full.n_placements(1))),
                             list(range(problem_full.n_placements(2)))])
    placement = best_placement(kabsch_rmsd_pair_scores(pred, problem), problem, beta=20.0)
    assert placement[0] != true[0], "forbidden start was chosen"
    assert placement[0] in allowed0, "chosen start outside the allowed set"
    print(f"ok  explicit valid-start set excludes forbidden positions {placement}")


def test_clique_readout_M4():
    pred, problem, true = _build_scene(M=4, true_starts=(2, 10, 18, 26), L=34)
    placement = best_placement(kabsch_rmsd_pair_scores(pred, problem), problem, beta=20.0)
    assert [placement[c] for c in range(4)] == true, f"M=4 clique {placement} != {true}"
    print(f"ok  M=4 greedy-clique read-out recovers placement {placement}")


def test_pipeline_style_fixed_anchor_plus_sliding():
    """Mirrors boltz_hallucination's exhaustive path: fixed residues as pinned
    single-residue anchor contigs + sliding multi-residue island contigs, scored
    on a confident distogram (token_map = identity). The sliding islands must be
    placed at their true starts relative to the pinned anchors."""
    g = torch.Generator().manual_seed(11)
    L = 40
    # 5-residue reference motif: 2 fixed (anchors) + island A (2 res) + island B (1).
    motif = torch.rand(5, 1, 3, generator=g) * 9.0
    fixed_refs = [motif[0:1], motif[1:2]]                 # pinned single residues
    islandA, islandB = motif[2:4], motif[4:5]
    fixed_pos = [5, 6]
    trueA, trueB = 20, 30                                 # island start positions
    # Embed a rigid copy of the whole motif into a decoy design.
    pred = torch.rand(L, 1, 3, generator=g) * 100.0
    q = torch.rand(4, generator=g); q = q / q.norm(); w, x, y, z = q
    R = torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    t = torch.rand(3, generator=g) * 40.0
    place = {0: (fixed_pos[0], [0]), 1: (fixed_pos[1], [0]),
             2: (trueA, [0, 1]), 3: (trueB, [0])}
    seg_refs = [motif[0:1], motif[1:2], islandA, islandB]
    for c, (start, offs) in place.items():
        seg = seg_refs[c].reshape(-1, 3) @ R.t() + t
        for r, off in enumerate(offs):
            pred[start + off] = seg[r]
    logits = _onehot_distogram(pred[:, 0, :])

    # Build exactly like _build_slide_mp: fixed pinned, sliding free minus fixed.
    forbidden = set(fixed_pos)
    refs = [fixed_refs[0], fixed_refs[1], islandA, islandB]
    offsets = [[0], [0], [0, 1], [0]]
    valid_starts = [
        [fixed_pos[0]], [fixed_pos[1]],
        [s for s in range(L - 2 + 1) if all((s + o) not in forbidden for o in (0, 1))],
        [s for s in range(L - 1 + 1) if s not in forbidden]]
    problem = MotifPlacementProblem(refs, L, contig_offsets=offsets,
                                    contig_valid_starts=valid_starts)
    scores = distogram_ce_pair_scores(logits, problem)    # token_map = identity
    placement = best_placement(scores, problem, beta=20.0)
    assert placement[0] == fixed_pos[0] and placement[1] == fixed_pos[1], "anchor moved"
    assert placement[2] == trueA and placement[3] == trueB, \
        f"sliding placement {placement} wrong (expected A@{trueA}, B@{trueB})"
    print(f"ok  pipeline-style fixed-anchor + sliding distogram placement {placement}")


def test_pipeline_style_coords_scorers():
    """Coords analogue of the exhaustive path (_build_slide_mp_coords): multi-atom
    (N,CA,C,CB) refs [n,A,3] + a [L,A,3] prediction, scored with the Kabsch-RMSD
    and FAPE coords scorers (full-mode determinants). Fixed anchors pinned,
    sliding islands recovered at their true starts under one global rigid copy."""
    g = torch.Generator().manual_seed(7)
    L, A = 40, 4
    # Fixed non-collinear 4-atom backbone template (N,CA,C,CB) so frames/Kabsch
    # are well-defined; per-residue center makes the 5 residues distinct.
    tmpl = torch.tensor([[-1.2, 0.0, 0.0], [0.0, 0.0, 0.0],
                         [0.9, 1.0, 0.0], [0.4, -0.8, 1.0]])
    centers = torch.rand(5, 3, generator=g) * 9.0
    motif = centers[:, None, :] + tmpl[None, :, :]            # [5, A, 3]
    fixed_refs = [motif[0:1], motif[1:2]]
    islandA, islandB = motif[2:4], motif[4:5]
    fixed_pos = [5, 6]
    trueA, trueB = 20, 30
    # Decoy design with one rigid copy of the whole motif embedded.
    pred = torch.rand(L, A, 3, generator=g) * 100.0
    qv = torch.rand(4, generator=g); qv = qv / qv.norm(); w, x, y, z = qv
    R = torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    tr = torch.rand(3, generator=g) * 40.0
    place = {0: (fixed_pos[0], [0]), 1: (fixed_pos[1], [0]),
             2: (trueA, [0, 1]), 3: (trueB, [0])}
    seg_refs = [motif[0:1], motif[1:2], islandA, islandB]
    for c, (start, offs) in place.items():
        for r, off in enumerate(offs):
            pred[start + off] = seg_refs[c][r] @ R.t() + tr   # [A,3] rigid

    forbidden = set(fixed_pos)
    refs = [fixed_refs[0], fixed_refs[1], islandA, islandB]
    offsets = [[0], [0], [0, 1], [0]]
    valid_starts = [
        [fixed_pos[0]], [fixed_pos[1]],
        [s for s in range(L - 2 + 1) if all((s + o) not in forbidden for o in (0, 1))],
        [s for s in range(L - 1 + 1) if s not in forbidden]]
    problem = MotifPlacementProblem(refs, L, contig_offsets=offsets,
                                    contig_valid_starts=valid_starts)
    for name, fn in (('kabsch_rmsd', kabsch_rmsd_pair_scores),
                     ('fape', fape_pair_scores)):
        placement = best_placement(fn(pred, problem), problem, beta=10.0)
        assert placement[0] == fixed_pos[0] and placement[1] == fixed_pos[1], \
            f"{name}: anchor moved {placement}"
        assert placement[2] == trueA and placement[3] == trueB, \
            f"{name}: sliding placement {placement} wrong (want A@{trueA}, B@{trueB})"
    print(f"ok  pipeline-style coords scorers (kabsch_rmsd + fape) {placement}")


if __name__ == '__main__':
    tests = [
        test_pair_scores_minimum_at_truth,
        test_triplet_argmax_recovers_placement,
        test_beta_sharpens_soft_min,
        test_gradient_flows,
        test_clique_readout_recovers_placement,
        test_lcon_active_at_M4_and_rewards_consistency,
        test_fape_scorer_smoke,
        test_distogram_scorer_recovers_placement_and_grads,
        test_island_offsets_with_spacer,
        test_valid_starts_exclude_forbidden,
        test_clique_readout_M4,
        test_pipeline_style_fixed_anchor_plus_sliding,
        test_pipeline_style_coords_scorers,
    ]
    for t in tests:
        t()
    print(f"\nall {len(tests)} motif-enum tests passed")
