"""Motif placement by exhaustive triplet enumeration (coordinate- or distogram-based).

A faithful re-implementation of the RFDesign "motif placement" loss
(``L_motif = L_sat + L_con``), adapted to the scores BoltzDesign actually
produces: coordinate RMSD / FAPE (``get_motif_coords_loss`` /
``get_motif_fape_loss``) and the trunk Cβ-distogram cross-entropy
(``get_motif_distogram_loss``).

The method rewards recapitulation of a discontiguous motif (M "contigs", which
here are BoltzDesign **islands**) at *any* location on a length-L design,
without committing to a placement up front:

  * For every contig pair (a,b) and every placement (i,j), a pairwise score
    H^{ab}_{ij} is computed from the network prediction -- Kabsch-RMSD, FAPE, or
    distogram cross-entropy -- against the reference motif geometry.  This is the
    per-pair score; the distogram-CE form is the closest analogue of RFDesign's
    original trRosetta 6D cross-entropy.
  * The 3-body energy is the (additive) sum H^{abc}_{ijk} = H^{ab}_{ij} +
    H^{bc}_{jk} + H^{ac}_{ik}.  Boltzmann weights p^{abc}_{ijk} =
    softmax(-beta * H^{abc}_{ijk}) give the per-triplet satisfaction L_sat^{abc}
    = sum p^{abc}_{ijk} H^{abc}_{ijk} (a soft-min over placements; -> min H as
    beta -> inf).  L_sat averages over all C(M,3) triplets.
  * L_con is the (symmetrized) cross-entropy between the (a,b)-pair-placement
    marginals p^{ab(c)}_{ij} = sum_k p^{abc}_{ijk} obtained through different
    third contigs c -- forcing triplets that share a pair to agree.  Non-trivial
    only at M >= 4.
  * ``best_placement`` / ``read_out_placement`` give the final, single
    non-overlapping contig -> start-position assignment: exact joint argmin for
    M <= 3 (one triplet IS the joint), greedy high-weight clique on the adjacency
    A_ij for M >= 4.

Generalizations over a plain contiguous-window motif (so it matches BoltzDesign
islands):

  (a) **Per-residue offsets.** A contig is a rigid *island*: residue r sits at
      design position ``start + offset[r]`` (default offsets = 0..len-1, i.e.
      contiguous).  The island's **span** = max(offset)+1 is what is reserved /
      tested for overlap (matching ``is_placement_valid``'s ``range(o, o+L)``),
      so internal fixed-gap spacers are honoured.
  (b) **Explicit valid-start sets.** Each contig carries the list of allowed
      start positions (default 0..L-span).  Pass a restricted set to exclude
      **forbidden** positions (claimed by fixed --motif_residues), or a singleton
      to pin a **fixed anchor** contig -- so sliding islands are placed relative
      to the fixed motif, exactly as the MCMC joint-Kabsch energy does.

Everything except the SVD inside Kabsch (detached, envelope-theorem scheme
matching ``kabsch_align``) and the distogram gather is in-graph, so the losses
carry a real gradient w.r.t. the prediction.  The engine is scorer-agnostic: it
consumes a ``{(a,b): grid}`` dict, so any per-pair score plugs in.

Standalone / importable; depends only on torch + the stdlib.
"""

from itertools import combinations

import torch

# Per-residue atom layout for coord tensors [..., A, 3], atom axis (N, CA, C, CB,
# ...).  RMSD uses all supplied atoms; FAPE uses 0,1,2 for the backbone frame;
# the distogram scorer uses the CA (pseudo-Cβ) channel for pairwise distances.
N_IDX, CA_IDX, C_IDX = 0, 1, 2

# Penalty for invalid placements (overlap / forbidden): softmax(-beta * _OVERLAP)
# underflows to 0, so they get zero Boltzmann weight without leaking gradient
# (the constant is detached).
_OVERLAP = 1.0e4
_EPS = 1.0e-12

# Distogram bin edges, matching boltz confidence.py / get_motif_distogram_loss.
_DIST_LO, _DIST_HI = 2.0, 22.0


class MotifPlacementProblem:
    """A discontiguous motif (M island-contigs) to be placed on a length-L design.

    contig_refs        : list of M reference-coordinate tensors, each [len_c, A, 3]
                         in Angstrom (A atoms/residue, shared across contigs;
                         CA/pseudo-Cβ-only -> A=1).  Residues in declaration order.
    length             : design length L.
    contig_offsets     : optional list of M per-residue intra-island offset lists
                         (default range(len_c) -- contiguous).  Residue r of
                         contig c lands at design position start + offset[r].
    contig_valid_starts: optional list of M allowed-start lists (default
                         0..L-span_c).  Restrict to exclude forbidden positions or
                         pin a fixed-anchor contig to a single start.
    """

    def __init__(self, contig_refs, length, contig_offsets=None,
                 contig_valid_starts=None):
        self.refs = [torch.as_tensor(r, dtype=torch.float32) for r in contig_refs]
        self.M = len(self.refs)
        # Keep the index tensors (offsets/starts) on the same device as the refs
        # so direct users like ``_overlap_mask`` stay device-consistent (the refs
        # may be built on cuda while the lists below default to CPU).
        _dev = self.refs[0].device if self.M else torch.device("cpu")
        self.L = int(length)
        if self.M < 1:
            raise ValueError("need at least one contig")
        self.A = int(self.refs[0].shape[1])
        if any(r.shape[1] != self.A for r in self.refs):
            raise ValueError("all contigs must share the same atoms/residue")
        self.lengths = [int(r.shape[0]) for r in self.refs]
        self.ca_idx = CA_IDX if self.A > CA_IDX else 0

        if contig_offsets is None:
            contig_offsets = [list(range(lc)) for lc in self.lengths]
        if len(contig_offsets) != self.M:
            raise ValueError("contig_offsets must have one entry per contig")
        self.offsets = []
        self.spans = []
        for c, off in enumerate(contig_offsets):
            off = [int(o) for o in off]
            if len(off) != self.lengths[c]:
                raise ValueError(f"contig {c}: offsets length != ref length")
            self.offsets.append(torch.as_tensor(off, dtype=torch.long, device=_dev))
            self.spans.append(max(off) + 1)

        if contig_valid_starts is None:
            contig_valid_starts = [list(range(self.L - self.spans[c] + 1))
                                   for c in range(self.M)]
        if len(contig_valid_starts) != self.M:
            raise ValueError("contig_valid_starts must have one entry per contig")
        self.starts = []
        for c, st in enumerate(contig_valid_starts):
            st = torch.as_tensor([int(s) for s in st], dtype=torch.long, device=_dev)
            if st.numel() == 0:
                raise ValueError(f"contig {c} has no valid start positions")
            if int(st.min()) < 0 or int(st.max()) + self.spans[c] > self.L:
                raise ValueError(f"contig {c} has a start that runs off the design")
            self.starts.append(st)

    def to(self, device):
        self.refs = [r.to(device) for r in self.refs]
        self.offsets = [o.to(device) for o in self.offsets]
        self.starts = [s.to(device) for s in self.starts]
        return self

    def n_placements(self, c):
        return int(self.starts[c].numel())


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #

def _gather(pred, problem, c):
    """All valid placements of contig c gathered from ``pred`` [L, A, 3]:
    returns [N_c, len_c, A, 3], position of residue r at placement i is
    ``starts[c][i] + offsets[c][r]``."""
    starts = problem.starts[c].to(pred.device)
    offsets = problem.offsets[c].to(pred.device)
    idx = starts[:, None] + offsets[None, :]                              # [N_c, len_c]
    return pred[idx]                                                      # [N_c, len_c, A, 3]


def _overlap_mask(problem, a, b):
    """[N_a, N_b] bool: True where the islands' spans overlap (share a position)."""
    sa = problem.starts[a][:, None]
    sb = problem.starts[b][None, :]
    return (sa < sb + problem.spans[b]) & (sb < sa + problem.spans[a])


def _batched_kabsch_rmsd(pred, ref):
    """Kabsch-aligned RMSD between ``pred`` [*, P, 3] and a single ``ref`` [P, 3].

    The optimal rotation is solved under ``no_grad`` and applied in-graph
    (envelope theorem; same scheme as ``kabsch_align`` / motif coords loss --
    correct forward value, gradient through ``pred`` without SVD-backward
    instability).  Returns RMSD [*] in Angstrom.
    """
    p_mean = pred.mean(dim=-2, keepdim=True)
    r_mean = ref.mean(dim=-2, keepdim=True)
    pc = pred - p_mean
    rc = ref - r_mean
    with torch.no_grad():
        H = pc.transpose(-1, -2) @ rc
        U, _, Vh = torch.linalg.svd(H, full_matrices=False)
        d = torch.sign(torch.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2)))
        D = torch.diag_embed(torch.stack(
            [torch.ones_like(d), torch.ones_like(d), d], dim=-1))
        R = Vh.transpose(-1, -2) @ D @ U.transpose(-1, -2)
    pc_al = pc @ R.transpose(-1, -2)
    err2 = ((pc_al - rc) ** 2).sum(dim=-1)
    return torch.sqrt(err2.mean(dim=-1) + _EPS)


def _rigid_frames_batched(n_xyz, ca_xyz, c_xyz, eps=1e-8):
    """AlphaFold-style frames from N, CA, C over arbitrary leading dims.
    Returns R [*, 3, 3] (columns e1,e2,e3) and origin t = CA [*, 3]."""
    e1 = c_xyz - ca_xyz
    e1 = e1 / (e1.norm(dim=-1, keepdim=True) + eps)
    u2 = n_xyz - ca_xyz
    u2 = u2 - e1 * (e1 * u2).sum(dim=-1, keepdim=True)
    e2 = u2 / (u2.norm(dim=-1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1), ca_xyz


# --------------------------------------------------------------------------- #
# Pairwise scorers  (H^{ab}_{ij})   -- the engine is agnostic to which is used
# --------------------------------------------------------------------------- #

def kabsch_rmsd_pair_scores(pred, problem):
    """Pairwise Kabsch-RMSD grids ``{(a,b): [N_a, N_b]}`` (Angstrom).

    H_ab[i,j] = RMSD of (contig-a atoms at start i) ++ (contig-b atoms at start j)
    onto (ref_a ++ ref_b).  Overlapping placements -> ``_OVERLAP``.
    """
    pred = pred.to(problem.refs[0].dtype)
    flat_segs = [_gather(pred, problem, c).reshape(problem.n_placements(c), -1, 3)
                 for c in range(problem.M)]                                # c: [N_c, len_c*A, 3]
    flat_refs = [problem.refs[c].reshape(-1, 3) for c in range(problem.M)]
    out = {}
    for a, b in combinations(range(problem.M), 2):
        na, nb = flat_segs[a].shape[0], flat_segs[b].shape[0]
        pa, pb = flat_segs[a].shape[1], flat_segs[b].shape[1]
        sa = flat_segs[a][:, None].expand(na, nb, pa, 3)
        sb = flat_segs[b][None, :].expand(na, nb, pb, 3)
        pred_pair = torch.cat([sa, sb], dim=2)                            # [na, nb, P, 3]
        ref_pair = torch.cat([flat_refs[a], flat_refs[b]], dim=0)         # [P, 3]
        rmsd = _batched_kabsch_rmsd(pred_pair, ref_pair)                  # [na, nb]
        out[(a, b)] = torch.where(_overlap_mask(problem, a, b),
                                  rmsd.new_full((), _OVERLAP), rmsd)
    return out


def fape_pair_scores(pred, problem, clamp=10.0):
    """Pairwise FAPE grids ``{(a,b): [N_a, N_b]}`` (Angstrom; drop-in for RMSD).

    For each pair, the union of both contigs' residues defines F = len_a+len_b
    backbone frames (atoms N_IDX, CA_IDX, C_IDX); every atom is expressed in every
    frame and compared, in local coords, to the constant reference; clamped at
    ``clamp`` and averaged over frame x atom.  Requires A >= 3.
    """
    if problem.A < 3:
        raise ValueError("FAPE needs at least N, CA, C atoms per residue (A >= 3)")
    pred = pred.to(problem.refs[0].dtype)
    segs = [_gather(pred, problem, c) for c in range(problem.M)]          # c: [N_c, len_c, A, 3]
    A = problem.A
    out = {}
    for a, b in combinations(range(problem.M), 2):
        sa, sb = segs[a], segs[b]
        na, la = sa.shape[0], sa.shape[1]
        nb, lb = sb.shape[0], sb.shape[1]
        ra = sa[:, None].expand(na, nb, la, A, 3)
        rb = sb[None, :].expand(na, nb, lb, A, 3)
        res = torch.cat([ra, rb], dim=2)                                 # [na,nb,F,A,3]
        F = la + lb
        R, t = _rigid_frames_batched(res[..., N_IDX, :], res[..., CA_IDX, :],
                                     res[..., C_IDX, :])
        pts = res.reshape(na, nb, F * A, 3)
        diff = pts[:, :, None, :, :] - t[:, :, :, None, :]               # [na,nb,F,Pt,3]
        loc = torch.einsum('abfij,abfpj->abfpi', R.transpose(-1, -2), diff)
        ref_pair = torch.cat([problem.refs[a], problem.refs[b]], dim=0)  # [F,A,3]
        with torch.no_grad():
            Rr, tr = _rigid_frames_batched(ref_pair[:, N_IDX], ref_pair[:, CA_IDX],
                                           ref_pair[:, C_IDX])
            ref_pts = ref_pair.reshape(F * A, 3)
            ref_loc = torch.einsum('fij,fpj->fpi', Rr.transpose(-1, -2),
                                   ref_pts[None] - tr[:, None])           # [F,Pt,3]
        d = torch.sqrt(((loc - ref_loc) ** 2).sum(dim=-1) + _EPS)
        fape = torch.clamp(d, max=clamp).mean(dim=(-1, -2))              # [na,nb]
        out[(a, b)] = torch.where(_overlap_mask(problem, a, b),
                                  fape.new_full((), _OVERLAP), fape)
    return out


def distogram_ce_pair_scores(pdistogram, problem, token_map=None):
    """Pairwise distogram cross-entropy grids ``{(a,b): [N_a, N_b]}``.

    The closest analogue of RFDesign's per-pair 6D cross-entropy, restricted to
    Boltz's Cβ-distance head and **fast-mode safe** (no diffusion coords needed).
    For contigs a,b at placements (i,j),

        H_ab[i,j] = sum_{r in a, q in b}  -log softmax(p)[t_a(i,r), t_b(j,q), b0_rq]

    where ``b0_rq`` is the reference bin of the motif's r-q distance (edges =
    linspace(2,22,nbins-1), the get_motif_distogram_loss / con-loss convention)
    and ``t_*`` maps a design position to its distogram token index.

    pdistogram : trunk logits [N, N, nbins] (or [1, N, N, nbins]).
    token_map  : [L] long, design position -> token index (default identity, for
                 the standalone case where design positions ARE token indices).
    """
    pd = pdistogram[0] if pdistogram.dim() == 4 else pdistogram            # [N,N,nbins]
    nbins = pd.shape[-1]
    logP = torch.log_softmax(pd, dim=-1)                                   # [N,N,nbins]
    edges = torch.linspace(_DIST_LO, _DIST_HI, nbins - 1, device=pd.device)
    if token_map is None:
        tok = torch.arange(problem.L, device=pd.device)
    else:
        tok = torch.as_tensor(token_map, dtype=torch.long, device=pd.device)
    ca = problem.ca_idx
    out = {}
    for a, b in combinations(range(problem.M), 2):
        d = torch.cdist(problem.refs[a][:, ca, :].to(pd.device),
                        problem.refs[b][:, ca, :].to(pd.device))           # [la, lb]
        ref_bin = (d.unsqueeze(-1) > edges).sum(dim=-1)                    # [la, lb]
        pos_a = (problem.starts[a].to(pd.device)[:, None]
                 + problem.offsets[a].to(pd.device)[None, :])              # [Na, la] design pos
        pos_b = (problem.starts[b].to(pd.device)[:, None]
                 + problem.offsets[b].to(pd.device)[None, :])              # [Nb, lb]
        tok_a, tok_b = tok[pos_a], tok[pos_b]                              # token indices
        na, la = tok_a.shape
        nb, lb = tok_b.shape
        ta = tok_a[:, None, :, None].expand(na, nb, la, lb)
        tb = tok_b[None, :, None, :].expand(na, nb, la, lb)
        rb = ref_bin[None, None, :, :].expand(na, nb, la, lb)
        ce = -logP[ta, tb, rb].sum(dim=(-1, -2))                          # [na, nb]
        out[(a, b)] = torch.where(_overlap_mask(problem, a, b),
                                  ce.new_full((), _OVERLAP), ce)
    return out


# --------------------------------------------------------------------------- #
# Enumeration engine  (scorer-agnostic)
# --------------------------------------------------------------------------- #

def _triplet_weights(pair_scores, a, b, c, beta):
    """3-body energy and Boltzmann weights for triplet a<b<c.

    H = H^{ab}[i,j] + H^{bc}[j,k] + H^{ac}[i,k]; p = softmax(-beta H) over the
    flattened (i,j,k) grid.  Both [N_a, N_b, N_c].
    """
    H = (pair_scores[(a, b)][:, :, None]
         + pair_scores[(b, c)][None, :, :]
         + pair_scores[(a, c)][:, None, :])
    p = torch.softmax(-beta * H.reshape(-1), dim=0).reshape(H.shape)
    return H, p


def placement_losses(pair_scores, problem, beta, return_marginals=False):
    """L_sat and L_con from precomputed pairwise score grids (requires M >= 3).

    L_sat = mean over C(M,3) triplets of sum_{ijk} p^{abc}_{ijk} H^{abc}_{ijk}.
    L_con = mean over (shared pair, distinct thirds {c,d}) of the symmetrized
            cross-entropy between the (a,b) marginals p^{ab(c)}, p^{ab(d)}.  Zero
            for M < 4.  Up to a constant this is RFDesign's 1/C(M,4) * 1/L^2 form.
    """
    if problem.M < 3:
        raise ValueError("placement_losses (triplet enumeration) needs M >= 3")
    sat_terms = []
    marg = {}                                                            # (a,b) -> {c: p^{ab(c)}}
    for a, b, c in combinations(range(problem.M), 3):
        H, p = _triplet_weights(pair_scores, a, b, c, beta)
        sat_terms.append((p * H).sum())
        marg.setdefault((a, b), {})[c] = p.sum(dim=2)
        marg.setdefault((a, c), {})[b] = p.sum(dim=1)
        marg.setdefault((b, c), {})[a] = p.sum(dim=0)
    L_sat = torch.stack(sat_terms).mean()

    con_terms = []
    for _pair, by_third in marg.items():
        for c, d in combinations(list(by_third), 2):
            pc, pd = by_third[c], by_third[d]
            con_terms.append(
                -(pc * torch.log(pd + _EPS) + pd * torch.log(pc + _EPS)).mean())
    L_con = torch.stack(con_terms).mean() if con_terms else L_sat.new_zeros(())

    if return_marginals:
        return L_sat, L_con, marg
    return L_sat, L_con


def motif_enum_loss(pred, problem, beta, w_con=1.0, pair_scores_fn=kabsch_rmsd_pair_scores):
    """Total motif-placement loss L_sat + w_con * L_con and a diagnostics dict.

    pred           : [L, A, 3] coords, or a distogram if pair_scores_fn expects one.
    beta           : inverse temperature (anneal 2 -> 20 over optimization).
    pair_scores_fn : kabsch_rmsd_pair_scores / fape_pair_scores / distogram_ce_pair_scores.
    """
    L_sat, L_con = placement_losses(pair_scores_fn(pred, problem), problem, beta)
    total = L_sat + w_con * L_con
    return total, {'L_sat': float(L_sat.detach()),
                   'L_con': float(L_con.detach()), 'beta': float(beta)}


# --------------------------------------------------------------------------- #
# Final read-out: exact joint (M<=3) or weighted-adjacency clique (M>=4)
# --------------------------------------------------------------------------- #

def build_adjacency(marg, problem):
    """Weighted adjacency A_ij over design positions (RFDesign clique matrix).

    A_ij = mean over all triplet pair-marginals of p^{..}_{ij}, scattered by
    (start-of-first-contig, start-of-second-contig) into [L, L], then
    symmetrized.  Returns A [L, L].
    """
    L = problem.L
    A = problem.refs[0].new_zeros(L, L)
    n = 0
    for (a, b), by_third in marg.items():
        sa, sb = problem.starts[a], problem.starts[b]
        for _c, p in by_third.items():
            pd = p.detach()
            rows = sa[:, None].expand_as(pd).reshape(-1)
            cols = sb[None, :].expand_as(pd).reshape(-1)
            A.index_put_((rows, cols), pd.reshape(-1), accumulate=True)
            n += 1
    A = A / max(n, 1)
    return A + A.t()


def contig_position_scores(marg, problem):
    """Per-contig start-position marginal q_c[s] (unary clique evidence),
    averaged over every pair-marginal involving contig c.  List of M [L] tensors."""
    L, M = problem.L, problem.M
    q = [problem.refs[0].new_zeros(L) for _ in range(M)]
    cnt = [0] * M
    for (a, b), by_third in marg.items():
        sa, sb = problem.starts[a], problem.starts[b]
        for _c, p in by_third.items():
            pd = p.detach()
            q[a].index_add_(0, sa, pd.sum(dim=1))
            q[b].index_add_(0, sb, pd.sum(dim=0))
            cnt[a] += 1
            cnt[b] += 1
    return [qc / max(cnt[c], 1) for c, qc in enumerate(q)]


def find_placement(marg, problem):
    """Greedy high-weight clique -> one non-overlapping start per contig (M >= 4).

    Combines unary q_c[s] with pairwise A_ij: contigs placed most-confident-first,
    each at the valid start maximizing q_c[s] + sum over placed contigs A[s,s_o],
    skipping starts whose span overlaps an already-placed contig.  Exact max-weight
    clique is NP-hard; this is the practical read-out.  Returns ``{c: start}``.
    """
    A = build_adjacency(marg, problem)
    q = contig_position_scores(marg, problem)
    order = sorted(range(problem.M), key=lambda c: float(q[c].max()), reverse=True)
    placed = {}
    occupied = [False] * problem.L
    for c in order:
        span = problem.spans[c]
        best_s, best_score = None, float('-inf')
        for s in problem.starts[c].tolist():
            if any(occupied[s:s + span]):
                continue
            score = float(q[c][s]) + sum(float(A[s, placed[o]]) for o in placed)
            if score > best_score:
                best_s, best_score = s, score
        if best_s is None:                                               # fully blocked
            valid = problem.starts[c]
            best_s = int(valid[int(torch.argmax(q[c][valid]))])
        placed[c] = best_s
        for t in range(best_s, min(best_s + span, problem.L)):
            occupied[t] = True
    return placed


def exact_joint_placement(pair_scores, problem):
    """Exact joint argmin over all placements (M in {2, 3}). Returns ``{c: start}``."""
    M = problem.M
    if M == 2:
        H = pair_scores[(0, 1)]
        i, j = divmod(int(torch.argmin(H)), H.shape[1])
        return {0: int(problem.starts[0][i]), 1: int(problem.starts[1][j])}
    if M == 3:
        H = (pair_scores[(0, 1)][:, :, None]
             + pair_scores[(1, 2)][None, :, :]
             + pair_scores[(0, 2)][:, None, :])
        n1, n2 = H.shape[1], H.shape[2]
        flat = int(torch.argmin(H))
        i, j, k = flat // (n1 * n2), (flat % (n1 * n2)) // n2, flat % n2
        return {0: int(problem.starts[0][i]), 1: int(problem.starts[1][j]),
                2: int(problem.starts[2][k])}
    raise ValueError("exact_joint_placement only for M in {2, 3}")


def best_placement(pair_scores, problem, beta=10.0):
    """Final contig -> start mapping: exact joint for M <= 3 (one triplet IS the
    joint), greedy adjacency clique for M >= 4."""
    if problem.M == 1:
        return {0: int(problem.starts[0][0])}
    if problem.M <= 3:
        return exact_joint_placement(pair_scores, problem)
    _, _, marg = placement_losses(pair_scores, problem, beta, return_marginals=True)
    return find_placement(marg, problem)


def read_out_placement(pred, problem, beta=10.0, pair_scores_fn=kabsch_rmsd_pair_scores):
    """End-to-end read-out: prediction -> contig -> start mapping."""
    return best_placement(pair_scores_fn(pred, problem), problem, beta)
