import os
import torch
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser


chain_to_number_ = {
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
    atom_to_token = batch['atom_to_token'] * (batch['entity_id']==chain_to_number_[binder_chain])
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
    average. Loss = sqrt(||binder_COM - mean_ORI||^2 + 1e-8) in Angstrom -- the
    L2 distance in the root-mean-square form of the motif coords / rg losses.

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
    # Euclidean distance in Å written as sqrt(sum_sq + 1e-8): the same root-mean-
    # square form as get_motif_coords_loss / add_rg_loss (an RMS over a single 3D
    # point). The +1e-8 also removes the NaN gradient torch.norm gives when the
    # centroids coincide exactly (grad of ||v|| at v=0 is 0/0).
    loss = torch.sqrt(((binder_com - mean_ori) ** 2).sum() + 1e-8)
    return loss, binder_com, mean_ori, all_oris


def get_mid_points(pdistogram):
    boundaries = torch.linspace(2, 22.0, 63)
    lower = torch.tensor([1.0])
    upper = torch.tensor([22.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(
        pdistogram.device
    )

    return mid_points


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


def _parse_atom_suffix(atomspec, flag, tok):
    """Parse the optional ``:ATOMS`` suffix of a motif token.

    Returns ``None`` (no suffix -> the loss's default atom set), the string
    ``'ALL'`` (every heavy atom), or a list of uppercased atom names. Atom
    names are alphanumeric (apostrophe allowed for nucleic-acid primes); a
    stray ':' is rejected with a hint to space-separate residue tokens.
    """
    s = (atomspec or '').strip()
    if s == '':
        return None
    if s.upper() == 'ALL':
        return 'ALL'
    atoms = []
    for a in s.split(','):
        a = a.strip().upper()
        if not a:
            continue
        if not a.replace("'", "").isalnum():
            raise ValueError(
                f"{flag} token '{tok}': bad atom name '{a}'. Atom names are "
                f"alphanumeric; when selecting atoms, separate residue tokens "
                f"with spaces (not commas).")
        atoms.append(a)
    return atoms or None


def _parse_residue_token(tok, flag='--motif_residues'):
    """Parse one ``CHAIN+RESNUM[:ATOMS]`` token into a list of
    ``(chain, resnum, atoms)`` -- one entry, or several when ``RESNUM`` is an
    inclusive range (``A10-14``). ``atoms`` is ``None`` / ``'ALL'`` / list of
    names (the shared ``:ATOMS`` selection). Shared by ``parse_motif_residue_spec``
    and ``parse_motif_islands_spec``.
    """
    body, _, atomspec = tok.partition(':')
    body = body.strip()
    i = 0
    while i < len(body) and body[i].isalpha():
        i += 1
    if i == 0:
        raise ValueError(
            f"{flag} token '{tok}' must be CHAIN+RESNUM (e.g. 'A57', 'A10-14', "
            f"or 'A57:CA,C'); bare residue numbers are not accepted -- prefix "
            f"the chain explicitly.")
    chain, rbody = body[:i], body[i:]
    if not rbody:
        raise ValueError(f"{flag} token '{tok}' has a chain but no residue number")
    atoms = _parse_atom_suffix(atomspec, flag, tok)
    if '-' in rbody and not rbody.startswith('-'):
        lo, hi = rbody.split('-')
        return [(chain, r, atoms) for r in range(int(lo), int(hi) + 1)]
    return [(chain, int(rbody), atoms)]


def parse_motif_residue_spec(spec, with_atoms=False):
    """Parse ``--motif_residues`` (chain-prefixed motif residue selection, with
    an optional per-residue atom selection).

    Each token is ``CHAIN+RESNUM[:ATOMS]``: leading letters = chain id, then an
    author/PDB residue number or inclusive range, then an optional ``:ATOMS``
    suffix selecting which atoms of that residue enter the coordinate AND FAPE
    motif losses (the SAME selection drives alignment + loss for both; there is
    no automatic backbone skipping -- only the listed atoms participate):
      * no suffix -> default set (backbone N,CA,C,CB)
      * ``:CA,C`` -> only those named atoms (comma-separated)
      * ``:ALL``  -> every heavy atom of the residue

    Grammar: residue tokens are **whitespace-separated** (commas are ONLY for
    grouping atoms inside ``:ATOMS`` -- comma-separated residue lists are not
    accepted). A **bare integer between two residues** is a fixed-gap spacer
    (see ``parse_motif_islands_spec``); for the flat residue list returned here
    it only affects the binder layout, not membership:
      ``"A57 A102 A195"``                        -> all chain A, default atoms
      ``"A47:SG  B53:CA,C  A195:ALL"``           -> per-residue atom selection
      ``"A10-14:CA"``                            -> range; atoms apply to each
      ``"A38:CB 3 A42:CB"``                      -> A38, A42 (3-residue gap)

    Returns ``(chain, resnum)`` tuples in declaration order by default; with
    ``with_atoms=True`` returns ``(chain, resnum, atoms)``. Spacers are flattened
    away (they carry no residue); the 2-tuple form preserves the original
    contract for residue-level callers (distogram target, binder mapping).
    """
    islands = parse_motif_islands_spec(spec, flag='--motif_residues')
    flat = [(r['chain'], r['resnum'], r['atoms']) for isl in islands for r in isl]
    return flat if with_atoms else [(c, r) for c, r, _ in flat]


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


def parse_motif_ligand_spec(spec, with_atoms=False):
    """Parse ``--motif_ligand_residues`` (chain-prefixed ligand residue
    selection, with an optional per-residue atom selection).

    Tokens are ``CHAIN+RESNUM[:ATOMS]`` (whitespace-separated; a token without
    ':' may be a legacy comma list), e.g. ``"B1"``, ``"B1 C401"``,
    ``"B1:FE,N1,N2"``, ``"C1:ALL"``. ``:ATOMS`` selects which ligand atoms are
    carried into the motif coords RMSD; no suffix and ``:ALL`` use every heavy
    atom. Ligand atoms are **never** part of the Kabsch fit -- they are merely
    carried by the transform fit on the protein motif atoms (decoupled
    alignment), so the count is independent of ``--motif_binder_positions``.

    Returns ``(chain, resnum)`` tuples by default, or ``(chain, resnum, atoms)``
    when ``with_atoms=True`` (``atoms`` is None / 'ALL' / list of names).
    """
    if spec is None or spec == '':
        return []
    out = []
    for ws_tok in str(spec).split():
        sub = [ws_tok] if ':' in ws_tok else [t for t in ws_tok.split(',') if t.strip()]
        for tok in sub:
            tok = tok.strip()
            if not tok:
                continue
            body, _, atomspec = tok.partition(':')
            body = body.strip()
            i = 0
            while i < len(body) and body[i].isalpha():
                i += 1
            if i == 0:
                raise ValueError(
                    f"--motif_ligand_residues token '{tok}' must be CHAIN+RESNUM "
                    f"(e.g. 'B1' = chain B, residue 1, or 'B1:FE')")
            chain, rbody = body[:i], body[i:]
            if not rbody or not rbody.lstrip('-').isdigit():
                raise ValueError(
                    f"--motif_ligand_residues token '{tok}': residue number "
                    f"must be an integer")
            atoms = _parse_atom_suffix(atomspec, '--motif_ligand_residues', tok)
            out.append((chain, int(rbody), atoms))
    return out if with_atoms else [(c, r) for c, r, _ in out]


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
def parse_motif_islands_spec(spec, flag='--motif_unindex_residues'):
    """Parse a motif residue spec into **islands** (rigid units), with the same
    ``CHAIN+RESNUM[:ATOMS]`` grammar as ``--motif_residues`` plus **fixed-gap
    spacers**. The primary parser; ``parse_motif_residue_spec`` flattens it.

    Tokens are whitespace-separated (commas are ONLY for grouping atoms inside
    ``:ATOMS``; comma-separated residue lists are not accepted).
    Residues are grouped into islands that slide / lay out as one rigid unit:
      * **bare integer N between two residues** = a fixed gap of N free residues
        (the restored spacer, new space-separated syntax): the next residue joins
        the current island at ``intra_offset += N + 1``.
      * a **consecutive run** (same chain, ``resnum`` +1 — e.g. the range
        ``A38-40``, or adjacent tokens ``A38 A39``) continues the current island.
      * any other break (chain change, non-consecutive resnum, no spacer) starts
        a new island.
    Examples:
      ``"A38:CB 3 A42:CB  A170:CB"`` -> island [A38@0, A42@4] (gap 3) + [A170@0]
      ``"A38-40 A57"``               -> island [A38,A39,A40] + island [A57]
      ``"A38 A42"``                  -> two independent 1-residue islands

    Returns a list of islands, each a list of dicts ``{'chain', 'resnum',
    'intra_offset', 'atoms'}`` (first residue at ``intra_offset == 0``; ``atoms``
    is None / 'ALL' / list of names, feeding the coords + FAPE motif losses).
    """
    if spec is None or spec == '':
        return []

    def _is_int(s):
        try:
            int(s); return True
        except ValueError:
            return False

    # Tokenize on whitespace only. Commas are reserved for the ':ATOMS' suffix;
    # a stray comma outside ':ATOMS' is rejected by _parse_residue_token.
    raw = str(spec).split()

    islands = []
    pending_spacer = None
    for tok in raw:
        if ':' not in tok and _is_int(tok):
            if not islands:
                raise ValueError(f"{flag}: leading spacer '{tok}' has no preceding residue")
            if pending_spacer is not None:
                raise ValueError(f"{flag}: two consecutive spacers near '{tok}'")
            sp = int(tok)
            if sp < 0:
                raise ValueError(f"{flag}: negative spacer '{tok}'")
            pending_spacer = sp
            continue
        for chain, resnum, atoms in _parse_residue_token(tok, flag):
            if islands and pending_spacer is not None:
                isl = islands[-1]
                isl.append({'chain': chain, 'resnum': resnum, 'atoms': atoms,
                            'intra_offset': isl[-1]['intra_offset'] + pending_spacer + 1})
                pending_spacer = None
            elif (islands and islands[-1][-1]['chain'] == chain
                  and resnum == islands[-1][-1]['resnum'] + 1):
                isl = islands[-1]
                isl.append({'chain': chain, 'resnum': resnum, 'atoms': atoms,
                            'intra_offset': isl[-1]['intra_offset'] + 1})
            else:
                islands.append([{'chain': chain, 'resnum': resnum, 'atoms': atoms,
                                 'intra_offset': 0}])
    if pending_spacer is not None:
        raise ValueError(f"{flag}: trailing spacer with no following residue")
    # Sanity: no duplicate residues across islands.
    seen = set()
    for isl in islands:
        for r in isl:
            key = (r['chain'], r['resnum'])
            if key in seen:
                raise ValueError(
                    f"{flag}: residue {r['chain']}{r['resnum']} appears more than once")
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


def get_motif_distogram_loss(pdistogram, motif_token_idx, target_dist, mid_pts,
                             method='cross_entropy'):
    """Supervised motif loss on the trunk distogram, restricted to motif token
    pairs. Two formulations (`method=`):

      'cross_entropy' (default) -- one-hot categorical cross-entropy on the
                    Cβ-Cβ distance distribution, supervised by the motif ground
                    truth. Each motif pair's reference distance is discretized
                    into its distogram bin b_ij = (target_dist_ij > edges).sum()
                    (edges = linspace(2, 22, num_bins-1), matching boltz
                    confidence.py), and the loss is

                        L = -mean_{i != j} log softmax(p_ij)[b_ij]

                    i.e. -(one_hot(b_ij) . log_softmax(p_ij)) averaged over all
                    off-diagonal motif pairs (both triangles -> the symmetric
                    double-counting; only i == j is dropped -- the same diagonal/
                    symmetry convention as the con/i_con contact losses). No
                    distance cutoff: every motif pair is supervised by its true
                    reference distance. Minimizing it maximizes the predicted
                    probability mass on the reference bin. Boltz1 predicts only the
                    Cβ DISTANCE feature, so this is the distance term of the
                    restraint (no orientation heads here).

      'mse' (a.k.a. 'expected') -- bin-center MSE: for each off-diagonal motif
                    pair, sum_k p_ijk (mid_pts_k - target_dist_ij)^2 [Å²].
                    Decomposes as Var(d_k|i,j) + (E[d_k|i,j] - target_dist_ij)^2,
                    so it drives both the expected distance to the reference AND
                    the prediction toward a sharp delta there, weighting errors by
                    squared bin-center distance.

    ``pdistogram`` is the raw predicted logits [1, N, N, num_bins]; the sub-block
    at the motif token indices is gathered so this works in the trunk-only fast
    path (distogram_only=True).
    """
    p = pdistogram[0].index_select(0, motif_token_idx).index_select(1, motif_token_idx)
    M = motif_token_idx.shape[0]
    offdiag = ~torch.eye(M, dtype=torch.bool, device=p.device)
    if method in ('cross_entropy', 'ce'):
        edges = torch.linspace(2.0, 22.0, p.shape[-1] - 1, device=p.device)
        ref_bin = (target_dist.unsqueeze(-1) > edges).sum(dim=-1)         # [M, M] in [0, num_bins-1]
        log_prob = torch.log_softmax(p, dim=-1)                           # [M, M, num_bins]
        ce = -log_prob.gather(-1, ref_bin.unsqueeze(-1)).squeeze(-1)      # [M, M]
        return ce[offdiag].mean()
    if method in ('mse', 'expected'):
        prob = torch.softmax(p, dim=-1)                                   # [M, M, num_bins]
        err2 = (mid_pts.view(1, 1, -1) - target_dist.unsqueeze(-1)) ** 2  # [M, M, num_bins]
        per_pair = (prob * err2).sum(dim=-1)                              # [M, M], Å²
        return per_pair[offdiag].mean()
    raise ValueError(
        f"unknown motif_distogram_loss_type '{method}' "
        "(expected 'cross_entropy' or 'mse')")


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


# ---------------------------------------------------------------------------
# Motif coords loss
# ---------------------------------------------------------------------------

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

    The Kabsch transform is fit from the **selected protein motif atoms** alone
    (``bb_pred_idx`` / ``bb_ref_xyz`` — the per-residue atom selection from
    ``--motif_residues``; backbone N,CA,C,CB by default). The same rigid
    transform is then applied to predicted **ligand atoms** (if provided), and
    a combined RMSD over protein + ligand is returned. The decoupled scheme is
    deliberate for enzyme/cofactor scaffolding (e.g. hemoprotein): if a ~40-atom
    heme were folded into the Kabsch fit it would dominate the alignment and
    absorb its own placement error, defeating the point of the loss. Holding the
    transform to the protein motif instead makes the ligand RMSD report ligand
    placement *relative to the motif framework*.

    The Kabsch SVD runs under ``torch.no_grad()`` (transform treated as a
    constant), and only the linear apply is in-graph — so there is **no
    backprop through the alignment** (envelope theorem; same scheme as
    ``kabsch_align`` / ``add_com_loss``, avoiding SVD-backward instability).
    The forward value is unchanged vs. an in-graph SVD.

    Backward-compat: pass ``lig_pred_idx=lig_ref_xyz=None`` for the
    protein-only path. Carries a gradient through ``sample_atom_coords`` only
    with ``--attach_coords True`` (mirrors rg_loss); otherwise inert but still
    reported.

    sample_atom_coords: [1, n_atoms, 3] from the diffusion sampler.
    bb_pred_idx:        [K] global atom indices (selected protein motif atoms).
    bb_ref_xyz:         [K, 3] reference coords (Å), same order.
    lig_pred_idx:       [L] global atom indices (motif ligand heavy atoms).
    lig_ref_xyz:        [L, 3] reference ligand coords (Å), same order.
    return_stats:       if True, also return a dict with per-component RMSDs
                        (``bb_rmsd``, ``lig_rmsd``) for diagnostics.
    """
    pred_bb = sample_atom_coords[0].index_select(0, bb_pred_idx)           # [K,3]
    with torch.no_grad():
        R, p_mean, r_mean = _kabsch_transform(pred_bb, bb_ref_xyz)        # detached
    pred_bb_al = (pred_bb - p_mean) @ R.transpose(-1, -2) + r_mean         # [K,3]
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
    if chain not in chain_to_number_:
        raise ValueError(f"unknown chain '{chain}' in atom-pair spec")
    chain_tokens = torch.where(entity_id == chain_to_number_[chain])[0]
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
                                 method='cross_entropy', return_stats=False):
    """Distogram-based distance restraint over the --atom_pairs token pairs.
    Three selectable formulations (`method=`):

      'cross_entropy' (default) -- categorical cross-entropy against the user-
                    specified target distance (the same supervised form as the
                    motif distogram CE; the former 'prob' method, extended to
                    handle point targets):
                    POINT TARGET (lo == hi == d_ref, 3-field spec "epA, epB,
                       d_ref"): one-hot CE at the reference bin,
                       -log softmax(logits)[bin(d_ref)] (edges = linspace(2, 22,
                       num_bins-1)) -- drives all mass onto the bin holding d_ref.
                    WINDOW (lo < hi, 4-field spec "epA, epB, lo, hi"):
                       -log P(d in [lo,hi]) over the in-window probability MASS of
                       the symmetrized token-pair distribution. Can be driven down
                       by parking a little mass in-window while the bulk -- and
                       hence the sampled structure -- stays far; its gradient also
                       collapses by ~20 A (eps clamp + softmax saturation).
      'expected' -- bin-center MSE on the predicted distribution:
                    POINT TARGET (lo == hi == d_ref):  sum_k p_k * (mid_pts_k -
                       d_ref)^2. Equals Var(d_k) + (E[d_k] - d_ref)^2, so it pulls
                       the expected distance to d_ref AND drives the prediction
                       toward a sharp delta at d_ref.
                    WINDOW (lo < hi): relu(lo - E[d])^2 + relu(E[d] - hi)^2 on the
                       expected distance E[d] = sum(prob * mid_pts). Flat-bottom:
                       zero once E[d] in [lo, hi]; only the mean is penalized
                       (variance is invisible).
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
        if method in ('cross_entropy', 'ce'):
            if s['lo'] == s['hi']:                        # point target -> one-hot CE at bin(d_ref)
                edges = torch.linspace(2.0, 22.0, logits.shape[-1] - 1,
                                       device=logits.device)
                ref_bin = int((s['lo'] > edges).sum())
                term = -torch.log_softmax(logits, dim=-1)[ref_bin]
            else:                                         # window -> -log P(d in [lo,hi])
                term = -torch.log(p_window + 1e-8)
        elif method == 'expected':
            if s['lo'] == s['hi']:                        # point target -> bin-center MSE
                # sum_k p_k * (mid_pts_k - d_ref)^2 = Var(d_k) + (E[d_k] - d_ref)^2
                # (penalizes both prediction spread AND mean-distance bias).
                d_ref = s['lo']
                term = (prob * (mid_pts - d_ref).pow(2)).sum()
            else:                                         # window -> flat-bottom on E[d]
                term = torch.relu(s['lo'] - exp_d) ** 2 + torch.relu(exp_d - s['hi']) ** 2
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
                "(expected 'cross_entropy', 'expected', or 'contact')")
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
    """Flat-bottom distance restraint on the sampled coords, reduced as an RMS
    violation in Å -- the SAME root-mean-square form as ``get_motif_coords_loss``
    (and ``add_rg_loss`` / ``add_com_loss``). Per pair the violation is the flat-
    bottom distance ``relu(lo - d) + relu(d - hi)`` (zero inside [lo,hi]) on the
    true Euclidean distance d = ||x_a - x_b||; the loss is
    ``sqrt(mean_pairs(violation^2) + 1e-8)``, so it is in Å (not Å^2). For a
    single point-target pair (lo == hi == d_ref) this reduces to |d - d_ref|.
    Atom-level on both sides (any named atom). Full mode only; carries a gradient
    only with --attach_coords True (mirrors rg_loss / motif_coords_loss).

    return_stats=True also returns per-pair diagnostics (detached): the true
    distance and the per-pair violation magnitude in Å (sqrt of the squared
    violation), so the logged values share the aggregate's Å units."""
    coords = sample_atom_coords[0]                        # [num_atoms, 3]
    terms, stats = [], []
    for s in pair_specs:
        d = torch.linalg.norm(coords[s['atom_a']] - coords[s['atom_b']])
        term = torch.relu(s['lo'] - d) ** 2 + torch.relu(d - s['hi']) ** 2  # violation^2 (Å^2)
        terms.append(term)
        if return_stats:
            stats.append({
                'label': s['label'],
                'dist': d.item(),
                'loss': torch.sqrt(term.detach() + 1e-8).item(),  # violation magnitude (Å)
                'lo': float(s['lo']), 'hi': float(s['hi']),
            })
    loss = torch.sqrt(torch.stack(terms).mean() + 1e-8)   # RMS violation (Å)
    return (loss, stats) if return_stats else loss


# ---- Explicit three-atom angle restraints (--atom_angles) -------------------
# The angle analogue of --atom_pairs: push the A-B-C angle (vertex at the middle
# endpoint B) toward a window [lo,hi] (or a single reference value). An angle is
# a function of the actual 3D positions -- it is NOT recoverable from the trunk
# distogram (that only predicts a *binned distribution* over Cb-Cb distances
# between tokens, not exact distances and not for arbitrary atoms; reconstructing
# an angle via the law of cosines on three *expected* distances is biased,
# restricted to Cb tokens, and capped by the ~24.5 A distogram ceiling). So this
# restraint is COORDS-ONLY: full mode (--distogram_only False), gradient only
# with --attach_coords True, exactly like get_atom_pair_coords_loss. Endpoints
# reuse the --atom_pairs CHAIN:SEL[@ATOM] grammar and resolver.
def parse_atom_angles_spec(spec):
    """Parse the --atom_angles string into raw endpoint dicts (no batch needed).

    Grammar (per ';'-separated chunk):
       5 fields -- "<epA>, <epB>, <epC>, <lo>, <hi>"   window restraint
       4 fields -- "<epA>, <epB>, <epC>, <a_ref>"      point target (lo=hi=a_ref)

    The restrained angle is A-B-C with the VERTEX at the MIDDLE endpoint epB,
    measured in degrees (0..180). Each endpoint is CHAIN:SEL[@ATOM] exactly as in
    --atom_pairs (SEL = 1-indexed residue position for polymer chains or an atom
    name for ligand chains; @ATOM optionally pins a named atom). lo/hi/a_ref are
    in degrees.
    """
    angles = []
    for chunk in spec.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        fields = [f.strip() for f in chunk.split(',')]
        if len(fields) == 5:
            epA, epB, epC, lo, hi = fields
            lo, hi = float(lo), float(hi)
            if hi < lo:
                raise ValueError(f"atom-angle '{chunk}': hi ({hi}) < lo ({lo})")
        elif len(fields) == 4:
            epA, epB, epC, a_ref = fields
            lo = hi = float(a_ref)
        else:
            raise ValueError(
                f"atom-angle '{chunk}' must have 4 fields 'epA, epB, epC, a_ref' "
                f"(point target) or 5 fields 'epA, epB, epC, lo, hi' (window)")
        for _nm, _v in (('lo', lo), ('hi', hi)):
            if not (0.0 <= _v <= 180.0):
                raise ValueError(
                    f"atom-angle '{chunk}': {_nm} ({_v}) must be in [0,180] degrees")
        cA, sA, aA = _parse_endpoint(epA)
        cB, sB, aB = _parse_endpoint(epB)
        cC, sC, aC = _parse_endpoint(epC)
        angles.append({'chainA': cA, 'selA': sA, 'atomA': aA,
                       'chainB': cB, 'selB': sB, 'atomB': aB,
                       'chainC': cC, 'selC': sC, 'atomC': aC,
                       'lo': lo, 'hi': hi})
    if not angles:
        raise ValueError("--atom_angles is set but no valid angles were parsed")
    return angles


def resolve_atom_angles(spec, batch, structure):
    """Parse + resolve --atom_angles against a built batch/structure.

    Returns dicts with the three atom indices on the sample_atom_coords axis
    (atom_a/atom_b/atom_c, where atom_b is the vertex), plus lo/hi (degrees) and
    a human-readable label. Coords-only -- an angle is undefined on the trunk
    distogram -- so only the atom indices are kept (no token axis). Resolved once
    per run (binder length fixed -> stable target indices)."""
    entity_id = batch['entity_id']
    if entity_id.dim() > 1:
        entity_id = entity_id[0]
    resolved = []
    for a in parse_atom_angles_spec(spec):
        _, aa, na = _resolve_endpoint_indices(a['chainA'], a['selA'], a['atomA'],
                                              entity_id, structure)
        _, ab, nb = _resolve_endpoint_indices(a['chainB'], a['selB'], a['atomB'],
                                              entity_id, structure)
        _, ac, nc = _resolve_endpoint_indices(a['chainC'], a['selC'], a['atomC'],
                                              entity_id, structure)
        labA = f"{a['chainA']}:{a['selA']}{('@'+a['atomA']) if a['atomA'] else ''}({na})"
        labB = f"{a['chainB']}:{a['selB']}{('@'+a['atomB']) if a['atomB'] else ''}({nb})"
        labC = f"{a['chainC']}:{a['selC']}{('@'+a['atomC']) if a['atomC'] else ''}({nc})"
        resolved.append({'atom_a': aa, 'atom_b': ab, 'atom_c': ac,
                         'lo': a['lo'], 'hi': a['hi'],
                         'label': f"{labA} - {labB} - {labC} [{a['lo']},{a['hi']}]deg"})
    return resolved


def get_atom_angle_coords_loss(sample_atom_coords, angle_specs, return_stats=False):
    """Flat-bottom three-atom angle restraint on the sampled coords, reduced as an
    RMS violation in DEGREES -- the angle analogue of ``get_atom_pair_coords_loss``
    (same flat-bottom + ``sqrt(mean(violation^2))`` form). Per angle, with the
    VERTEX at the middle endpoint B,

        theta = acos( (u . v) / (|u| |v|) ),   u = x_a - x_b,  v = x_c - x_b

    is the A-B-C angle in degrees; the flat-bottom violation is
    ``relu(lo - theta) + relu(theta - hi)`` (zero inside [lo,hi]); the loss is
    ``sqrt(mean_angles(violation^2) + 1e-8)``, so it is in degrees (not deg^2).
    For a single point-target angle (lo == hi == a_ref) this reduces to
    ``|theta - a_ref|``. The cosine is clamped to ``[-1+1e-6, 1-1e-6]`` before
    ``acos`` so the gradient stays finite at the 0/180 deg extremes (where acos'
    diverges). Coords-only -- an angle is undefined on the trunk distogram -- so
    there is no distogram variant: full mode only, and a gradient flows only with
    --attach_coords True (mirrors rg_loss / atom_pair_coords_loss).

    return_stats=True also returns per-angle diagnostics (detached): the true
    angle (deg) and the per-angle violation magnitude in degrees."""
    coords = sample_atom_coords[0]                        # [num_atoms, 3]
    terms, stats = [], []
    for s in angle_specs:
        u = coords[s['atom_a']] - coords[s['atom_b']]     # vertex at atom_b
        v = coords[s['atom_c']] - coords[s['atom_b']]
        cos = (u * v).sum() / (torch.linalg.norm(u) * torch.linalg.norm(v) + 1e-8)
        cos = torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6)   # keep acos' finite
        theta = torch.rad2deg(torch.arccos(cos))          # degrees
        term = torch.relu(s['lo'] - theta) ** 2 + torch.relu(theta - s['hi']) ** 2  # violation^2 (deg^2)
        terms.append(term)
        if return_stats:
            stats.append({
                'label': s['label'],
                'angle': theta.item(),
                'loss': torch.sqrt(term.detach() + 1e-8).item(),  # violation magnitude (deg)
                'lo': float(s['lo']), 'hi': float(s['hi']),
            })
    loss = torch.sqrt(torch.stack(terms).mean() + 1e-8)   # RMS violation (deg)
    return (loss, stats) if return_stats else loss
