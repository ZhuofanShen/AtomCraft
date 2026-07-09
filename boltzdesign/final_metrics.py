"""Standardized final-fold design metrics, comparable across runs.

Computed on the FINAL full 200-step diffusion validation fold (the predict_step
`output`), not the gradient epochs, and with a fixed per-target-type-default
contact config (NOT the run's --num_inter_contacts / --i_con_loss / sampler
tuning) so the numbers compare across differently-tuned runs. Raw quantities are
reported wherever they beat a loss (pLDDT, PAE, ipTM, atom-pair distances,
angles); con / i_con / helix keep a standardized loss alongside structural
contact stats. Reuses the optimizer's own loss/geometry functions from
``loss_functions`` rather than re-deriving them.

Stand-alone: ``from final_metrics import compute_final_metrics, save_metrics_json``.
"""
import json
import os

import torch


from loss_functions import chain_to_number_, get_con_loss, _get_helix_loss, \
    get_mid_points, add_rg_loss, add_com_loss, parse_pdb_ori_atoms, \
    resolve_atom_pairs, resolve_atom_angles, get_atom_pair_distogram_loss, \
    get_atom_pair_coords_loss, get_atom_angle_coords_loss



class _NoModel(Exception):
    """Sentinel: boltz_model is None (offline backfill) -> skip the distogram
    block quietly rather than reporting it as a failure."""


def _masked_mean(values_1d, mask_1d):
    """Mean of a 1-D tensor over a 0/1 (or bool) mask; nan when the mask is empty."""
    mask = mask_1d.to(values_1d.dtype)
    denom = float(mask.sum().item())
    if denom <= 0:
        return float('nan')
    return float((values_1d * mask).sum().item() / denom)


def _interface_contact_stats(coords, atom_ent, atom_tok, binder_eid, target_eid, cutoff):
    """Structure-derived binder<->target-chain stats: (min heavy-atom distance A,
    number of binder residues with a heavy atom within `cutoff` of the target)."""
    b = atom_ent == binder_eid
    t = atom_ent == target_eid
    if int(b.sum()) == 0 or int(t.sum()) == 0:
        return float('nan'), 0
    d = torch.cdist(coords[b], coords[t])                  # [Nb, Nt]
    close = d.min(dim=1).values < cutoff                   # [Nb] per binder atom
    contact_tokens = torch.unique(atom_tok[b][close])
    return float(d.min().item()), int(contact_tokens.numel())


def compute_final_metrics(boltz_model, output, best_batch, best_structure, *,
                          binder_chain='A', target_chain_ids=None, length=None,
                          atom_pairs=None, atom_angles=None,
                          com_loss_weight=0.0, pdb_path='',
                          metric_config=None, interface_contact_cutoff=5.0):
    """Standardized, cross-run-comparable metrics from the final 200-step fold.

    `output` is the predict_step dict (coords/plddt/pae/iptm/ptm/...); `best_batch`
    / `best_structure` are the real-sequence featurization that produced it, so
    atom-pair / atom-angle / com specs are RE-resolved against them (the design-
    time specs were resolved on the 'X'*length layout). `metric_config` carries
    the per-target-type-default contact settings. Empty `target_chain_ids` => apo
    (binder-alone) reduced set. Every block is independently guarded so one
    failure never drops the rest."""
    target_chain_ids = list(target_chain_ids or [])
    metric_config = metric_config or {}
    coords = output['coords']
    device = coords.device
    binder_eid = chain_to_number_[binder_chain]

    entity_id_b = best_batch['entity_id']                  # [1, T]
    if entity_id_b.dim() == 1:
        entity_id_b = entity_id_b.unsqueeze(0)
    entity_id = entity_id_b[0]                              # [T]
    n_token = entity_id.numel()
    chain_mask_b = (entity_id_b == binder_eid).to(torch.float32)   # [1, T]
    chain_mask = chain_mask_b[0]                            # [T]
    n_binder = float(chain_mask.sum().item())

    metrics = {
        'binder_chain': binder_chain,
        'target_chains': target_chain_ids,
        'interface_contact_cutoff_A': interface_contact_cutoff,
        'units': {'plddt': '0-1 higher=better', 'pae': 'Angstrom lower=better',
                  'distance': 'Angstrom', 'angle': 'degrees', 'rg': 'Angstrom',
                  'con/i_con/helix': 'cross-entropy loss (standardized settings)'},
        'confidence': {}, 'geometry': {}, 'standardized_loss': {},
    }
    if length is not None:
        metrics['binder_length'] = int(length)

    # ---- Confidence quantities (report value, not 1-plddt) ------------------
    try:
        conf = metrics['confidence']
        plddt = output['plddt']
        plddt_flat = plddt[0] if plddt.dim() > 1 else plddt
        conf['binder_plddt'] = _masked_mean(plddt_flat, chain_mask)
        for key in ('complex_plddt', 'complex_iplddt', 'ptm', 'iptm', 'ligand_iptm',
                    'protein_iptm', 'complex_pde', 'complex_ipde', 'confidence_score'):
            if key in output:
                conf[key] = float(output[key].item())
        pae = None
        if 'pae' in output:
            pae = output['pae']
            pae = pae[0] if pae.dim() > 2 else pae
            pae = (pae + pae.transpose(-2, -1)) / 2
            bb = (chain_mask[:, None] * chain_mask[None, :]).reshape(-1)
            conf['binder_pae'] = _masked_mean(pae.reshape(-1), bb)
        tgt_plddt, tgt_pae, tgt_iptm = {}, {}, {}
        for cid in target_chain_ids:
            teid = chain_to_number_[cid]
            tmask = (entity_id == teid).to(torch.float32)
            tgt_plddt[cid] = _masked_mean(plddt_flat, tmask)
            if pae is not None:
                inter = (chain_mask[:, None] * tmask[None, :]
                         + tmask[:, None] * chain_mask[None, :]).clamp(max=1.0)
                tgt_pae[cid] = _masked_mean(pae.reshape(-1), inter.reshape(-1))
            pci = output.get('pair_chains_iptm')
            if pci is not None:
                try:
                    tgt_iptm[cid] = float(pci[binder_eid][teid].item())
                except Exception:
                    pass
        if tgt_plddt:
            conf['target_plddt'] = tgt_plddt
        if tgt_pae:
            conf['interface_pae'] = tgt_pae
        if tgt_iptm:
            conf['interface_iptm'] = tgt_iptm
    except Exception as e:
        print(f"[metrics] confidence block failed: {type(e).__name__}: {e}")

    # ---- coords-derived: atom->chain map, Rg, interface contacts ------------
    atom_tok = atom_ent = None
    try:
        atom_tok = best_batch['atom_to_token'][0].argmax(dim=-1)   # [n_atom]
        atom_ent = entity_id[atom_tok]
    except Exception as e:
        print(f"[metrics] atom->chain map failed: {type(e).__name__}: {e}")

    try:
        _, rg = add_rg_loss(coords, best_batch, length or int(n_binder),
                            binder_chain=binder_chain)
        metrics['geometry']['rg'] = float(rg.item())
        metrics['geometry']['rg_threshold'] = 2.38 * n_binder ** 0.365
    except Exception as e:
        print(f"[metrics] rg failed: {type(e).__name__}: {e}")

    if atom_ent is not None and target_chain_ids:
        ncon, mind = {}, {}
        coords_flat = coords[0][:atom_ent.numel()]
        for cid in target_chain_ids:
            try:
                d, n = _interface_contact_stats(
                    coords_flat, atom_ent, atom_tok, binder_eid,
                    chain_to_number_[cid], interface_contact_cutoff)
                mind[cid], ncon[cid] = d, n
            except Exception as e:
                print(f"[metrics] interface stats {cid} failed: {type(e).__name__}: {e}")
        if ncon:
            metrics['geometry']['n_interface_contacts'] = ncon
            metrics['geometry']['min_interface_distance'] = mind

    # COM distance (binder centroid -> mean Kabsch-transformed ORI), pdb_path src.
    if com_loss_weight and com_loss_weight > 0 and pdb_path and atom_ent is not None:
        try:
            ori = parse_pdb_ori_atoms(pdb_path)
            nb = torch.nonzero((atom_ent != binder_eid) & (atom_tok < n_token),
                               as_tuple=True)[0]
            ref_xyz = best_structure.atoms['coords'][nb.detach().cpu().numpy(), :]
            if ori.shape[0] > 0 and ref_xyz.shape[0] >= 3:
                com_loss, _, _, _ = add_com_loss(coords, best_batch, [{
                    'label': '--pdb_path',
                    'anchor_idx': nb.to(device).long(),
                    'anchor_ref': torch.as_tensor(ref_xyz, dtype=torch.float32, device=device),
                    'ori_ref': torch.as_tensor(ori, dtype=torch.float32, device=device),
                }], binder_chain=binder_chain)
                metrics['geometry']['com_distance'] = float(com_loss.item())
        except Exception as e:
            print(f"[metrics] com block failed: {type(e).__name__}: {e}")

    # ---- distogram pass: standardized con/i_con/helix + expected pair dists --
    # boltz_model is None in the offline backfill path (no trunk available): the
    # standardized con/i_con/helix losses and the distogram-expected atom-pair
    # distance are simply skipped, not an error.
    pdistogram = mid_pts = None
    try:
        if boltz_model is None:
            raise _NoModel
        with torch.no_grad():
            dist_out, _s, _z, _si = boltz_model.get_distogram(best_batch)
        pdistogram = dist_out['pdistogram']
        mid_pts = get_mid_points(pdistogram)
        std = metrics['standardized_loss']
        num_intra = metric_config.get('num_intra_contacts', 2)
        intra_cut = metric_config.get('intra_chain_cutoff', 14.0)
        std['settings'] = {'num_intra_contacts': num_intra, 'intra_chain_cutoff': intra_cut}
        std['con_loss'] = float(get_con_loss(
            pdistogram, mid_pts, num=num_intra, seqsep=9, cutoff=intra_cut,
            binary=False, mask_1d=chain_mask_b, mask_1b=chain_mask_b).item())
        mask_2d = chain_mask_b[:, :, None] * chain_mask_b[:, None, :]
        std['helix_loss'] = float(_get_helix_loss(
            pdistogram, mid_pts, offset=None, mask_2d=mask_2d, binary=True).item())

        if target_chain_ids:
            nt = len(target_chain_ids)
            num_inter = metric_config.get('num_inter_contacts', [2] * nt)
            inter_cut = metric_config.get('inter_chain_cutoff', [20.0] * nt)
            opt_pos = metric_config.get('optimize_contact_per_binder_pos', [False] * nt)
            std['settings']['num_inter_contacts'] = list(num_inter)
            std['settings']['inter_chain_cutoff'] = list(inter_cut)
            std['settings']['optimize_contact_per_binder_pos'] = list(opt_pos)
            per_target, i_con_total = {}, 0.0
            for ti, cid in enumerate(target_chain_ids):
                tmask_b = (entity_id_b == chain_to_number_[cid]).to(torch.float32)
                if opt_pos[ti]:
                    li = get_con_loss(pdistogram, mid_pts, num=num_inter[ti], seqsep=0,
                                      cutoff=inter_cut[ti], binary=False,
                                      mask_1d=chain_mask_b, mask_1b=tmask_b)
                else:
                    li = get_con_loss(pdistogram, mid_pts, num=num_inter[ti], seqsep=0,
                                      cutoff=inter_cut[ti], binary=False,
                                      mask_1d=tmask_b, mask_1b=chain_mask_b)
                per_target[cid] = float(li.item())
                i_con_total += float(li.item())
            std['i_con_loss'] = i_con_total
            std['i_con_loss_per_target'] = per_target
    except _NoModel:
        pass  # backfill: no trunk -> standardized losses intentionally omitted
    except Exception as e:
        print(f"[metrics] distogram block failed: {type(e).__name__}: {e}")

    # ---- atom-pair: actual (coords) + expected (distogram); atom-angle -------
    if atom_pairs:
        try:
            pair_specs = resolve_atom_pairs(atom_pairs, best_batch, best_structure)
            _, coord_stats = get_atom_pair_coords_loss(coords, pair_specs, return_stats=True)
            exp_by_label = {}
            if pdistogram is not None:
                _, exp_stats = get_atom_pair_distogram_loss(
                    pdistogram, pair_specs, mid_pts, method='expected', return_stats=True)
                exp_by_label = {s['label']: s['exp_dist'] for s in exp_stats}
            metrics['geometry']['atom_pairs'] = [{
                'label': s['label'], 'distance_struct': s['dist'],
                'distance_expected': exp_by_label.get(s['label']),
                'lo': s['lo'], 'hi': s['hi'],
                'in_window': bool(s['lo'] - 1e-6 <= s['dist'] <= s['hi'] + 1e-6),
            } for s in coord_stats]
        except Exception as e:
            print(f"[metrics] atom_pair block failed: {type(e).__name__}: {e}")

    if atom_angles:
        try:
            angle_specs = resolve_atom_angles(atom_angles, best_batch, best_structure)
            _, ang_stats = get_atom_angle_coords_loss(coords, angle_specs, return_stats=True)
            metrics['geometry']['atom_angles'] = [{
                'label': s['label'], 'angle_struct': s['angle'],
                'lo': s['lo'], 'hi': s['hi'],
                'in_window': bool(s['lo'] - 1e-6 <= s['angle'] <= s['hi'] + 1e-6),
            } for s in ang_stats]
        except Exception as e:
            print(f"[metrics] atom_angle block failed: {type(e).__name__}: {e}")

    return metrics


def save_metrics_json(folder_dir, output_metrics, name, model_idx=0):
    """Write a final-fold metrics dict next to the confidence JSON for `name`."""
    output_dir = os.path.join(folder_dir, f"boltz_results_{name}", "predictions", name)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"metrics_{name}_model_{model_idx}.json")
    with open(path, 'w') as f:
        json.dump(output_metrics, f, indent=4)
    return path
