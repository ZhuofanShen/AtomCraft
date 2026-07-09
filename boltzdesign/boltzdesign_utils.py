import os
import torch
import subprocess
import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional
import copy
import math
import random
from boltz.data.types import MSA, Connection, Input, Structure, Interface
from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
from loss_functions import get_mid_points, align_points, np_rmsd, parse_motif_ligand_spec, extract_motif_ligand_atoms, parse_motif_residue_spec, extract_motif_coords, parse_residue_spec, extract_motif_residue_atoms, _rigid_frames, parse_motif_islands_spec, island_length, random_valid_placement, get_motif_target_distances, _placement_to_positions, propose_mcmc_move, _find_named_atom, resolve_atom_pairs, resolve_atom_angles, parse_pdb_ori_atoms, _get_helix_loss, get_con_loss, get_plddt_loss, get_pae_loss, get_ca_coords, add_rg_loss, add_com_loss, get_motif_distogram_loss, get_motif_coords_loss, get_motif_fape_loss, kabsch_align, get_atom_pair_distogram_loss, get_atom_pair_coords_loss, get_atom_angle_coords_loss
from plot_utils import _LOSS_GROUPS, _plot_loss_group, _atom_pair_subplot_augments, _atom_angle_subplot_augments, plot_total_loss_history, visualize_training_history
from final_metrics import compute_final_metrics, save_metrics_json
import yaml
import shutil
import matplotlib.pyplot as plt
import numpy as np
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


@dataclass
class SamplerConfig:
    """Diffusion-sampler settings for one forward-pass phase.

    Bundles the per-call ``predict_args`` knobs (recycling / sampling step
    counts) with the ``AtomDiffusion`` runtime attributes (step_scale, gamma_0,
    use_heun, deterministic_sampler, attach_coords). Two phases use it:

      * the gradient **design epochs** -- per-run tuning lives here;
      * the **final prediction** + semi-greedy evals -- stock Boltz defaults.

    The defaults below ARE the stock Boltz defaults (== ``BoltzDiffusionParams``
    for the sampler attrs, recycling 3 / sampling 200 for the step counts), so
    ``SamplerConfig()`` is the final-prediction profile and the design profile
    is built by overriding only the knobs the user set.
    """

    recycling_steps: int = 3
    sampling_steps: int = 200
    step_scale: float = 1.638
    gamma_0: float = 0.605
    use_heun: bool = False
    deterministic_sampler: bool = False
    attach_coords: bool = False
    diffusion_samples: int = 1
    write_confidence_summary: bool = True
    write_full_pae: bool = False
    write_full_pde: bool = False

    def predict_args(self) -> dict:
        """The ``model.predict_args`` dict for this phase (recycling / sampling
        step counts + writer flags). The sampler attrs are applied separately
        via :func:`apply_sampler_config` because they live on the module, not
        in predict_args."""
        return {
            "recycling_steps": self.recycling_steps,
            "sampling_steps": self.sampling_steps,
            "diffusion_samples": self.diffusion_samples,
            "write_confidence_summary": self.write_confidence_summary,
            "write_full_pae": self.write_full_pae,
            "write_full_pde": self.write_full_pde,
        }


def apply_sampler_config(boltz_model, cfg: SamplerConfig):
    """Install a :class:`SamplerConfig` onto the model.

    Sets the ``AtomDiffusion`` runtime attributes (read fresh inside
    ``sample()`` every forward pass, so this takes effect immediately and is
    fully reversible) and ``model.predict_args``. Returns the predict_args dict
    for convenience. This is the single point where per-phase sampler settings
    are switched, so design-epoch tuning never leaks into the final fold.
    """
    sm = boltz_model.structure_module
    sm.step_scale = cfg.step_scale            # eta; 1.0 = velocity-consistent ODE step
    sm.gamma_0 = cfg.gamma_0                   # EDM churn; 0.0 = noise-free ODE sampler
    sm.use_heun = cfg.use_heun                 # 2nd-order corrector
    sm.deterministic_sampler = cfg.deterministic_sampler  # aug/SVD/churn-free for backprop
    # Drops the sampler's no_grad wrapper; paired with disconnect_coords=False
    # in confidence_args so gradients reach the confidence head through coords.
    sm.attach_coords = cfg.attach_coords
    boltz_model.predict_args = cfg.predict_args()
    return boltz_model.predict_args


def get_boltz_model(checkpoint: Optional[str] = None, predict_args=None, device: Optional[str] = None) -> Boltz1:
    """Load Boltz1 with **stock** diffusion defaults.

    Per-run sampler tuning is no longer baked in here: the design loop installs
    its settings via :func:`apply_sampler_config` for the gradient epochs and
    restores these stock defaults before the final prediction. Keeping the model
    at stock defaults means any forward pass that bypasses the design loop (e.g.
    the final fold) uses the validated Boltz sampler.
    """
    torch.set_grad_enabled(True)
    torch.set_float32_matmul_precision("highest")
    diffusion_params = BoltzDiffusionParams()
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
    return model_module



def boltz_hallucination(
    # Required arguments
    boltz_model,
    yaml_path,
    ccd_lib,
    length=100,
    binder_chain='A',
    design_algorithm="3stages",
    # Diffusion-sampler settings for the gradient DESIGN EPOCHS only. They are
    # installed on the model for the design loop via apply_sampler_config(); the
    # semi-greedy evals + final prediction restore stock Boltz defaults. Defaults
    # here mirror the stock values so an unset knob == stock behavior.
    recycling_steps=0,
    sampling_steps=200,
    step_scale=1.638,
    gamma_0=0.605,
    use_heun=False,
    deterministic_sampler=False,
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
    # Optional per-target residue selections that restrict WHICH target residues
    # are visible to the inter-chain losses (epitope targeting). Chain-prefixed,
    # 1-indexed residue positions within each target chain, range-aware:
    # "B:50-70 B:95 C:1-10". The two selections are INDEPENDENT:
    #   i_con_target_residues -> restricts the inter-chain CONTACT loss target side
    #   i_pae_target_residues -> restricts the inter-chain PAE loss target side
    # A target chain absent from a given spec keeps its FULL sequence for that
    # loss (so with chains B,C and i_con_target_residues="B:50-70 B:95", chain C
    # is still fully contacted). None/empty => full sequence for every target
    # (legacy behavior, byte-identical). Intended for polymer targets (protein/
    # peptide/dna/rna), where 1 token == 1 residue; for small_molecule/metal
    # targets use the i_con_loss_weights / i_pae_loss_weights knobs to turn a
    # term on/off rather than selecting residues.
    i_con_target_residues=None,
    i_pae_target_residues=None,
    # Auxiliary INTER-TARGET losses: optimize contacts / interface-PAE BETWEEN
    # two target chains (not the binder). Each entry is a pair-string
    # "<sideA> | <sideB>" (also "<sideA> and <sideB>"), where each side is one or
    # more CHAIN:RESNUM / CHAIN:start-end selections (comma/space separated) and
    # both sides are target chains, e.g. "B:45-47 | C:52" or "B:103 | D:82-91,D:98".
    # Computed with the SAME get_con_loss / get_pae_loss machinery as the
    # binder<->target i_con/i_pae (the two target-residue selections become
    # mask_1d / mask_1b). The two specs are independent; None/empty => no
    # inter-target term (legacy). num/cutoff below mirror num_inter_contacts /
    # inter_chain_cutoff for the contact term; the per-loss WEIGHTS live in
    # loss_scales as 'inter_target_con_loss' / 'inter_target_pae_loss'.
    inter_target_con_pairs=None,
    inter_target_pae_pairs=None,
    inter_target_num_contacts=2,
    inter_target_cutoff=14.0,
    e_soft=0.8,
    e_soft_1=0.8,
    e_soft_2=1.0,
    alpha=2.0,
    pre_run=False,
    set_train=True,
    use_temp=False,
    disconnect_feats=False,
    disconnect_pairformer=False,
    # Independent grad-flow control from the diffusion sampler back to the
    # sequence (vs. disconnect_feats/disconnect_pairformer which gate only the
    # confidence head). disconnect_feats_structure detaches s_inputs into the
    # sampler; disconnect_pairformer_structure detaches s_trunk/z_trunk. The
    # s_inputs bypass is cut by default; the through-trunk route stays connected
    # (mirrors the confidence head). Only matter in full mode with attach_coords=True.
    disconnect_feats_structure=True,
    disconnect_pairformer_structure=False,
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
    # Placement method for the sliding islands: 'mcmc' (Metropolis on coords
    # Kabsch RMSD, the existing path) or 'exhaustive' (deterministic exhaustive
    # triplet enumeration on the trunk distogram; see motif_enum). Selected by
    # --motif_slide_method. Inert without --motif_unindex_residues.
    motif_slide_method='mcmc',
    # Determinant score for the placement search (--motif_slide_loss):
    # 'distogram' (fast-mode safe, every epoch), 'rmsd' or 'fape' (coords, full
    # mode only). Independent of the motif_*_loss gradient weights.
    motif_slide_loss='distogram',
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
    # How the motif distogram restraint is computed: 'cross_entropy' (default;
    # one-hot categorical CE supervised by the motif ground-truth distance, with
    # the same diagonal/symmetry convention as the con/i_con contact losses) or
    # 'mse' (bin-center MSE). See get_motif_distogram_loss.
    motif_distogram_loss_type='cross_entropy',
    # Optional mutable dict the caller passes to receive the motif-residue ->
    # final-binder-position mapping (e.g. {"A30": "A27", ...}); populated after
    # the design loop (final slide_state) for both fixed + sliding motifs. The
    # caller writes it to JSON. None = don't build it (no overhead).
    motif_mapping_out=None,
    # Explicit atom/residue distance restraints (--atom_pairs). Inactive unless
    # set, so existing runs are byte-for-byte unchanged. Weights come via
    # loss_scales['atom_pair_distogram_loss'/'atom_pair_coords_loss'].
    atom_pairs='',
    # How the --atom_pairs distogram restraint is computed: 'cross_entropy'
    # (default; one-hot CE at the reference bin for a point target, -log P(d in
    # window) for a window), 'expected' (bin-center MSE for a point target,
    # flat-bottom on E[d] for a window), or 'contact' (BoltzDesign1's categorical
    # contact loss; robust attractive gradient, large when far / tapering near).
    # See get_atom_pair_distogram_loss.
    atom_pair_distogram_loss_type='cross_entropy',
    # Explicit three-atom angle restraints (--atom_angles). Inactive unless set.
    # Coords-only (an angle is undefined on the trunk distogram): contributes a
    # gradient only in full mode with --attach_coords True. Weight comes via
    # loss_scales['atom_angle_coords_loss']. See get_atom_angle_coords_loss.
    atom_angles='',
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
    # Standardized final-fold metric settings (per-target-type defaults, NOT the
    # run's optimization knobs). Passed straight to compute_final_metrics so the
    # final con/i_con/helix loss values compare across differently-tuned runs.
    # None => compute_final_metrics falls back to canonical defaults.
    metric_config=None,
):

    # Sampler profile for the gradient design epochs. Installed on the model now
    # (the design-loop forwards read these) and reused for nothing else: the
    # semi-greedy evals + final prediction restore stock defaults (final_sampler
    # below). predict_args drives the design() calls' confidence_args; attach_coords
    # is read directly from the closure in get_model_loss for disconnect_coords.
    design_sampler = SamplerConfig(
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
        step_scale=step_scale,
        gamma_0=gamma_0,
        use_heun=use_heun,
        deterministic_sampler=deterministic_sampler,
        attach_coords=attach_coords,
    )
    predict_args = apply_sampler_config(boltz_model, design_sampler)

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

    # stage_name -> last-epoch argmax sequence string for the binder chain.
    # Populated after each design() call in the algorithm branches below;
    # consumed after final_sampler is applied to write {stage}_last.pdb into
    # intermediate_dir (always-on, independent of save_intermediate_structures).
    stage_lasts = {}

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

    # ---- Residue-restricted target masks for the inter-chain losses ----------
    # Epitope targeting: build a residue-subset version of each target_masks[ti]
    # for the inter-chain CONTACT and PAE terms, independently. A polymer chain
    # contributes its tokens in residue order, so residue r (1-indexed) is the
    # (r-1)-th token of that chain. Chains not named in a spec keep their full
    # mask, so a subset selection on one chain never silently masks the others.
    def _chain_token_positions(c):
        return torch.where(batch['entity_id'][0] == chain_to_number[c])[0]

    def _parse_residue_selection(spec):
        """'B:50-70 B:95 C:1-10' (also comma/semicolon separated) ->
        {chain: set(1-indexed resnums)}. Empty/None -> {}."""
        import re
        sel = {}
        if not spec:
            return sel
        for tok in (t for t in re.split(r'[\s,;]+', str(spec).strip()) if t):
            if ':' not in tok:
                raise ValueError(
                    f"residue selection token {tok!r} must be CHAIN:RESNUM "
                    f"or CHAIN:start-end (e.g. B:95 or B:50-70)")
            ch, rng = tok.split(':', 1)
            ch = ch.strip()
            if '-' in rng:
                lo, hi = (int(x) for x in rng.split('-', 1))
                lo, hi = min(lo, hi), max(lo, hi)
                rs = range(lo, hi + 1)
            else:
                rs = [int(rng)]
            sel.setdefault(ch, set()).update(rs)
        return sel

    def _build_restricted_masks(spec, label):
        """Per-target list of token masks: the full target mask for any chain
        absent from `spec`, else that chain's mask zeroed outside the selected
        residues. Mirrors target_masks shape/dtype/device exactly."""
        sel = _parse_residue_selection(spec)
        unknown = set(sel) - set(target_chain_ids)
        if unknown:
            raise ValueError(
                f"{label}: chain(s) {sorted(unknown)} are not target chains "
                f"{target_chain_ids}")
        out = []
        for ti, c in enumerate(target_chain_ids):
            if c not in sel:
                out.append(target_masks[ti])          # full chain (unchanged)
                continue
            positions = _chain_token_positions(c)      # token idx, residue order
            L = int(positions.numel())
            chosen = []
            for r in sorted(sel[c]):
                if not (1 <= r <= L):
                    raise ValueError(
                        f"{label}: chain {c} residue {r} out of range "
                        f"[1, {L}] (chain has {L} token(s))")
                chosen.append(positions[r - 1])
            sub = torch.zeros_like(target_masks[ti])
            sub[0, torch.stack(chosen)] = 1
            out.append(sub)
            print(f"[inter-mask] {label}: chain {c} restricted to "
                  f"{len(chosen)}/{L} residues {sorted(sel[c])}")
        return out

    i_con_target_masks = _build_restricted_masks(
        i_con_target_residues, 'i_con_target_residues')
    i_pae_target_masks = _build_restricted_masks(
        i_pae_target_residues, 'i_pae_target_residues')

    # ---- Inter-target residue-pair masks (auxiliary cross-target losses) ------
    # Optimize contacts / interface-PAE BETWEEN two target chains (never the
    # binder). Each pair is "<sideA> | <sideB>" (also "<sideA> and <sideB>" /
    # "<sideA> vs <sideB>"); each side is one or more CHAIN:RESNUM /
    # CHAIN:start-end selections (comma/space separated). Both sides must be
    # TARGET chains. We reuse the SAME residue->token resolution as the epitope
    # masks above, then hand the two side-masks to the unchanged get_con_loss /
    # get_pae_loss as mask_1d / mask_1b, so the math is identical to i_con/i_pae.
    import re as _re_pair
    _PAIR_SEP = _re_pair.compile(r'\s*\|\s*|\s+and\s+|\s+vs\s+', _re_pair.IGNORECASE)

    def _selection_to_mask(sel, side_label):
        """{chain: set(resnums)} -> [1, N] token mask over the selected target
        residues (union across chains). Chains must be target chains."""
        mask = torch.zeros_like(chain_mask)
        for c in sorted(sel):
            if c not in target_chain_ids:
                raise ValueError(
                    f"inter-target pair ({side_label}): chain {c} is not a "
                    f"target chain {target_chain_ids}; inter-target losses are "
                    f"target<->target only (use --i_con_target_residues / "
                    f"--i_pae_target_residues for binder<->target selections)")
            positions = _chain_token_positions(c)
            L = int(positions.numel())
            for r in sorted(sel[c]):
                if not (1 <= r <= L):
                    raise ValueError(
                        f"inter-target pair ({side_label}): chain {c} residue {r}"
                        f" out of range [1, {L}] (chain has {L} token(s))")
                mask[0, positions[r - 1]] = 1
        return mask

    def _build_inter_target_pair_masks(pairs, label):
        """List of (maskA, maskB, descriptor) per "<sideA>|<sideB>" pair.
        `pairs` may be a list of pair-strings or a single string; empty -> []."""
        if not pairs:
            return []
        if isinstance(pairs, str):
            pairs = [pairs]
        out = []
        for raw in pairs:
            raw = str(raw).strip()
            if not raw:
                continue
            parts = _PAIR_SEP.split(raw, maxsplit=1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError(
                    f"{label}: pair {raw!r} must have two sides separated by "
                    f"'|' (or 'and'/'vs'), e.g. 'B:45-47 | C:52'")
            mA = _selection_to_mask(_parse_residue_selection(parts[0]),
                                    f"{label} sideA of {raw!r}")
            mB = _selection_to_mask(_parse_residue_selection(parts[1]),
                                    f"{label} sideB of {raw!r}")
            desc = f"{parts[0].strip()} | {parts[1].strip()}"
            out.append((mA, mB, desc))
            print(f"[inter-target] {label}: {desc}  "
                  f"(|A|={int(mA.sum().item())} tok, |B|={int(mB.sum().item())} tok)")
        return out

    inter_target_con_pair_masks = _build_inter_target_pair_masks(
        inter_target_con_pairs, 'inter_target_con_pairs')
    inter_target_pae_pair_masks = _build_inter_target_pair_masks(
        inter_target_pae_pairs, 'inter_target_pae_pairs')

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
    if motif_slide_active:
        if motif_slide_method not in ('mcmc', 'exhaustive'):
            raise ValueError(
                f"--motif_slide_method must be 'mcmc' or 'exhaustive', "
                f"got {motif_slide_method!r}")
        if motif_slide_loss not in ('distogram', 'rmsd', 'fape'):
            raise ValueError(
                f"--motif_slide_loss must be 'distogram', 'rmsd' or 'fape', "
                f"got {motif_slide_loss!r}")
        # Exhaustive enumeration runs the distogram determinant every epoch
        # (fast-mode safe) and the rmsd/fape determinants on full-mode epochs
        # (coords needed). The rmsd/fape exhaustive path supports only the
        # default backbone selection (uniform atom grid); it is validated
        # eagerly in `_build_slide_mp_coords` below.
    motif_token_idx = None        # LongTensor of token-axis indices (the binder
                                  # residues that hold the motif). Rebuilt per
                                  # epoch when sliding is active.
    motif_target_dist = None      # [M, M] raw reference pairwise distances (A)
    motif_onehot = None           # [M, V] hard res_type for sequence pinning
    # slide_state captured at the moment of the last sequence pin (start of the
    # final epoch). The placement search (MCMC/exhaustive) moves slide_state
    # *after* the pin within get_model_loss, so the final folded sequence is
    # pinned at THIS state, not the post-move one. The motif_mapping write and
    # the semi-greedy mutation guard both read it so they match the actual
    # binder sequence. None until the first pinned epoch (or if fix_motif_seq
    # is off / no sliding motif).
    _last_pin_slide_state = [None]
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

        # Default atom set when a residue carries no explicit :ATOMS selection.
        # The SAME resolved selection drives BOTH the coords Kabsch fit/RMSD and
        # the FAPE scored set -- atom membership (incl. backbone N/CA/C/O) is
        # controlled ONLY by the explicit selection; there is no hidden skip.
        _BB = ['N', 'CA', 'C', 'CB']
        V = batch['res_type'].shape[-1]
        N = batch['res_type'].shape[1]

        def _resolve_sel(sel, atoms_dict, default_keys, label=''):
            """Resolve a per-residue atom selection against the reference atom
            dict. ``sel`` is None (-> ``default_keys``), 'ALL' (every heavy atom
            present), or an explicit list of names. Warns on explicitly-requested
            atoms that are absent in the reference."""
            if sel is None:
                names = [nm for nm in default_keys if nm in atoms_dict]
            elif sel == 'ALL':
                names = list(atoms_dict.keys())
            else:
                names = [nm for nm in sel if nm in atoms_dict]
                _missing = [nm for nm in sel if nm not in atoms_dict]
                if _missing:
                    print(f"[motif] WARNING: {label} requested atom(s) "
                          f"{_missing} absent in reference (have "
                          f"{sorted(atoms_dict)}); skipped")
            return names

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
        # FAPE protos. FIXED residues contribute CONSTANT frames + scored atoms
        # (collected in the fixed loop). SLIDING residues contribute DYNAMIC ones
        # (collected in the sliding setup; their pred indices are rebuilt per
        # epoch from slide_state). Carried ligand atoms are appended in the FAPE
        # assembly after ligand parsing. loc_ref (the reference) is constant in
        # all cases -- only sliding pred indices move.
        _fape_fr_pred, _fape_fr_ref, _fape_sc_pred, _fape_sc_ref = [], [], [], []
        _fape_fr_slide_slot, _fape_fr_ref_slide = [], []
        _fape_sc_slide_slot, _fape_sc_slide_col, _fape_sc_ref_slide = [], [], []
        _fape_fr_slide_cols = None
        if motif_fixed_active:
            _parsed_motif = parse_motif_residue_spec(motif_residues, with_atoms=True)
            if not _parsed_motif:
                raise ValueError("--motif_residues set but parsed to no entries")
            # Residue-level keys (distogram target, binder map, sliding conflict
            # check) + the parallel per-residue atom selection (coords / FAPE).
            m_chain_residues = [(c, r) for c, r, _ in _parsed_motif]
            motif_atom_sel = [a for _, _, a in _parsed_motif]
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
                # No explicit positions: lay islands out back-to-back, honoring
                # the per-island intra-offsets (so a fixed-gap spacer like
                # "A38 3 A42" keeps A38, A42 four positions apart). With no
                # spacers/ranges this reduces to range(M_fix) (legacy behavior).
                _fixed_islands = parse_motif_islands_spec(motif_residues,
                                                          flag='--motif_residues')
                binder_pos, _base = [], 0
                for _isl in _fixed_islands:
                    for _rr in _isl:
                        binder_pos.append(_base + _rr['intra_offset'])
                    _base += island_length(_isl)
            if max(binder_pos) >= length:
                raise ValueError(
                    f"motif binder position {max(binder_pos)+1} exceeds binder "
                    f"length {length}; raise --length_min/--length_max")
            fixed_positions = set(binder_pos)
            fixed_token_idx_list = [int(binder_tok[p]) for p in binder_pos]

            _ref_res_atoms = extract_motif_residue_atoms(motif_pdb, m_chain_residues)
            _unresolved = []  # explicitly-selected atoms absent on the UNK binder
            for _k, _pos in enumerate(binder_pos):
                _ratoms = _ref_res_atoms[_k]['atoms']
                _sel = motif_atom_sel[_k]
                _c, _r = m_chain_residues[_k]
                # ONE selected atom set per residue (default N,CA,C,CB) feeds BOTH
                # the coords Kabsch fit/RMSD AND the FAPE scored set -- membership
                # is controlled solely by the :ATOMS selection (no hidden skip).
                _sel_names = _resolve_sel(_sel, _ratoms, _BB,
                                          label=f"--motif_residues {_c}{_r}")
                for _nm in _sel_names:
                    _pi = _pred_atom_idx(_pos, _nm)
                    if _pi is not None:
                        fixed_bb_pred_list.append(_pi)      # coords fit + RMSD
                        fixed_bb_ref_list.append(_ratoms[_nm])
                        _fape_sc_pred.append(_pi)            # FAPE scored atom
                        _fape_sc_ref.append(_ratoms[_nm])
                    elif _sel not in (None, 'ALL'):
                        _unresolved.append(f"{_c}{_r}:{_nm}")
                # FAPE frame per FIXED residue (needs N,CA,C; independent of the
                # scored selection). CONSTANT pred indices.
                _pN, _pCA, _pC = (_pred_atom_idx(_pos, _n) for _n in ('N', 'CA', 'C'))
                if (None not in (_pN, _pCA, _pC)
                        and all(_n in _ratoms for _n in ('N', 'CA', 'C'))):
                    _fape_fr_pred.append([_pN, _pCA, _pC])
                    _fape_fr_ref.append([_ratoms['N'], _ratoms['CA'], _ratoms['C']])
            if _unresolved:
                print(f"[motif] WARNING: selected coords atom(s) not present on "
                      f"the (UNK) binder, so they do not contribute: "
                      f"{_unresolved}. Build the residue type (--fix_motif_seq) "
                      f"or pick atoms the binder has.")
            if not fixed_bb_pred_list:
                raise ValueError("motif: no fixed-mode protein motif atoms could be resolved")

        # ----- Sliding-mode setup (--motif_unindex_residues) -----
        slide_islands = []
        slide_state = None
        slide_island_lengths = []
        slide_motif_seq = ''
        slide_ref_xyz = np.zeros((0, 3), dtype=np.float32)
        slide_atom_res_idx = slide_atom_within_idx = slide_atom_ref_xyz = None
        binder_slide_atom_idx = None
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

            # Per-residue atom selection (shared :ATOMS grammar), aligned with
            # slide_chain_residues / _slide_ref_res_atoms (island-then-residue).
            slide_atom_sel = [r['atoms'] for isl in slide_islands for r in isl]

            # Collect, per sliding residue (keyed by motif slot + atom NAME, so
            # the binder position can be resolved per epoch via slide_state). ONE
            # selected atom set per residue (default N,CA,C,CB) feeds BOTH the
            # coords loss AND the FAPE scored set; the FAPE frame additionally
            # needs N,CA,C. Binder presence is filtered after the index map.
            _slide_specs = []                       # (slot, name, ref_xyz)  coords + FAPE scored
            _slide_names = {'N', 'CA', 'C'}         # frames always need N,CA,C
            _fr_slide_proto = []                    # (slot, [N_ref,CA_ref,C_ref])
            for _ri, ent in enumerate(_slide_ref_res_atoms):
                _c, _r = slide_chain_residues[_ri]
                _ra = ent['atoms']
                for _nm in _resolve_sel(slide_atom_sel[_ri], _ra, _BB,
                                        label=f"--motif_unindex_residues {_c}{_r}"):
                    _slide_specs.append((_ri, _nm, _ra[_nm]))
                    _slide_names.add(_nm)
                if all(_n in _ra for _n in ('N', 'CA', 'C')):
                    _fr_slide_proto.append((_ri, [_ra['N'], _ra['CA'], _ra['C']]))
            _sc_slide_proto = _slide_specs          # FAPE scored == coords selection
            if not _slide_specs:
                raise ValueError("sliding motif: no usable atoms in the motif PDB")

            # Binder atom-index map [length, n_names] over the UNION of needed
            # names (coords + N,CA,C frames + FAPE sidechain): binder position +
            # atom name -> global atom index (-1 if absent). Stable across MCMC.
            _slide_name_list = sorted(_slide_names)
            _slide_name_col = {nm: c for c, nm in enumerate(_slide_name_list)}
            binder_slide_atom_idx = torch.full((length, len(_slide_name_list)), -1,
                                               dtype=torch.long, device=device)
            for _p in range(length):
                for _nm, _c in _slide_name_col.items():
                    _idx = _pred_atom_idx(_p, _nm)
                    if _idx is not None:
                        binder_slide_atom_idx[_p, _c] = _idx

            def _on_binder(nm):
                _c = _slide_name_col.get(nm)
                return _c is not None and bool((binder_slide_atom_idx[:, _c] >= 0).all())

            # Coords specs: keep atoms present at every binder position (UNK is uniform).
            _kept = []
            for (_ri, _nm, _xyz) in _slide_specs:
                if _on_binder(_nm):
                    _kept.append((_ri, _slide_name_col[_nm], _xyz))
                else:
                    print(f"[motif/slide] WARNING: atom '{_nm}' not on the UNK "
                          f"binder; dropped from the coords loss")
            if not _kept:
                raise ValueError("sliding motif: no selected atoms present on the binder")
            slide_atom_res_idx = torch.as_tensor([s[0] for s in _kept],
                                                  dtype=torch.long, device=device)
            slide_atom_within_idx = torch.as_tensor([s[1] for s in _kept],
                                                     dtype=torch.long, device=device)
            slide_atom_ref_xyz = torch.as_tensor(np.stack([s[2] for s in _kept]),
                                                  dtype=torch.float32, device=device)

            # Sliding FAPE protos (DYNAMIC pred idx; filtered to binder presence).
            # Frames need N,CA,C present on the binder (uniform for UNK).
            if all(_on_binder(_n) for _n in ('N', 'CA', 'C')):
                _fape_fr_slide_cols = [_slide_name_col[_n] for _n in ('N', 'CA', 'C')]
                for (_ri, _ref3) in _fr_slide_proto:
                    _fape_fr_slide_slot.append(_ri)
                    _fape_fr_ref_slide.append(_ref3)
            for (_ri, _nm, _xyz) in _sc_slide_proto:
                if _on_binder(_nm):
                    _fape_sc_slide_slot.append(_ri)
                    _fape_sc_slide_col.append(_slide_name_col[_nm])
                    _fape_sc_ref_slide.append(_xyz)

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

        # res_type column for each one-letter AA, matching the alphabet used
        # throughout boltz_hallucination: list('XXARNDCQEGHILKMFPSTWYV-').
        _ALPHABET = list('XXARNDCQEGHILKMFPSTWYV-')
        _AA_TO_COL = {aa: i for i, aa in enumerate(_ALPHABET) if i >= 2 and aa not in ('X', '-')}

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
        # The Kabsch fit (motif_coords_loss) is over these selected protein
        # atoms; it needs >=3 non-collinear points for a defined rotation.
        if _K_total < 3:
            raise ValueError(
                f"motif_coords_loss Kabsch fit needs >=3 selected protein "
                f"atoms; got {_K_total}. Widen the --motif_residues atom "
                f"selection (ligand atoms do not count toward the fit).")
        with torch.no_grad():
            _ref_centered = motif_bb_ref_xyz - motif_bb_ref_xyz.mean(0, keepdim=True)
            if int(torch.linalg.matrix_rank(_ref_centered)) < 2:
                raise ValueError(
                    "motif_coords_loss: selected protein atoms are collinear "
                    "(rank<2); the Kabsch rotation is undefined. Select atoms "
                    "that span more than a line.")
        # Static fixed backbone idx (used by the dynamic rebuilder + MCMC energy)
        motif_bb_pred_idx_fixed_static = (
            torch.as_tensor(fixed_bb_pred_list, dtype=torch.long, device=device)
            if fixed_bb_pred_list else torch.zeros(0, dtype=torch.long, device=device))

        # ----- Mutable dynamic tensors (refreshed per MCMC step) -----
        motif_token_idx = torch.zeros(M, dtype=torch.long, device=device)
        motif_bb_pred_idx = torch.zeros(_K_total, dtype=torch.long, device=device)
        motif_row_mask = torch.zeros(1, N, 1, device=device, dtype=batch['res_type'].dtype)
        motif_res_type_full = torch.zeros(1, N, V, device=device, dtype=batch['res_type'].dtype)
        # Holder for the sliding-FAPE rebuild closure (set in the FAPE assembly
        # below, after ligand parsing). Called after each MCMC step so the FAPE
        # frame/scored pred indices track slide_state. None when FAPE is inactive.
        _slide_fape_rebuild = [None]

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
                    motif_bb_pred_idx[_K_fix:] = binder_slide_atom_idx[
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

        # ---- Candidate-placement scoring for the placement search ----
        # The determinant (--motif_slide_loss) is computed for a *candidate*
        # slide_state with the SAME motif loss kernel as the matching gradient
        # term, over the WHOLE motif (fixed anchors ++ sliding-at-candidate), so
        # sliding islands are placed relative to the fixed motif -- exactly what
        # the old coords-only energy did via the joint Kabsch.
        def _candidate_bb_pred_idx(state):
            positions_t = torch.as_tensor(
                _placement_to_positions(state, slide_islands),
                dtype=torch.long, device=device)
            slide_bb_pred = binder_slide_atom_idx[
                positions_t[slide_atom_res_idx], slide_atom_within_idx]
            if motif_fixed_active:
                return torch.cat([motif_bb_pred_idx_fixed_static, slide_bb_pred], dim=0)
            return slide_bb_pred

        def _candidate_token_idx(state):
            tok = motif_token_idx.clone()                # fixed prefix is constant
            positions_t = torch.as_tensor(
                _placement_to_positions(state, slide_islands),
                dtype=torch.long, device=device)
            tok[M_fix:] = binder_tok[positions_t]
            return tok

        def _candidate_fape_idx(state):
            fr = motif_frame_pred_idx.clone()
            sc = motif_fape_sc_pred_idx.clone()
            positions_t = torch.as_tensor(
                _placement_to_positions(state, slide_islands),
                dtype=torch.long, device=device)
            if _F_slide and _fr_slide_cols_t is not None:
                fr[_F_fix:] = binder_slide_atom_idx[
                    positions_t[_fr_slide_slot_t]][:, _fr_slide_cols_t]
            if _S_slide:
                sc[_sc_const_n:] = binder_slide_atom_idx[
                    positions_t[_sc_slide_slot_t], _sc_slide_col_t]
            return fr, sc

        def _placement_energy(state, dict_out, pdist, mid_pts):
            """Determinant score (lower = better fit) for a candidate placement.
            Returns None when --motif_slide_loss needs the diffusion coords but
            they are unavailable this epoch (fast mode)."""
            with torch.no_grad():
                if motif_slide_loss == 'distogram':
                    return float(get_motif_distogram_loss(
                        pdist, _candidate_token_idx(state), motif_target_dist,
                        mid_pts, method=motif_distogram_loss_type))
                coords = dict_out.get('sample_atom_coords')
                if coords is None:
                    return None
                if motif_slide_loss == 'rmsd':
                    return float(get_motif_coords_loss(
                        coords, _candidate_bb_pred_idx(state), motif_bb_ref_xyz))
                if motif_slide_loss == 'fape':
                    if not motif_fape_active:
                        return None
                    fr, sc = _candidate_fape_idx(state)
                    return float(get_motif_fape_loss(
                        coords, fr, sc, motif_fape_loc_ref))
            return None

        def _mcmc_step(dict_out, pdist, mid_pts):
            """One epoch of Metropolis + simulated-annealing moves on slide_state
            using the --motif_slide_loss determinant as the energy. No-op (and no
            call-count increment) when the determinant is unavailable this epoch,
            so the T schedule advances only on epochs that actually search."""
            E_cur = _placement_energy(slide_state, dict_out, pdist, mid_pts)
            if E_cur is None:
                return
            t = min(slide_runtime['call_count']
                    / float(slide_runtime['total_calls']), 1.0)
            T = motif_slide_T_init * (motif_slide_T_final
                                       / motif_slide_T_init) ** t
            for _ in range(int(motif_slide_steps_per_epoch)):
                slide_runtime['attempts'] += 1
                prop = propose_mcmc_move(slide_state, slide_island_lengths,
                                          length, fixed_positions,
                                          motif_slide_shift_max,
                                          motif_slide_swap_prob,
                                          slide_runtime['rng'])
                if prop is None:
                    continue
                E_prop = _placement_energy(prop, dict_out, pdist, mid_pts)
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
            if _slide_fape_rebuild[0] is not None:
                _slide_fape_rebuild[0]()

        # ---- Exhaustive placement (motif_enum), distogram determinant ----
        # Built lazily once: contigs = fixed anchors (single pinned start each) ++
        # sliding islands (free, minus fixed-occupied positions). Holder stores
        # (problem, sliding_contig_indices, token_map).
        _slide_mp_holder = [None]

        def _build_slide_mp():
            import motif_enum as _me
            refs, offsets, valid_starts, sliding_idx = [], [], [], []
            # Fixed motif residues anchor the sliding placement: one pinned
            # single-residue contig each, at its exact binder position (robust to
            # the --motif_binder_positions layout). Their pairwise geometry to the
            # sliding islands is what makes the search place relative to the fixed
            # motif, mirroring the MCMC joint-Kabsch energy.
            if motif_fixed_active:
                for _k, _pos in enumerate(binder_pos):
                    refs.append(torch.as_tensor(
                        fixed_ref_xyz[_k], dtype=torch.float32,
                        device=device).reshape(1, 1, 3))
                    offsets.append([0])
                    valid_starts.append([int(_pos)])
            _ri = 0
            for _isl in slide_islands:
                _n = len(_isl)
                _off = [r['intra_offset'] for r in _isl]
                _span = max(_off) + 1
                refs.append(torch.as_tensor(
                    slide_ref_xyz[_ri:_ri + _n], dtype=torch.float32,
                    device=device).reshape(_n, 1, 3))
                offsets.append(_off)
                valid_starts.append([
                    s for s in range(length - _span + 1)
                    if all((s + o) not in fixed_positions for o in range(_span))])
                sliding_idx.append(len(refs) - 1)
                _ri += _n
            problem = _me.MotifPlacementProblem(
                refs, length, contig_offsets=offsets, contig_valid_starts=valid_starts)
            return problem, sliding_idx, binder_tok

        # ---- Coords variant (rmsd/fape determinant): identical contig layout
        # to _build_slide_mp, but the refs carry the backbone atom set [n,A,3]
        # and we also resolve a [length, A] gather index into sample_atom_coords.
        # Default-backbone selection only -- the engine's coords scorers need a
        # uniform per-residue atom grid (A = the subset of N,CA,C,CB present in
        # every motif reference residue AND on the UNK binder).
        _slide_mp_coords_holder = [None]

        def _build_slide_mp_coords():
            import motif_enum as _me
            _sels = (list(motif_atom_sel) if motif_fixed_active else []) \
                + list(slide_atom_sel)
            if any(s is not None for s in _sels):
                raise ValueError(
                    "--motif_slide_method exhaustive with --motif_slide_loss "
                    "rmsd/fape supports only the default backbone selection "
                    "(no per-residue :ATOMS); use --motif_slide_method mcmc for "
                    "custom atom selections")
            _ref_dicts = ([e['atoms'] for e in _ref_res_atoms]
                          if motif_fixed_active else [])
            _ref_dicts += [e['atoms'] for e in _slide_ref_res_atoms]
            names = [nm for nm in _BB
                     if all(nm in d for d in _ref_dicts)
                     and nm in _slide_name_col
                     and bool((binder_slide_atom_idx[:, _slide_name_col[nm]]
                               >= 0).all())]
            if len(names) < 3:
                raise ValueError(
                    f"exhaustive rmsd/fape needs >=3 uniform backbone atoms "
                    f"across all motif residues + the binder; resolved {names}. "
                    f"Use --motif_slide_method mcmc.")
            cols = torch.as_tensor([_slide_name_col[nm] for nm in names],
                                   dtype=torch.long, device=device)
            A = len(names)
            pred_idx = binder_slide_atom_idx[:, cols]                # [length, A]
            refs, offsets, valid_starts, sliding_idx = [], [], [], []
            if motif_fixed_active:
                for _k, _pos in enumerate(binder_pos):
                    _d = _ref_res_atoms[_k]['atoms']
                    refs.append(torch.as_tensor(
                        np.stack([_d[nm] for nm in names]),
                        dtype=torch.float32, device=device).reshape(1, A, 3))
                    offsets.append([0])
                    valid_starts.append([int(_pos)])
            _ri = 0
            for _isl in slide_islands:
                _n = len(_isl)
                _off = [r['intra_offset'] for r in _isl]
                _span = max(_off) + 1
                _isl_ref = np.stack([
                    np.stack([_slide_ref_res_atoms[_ri + _j]['atoms'][nm]
                              for nm in names])
                    for _j in range(_n)])                            # [n, A, 3]
                refs.append(torch.as_tensor(
                    _isl_ref, dtype=torch.float32, device=device))
                offsets.append(_off)
                valid_starts.append([
                    s for s in range(length - _span + 1)
                    if all((s + o) not in fixed_positions for o in range(_span))])
                sliding_idx.append(len(refs) - 1)
                _ri += _n
            problem = _me.MotifPlacementProblem(
                refs, length, contig_offsets=offsets, contig_valid_starts=valid_starts)
            return problem, sliding_idx, pred_idx, names

        def _apply_slide_placement(placement, sliding_idx, beta):
            """Write an enumeration result into slide_state and refresh the
            dynamic motif tensors (shared by both exhaustive determinants)."""
            new_state = [int(placement[ci]) for ci in sliding_idx]
            slide_runtime['call_count'] += 1
            slide_runtime['current_T'] = beta
            if new_state != list(slide_state):
                for i in range(len(slide_state)):
                    slide_state[i] = new_state[i]
                _rebuild_motif_dynamic()
                if _slide_fape_rebuild[0] is not None:
                    _slide_fape_rebuild[0]()

        def _exhaustive_step(dict_out, pdist, mid_pts):
            """Deterministic placement by exhaustive triplet enumeration
            (motif_enum); writes the best joint placement to slide_state. beta
            anneals 2 -> 20 over the run. The distogram determinant runs every
            epoch; rmsd/fape need the diffusion coords (full-mode epochs only)
            and are a no-op otherwise."""
            import motif_enum as _me
            t = min(slide_runtime['call_count']
                    / float(slide_runtime['total_calls']), 1.0)
            beta = 2.0 + (20.0 - 2.0) * t
            if motif_slide_loss == 'distogram':
                if _slide_mp_holder[0] is None:
                    _slide_mp_holder[0] = _build_slide_mp()
                problem, sliding_idx, token_map = _slide_mp_holder[0]
                with torch.no_grad():
                    scores = _me.distogram_ce_pair_scores(
                        pdist.detach(), problem, token_map=token_map)
                    placement = _me.best_placement(scores, problem, beta=beta)
                _apply_slide_placement(placement, sliding_idx, beta)
                return
            # rmsd / fape determinant: coords needed -> full-mode epochs only.
            coords = dict_out.get('sample_atom_coords')
            if coords is None:
                return
            if _slide_mp_coords_holder[0] is None:
                _slide_mp_coords_holder[0] = _build_slide_mp_coords()
            problem, sliding_idx, pred_idx, _names = _slide_mp_coords_holder[0]
            pred = coords[0][pred_idx]                                # [length, A, 3]
            score_fn = (_me.fape_pair_scores if motif_slide_loss == 'fape'
                        else _me.kabsch_rmsd_pair_scores)
            with torch.no_grad():
                scores = score_fn(pred, problem)
                placement = _me.best_placement(scores, problem, beta=beta)
            _apply_slide_placement(placement, sliding_idx, beta)

        def _run_slide_placement(dict_out, pdist, mid_pts):
            """Per-epoch placement update (called before the motif gradient losses
            so they see the new placement). Dispatches on --motif_slide_method."""
            if not motif_slide_active:
                return
            if motif_slide_method == 'mcmc':
                _mcmc_step(dict_out, pdist, mid_pts)
            else:
                _exhaustive_step(dict_out, pdist, mid_pts)

        # Initialize dynamic tensors from the initial slide_state.
        _rebuild_motif_dynamic()

        # Front-load the coords-exhaustive validation (default-backbone, uniform
        # atom grid) so a bad config errors at setup, not on the first full-mode
        # epoch. The build needs no coords (refs + indices only), so it is safe
        # to run here and cache.
        if (motif_slide_active and motif_slide_method == 'exhaustive'
                and motif_slide_loss in ('rmsd', 'fape')):
            _slide_mp_coords_holder[0] = _build_slide_mp_coords()

        # Optional ligand carry-along (--motif_ligand_residues). The Kabsch
        # transform is fit from the motif backbone above (joint over fixed +
        # sliding); the ligand atoms are simply *carried* by that transform.
        motif_lig_active = bool(motif_ligand_residues)
        motif_lig_pred_idx = motif_lig_ref_xyz = None
        if motif_lig_active:
            _lig_specs_full = parse_motif_ligand_spec(motif_ligand_residues, with_atoms=True)
            if not _lig_specs_full:
                raise ValueError("--motif_ligand_residues set but parsed to no entries")
            _lig_specs = [(c, r) for c, r, _ in _lig_specs_full]
            _lig_atom_sel = [a for _, _, a in _lig_specs_full]
            _ref_lig = extract_motif_ligand_atoms(motif_pdb, _lig_specs)
            _lig_pred, _lig_ref, _lig_summary = [], [], []
            for _ei, _ent in enumerate(_ref_lig):
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
                _use_names = _resolve_sel(
                    _lig_atom_sel[_ei], _ent['atoms'], list(_ent['atoms'].keys()),
                    label=f"--motif_ligand_residues {_ent['chain']}{_ent['resnum']}")
                for _nm in _use_names:
                    try:
                        _pi = _find_named_atom(structure,
                                               int(_match_res['atom_idx']),
                                               int(_match_res['atom_num']),
                                               _nm, str(_match_chain['name']), None)
                    except ValueError:
                        continue
                    _lig_pred.append(_pi)
                    _lig_ref.append(_ent['atoms'][_nm])
                    _n_match += 1
                _lig_summary.append(
                    f"{_ent['chain']}{_ent['resnum']}({_ent['resname']})->"
                    f"{str(_match_chain['name'])}:"
                    f"{_n_match}/{len(_use_names)}atoms")
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

        # ----- FAPE assembly (fixed + sliding + ligand) -----
        # Canonical order: frames = [fixed frames ++ sliding frames]; scored
        # atoms = [fixed sidechain ++ ligand ++ sliding sidechain]. loc_ref (the
        # reference, motif PDB geometry) is CONSTANT; only the SLIDING pred
        # indices move with slide_state, rebuilt by `_fill_slide_fape` (init +
        # after each MCMC step). The SAME flags that feed motif_coords_loss
        # (--motif_residues / --motif_unindex_residues / --motif_ligand_residues)
        # thus all feed motif_fape_loss; ligand atoms are scored as points in
        # the motif frames (frame-aligned ligand placement).
        _F_fix, _F_slide = len(_fape_fr_pred), len(_fape_fr_slide_slot)
        _S_fix = len(_fape_sc_pred)
        _n_lig_fape = (int(motif_lig_pred_idx.shape[0])
                       if (motif_lig_active and motif_lig_pred_idx is not None) else 0)
        _S_slide = len(_fape_sc_slide_slot)
        motif_fape_active = ((_F_fix + _F_slide) > 0
                             and (_S_fix + _n_lig_fape + _S_slide) > 0)
        if motif_fape_active:
            _fr_ref_all = list(_fape_fr_ref) + list(_fape_fr_ref_slide)          # [F,3,3]
            _sc_ref_all = list(_fape_sc_ref)                                     # fixed sc
            if _n_lig_fape:
                _sc_ref_all = _sc_ref_all + [r for r in motif_lig_ref_xyz.cpu().numpy()]
            _sc_ref_all = _sc_ref_all + list(_fape_sc_ref_slide)                 # [S,3]
            _fr_ref_t = torch.as_tensor(np.asarray(_fr_ref_all), dtype=torch.float32, device=device)
            _sc_ref_t = torch.as_tensor(np.asarray(_sc_ref_all), dtype=torch.float32, device=device)
            with torch.no_grad():
                _Rr, _tr = _rigid_frames(_fr_ref_t[:, 0], _fr_ref_t[:, 1], _fr_ref_t[:, 2])
                _diff = _sc_ref_t[None, :, :] - _tr[:, None, :]
                motif_fape_loc_ref = torch.einsum('fij,fsj->fsi',
                                                  _Rr.transpose(-1, -2), _diff)
            # Mutable pred-index tensors: constant prefixes set now, sliding
            # suffixes filled by _fill_slide_fape (init + per MCMC step).
            _F, _S = _F_fix + _F_slide, _S_fix + _n_lig_fape + _S_slide
            motif_frame_pred_idx = torch.zeros(_F, 3, dtype=torch.long, device=device)
            motif_fape_sc_pred_idx = torch.zeros(_S, dtype=torch.long, device=device)
            if _F_fix:
                motif_frame_pred_idx[:_F_fix] = torch.as_tensor(_fape_fr_pred, dtype=torch.long, device=device)
            _sc_const = list(_fape_sc_pred) + (motif_lig_pred_idx.tolist() if _n_lig_fape else [])
            if _sc_const:
                motif_fape_sc_pred_idx[:len(_sc_const)] = torch.as_tensor(_sc_const, dtype=torch.long, device=device)
            _sc_const_n = len(_sc_const)
            _fr_slide_slot_t = (torch.as_tensor(_fape_fr_slide_slot, dtype=torch.long, device=device)
                                if _F_slide else None)
            _fr_slide_cols_t = (torch.as_tensor(_fape_fr_slide_cols, dtype=torch.long, device=device)
                                if (_F_slide and _fape_fr_slide_cols is not None) else None)
            _sc_slide_slot_t = (torch.as_tensor(_fape_sc_slide_slot, dtype=torch.long, device=device)
                                if _S_slide else None)
            _sc_slide_col_t = (torch.as_tensor(_fape_sc_slide_col, dtype=torch.long, device=device)
                               if _S_slide else None)

            def _fill_slide_fape():
                """Rebuild the SLIDING portion of the FAPE pred-index tensors from
                the current slide_state (constant prefixes untouched; loc_ref is
                constant). No-op without sliding."""
                if not motif_slide_active:
                    return
                with torch.no_grad():
                    positions_t = torch.as_tensor(
                        _placement_to_positions(slide_state, slide_islands),
                        dtype=torch.long, device=device)
                    if _F_slide and _fr_slide_cols_t is not None:
                        motif_frame_pred_idx[_F_fix:] = \
                            binder_slide_atom_idx[positions_t[_fr_slide_slot_t]][:, _fr_slide_cols_t]
                    if _S_slide:
                        motif_fape_sc_pred_idx[_sc_const_n:] = \
                            binder_slide_atom_idx[positions_t[_sc_slide_slot_t], _sc_slide_col_t]

            _slide_fape_rebuild[0] = _fill_slide_fape
            _fill_slide_fape()   # initial fill from the initial slide_state

        # ----- Startup summary -----
        if motif_fixed_active:
            _motif_chains = sorted({c for c, _ in m_chain_residues})
            _any_sel = any(a is not None for a in motif_atom_sel)
            print(f"[motif/fixed] {M_fix} residues from {motif_pdb} "
                  f"chain(s) {','.join(_motif_chains)} "
                  f"({','.join(f'{c}{r}' for c, r in m_chain_residues)}) "
                  f"-> binder positions {[p+1 for p in binder_pos]}; "
                  f"seq={fixed_motif_seq}; fix_seq={bool(fix_motif_seq)}; "
                  f"atom_sel={'per-residue' if _any_sel else 'default(N,CA,C,CB / sidechain)'}; "
                  f"Kabsch-fit atoms={_K_total}")
        if motif_fape_active:
            _n_fr = int(motif_frame_pred_idx.shape[0])
            _n_sc = int(motif_fape_sc_pred_idx.shape[0])
            print(f"[motif] FAPE over {_n_fr} frame(s) ({_F_fix} fixed + {_F_slide} "
                  f"sliding) x {_n_sc} scored atom(s) ({_S_fix} fixed sc + "
                  f"{_n_lig_fape} ligand + {_S_slide} sliding sc)")
        if motif_slide_active:
            _slide_chains = sorted({r['chain'] for isl in slide_islands for r in isl})
            print(f"[motif/slide] {len(slide_islands)} island(s), {M_slide} "
                  f"residues from chain(s) {','.join(_slide_chains)}; "
                  f"island_lengths={slide_island_lengths}; "
                  f"init state={slide_state}; method={motif_slide_method}; "
                  f"determinant={motif_slide_loss}")
            if motif_slide_method == 'mcmc':
                print(f"[motif/slide] MCMC T schedule "
                      f"{motif_slide_T_init} -> {motif_slide_T_final} over "
                      f"{slide_runtime['total_calls']} calls; "
                      f"steps/call={motif_slide_steps_per_epoch}; "
                      f"swap_prob={motif_slide_swap_prob}, "
                      f"shift_max={motif_slide_shift_max}; "
                      + ("" if motif_slide_loss == 'distogram' else
                         "(coords determinant -> placement updates on full-mode "
                         "epochs only); ")
                      + f"fix_seq={bool(fix_motif_seq)} (pin moves with state)")
            else:
                print(f"[motif/slide] exhaustive triplet enumeration; "
                      f"determinant={motif_slide_loss}; beta 2 -> 20 over "
                      f"{slide_runtime['total_calls']} calls; "
                      + ("" if motif_slide_loss == 'distogram' else
                         "(coords determinant -> placement updates on full-mode "
                         "epochs only); ")
                      + f"fix_seq={bool(fix_motif_seq)} (pin moves with state)")

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

    # ---- Explicit three-atom angle restraints (--atom_angles) ---------------
    # Coords-only (no distogram variant); resolved once here like --atom_pairs.
    atom_angles_active = bool(atom_angles)
    atom_angle_specs = None
    if atom_angles_active:
        atom_angle_specs = resolve_atom_angles(atom_angles, batch, structure)
        print(f"[atom_angles] {len(atom_angle_specs)} angle restraint(s):")
        for _s in atom_angle_specs:
            print(f"  {_s['label']}  atom({_s['atom_a']},{_s['atom_b']},{_s['atom_c']})")

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
        # i_pae_w, tplddt_w, target_masks, i_con_target_masks, i_pae_target_masks,
        # inter_target_con_pair_masks, inter_target_pae_pair_masks,
        # target_chain_ids) are captured from the enclosing boltz_hallucination
        # scope; get_model_loss reads them directly so the long arg list doesn't
        # need to thread per-target lists.
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
                'disconnect_feats_structure': disconnect_feats_structure,
                'disconnect_pairformer_structure': disconnect_pairformer_structure,
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
                # Epitope targeting: the target SIDE of the contact term uses the
                # i_con residue-restricted mask (full chain if this target was not
                # named in --i_con_target_residues). It is mask_1d in the standard
                # (per-target-residue) branch and mask_1b in the per-binder-pos
                # branch, so a single substitution covers both.
                t_mask = i_con_target_masks[ti]
                if optimize_contact_per_binder_pos_pt[ti]:
                    # Per-binder-position contact objective (count from the binder
                    # side: mask_1d=chain_mask). num_pos selects how many binder
                    # positions must contact this target. Faithful to the original
                    # boltz_hallucination: a FINITE num_pos (the annealed schedule)
                    # is used ONLY when increasing_contact_over_itr is on; otherwise
                    # num_pos stays inf so every binder position counts. Note
                    # increasing_contact_over_itr and num_optimizing_binder_pos are
                    # GLOBAL (not per-target) -- only the optimize flag is per-target.
                    if increasing_contact_over_itr:
                        n_pos = 0 if pre_run else num_optimizing_binder_pos
                    else:
                        n_pos = float("inf")
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

            # Auxiliary INTER-TARGET contact loss: optimize contacts BETWEEN two
            # target chains. Each pair reuses the exact get_con_loss term as the
            # binder<->target i_con (the two target-residue selections are
            # mask_1d / mask_1b); pair losses are summed. Like con_loss/i_con it
            # is a distogram loss, so it is computed in pre_run and full mode. The
            # overall weight is loss_scales['inter_target_con_loss'].
            if inter_target_con_pair_masks:
                it_con_loss = pdist.new_zeros(())
                for _mA, _mB, _desc in inter_target_con_pair_masks:
                    it_con_loss = it_con_loss + get_con_loss(
                        pdist, mid_pts,
                        num=inter_target_num_contacts, seqsep=0,
                        cutoff=inter_target_cutoff, binary=False,
                        mask_1d=_mA, mask_1b=_mB)
                losses['inter_target_con_loss'] = it_con_loss

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
                    # Epitope targeting: restrict the target rows entering the
                    # inter-chain PAE to the i_pae selection (full chain if this
                    # target was not named in --i_pae_target_residues). pae is
                    # symmetrized just above, so restricting mask_1d alone already
                    # covers both directions of the selected-target<->binder block.
                    li = get_pae_loss(pae, mask_1d=i_pae_target_masks[ti], mask_1b=chain_mask)
                    i_pae_loss = i_pae_loss + w * li
                    i_pae_any = True

                losses.update({
                    'plddt_loss': plddt_loss,
                    'pae_loss': pae_loss,
                    'rg_loss': rg_loss,
                })
                if i_pae_any:
                    losses['i_pae_loss'] = i_pae_loss

                # Auxiliary INTER-TARGET interface-PAE loss: same get_pae_loss
                # term as the binder<->target i_pae, between two target-residue
                # selections (mask_1d / mask_1b). pae is symmetrized above, so a
                # single call per pair captures both directions of the
                # sideA<->sideB block; pair losses are summed. Full-mode only
                # (pae is unavailable in pre_run / distogram_only), matching
                # i_pae. Weight: loss_scales['inter_target_pae_loss'].
                if inter_target_pae_pair_masks:
                    it_pae_loss = pdist.new_zeros(())
                    for _mA, _mB, _desc in inter_target_pae_pair_masks:
                        it_pae_loss = it_pae_loss + get_pae_loss(
                            pae, mask_1d=_mA, mask_1b=_mB)
                    losses['inter_target_pae_loss'] = it_pae_loss

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
            #   - `motif_distogram_loss`: trunk-distogram restraint over motif
            #     token pairs (fast-mode safe); one-hot categorical CE by default
            #     (--motif_distogram_loss_type), or bin-center MSE.
            #   - `motif_coords_loss`:    Kabsch (backbone-only) + RMSD over
            #     backbone (and ligand if --motif_ligand_residues set), full-
            #     mode only; gradient needs --attach_coords True like rg_loss.
            #   - `motif_fape_loss`:      AF-style frame-aligned sidechain
            #     error (full-mode only; same gating).
            if motif_active:
                # Update the sliding-island placement (MCMC or exhaustive) BEFORE
                # the supervised terms, so they (and the token/atom index tensors)
                # see the freshly-chosen placement. No-op for fixed-only motifs or
                # when the determinant is unavailable this epoch.
                if motif_slide_active:
                    _run_slide_placement(dict_out, pdist, mid_pts)
                losses['motif_distogram_loss'] = get_motif_distogram_loss(
                    pdist, motif_token_idx, motif_target_dist, mid_pts,
                    method=motif_distogram_loss_type)
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

            # Explicit three-atom angle restraints (--atom_angles). Coords-only
            # (an angle is undefined on the trunk distogram): needs the sampler
            # (full mode) and carries a gradient only with --attach_coords True.
            if (atom_angles_active and not pre_run and not distogram_only
                    and 'sample_atom_coords' in dict_out):
                aa_coords_loss, aa_coords_stats = get_atom_angle_coords_loss(
                    dict_out['sample_atom_coords'], atom_angle_specs,
                    return_stats=True)
                losses['atom_angle_coords_loss'] = aa_coords_loss
                for _st in aa_coords_stats:
                    _lab = _st['label']
                    loss_component_history.setdefault(f"atom_angle|cang|{_lab}", []).append(_st['angle'])
                    loss_component_history.setdefault(f"atom_angle|closs|{_lab}", []).append(_st['loss'])

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
                    'inter_target_con_loss': 1.0,
                    'inter_target_pae_loss': 0.1,
                    'rg_loss': 0.3,
                    'target_plddt_loss': 0.1,
                    'motif_distogram_loss': 0.0,
                    'motif_coords_loss': 0.0,
                    'motif_fape_loss': 0.0,
                    'atom_pair_distogram_loss': 1.0,
                    'atom_pair_coords_loss': 1.0,
                    'atom_angle_coords_loss': 1.0,
                }

            # Defensive: these keys may be present in `losses` while a
            # caller-supplied loss_scales (configs/CLI) predates the feature.
            for _k in ('target_plddt_loss', 'motif_distogram_loss', 'motif_coords_loss',
                       'motif_fape_loss',
                       'atom_pair_distogram_loss', 'atom_pair_coords_loss',
                       'atom_angle_coords_loss', 'com_loss',
                       'inter_target_con_loss', 'inter_target_pae_loss'):
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
                # Record the placement the pin was just applied at. The MCMC/
                # exhaustive move happens later in get_model_loss, so this is the
                # state the FINAL sequence is pinned to (the mapping uses it).
                if motif_slide_active:
                    _last_pin_slide_state[0] = list(slide_state)

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

        # Snapshot the last-epoch argmax sequence for this stage; outer block
        # re-folds at stock sampler and writes {stage}_last.pdb. Only stages
        # that actually ran (iters>0) are captured -- an iters=0 no-op would
        # otherwise alias the previous stage's sequence.
        if iters > 0:
            stage_lasts[stage_name] = ''.join([
                alphabet[k] for k in torch.argmax(
                    batch['res_type'][batch['entity_id']==chain_to_number[binder_chain],:],
                    dim=-1).detach().cpu().numpy()
            ])

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

    # visualize_results(plots)

    if pre_run:
        # Warm-up returns the logits only; no final fold runs here, so no
        # predict_args/sampler change is needed.
        best_logits = batch['res_type_logits']
        best_seq = ''.join([alphabet[i] for i in torch.argmax(batch['res_type'][batch['entity_id']==chain_to_number[binder_chain],:], dim=-1).detach().cpu().numpy()])
        data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = best_seq
        return batch['res_type'].detach().cpu().numpy(), plots, loss_history, distogram_history, sequence_history, traj_coords_list, traj_plddt_list

    # Motif residue -> FINAL binder position mapping (the design loop is done, so
    # slide_state is final). Keys: original motif residue (chain+resnum in the
    # motif PDB); values: binder chain + 1-indexed residue. Covers --motif_residues
    # (fixed) and --motif_unindex_residues (sliding) alike. The caller writes JSON.
    if motif_active and motif_mapping_out is not None:
        if motif_fixed_active:
            for _k, (_mc, _mr) in enumerate(m_chain_residues):
                motif_mapping_out[f"{_mc}{_mr}"] = f"{binder_chain}{binder_pos[_k] + 1}"
        if motif_slide_active:
            # Use the slide_state the sequence was actually pinned at (the in-loop
            # pin precedes the final MCMC/exhaustive move), so the mapping matches
            # the folded binder. Falls back to the current state when fix_motif_seq
            # is off (no pin -> the geometry placement is the only truth).
            _map_state = (_last_pin_slide_state[0]
                          if _last_pin_slide_state[0] is not None else slide_state)
            _final_pos = _placement_to_positions(_map_state, slide_islands)
            for _k, (_mc, _mr) in enumerate(slide_chain_residues):
                motif_mapping_out[f"{_mc}{_mr}"] = f"{binder_chain}{_final_pos[_k] + 1}"

    boltz_model.eval()

    if best_batch is None:
        if first_step_best_batch is not None:
            best_batch = first_step_best_batch
        else:
            best_batch = batch  

    # Restore stock Boltz sampler defaults for everything past the design loop.
    # The semi-greedy MCMC evals AND the final folds all run this full 200-step
    # stochastic sampler (step_scale 1.638, gamma_0 0.605, no Heun, coords
    # detached) -- matching the original boltz_hallucination, where semi-greedy
    # and the final prediction share one predict_args. Per-run design tuning
    # (deterministic_sampler / step_scale / sampling_steps / attach_coords / ...)
    # never reaches this point.
    final_sampler = SamplerConfig(write_full_pae=True)
    final_predict_args = apply_sampler_config(boltz_model, final_sampler)

    # Binder positions occupied by the pinned motif (final, pin-consistent state).
    # Semi-greedy accepts mutations on iPTM alone -- it never sees the motif loss
    # -- so without this guard it could silently mutate a pinned motif residue.
    _motif_binder_pos_arr = None
    if motif_active and fix_motif_seq:
        _mbp = list(binder_pos) if motif_fixed_active else []
        if motif_slide_active:
            _mstate = (_last_pin_slide_state[0]
                       if _last_pin_slide_state[0] is not None else slide_state)
            _mbp += list(_placement_to_positions(_mstate, slide_islands))
        _motif_binder_pos_arr = np.array(sorted(set(_mbp)), dtype=int)

    def _mutate(sequence, best_logits, i_prob):
        mutated_sequence = list(sequence) # Create a copy of the input tensor
        i_prob = np.array(i_prob, dtype=float)
        if _motif_binder_pos_arr is not None and _motif_binder_pos_arr.size:
            i_prob[_motif_binder_pos_arr] = 0.0   # never mutate pinned motif residues
            if i_prob.sum() <= 0:                 # degenerate: spread over the rest
                i_prob = np.ones(length, dtype=float)
                i_prob[_motif_binder_pos_arr] = 0.0
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

    def _dump_pdb_to_intermediate(out_name, coords, plddt, structure_obj):
        """Write coords + plddt to intermediate_dir/<out_name>, restoring the
        structure object's atoms/is_present afterwards. No-op if
        intermediate_dir is None."""
        if intermediate_dir is None:
            return
        try:
            os.makedirs(intermediate_dir, exist_ok=True)
            n_atoms = structure_obj.atoms['coords'].shape[0]
            coords_np = coords[:n_atoms, :]
            saved_coords = structure_obj.atoms['coords'].copy()
            saved_present = structure_obj.atoms['is_present'].copy()
            try:
                structure_obj.atoms['coords'] = coords_np
                structure_obj.atoms['is_present'] = True
                with open(os.path.join(intermediate_dir, out_name), 'w') as _f:
                    _f.write(to_pdb(structure_obj, plddts=plddt))
            finally:
                structure_obj.atoms['coords'] = saved_coords
                structure_obj.atoms['is_present'] = saved_present
        except Exception as _e:
            print(f"[stage_last] save failed for {out_name}: "
                  f"{type(_e).__name__}: {_e}")

    # Stage-last PDB dump (always-on; independent of save_intermediate_structures).
    # For each stage that actually ran (iters>0), re-fold its last-epoch argmax
    # sequence at the stock final sampler and write {stage}_last.pdb. The current
    # `data` already carries the LAST stage's sequence (best_seq set above), so
    # we temporarily swap in each stage's sequence and restore afterwards so the
    # seed fold + semi-greedy code below see unchanged `data`.
    if intermediate_dir is not None and stage_lasts:
        _saved_data_seq = data['sequences'][chain_to_number[binder_chain]]['protein']['sequence']
        try:
            for _stage_name, _stage_seq in stage_lasts.items():
                try:
                    data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = _stage_seq
                    _stage_target = parse_boltz_schema(name, data, ccd_lib)
                    _stage_batch, _stage_structure = get_batch(
                        _stage_target, msa_max_seqs, length, keep_record=True)
                    _stage_batch = {k: v.unsqueeze(0).to(device) if k != 'record' else v
                                    for k, v in _stage_batch.items()}
                    _stage_out = _run_model(boltz_model, _stage_batch, final_predict_args)
                    # predict_step returns the coords under key 'coords' (it
                    # renames the model's `sample_atom_coords` for the writer).
                    if _stage_out is None or 'coords' not in _stage_out:
                        print(f"[stage_last] fold unavailable for "
                              f"{_stage_name}_last.pdb; skipping")
                        continue
                    _coords = _stage_out['coords'][0].detach().cpu().numpy()
                    _plddt = (_stage_out['plddt'][0].detach().cpu().numpy()
                              if 'plddt' in _stage_out else None)
                    _dump_pdb_to_intermediate(
                        f"{_stage_name}_last.pdb", _coords, _plddt, _stage_structure)
                except Exception as _e:
                    print(f"[stage_last] failure for {_stage_name}: "
                          f"{type(_e).__name__}: {_e}")
        finally:
            data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = _saved_data_seq

    # Standardized final-fold metrics (holo + apo). Computed below from the LAST
    # successful 200-step fold; stays None if the fold failed (early returns).
    final_metrics = None

    def _compute_final_metrics(out_holo, bb_holo, bs_holo, out_apo, bb_apo, bs_apo):
        """Best-effort holo+apo metrics from the final fold; never raises."""
        result = {}
        try:
            if out_holo is not None:
                result['holo'] = compute_final_metrics(
                    boltz_model, out_holo, bb_holo, bs_holo,
                    binder_chain=binder_chain,
                    target_chain_ids=list(target_chain_ids or []), length=length,
                    atom_pairs=atom_pairs, atom_angles=atom_angles,
                    com_loss_weight=com_loss_weight, pdb_path=pdb_path,
                    metric_config=metric_config)
            if out_apo is not None:
                result['apo'] = compute_final_metrics(
                    boltz_model, out_apo, bb_apo, bs_apo,
                    binder_chain=binder_chain, target_chain_ids=[], length=length,
                    atom_pairs=None, atom_angles=None, com_loss_weight=0.0,
                    pdb_path='', metric_config=metric_config)
        except Exception as e:
            print(f"[metrics] compute_final_metrics failed: {type(e).__name__}: {e}")
        return result or None

    # final_predict_args / final_sampler were installed above (stock defaults);
    # all folds from here on -- the seed fold, the semi-greedy per-mutation evals
    # and the post-semi-greedy fold -- use them.
    best_batch, best_batch_apo, best_structure, best_structure_apo = _update_batches(data, data_apo)
    output = _run_model(boltz_model, best_batch, final_predict_args)
    output_apo = _run_model(boltz_model, best_batch_apo, final_predict_args)

    # If the final fold OOM'd, skip semi-greedy and return histories so the
    # caller can still write loss plots and the training animation.
    if output is None or output_apo is None:
        print("[boltz_hallucination] final-fold output unavailable; skipping "
              "semi-greedy and returning collected histories for plotting.")
        return output, output_apo, best_batch, best_batch_apo, best_structure, best_structure_apo, distogram_history, sequence_history, loss_history, con_loss_history, i_con_loss_history, plddt_loss_history, traj_coords_list, traj_plddt_list, structure, loss_component_history, plddt_history, pae_history, final_metrics

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
            output = _run_model(boltz_model, best_batch, final_predict_args)
            if output is None:
                print(f"[semi-greedy] step {step} epoch {t}: fold failed; "
                      f"aborting semi-greedy with histories collected.")
                return output, output_apo, best_batch, best_batch_apo, best_structure, best_structure_apo, distogram_history, sequence_history, loss_history, con_loss_history, i_con_loss_history, plddt_loss_history, traj_coords_list, traj_plddt_list, structure, loss_component_history, plddt_history, pae_history, final_metrics

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

            # semi_greedy_last.pdb: dump the post-semi-greedy fold so it sits
            # alongside soft_last.pdb / temp_last.pdb / hard_last.pdb under
            # intermediate_dir. Only fires when semi-greedy actually ran
            # (semi_greedy_steps>0) and the fold succeeded.
            # predict_step returns the coords under key 'coords' (it renames
            # the model's `sample_atom_coords` for the writer).
            if output is not None and 'coords' in output:
                _coords = output['coords'][0].detach().cpu().numpy()
                _plddt = (output['plddt'][0].detach().cpu().numpy()
                          if 'plddt' in output else None)
                _dump_pdb_to_intermediate(
                    "semi_greedy_last.pdb", _coords, _plddt, best_structure)

    final_metrics = _compute_final_metrics(
        output, best_batch, best_structure,
        output_apo, best_batch_apo, best_structure_apo)

    return output, output_apo, best_batch, best_batch_apo, best_structure, best_structure_apo, distogram_history, sequence_history, loss_history, con_loss_history, i_con_loss_history, plddt_loss_history, traj_coords_list, traj_plddt_list, structure, loss_component_history, plddt_history, pae_history, final_metrics


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
    metric_config=None,
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
            'disconnect_feats_structure': True,
            'disconnect_pairformer_structure': False,
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
            'i_con_target_residues': None,
            'i_pae_target_residues': None,
            'inter_target_con_pairs': [],
            'inter_target_pae_pairs': [],
            'inter_target_num_contacts': 2,
            'inter_target_cutoff': 14.0,
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
    # Motif residue -> final binder position maps (one JSON per design), written
    # whenever a motif is active (--motif_residues and/or --motif_unindex_residues).
    motif_mapping_dir = os.path.join(version_dir, 'motif_mapping')
    # Always create the intermediate_structures root: it holds the always-on
    # {stage}_last.pdb / semi_greedy_last.pdb dumps regardless of
    # save_intermediate_structures (which only gates per-epoch dumps).
    intermediate_structures_root = os.path.join(version_dir, 'intermediate_structures')

    _dirs_to_make = [results_yaml_dir, results_final_dir, results_yaml_dir_apo, results_final_dir_apo, loss_dir, animation_save_dir, fasta_dir, motif_mapping_dir]
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
                    # Receives the motif residue -> final binder position mapping
                    # (fixed + sliding); written to JSON below when non-empty.
                    _motif_map = {}
                    output, output_apo, best_batch, best_batch_apo, best_structure, best_structure_apo ,distogram_history_2, sequence_history_2, loss_history_2, con_loss_history, i_con_loss_history, plddt_loss_history, traj_coords_list_2, traj_plddt_list_2, structure, loss_component_history, plddt_history, pae_history, final_metrics = boltz_hallucination(
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
                        metric_config=metric_config,
                        motif_mapping_out=_motif_map,
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
                        plot_total_loss_history(loss_component_history, loss_history, loss_dir, save_filename=f"{target_binder_input}_total_loss_history_itr{itr + 1}_length{config['length']}.png")

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

                    # Total-loss CSV. total_loss only -- the contact losses are
                    # written to the distogram-category CSV as con_loss/i_con_loss
                    # (no longer aliased here as intra/inter_contact_loss).
                    try:
                        if loss_dir:
                            main_series = {
                                'total_loss': loss_component_history.get('total_loss', loss_history),
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
                    # The atom-pair per-pair distance diagnostics (+ shaded target
                    # window) are overlaid on a twin axis of the Atom-Pair *Loss
                    # subplots via `augment`, so no separate atom-pair figure is
                    # needed.
                    try:
                        _apd_method = config.get("atom_pair_distogram_loss_type", "cross_entropy")
                        _overrides_dist = {
                            'atom_pair_distogram_loss': f'atom_pair_distogram_loss_{_apd_method}',
                        }
                        _ap_aug_dist, _ap_aug_coords = _atom_pair_subplot_augments(
                            loss_component_history)
                        _aa_aug_coords = _atom_angle_subplot_augments(
                            loss_component_history)
                        _suffix = f'_itr{itr + 1}_length{config["length"]}'
                        for _gname, _specs, _title, _overrides, _augment in [
                            ('distogram',  _LOSS_GROUPS['distogram'],
                             'Distogram Losses', _overrides_dist,
                             {'atom_pair_distogram_loss': _ap_aug_dist}),
                            ('confidence', _LOSS_GROUPS['confidence'],
                             'Confidence Losses', None, None),
                            ('coords',     _LOSS_GROUPS['coords'],
                             'Coordinate Losses', None,
                             {'atom_pair_coords_loss': _ap_aug_coords,
                              'atom_angle_coords_loss': _aa_aug_coords}),
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
                                csv_col_overrides=_overrides, augment=_augment)
                            if wrote:
                                print(f"[loss] {_gname} loss history -> {_csv}")
                    except Exception as e:
                        print(f"Error plotting category loss histories: {str(e)}")

                    # ---- Atom-pair per-pair diagnostics CSV (--atom_pairs) -----
                    # The per-pair distance/window curves are overlaid on the
                    # Atom-Pair *Loss subplots of the category figures above (see
                    # `augment`); here we just dump the full per-pair table
                    # (distance, target window, loss for the distogram restraint
                    # and, in full mode, the coord restraint) into the loss folder.
                    # Own try/except (warn only).
                    try:
                        import re as _re
                        ap_keys = [k for k in loss_component_history
                                   if k.startswith('atom_pair|dist|')]
                        if ap_keys and loss_dir:
                            labels = [k.split('atom_pair|dist|', 1)[1] for k in ap_keys]
                            _apd_method = config.get("atom_pair_distogram_loss_type", "cross_entropy")

                            def _win(lab):
                                m = _re.search(r'\[([-\d.]+),\s*([-\d.]+)\]A\s*$', lab)
                                return (float(m.group(1)), float(m.group(2))) if m else (None, None)

                            def _h(kind, lab):
                                return loss_component_history.get(f'atom_pair|{kind}|{lab}', [])

                            # CSV: one row per (logged step, pair) with everything.
                            ap_csv = os.path.join(
                                loss_dir,
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
                            print(f"[atom_pairs] per-pair diagnostics -> {ap_csv}")
                    except Exception as e:
                        print(f"Error writing atom-pair diagnostics CSV: {str(e)}")

                    # ---- Atom-angle per-angle diagnostics CSV (--atom_angles) --
                    # The per-angle trace is overlaid on the Atom-Angle Coords
                    # Loss subplot of the coords category figure (see `augment`);
                    # here we dump the full per-angle table (angle, target window,
                    # loss) into the loss folder. Coords-only, so the columns are
                    # the full-mode angle + violation. Own try/except (warn only).
                    try:
                        import re as _re
                        aa_keys = [k for k in loss_component_history
                                   if k.startswith('atom_angle|cang|')]
                        if aa_keys and loss_dir:
                            labels = [k.split('atom_angle|cang|', 1)[1] for k in aa_keys]

                            def _winA(lab):
                                m = _re.search(r'\[([-\d.]+),\s*([-\d.]+)\]deg\s*$', lab)
                                return (float(m.group(1)), float(m.group(2))) if m else (None, None)

                            def _hA(kind, lab):
                                return loss_component_history.get(f'atom_angle|{kind}|{lab}', [])

                            aa_csv = os.path.join(
                                loss_dir,
                                f'{target_binder_input}_atom_angle_itr{itr + 1}_length{config["length"]}.csv')
                            with open(aa_csv, 'w', newline='') as f:
                                w = csv.writer(f)
                                w.writerow(['step', 'angle_spec', 'lo', 'hi',
                                            'angle_deg', 'coord_loss'])
                                for lab in labels:
                                    lo, hi = _winA(lab)
                                    short = lab.split(' [')[0]
                                    ang = _hA('cang', lab); closs = _hA('closs', lab)
                                    for t in range(len(ang)):
                                        g = lambda s, j: (f'{s[j]:.4f}' if j < len(s) else '')
                                        w.writerow([t, short, lo, hi, g(ang, t), g(closs, t)])
                            print(f"[atom_angles] per-angle diagnostics -> {aa_csv}")
                    except Exception as e:
                        print(f"Error writing atom-angle diagnostics CSV: {str(e)}")

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

                    # Motif residue -> final binder position map (one JSON per
                    # design), e.g. {"A30": "A27", ...}. Written whenever a motif
                    # is active (fixed --motif_residues and/or sliding
                    # --motif_unindex_residues); empty dict => no motif, skip.
                    if _motif_map:
                        motif_map_path = os.path.join(motif_mapping_dir, f"{fasta_tag}.json")
                        with open(motif_map_path, 'w') as mf:
                            json.dump(_motif_map, mf, indent=2)
                        print(f"Wrote motif mapping: {motif_map_path}")


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

                    # Standardized cross-run metrics from the final 200-step fold
                    # (holo + apo), written next to the confidence JSON. Independent
                    # of redo_boltz_predict: final_metrics came from the in-memory
                    # 200-step validation fold inside boltz_hallucination.
                    if final_metrics:
                        _mname = f"{target_binder_input}_results_itr{itr + 1}_length{config['length']}"
                        if final_metrics.get('holo') is not None:
                            print(f"[metrics] wrote {save_metrics_json(results_final_dir, final_metrics['holo'], _mname, 0)}")
                        if final_metrics.get('apo') is not None:
                            save_metrics_json(results_final_dir_apo, final_metrics['apo'], _mname, 0)
                    gc.collect()
                    torch.cuda.empty_cache()
