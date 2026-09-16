# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

BoltzDesign1 is a protein binder design pipeline that *inverts* the Boltz1 all-atom structure prediction model. It optimizes a binder sequence (represented as soft logits) by backpropagating through Boltz1's trunk and confidence head to maximize inter/intra-chain contact and pLDDT scores. After structural optimization, LigandMPNN or ProteinMPNN redesigns the sequence, and optionally AlphaFold3 cross-validates the result.

## Setup

```bash
conda activate boltz_design   # environment created by setup.sh
```

The boltzdesign/ dir isn't a package; add its dir to sys.path directly:
● Bash(source /ai/share/workspace/zhuofanshen/anaconda3/etc/profile.d/conda.sh && conda activate boltz_design && python -c "import sys, os, importlib.util…)

Target types: `small_molecule`, `dna`, `rna`, `ppi`, `peptide`, `metal`  
Per-type default configs live in `boltzdesign/configs/default_{sm,ppi,na,pep,metal}_config.yaml`.

## Architecture

### Pipeline flow (`boltzdesign.py`)

1. Parse args → download/load PDB → generate YAML input for Boltz1
2. `run_boltz_design()` → `boltz_hallucination()` → optimization loop
3. Convert best logits to sequence → run LigandMPNN/ProteinMPNN redesign
4. (Optionally) run AlphaFold3 cross-validation

### Module responsibilities (`boltzdesign/`)

| File | Purpose |
|---|---|
| `boltzdesign_utils.py` | Core design logic: `get_boltz_model`, `boltz_hallucination`, `run_boltz_design`, all loss functions |
| `input_utils.py` | PDB download, YAML generation, chain parsing, MSA setup |
| `ligandmpnn_utils.py` | LigandMPNN/ProteinMPNN sequence redesign, CIF→PDB conversion, interface detection |
| `alphafold_utils.py` | AF3 input preparation, Docker invocation, result parsing |
| `utils.py` | CIF/PDB conversion helpers |
| `configs/` | Per-target-type YAML configs with default hyperparameters |

### The optimization loop (`boltz_hallucination`)

The binder sequence is represented as floating-point logits over the 20 amino acids. Three optimization stages controlled by `design_algorithm`:

- **Pre-iteration** (`pre_iteration=30`): pure logit optimization, `soft=1.0`
- **Soft stage** (`soft_iteration=75`): logits → softmax (temperature annealing from 1→`e_soft`)
- **Temp stage** (`temp_iteration=45`): softmax with temperature annealing to 0.01
- **Hard stage** (`hard_iteration=5`): one-hot encoding (argmax)
- **Semi-greedy** (`semi_greedy_steps`): MCMC using iPTM score as energy

`design_algorithm="3stages_extra"` adds a second soft stage before temp; `"hard_only"` skips soft entirely.

### Two forward modes through Boltz1

**Fast mode** (`distogram_only=True`, the default): calls `model.get_distogram(batch)` — runs input embedder + MSA module + Pairformer trunk only, returns `pdistogram`. No diffusion. Used for all gradient steps.

**Full mode** (`distogram_only=False`): calls `model.get_distogram_confidence(batch, ...)` — runs trunk + diffusion sampling (`structure_module.sample()`) + confidence head. Used for evaluation/trajectory snapshots. Much slower.

Key flags:
- `disconnect_feats=True` (default): detaches the feature dict before the confidence head so gradients flow through trunk outputs but not raw features
- `disconnect_pairformer=True`: further detaches `s`, `z` before the confidence head

### Boltz1 model (`boltz/src/boltz/model/`)

```
model.py (Boltz1)
├── input_embedder         — token + atom feature embedding
├── msa_module             — MSA row/column attention
├── pairformer_module      — 48-layer Pairformer (trunk), run recycling_steps+1 times
├── distogram_module       — predicts Cβ distance bins from z
├── structure_module       — AtomDiffusion (EDM, 5 steps default)
│   └── DiffusionModule    — atom encoder → 24-layer DiffusionTransformer → atom decoder
└── confidence_module      — pLDDT, PAE, iPTM from trunk outputs + diffusion token repr
```

Key methods on `Boltz1`:
- `get_distogram(feats)` → `{pdistogram, s, z, s_inputs}` — trunk only
- `get_distogram_confidence(feats, ...)` → `{pdistogram, sample_atom_coords, plddt, iptm, ...}` — full pipeline

The structure module (`modules/diffusion.py`) uses EDM (Karras et al. 2022): preconditioning via `c_skip/c_out/c_in/c_noise`, rho-power-law sigma schedule (`rho=7`, `sigma_max=160`), log-normal training noise (`P_mean=-1.2`, `P_std=1.5`), and a stochastic Euler sampler with gamma noise injection. Default `num_sampling_steps=5` (vs AF3's ~200) — EDM's direct-denoiser preconditioning, log-normal training distribution, and `step_scale=1.5` over-relaxation make this viable. AF3 uses the same EDM training but simply discretizes inference more finely; the difference is an inference-time choice, not a training-framework difference.

Both knobs are CLI-adjustable in BoltzDesign1: `--num_sampling_steps` (total EDM step count; default 200 in BoltzDesign1's `predict_args`, not the model-default 5) and `--step_scale` (per-step over-relaxation factor in the Euler/Heun update, `atom_coords_noisy + step_scale·(sigma_t − t_hat)·denoised_over_sigma`; default 1.638). `--num_sampling_steps` follows the `--recycling_steps` plumbing (argparse → `predict_args["sampling_steps"]` + `advanced_params`/config → `boltz_hallucination(num_sampling_steps=…)` → its internal `predict_args` → `confidence_args['num_sampling_steps']` → `structure_module.sample`); it bites only on the full pipeline (`distogram_only=False`, trajectory snapshots, final structure prediction). `--step_scale` follows the `--use_heun` plumbing (argparse → `get_boltz_model(step_scale=…)` → `BoltzDiffusionParams.step_scale` → `AtomDiffusion`); it is baked into the model at load time and is *not* in the run config. Defaults reproduce prior behavior exactly (no change unless the flags are passed).

#### Optional grad-through-sampler (`attach_coords`)

`--attach_coords True` lifts both gating points so coord gradients reach the trunk:
1. `boltz/src/boltz/model/modules/diffusion.py` — `AtomDiffusion.sample()` reads `getattr(self, "attach_coords", False)` and swaps the `torch.no_grad()` wrapper around the predictor (and Heun corrector) forward passes for `contextlib.nullcontext` when the flag is set.
2. `boltz/src/boltz/model/model.py` — `get_distogram_confidence(..., disconnect_coords=True)` gains a `disconnect_coords` kwarg; when False, the post-sample `.detach()` over `structure_out` is skipped so `sample_atom_coords` enters the confidence head with its graph intact.

Plumbing (BoltzDesign1 entry-point only — `boltz/main.py` and `BoltzDiffusionParams` are untouched):
- `boltzdesign.py` — `--attach_coords` argparse flag, threaded into `get_boltz_model` and added to the run config.
- `boltzdesign/boltzdesign_utils.py` — `get_boltz_model(..., attach_coords=False)` sets `model.structure_module.attach_coords` directly on the loaded module. `boltz_hallucination` adds `'disconnect_coords': not attach_coords` to `confidence_args`.

**When it's meaningful:** only with `distogram_only=False` (the optimization path that actually calls `get_distogram_confidence`). In the default fast mode the sampler is not invoked, so `--attach_coords` is a no-op. With `save_trajectory=True` snapshots, the coords are differentiable but the snapshot is read out and not used in any loss, so still nothing trains differently.

**Cost:** keeping the 5-step (or more, with `--use_heun`) sampler in the autograd graph adds significant memory and compute overhead per step. Combine with reduced `num_sampling_steps` if you hit OOM.

#### Optional deterministic sampler (`deterministic_sampler`)

`--deterministic_sampler True` removes the three stochastic/ill-conditioned operations from the reverse sampler so the `sequence → sample_atom_coords` map is smooth, low-variance and SVD-free — intended for stable gradient flow with `--attach_coords True` (e.g. `rg_loss`/confidence-loss backprop). When set, in `AtomDiffusion.sample()` (`diffusion.py`):

1. **Per-step random augmentation off, centering kept**: `center_random_augmentation(..., augmentation=not deterministic_sampler)` — `centering=True` still removes the centroid (the network expects centered input; `rg` is translation-invariant anyway), but the random rotation/translation that makes the sampler a random function is dropped → **gradient variance reduction**.
2. **Kabsch/SVD alignment skipped**: the `alignment_reverse_diff` block (`weighted_rigid_align`, a `torch.linalg.svd` whose backward blows up on near-degenerate singular values) is gated off → **gradient conditioning fix**. (1) and (2) are a coupled pair — the alignment exists to undo the frame mismatch that augmentation + the non-equivariant network introduce; disabling only one is incoherent (align-only → blow-up; aug-only → wasted near-identity SVD), so the flag toggles both together.
3. **EDM churn zeroed**: `gammas → 0`, so `t_hat == sigma_tm` and the `eps = sqrt(t_hat² − sigma_tm²)·randn` injection vanishes — removes the third (otherwise-residual) stochastic source.

**Trade-off:** the augmentation+align+churn machinery is how Boltz/AF3 squeeze single-sample fidelity out of a non-equivariant network; disabling it introduces a systematic orientation bias and degrades fine geometry. But for gradient-based *sequence optimization* a deterministic, consistently-biased, low-variance signal generally beats a high-variance ill-conditioned one, and `rg` (a rotation/translation-invariant global compactness scalar) tolerates the fine-geometry bias. Validate empirically per target (compare full-mode structures / AF3 success), not just the rg curve.

Plumbing — follows the `--use_heun` path exactly (baked into the model at load time, **not** in the run config):
- `boltz/src/boltz/model/modules/diffusion.py` — `AtomDiffusion.__init__(deterministic_sampler=False)`, `self.deterministic_sampler`, three `getattr`-gated points in `sample()`.
- `boltz/src/boltz/main.py` — `BoltzDiffusionParams.deterministic_sampler: bool = False`, `--deterministic_sampler` Click flag on `predict()`, set into `diffusion_params` before load.
- `boltzdesign/boltzdesign_utils.py` — `get_boltz_model(..., deterministic_sampler=False)` sets `diffusion_params.deterministic_sampler`.
- `boltzdesign.py` — `--deterministic_sampler` argparse flag passed into `get_boltz_model`.

Like `--use_heun`/`--step_scale`, it only bites on the full pipeline (`distogram_only=False` / trajectory snapshots / final prediction); in default fast mode the sampler is not invoked. Default `False` reproduces prior behavior exactly.

### Loss function (`get_model_loss` in `boltzdesign_utils.py`)

Losses computed from distogram predictions during optimization:
- **Inter-contact loss**: contacts between binder and target within `inter_chain_cutoff` (default 20 Å)
- **Intra-contact loss**: contacts within binder within `intra_chain_cutoff` (default 14 Å)
- **Helix loss**: penalizes or rewards helix content (controlled by `helix_loss_min/max`)
- **pLDDT loss**: from confidence head (only when `distogram_only=False`). Masked to the **binder chain only** (`mask_1d=chain_mask`), so a target ligand's own pLDDT is *not* optimized by this term — see Target pLDDT loss below.
- **Target pLDDT loss** (`--target_plddt_chains`, weight `--target_plddt_loss` default 0.1): a separately-weighted `mean(1 − plddt)` over the tokens of named **target** chain(s) (comma/space-separated chain IDs, e.g. `"C"`). The default pLDDT loss is binder-only, so this is the only term that pushes the model to be confident about a **cofactor/ligand's own placement** (e.g. a heme). Plotted/CSV'd as `target_plddt_loss` in the aux-loss outputs.
- **PAE loss**: from confidence head (only when `distogram_only=False`)
- **Rg loss** (`add_rg_loss`): `elu(rg - 2.38·N^0.365)` over binder Cα. The Rg is computed over the **binder CAs only** (`get_ca_coords` masks `atom_to_token` by `entity_id == binder_chain` before picking CAs); the threshold `2.38·N^0.365` uses `N_binder` only — so the loss treats the binder as a monomeric protein, ignoring the target. A high `rg_loss` is a *size* signal, not a topology signal — a binder threaded/interleaved with the target can have a normal Rg, so `--num_intra_contacts` and `--inter_chain_cutoff` are usually the stronger globularity levers.
- **COM loss** (`add_com_loss`, `--com_loss` default 0.0): pulls the binder Cα centroid toward the **average of HETATM ORI atoms** parsed from `--pdb_path` and/or `--motif_pdb`. Each source's ORI is rigid-body Kabsch-transformed from its input PDB frame into the live co-fold frame; multiple sources are concatenated and averaged before the L2 distance to the binder COM. Loss value is the distance in Å. Computed only when `distogram_only=False` (needs `sample_atom_coords`); real gradient only with `--attach_coords True` (otherwise reported but inert, mirroring `rg_loss`). Inactive when no ORI atoms are found in either PDB (warning printed). **Anchors used for the Kabsch fit per source:**
  - `--pdb_path` source — all atoms whose token belongs to a non-binder chain (the input-frame xyz comes from `structure.atoms['coords']`).
  - `--motif_pdb` source — the protein motif backbone (`motif_bb_pred_idx_fixed_static` / `_bb_ref_fixed_t`, reusing the existing motif plumbing); only used when `--motif_pdb` is set.
  - Need ≥3 anchor atoms; otherwise the source is skipped with a warning.

  **Kabsch fit is off-graph and ORI is treated as a constant target each step** (`torch.no_grad()` around `_kabsch_transform` + `.detach()` on `all_oris`). Gradient flows ONLY through `binder_COM` → binder CAs → binder logits. Two reasons: (1) physically, we want the binder to *move toward* where the ORI is in the current target frame, not "rearrange the target's prediction so the ORI lands closer to wherever the binder happens to be"; (2) numerically, SVD backward is pathological near degenerate singular values (the same instability the `--deterministic_sampler` flag avoids in the diffusion sampler).

### Motif scaffolding (ColabDesign `partial` protocol)

Design a binder that binds the target **and** retains a given structural motif (e.g. a catalytic triad or a cofactor-binding site), so the binder is itself an enzyme / small-molecule binder. Note: **BindCraft does not have this** — it is pure de-novo binder design (`mk_afdesign_model(protocol="binder")` + MPNN interface fixing). Motif scaffolding lives in the upstream ColabDesign AFDesign `partial` protocol; this is a faithful port of its core supervised loss (`af/loss.py::get_dgram_loss` / `_loss_partial`).

Two complementary supervised motif losses (both active when `--motif_pdb` is set):

1. **Distogram CE** (`--motif_distogram_loss`, default 1.0): categorical cross-entropy between the predicted `pdistogram` restricted to motif token pairs and a one-hot target built from the reference structure's pseudo-Cβ coordinates (CB, CA for Gly/missing). Boltz1's distogram is 64 bins with edges `linspace(2,22,63)` and binning `(d > edges).sum(-1)` (`boltz/.../confidence.py:81,295`), identical to ColabDesign's discretization, so the port is exact. Runs in **fast mode** (`distogram_only=True`) — no diffusion/coords required. Bin-resolution limited (~0.32 Å bin width); rotation/reflection-invariant; constrains only pairwise Cβ distances, not orientation, sidechain rotamers, or motif-vs-binder pose.

2. **Backbone coord RMSD (+ optional ligand carry-along)** (`--motif_coords_loss`, default 1.0): Decoupled-alignment Kabsch RMSD between predicted motif positions (from `dict_out['sample_atom_coords']`) and reference. Alignment is fit on the **protein motif backbone N, CA, C, CB only** (~4·M points, much better conditioned than the sampler's internal SVD) and the same rigid transform is then **applied to any predicted ligand atoms named by `--motif_ligand_residues`**; the combined RMSD over backbone + ligand is the optimized scalar (per-component `motif_bb_rmsd` / `motif_lig_rmsd` also logged for diagnostics). The decoupling — rather than a joint Kabsch over backbone + ligand — is the design choice that makes this work for enzyme/cofactor scaffolding (hemoprotein etc.): a joint fit would let a ~40-atom heme dominate alignment and absorb its own placement error, defeating the very objective. Backbone closes the gaps the distance-only distogram CE can't see — crucially motif **orientation** (the 6D analog), plus sub-Å geometry and rigid-body float. Requires `--distogram_only False` for coords to exist, and `--attach_coords True` to carry a gradient (otherwise reported but inert, mirroring `rg_loss`). Pair with `--deterministic_sampler True` for stable backprop. **Predicted ligand lookup:** the designed system is scanned for a residue whose **resname** matches the motif-PDB ligand resname (CCD code) — robust to chain-ID drift between the motif PDB and the YAML-built design. Atom-name matches between motif PDB and the designed CCD; atoms present on one side only are silently skipped (with a coverage print at startup).

3. **Sidechain FAPE** (`--motif_fape_loss`, default 1.0): AlphaFold frame-aligned point error restricted to motif residues — backbone frames (N, CA, C) per residue, every motif sidechain heavy atom expressed in every motif frame, compared in local-frame coords to the reference, clamped at 10 Å. This is the genuinely-missing piece relative to the trRosetta 6D path's sidechain supervision (the AF KSI run used sidechain FAPE here). Same gates as the coord RMSD (`--distogram_only False`, gradient needs `--attach_coords True`, pair with `--deterministic_sampler True`). **Atom-set caveat:** because the binder is built as UNK (`N,CA,C,O,CB,CG` only), the FAPE matches just the shared sidechain heavy atoms — by default **CB,CG** (sidechain-base orientation). A full-rotamer FAPE would require building the binder with the reference residue types at the motif positions (not done at present). Inert/skipped if no shared sidechain atoms (e.g. an all-Gly motif).

- `--motif_residues` — chain-prefixed motif residue selection, e.g. `"A57,A102,A195"` (single chain) or `"A10-14,B57,C195"` (multi-chain motif PDB). Each comma-separated token is CHAIN+RESNUM (author/PDB) with optional ranges. **Bare residue numbers are not accepted** — every entry must prefix its chain explicitly. **Protein motif residues only**: count must match `--motif_binder_positions` 1-to-1. Ligand residues go in `--motif_ligand_residues` instead.
- `--motif_binder_positions` — 1-indexed binder positions the motif maps to (same count/order); default is the N-terminal contiguous block. Positions must be `< length`, so set `--length_min` high enough.
- `--motif_ligand_residues` (optional) — comma-separated chain-prefixed entries `CHAIN+RESNUM` (author/PDB) naming **ligand residues in the motif PDB** to be carried by the backbone-only Kabsch transform: `"B1"` for chain B residue 1; `"B1,C401"` for multiple. Does NOT consume `--motif_binder_positions` slots (the ligand lives in its own target chain in the designed system, added via `--target_mols`). Predicted ligand is found in the designed system by **resname match** (e.g. `HEM`), then atoms matched by **name**; chain IDs can differ between motif PDB and the YAML build. **Requires the target ligand to be a CCD ligand** (resname = CCD code, canonical atom names) — i.e. `--target_mols` must be a CCD code like `HEM`, not a non-CCD analog that falls back to a SMILES-built `LIG`; see "Ligand identifier handling — SMILES vs CCD" above. Inactive when empty; backward-compat byte-identical.
- `--fix_motif_seq True` (default) — pins those binder residues to the reference sequence: their `res_type` is overwritten with the reference one-hot every step (constant write, so it survives the soft/hard/omit machinery — including residues like Cys that BoltzDesign1 otherwise excludes from design) and their `res_type_logits.grad` is zeroed before `norm_seq_grad` so they neither move nor skew the gradient normalization of the free positions. `False` = scaffold geometry only, sequence stays designable. Nonstandard reference residues fall back to geometry-only automatically.

### Explicit sequence pinning (`--fix_seq_positions` / `--fix_seq`)

The sequence-only counterpart to `--fix_motif_seq`, with **no motif PDB and no geometry restraint**: a plain 1-to-1 map from binder positions to amino acids. Use it to hard-code residues the design must carry (a catalytic Cys, an engineered salt bridge, a fixed framework residue) while leaving their structure entirely to the optimizer.

```bash
--fix_seq_positions 35 67-70 --fix_seq E GKDF     # A35=E, A67=G, A68=K, A69=D, A70=F
```

- `--fix_seq_positions` — space-separated 1-indexed positions within the binder chain, inclusive ranges allowed. Same grammar as `--motif_binder_positions` / `--i_con_binder_residues` (no chain prefix — the binder is always `--binder_id`); the legacy comma form `"35,67-70"` also works. Positions must be `<= length`, so set `--length_min` high enough.
- `--fix_seq` — one one-letter code per selected position, **in the same order**. Whitespace/comma group separators are stripped (`E GKDF` == `E,GKDF` == `EGKDF`), so group the letters to mirror `--fix_seq_positions` for readability. Only the 20 standard amino acids are accepted.
- Both flags must be given together and the counts must match 1-to-1; mismatched counts, non-standard letters, out-of-range positions and repeated positions all raise at setup, before any GPU work.

**Mechanism** (identical to the motif pin, in `boltz_hallucination`): the selected `res_type` rows are overwritten with a constant one-hot every step inside `update_sequence` — a constant write, so it survives the soft/hard/omit machinery, including residues like Cys that BoltzDesign1 otherwise excludes from design — and their `res_type_logits.grad` is zeroed before `norm_seq_grad` so they neither move nor skew the gradient normalization of the free positions. The positions are also excluded from `_mutate`'s sampling distribution, so the semi-greedy stage (which accepts on iPTM alone and sees no sequence constraint at all) can't silently mutate them.

**Interaction with the motif pin:** `--fix_seq` is applied *after* the motif pin, so it wins on any position it names. An overlap with a *fixed* motif (`--motif_residues` + `--fix_motif_seq True`) is a spec conflict known up-front and raises at setup; an overlap with a *sliding* motif (`--motif_unindex_residues`) can only be decided per-epoch by the placement search, so it prints a warning instead.

**Scope:** the pin covers the hallucination stage only (design loop + semi-greedy + the final fold). The downstream LigandMPNN/ProteinMPNN redesign is a separate stage and does **not** see it — it will freely reassign those positions unless you also constrain it there.

Both flags empty (the default) => inactive, byte-identical to pre-feature runs.

### Shared Kabsch helper (`_kabsch_transform`)

A single SVD core in `boltzdesign_utils.py` underlies both the **motif coords loss** (`get_motif_coords_loss` via `kabsch_align`) and the **COM loss** (`add_com_loss`). The helper returns `(R, src_mean, dst_mean)` such that `aligned = (src − src_mean) @ R.T + dst_mean`. Both call sites run the SVD under `torch.no_grad()` (off-graph) and apply the detached R/t through the linear formula, which keeps `pred`'s gradient flowing through the apply but never through the SVD.

**Why off-graph is the right shape for both losses:**
- Kabsch is the closed-form **inner optimum** of `min_{R,t} ||(pred − t1) @ R.T − ref||²`. By the envelope theorem, `d(RMSD)/d(pred)` evaluated with R/t held constant *at their optimal values* equals the gradient with R/t propagated through — because at the optimum `∂RMSD/∂R = ∂RMSD/∂t = 0`. So differentiable Kabsch buys nothing in expectation; it only buys numerical risk (SVD backward is pathological near degenerate singular values — same instability the `--deterministic_sampler` flag avoids in the diffusion sampler's `weighted_rigid_align`).
- For COM loss specifically, off-graph fit *also* gives the physically correct gradient: the binder should move toward where the ORI is in the current target frame, not "rearrange the target's prediction so the ORI happens to land near wherever the binder is."
- For the motif coords loss with `--motif_ligand_residues`, the ligand RMSD is computed under a **backbone-best** alignment (not the ligand's own Kabsch optimum). With off-graph R/t, the backbone gradient is unchanged (envelope theorem at the backbone optimum), while the ligand RMSD no longer gets to push backbone atoms around through R to find a "more ligand-favorable" alignment frame. Backbone is pressured by backbone_ref, ligand by ligand_ref under the backbone-derived alignment — the decoupled story the motif-coords-loss docstring describes.

The motif FAPE loss (`get_motif_fape_loss`) doesn't use Kabsch — it builds per-residue frames from N/CA/C and compares atoms in local frames — so it's unaffected.

### Output structure

```
outputs/{target_name}_{suffix}/
├── results_final/          — Boltz1 design outputs (CIF)
├── results_yaml/           — per-design YAML inputs
├── ligandmpnn_cutoff_*/
│   ├── 01_pdb/             — redesigned PDB files
│   ├── 02_af_input/        — AF3 input JSONs
│   └── 03_af_pdb_success/  — validated designs + high_iptm_confidence_scores.csv
├── loss/                   — loss curves; every loss PNG has a matching `.csv` of the plotted series. Four figures per design iteration: `<input>_loss_history_itr<N>_length<L>.png` (total loss only, single panel); `<input>_distogram_loss_history_itr<N>_length<L>.png` (con_loss, i_con_loss, helix_loss, motif_distogram_loss, atom_pair_distogram_loss — trunk-distogram losses, fast-mode-safe); `<input>_confidence_loss_history_itr<N>_length<L>.png` (plddt_loss, pae_loss, i_pae_loss, target_plddt_loss — confidence-head, full-mode only); `<input>_coords_loss_history_itr<N>_length<L>.png` (rg_loss, com_loss, motif_coords_loss, motif_bb_rmsd, motif_lig_rmsd, motif_fape_loss, atom_pair_coords_loss — sample-coords losses, in Å, gradient needs `--attach_coords True`). Each subplot has its OWN y-axis (not shared) so widely-different magnitudes coexist cleanly. Only loss series with data this run are plotted (silent skip otherwise). Categorization and the `_plot_loss_group` helper live at the top of `boltzdesign_utils.py` (`_LOSS_GROUPS` dict). The unified `loss_component_history` dict is the single source of truth — every per-epoch loss component (incl. `total_loss`) is appended into it inside `get_model_loss`; the legacy `loss_history` / `con_loss_history` / `i_con_loss_history` / `plddt_loss_history` lists are kept as mirrors for the return-tuple shape only.
├── fasta/                  — `<input>_itr<N>_length<L>.fasta` (status=ok|fold_failed)
├── animation/              — per-iter trajectory animations
└── intermediate_structures/ — per-epoch full-mode folds (only if --save_intermediate_structures True)
    └── <input>_itr<N>_length<L>/
        └── <stage>_epoch<NNNN>.pdb  — stage ∈ {pre,soft,soft1,soft2,temp,hard,hard_only}
```

### Final-predict vs optimization-loop forward (decoupled)

`boltzdesign_utils.py:boltz_hallucination` keeps two distinct `predict_args` dicts:
- The **per-epoch optimization** dict (built from the user's `--recycling_steps` and `--num_sampling_steps`) is set on the model and consumed by `get_distogram` (fast mode) and `get_distogram_confidence` (full mode) inside the design loop. Keep these low for in-loop memory/throughput.
- The **stand-alone final-validation** dict (`final_predict_args`, hardcoded `recycling_steps=3, sampling_steps=200`) is used only for the final two `_run_model(best_batch_*, ...)` calls (and the post-semi-greedy ones), so the final fold quality doesn't depend on how cheap the optimization-loop forwards were. These values are intentionally **not** surfaced as CLI flags — they are internal validation hyperparameters. The per-mutation semi-greedy eval still uses the cheaper optimization dict.

Historical note: prior to this split, `--recycling_steps` was effectively dead during the optimization loop because `predict_args["recycling_steps"]` was hardcoded to 3. The split also fixes that — `--recycling_steps N` now actually takes effect in the optimization-loop forwards.

## Tuning tips (from README)

- Binder too unstructured → increase `num_intra_contacts` (e.g. 2→4)
- No target interaction → increase `num_inter_contacts` (e.g. 2→4)
- Too many helices → set `helix_loss_max=-0.3`, `helix_loss_min=-0.6`
- No interface features with `recycling_steps=0` → try `recycling_steps=1`
