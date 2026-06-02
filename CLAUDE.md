# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

BoltzDesign1 is a protein binder design pipeline that *inverts* the Boltz1 all-atom structure prediction model. It optimizes a binder sequence (represented as soft logits) by backpropagating through Boltz1's trunk and confidence head to maximize inter/intra-chain contact and pLDDT scores. After structural optimization, LigandMPNN or ProteinMPNN redesigns the sequence, and optionally AlphaFold3 cross-validates the result.

## Setup

```bash
conda activate boltz_design   # environment created by setup.sh
```

The setup script (`setup.sh`) creates the `boltz_design` conda environment, installs `boltz/` as an editable package (`pip install -e ./boltz`), downloads Boltz weights to `~/.boltz/`, and fetches LigandMPNN model parameters.

Boltz model weights location: `~/.boltz/`  
CCD (chemical component dictionary): `~/.boltz/ccd.pkl`

## Running the pipeline

```bash
# Small molecule target (fetches PDB automatically)
python boltzdesign.py --target_name 7v11 --target_type small_molecule --target_mols OQO --gpu_id 0 --design_samples 2 --suffix 1

# DNA/RNA target
python boltzdesign.py --target_name 5zmc --target_type dna --pdb_target_ids C,D --gpu_id 0 --design_samples 5 --suffix 1

# Custom PDB file
python boltzdesign.py --target_name 7v11 --pdb_path your_pdb_path --target_type small_molecule --target_mols OQO --gpu_id 0

# Disable AlphaFold3 cross-validation
python boltzdesign.py ... --run_alphafold False

# Enable Heun corrector in the diffusion sampler (only affects full-mode evals/snapshots; see Architecture)
python boltzdesign.py ... --use_heun True

# Let confidence-head losses backprop through the diffusion sampler (only meaningful with distogram_only=False)
python boltzdesign.py ... --distogram_only False --attach_coords True

# Tune the diffusion sampler: total step count and per-step length/over-relaxation
python boltzdesign.py ... --num_sampling_steps 50 --step_scale 1.5

# Deterministic sampler for stable backprop through the diffusion module
python boltzdesign.py ... --distogram_only False --attach_coords True --rg_loss 0.3 --deterministic_sampler True --use_heun True --num_sampling_steps 3

# Motif scaffolding: bind the target while retaining a motif (enzyme/glue design; see Architecture)
python boltzdesign.py ... --motif_pdb ref.pdb --motif_residues "A57,A102,A195" --fix_motif_seq True

# Multi-target: one binder contacting several targets of MIXED types at once
# (e.g. a scaffolded peroxidase that binds a target protein AND holds its heme cofactor)
python boltzdesign.py --target_name 1abc --target_types "protein small_molecule" \
  --pdb_target_ids C --target_mols HEM --gpu_id 0

# Pull the binder COM toward an "ORI" pseudo-atom inserted in --pdb_path / --motif_pdb
# (HETATM lines with atom name "ORI", e.g.
#   HETATM 1522 ORI  ORI z   1       4.097   2.872   4.174  0.00  0.00      z   ORI ).
# The ORI is rigid-body Kabsch-transformed from its source PDB frame into the
# live co-fold frame each epoch; with multiple ORIs (across both PDBs) the per-
# source transforms are averaged. Full-mode + --attach_coords needed for a live
# gradient (mirrors --rg_loss).
python boltzdesign.py ... --pdb_path complex.pdb --com_loss 0.3 \
  --distogram_only False --attach_coords True --deterministic_sampler True
```

Target types: `small_molecule`, `dna`, `rna`, `ppi`, `peptide`, `metal`  
Per-type default configs live in `boltzdesign/configs/default_{sm,ppi,na,pep,metal}_config.yaml`.

### Ligand identifier handling — SMILES vs CCD (`--target_mols`)

How a `small_molecule` target is emitted into the Boltz YAML depends on whether its `--target_mols` identifier is a **CCD code present in `~/.boltz/ccd.pkl`**:

- **CCD code** (e.g. `HEM`) → emitted as `ligand: {ccd: HEM}`. Boltz builds it from the CCD: **resname = the code**, **canonical atom names**, exact bond orders. No PDB ligand parse happens (the chemistry comes from the CCD pickle, not your input PDB — the PDB only needs the ligand present if you also reference it via `--motif_ligand_residues`/`--atom_pairs`).
- **Non-CCD identifier** (e.g. `6TR`) → RDKit perceives a **SMILES** from the PDB ligand block, emitted as `ligand: {smiles: …}`. Boltz builds it as **resname `LIG`** with **element-renamed atoms** (`C1`, `N1`, `FE1`…); the original names and code are lost.

**This matters because resname/atom-name features only work on CCD ligands:** the motif ligand carry-along (`--motif_ligand_residues`) finds the predicted ligand by **resname** and matches atoms by **name**, and `--atom_pairs` addresses ligand atoms by name. Both therefore require the target to be a **CCD ligand** — a SMILES-built `LIG` cannot be resolved. So the hemoprotein / cofactor-scaffolding workflows below are only wired for cofactors that exist in the CCD (`HEM`, etc.); a non-CCD analog (`6TR`) has no CCD entry and stays on the SMILES path, so its carry-along is not supported as-is. If your input PDB's ligand is a renamed canonical cofactor, rename the HETATM resname to the CCD code (e.g. `HEM`) and pass `--target_mols HEM` — the PDB's atom names must then match the CCD's (true for a standard heme).

The decision keys off the **identifier** (unambiguous intent), not the emitted value: a custom SMILES that happens to collide with a CCD code (e.g. `CO` = methanol as SMILES vs CCD `CO` = cobalt) is **not** misrouted, because the CCD check runs on the `--target_mols` token, and a perceived SMILES from RDKit is never a bare CCD code.

**Plumbing** (inert for non-CCD identifiers — `6TR`-style runs are byte-for-byte unchanged):
- `boltzdesign/input_utils.py` — `is_ccd_code(name)` (membership in the Boltz CCD pickle); `build_chain_dict` accepts an explicit `{'ccd': code}` / `{'smiles': str}` target value (the caller states the key — no value-based guessing), else falls back to the type-driven default.
- `boltzdesign.py` — both `small_molecule` branches in `generate_yaml_config` (single + multi) emit `{'ccd': code}` when the `--target_mols` token `is_ccd_code(...)`, else perceive SMILES via the (lazily-built) PDB ligand lookup.
- `boltzdesign/alphafold_utils.py` — the `process_yaml_files` `small_molecule` branch routes a `ccd` ligand to AF3's `ccdCodes` (via `build_json_sequence(metal=…)`) instead of unconditionally reading `['smiles']` (which would `KeyError` on a CCD ligand).

`metal` targets were already `ccd:` (unchanged); `dna`/`rna`/`protein` unaffected.

### Multi-target mode (`--target_types`)

A single binder can be designed against **several targets of different types at once** — e.g. design an enzyme that binds a partner protein while also retaining its small-molecule cofactor. The design loop is already target-agnostic: it splits chains into binder (`--binder_id`) vs. everything-else and computes the inter-contact loss against **all** non-binder chains symmetrically, so there is no "primary" target — every target is contacted equally. The single-type assumption lived only in the input/setup + downstream-validation layers; `--target_types` lifts it.

- `--target_types "protein small_molecule"` — space- or comma-separated list, **one entry per target, in chain order**. Overrides `--target_type` when set. Targets are assigned chains `B, C, …` (skipping `--binder_id`) in this order.
- Identifier sourcing (pdb input): `protein`/`dna`/`rna` targets consume `--pdb_target_ids` in order; `small_molecule`/`metal` targets consume `--target_mols` in order. Counts are validated up front (clear error if they don't match). All pdb-mode targets are read from the one PDB (`--pdb_path` or the auto-fetched `--target_name`).
- Custom input (`--input_type custom`): `--custom_target_input` entries map one-to-one to `--target_types`, in order.
- `--target_type` (singular) is **not** used to type the targets in multi mode; it only selects the base default-config file (`default_{ppi,sm,…}_config.yaml`, just seeds hyperparameters you override via CLI) and the output-folder label. Its default `protein` (→ ppi config) is the right base whenever a protein interface is present, so the peroxidase+protein case needs no extra flag. Override individual hyperparameters via their own CLI flags.
- `non_protein_target` (the no-MSA single-sequence path) is derived as "no target is a protein": `True` only when **none** of `--target_types` is `protein`. So a mixed protein+ligand run uses the protein-MSA path; a sm+metal run stays single-sequence. Reproduces the old `target_type != 'protein'` exactly for single-type runs.

Combine freely with motif scaffolding to get the full enzyme-design objective (scaffold the catalytic motif + hold the cofactor ligand + bind a partner protein) in one run:
```bash
python boltzdesign.py --target_name complex --pdb_path ref.pdb \
  --target_types "protein small_molecule" --pdb_target_ids C --target_mols HEM \
  --motif_pdb ref.pdb --motif_residues "A57,A102,A195" \
  --motif_binder_positions "30,75,110" --fix_motif_seq True \
  --motif_distogram_loss 1.0 --length_min 150 --length_max 180
```

To also drive up the **cofactor's own confidence** (the heme is target chain `C` here), add a separately-weighted target-chain pLDDT loss. The default pLDDT loss is binder-only, so the ligand's pLDDT is otherwise never optimized; increasing `--num_inter_contacts` only buys proximity, not confidence. Requires full mode (`--distogram_only False`); carries a real trunk gradient without `--attach_coords`:
```bash
python boltzdesign.py --target_name complex --pdb_path ref.pdb \
  --target_types "protein small_molecule" --pdb_target_ids B --target_mols HEM \
  --target_plddt_chains C --target_plddt_loss 0.3 --distogram_only False
```

**Plumbing** (single-type path is byte-for-byte unchanged; everything keys off `get_all_target_types(args)`):
- `boltzdesign.py` — `--target_types` argparse flag; `get_all_target_types()` / `_split_csv()` helpers; multi-target target building in `generate_yaml_config` (per-type, lazily-cached PDB lookups, cursor over `pdb_target_ids`/`target_mols`); `non_protein_target` (and the LigandMPNN/Rosetta gates) derived from the type list; AF3 step passes `target_type='multi'` when `--target_types` is set. Config file + folder label still come from the singular `--target_type`.
- `boltzdesign/input_utils.py` — `build_chain_dict` / `generate_yaml_for_target_binder` accept a per-target type **list** (a plain `str` still broadcasts to all targets, so existing callers are unaffected).
- `boltzdesign/alphafold_utils.py` — `process_yaml_files` gains a `target_type == 'multi'` branch that reconstructs **every** entity present in the boltz YAML (binder + any mix of protein/ligand/metal/rna/dna) into one AF3 holo JSON instead of dropping off-type chains; apo stays binder-only.
- `boltzdesign/boltzdesign_utils.py` — **unchanged** (the hallucination loop already treats all non-binder chains as target).

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

#### Optional Heun second-order corrector (`use_heun`)

`AtomDiffusion` accepts `use_heun: bool = False` (`diffusion.py`). When enabled, after each Euler predictor step the sampler does a second forward pass at `sigma_t` on the predicted coords and averages the two scores before recomputing the step from `atom_coords_noisy`. Skipped at the final step where `sigma_t == 0`. Trades ~2× NFE per step for better trajectory accuracy — useful when reducing `num_sampling_steps`.

Plumbing chain (already wired end-to-end):
- `boltz/src/boltz/model/modules/diffusion.py` — `AtomDiffusion.__init__(use_heun=False)`, corrector block in `sample()`. Both predictor and corrector forward passes are under `torch.no_grad()`.
- `boltz/src/boltz/main.py` — `BoltzDiffusionParams.use_heun: bool = False`, `--use_heun` Click flag on `predict()`, set into `diffusion_params` before `Boltz1.load_from_checkpoint`. `Boltz1.__init__` already splats `**diffusion_process_args` into `AtomDiffusion`, so no intermediate change is needed.
- `boltzdesign/boltzdesign_utils.py` — `get_boltz_model(..., use_heun=False)` sets `diffusion_params.use_heun`.
- `boltzdesign.py` — `--use_heun` argparse flag passed into `get_boltz_model`.

**Where Heun actually fires in BoltzDesign1:** only on the full-mode path (`distogram_only=False`) and on `save_trajectory=True` snapshots — both of which call `get_distogram_confidence`. The default fast-mode optimization path calls `get_distogram` (trunk only, no diffusion sampler), so `--use_heun` is dead code there.

**Effect on full-mode losses (gradient graph vs numerics):**
- By default (`attach_coords=False`), sampler outputs (`sample_atom_coords`) are detached in `model.py` before reaching the confidence head, and the entire sampler runs under `no_grad`. So Heun **does not change the gradient path** — gradients still flow only through `binder_logits → input_embedder → trunk (s, z, s_inputs) → confidence_module`.
- However, the confidence head consumes `x_pred` (the detached sampled coords) as a parameterizing input. Heun changes that input's value, so `plddt_loss`, `i_pae_loss`, `pae_loss` evaluate to different scalars **and** their gradients w.r.t. trunk activations evaluate to different numbers. `rg_loss` is computed directly from the detached coords, so its value shifts but its gradient w.r.t. trunk params is zero either way (no contribution to optimization).

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
- **Target pLDDT loss** (`--target_plddt_chains`, weight `--target_plddt_loss` default 0.1): a separately-weighted `mean(1 − plddt)` over the tokens of named **target** chain(s) (comma/space-separated chain IDs, e.g. `"C"`). The default pLDDT loss is binder-only, so this is the only term that pushes the model to be confident about a **cofactor/ligand's own placement** (e.g. a heme). Built from a `entity_id`-based token mask in `boltz_hallucination`; computed only when `distogram_only=False` (confidence head). Unlike `rg_loss`, it carries a **real trunk gradient without `--attach_coords`** (it's a function of the trunk `s,z` plus the detached coords, like the other confidence losses). Inert unless `--target_plddt_chains` is set; non-set runs are byte-for-byte unchanged. Note: increasing `--num_inter_contacts` only enforces binder↔target *proximity* in the distogram, not *confidence* — this term is the confidence lever. Plotted/CSV'd as `target_plddt_loss` in the aux-loss outputs.
- **PAE loss**: from confidence head (only when `distogram_only=False`)
- **Rg loss** (`add_rg_loss`): `elu(rg - 2.38·N^0.365)` over binder Cα, a faithful PyTorch port of BindCraft's `add_rg_loss`. Default weight `--rg_loss 0.3` (matches BindCraft `weights_rg`/`use_rg_loss=true`). Computed only when `distogram_only=False` (needs `sample_atom_coords`); contributes a real gradient only when `--attach_coords True` (otherwise the coords are detached and rg_loss is reported but inert — BindCraft's AF2 IPA is always differentiable, so its rg_loss is always a live gradient). To reproduce BindCraft's rg behavior exactly: `--distogram_only False --attach_coords True --rg_loss 0.3`. The Rg is computed over the **binder CAs only** (`get_ca_coords` masks `atom_to_token` by `entity_id == binder_chain` before picking CAs); the threshold `2.38·N^0.365` uses `N_binder` only — so the loss treats the binder as a monomeric protein, ignoring the target. A high `rg_loss` is a *size* signal, not a topology signal — a binder threaded/interleaved with the target can have a normal Rg, so `--num_intra_contacts` and `--inter_chain_cutoff` are usually the stronger globularity levers.
- **COM loss** (`add_com_loss`, `--com_loss` default 0.0): pulls the binder Cα centroid toward the **average of HETATM ORI atoms** parsed from `--pdb_path` and/or `--motif_pdb`. Each source's ORI is rigid-body Kabsch-transformed from its input PDB frame into the live co-fold frame; multiple sources are concatenated and averaged before the L2 distance to the binder COM. Loss value is the distance in Å. Computed only when `distogram_only=False` (needs `sample_atom_coords`); real gradient only with `--attach_coords True` (otherwise reported but inert, mirroring `rg_loss`). Inactive when no ORI atoms are found in either PDB (warning printed). **Anchors used for the Kabsch fit per source:**
  - `--pdb_path` source — all atoms whose token belongs to a non-binder chain (the input-frame xyz comes from `structure.atoms['coords']`).
  - `--motif_pdb` source — the protein motif backbone (`motif_bb_pred_idx_fixed_static` / `_bb_ref_fixed_t`, reusing the existing motif plumbing); only used when `--motif_pdb` is set.
  - Need ≥3 anchor atoms; otherwise the source is skipped with a warning.

  **Kabsch fit is off-graph and ORI is treated as a constant target each step** (`torch.no_grad()` around `_kabsch_transform` + `.detach()` on `all_oris`). Gradient flows ONLY through `binder_COM` → binder CAs → binder logits. Two reasons: (1) physically, we want the binder to *move toward* where the ORI is in the current target frame, not "rearrange the target's prediction so the ORI lands closer to wherever the binder happens to be"; (2) numerically, SVD backward is pathological near degenerate singular values (the same instability the `--deterministic_sampler` flag avoids in the diffusion sampler).

  **Inserting an ORI in your PDB:** add a HETATM line whose atom name is `ORI`, e.g. `HETATM 1522 ORI  ORI z   1       4.097   2.872   4.174  0.00  0.00      z   ORI`. Multiple ORIs in the same PDB are all picked up and averaged after their per-source transform.
- **Motif distogram loss** (`get_motif_distogram_loss`, `--motif_distogram_loss` default 1.0; formerly `--motif_loss`): trunk-distogram CE over motif token pairs. Fast-mode safe.
- **Motif coords loss** (`get_motif_coords_loss`, `--motif_coords_loss` default 1.0): **Decoupled-alignment** Kabsch RMSD on `sample_atom_coords`. The rigid transform is fit from the protein motif **backbone (N, CA, C, CB)** alone — typically ~4·M atoms, a well-conditioned rigid frame — then applied to any **ligand atoms** named by `--motif_ligand_residues`; the combined RMSD over backbone + ligand is the optimized scalar. Decoupling is deliberate: a joint Kabsch over backbone + a ~40-atom heme would let the ligand dominate the alignment and absorb its own placement error, the wrong incentive for enzyme/cofactor scaffolding (hemoprotein etc.). Holding the transform to the backbone instead makes the **ligand RMSD a real signal about ligand placement relative to the motif framework**. Predicted atoms are resolved by name (`_find_named_atom`) and matched to the reference by name; predicted ligand atoms are found in the designed system by **resname match** (chain IDs may differ between motif PDB and the YAML-built design) — which **requires the target ligand to be a CCD ligand** (canonical resname + atom names; see "Ligand identifier handling — SMILES vs CCD"), since a SMILES-built `LIG` loses both. Per-component `motif_bb_rmsd` / `motif_lig_rmsd` recorded separately in `loss_component_history`. Computed only when `distogram_only=False`; real gradient only with `--attach_coords True` (otherwise reported but inert, mirroring `rg_loss`). Combine with `--deterministic_sampler True` for stable backprop. See Motif scaffolding below. **Kabsch fit is off-graph** (`kabsch_align` runs the SVD under `torch.no_grad`, then applies the detached R/t via `(pred − p_mean) @ R.T + r_mean` so pred's gradient still flows through the linear apply); the envelope theorem says this gives the *same* backbone gradient as a fully differentiable Kabsch (Kabsch is the closed-form inner optimum, so ∂RMSD/∂R, ∂RMSD/∂t vanish there), with the bonus that the ligand RMSD no longer pulls backbone atoms around through R to find a "better-fitting alignment frame" — backbone is now pressured only by backbone_ref, ligand only by ligand_ref under the backbone-derived alignment. See "Shared Kabsch helper" below.
- **Motif sidechain FAPE loss** (`get_motif_fape_loss`, `--motif_fape_loss` default 1.0): AlphaFold-style frame-aligned point error — a backbone frame (N, CA, C; `_rigid_frames`) is built per motif residue and every motif sidechain heavy atom is expressed in *every* motif frame and compared in local-frame coords to the reference (precomputed once, since the reference is constant), clamped at 10 Å, averaged over (frame, atom) pairs, in Å. Pose-invariant; captures relative sidechain placement. Computed only when `distogram_only=False` AND shared sidechain heavy atoms exist; real gradient only with `--attach_coords True`. **Caveat:** in the design loop the binder is built as UNK (`'X'`), whose only heavy atoms are `N,CA,C,O,CB,CG`, so the FAPE atom set is the intersection with the reference — typically just **CB,CG** (sidechain-base orientation), not full rotamers. Residue-specific sidechain atoms are absent unless the binder is built with the reference residue types. Inert (skipped) if no shared sidechain atoms. See Motif scaffolding below.
- **Atom-pair distance restraints** (`get_atom_pair_distogram_loss` / `get_atom_pair_coords_loss`, `--atom_pair_distogram_loss` / `--atom_pair_coords_loss`, both default 1.0): explicit per-pair distance windows between **any two specified points**, independent of the general binder–target contact loss and usable across **any two chains incl. target↔target**. The distogram term's formulation is selectable via `--atom_pair_distogram_loss_type {prob,expected,contact}` (default `prob`; `contact` gives a robust long-range attractive pull). Per-pair distance/loss diagnostics are dumped to the animation folder. Inert unless `--atom_pairs` is set. See "Atom-pair distance restraints" below.

### Atom-pair distance restraints (`--atom_pairs`)

A restraint term that pushes the distance between two *named* points into a window `[lo,hi]` Å. Unlike the motif losses (one contiguous motif from one reference structure, either small-molecule- or protein-oriented but not both) and unlike the general contact loss (binder↔pooled-target), this targets **specific atom/residue pairs across any chains** — including **target↔target**, e.g. holding a cofactor near a particular target-protein residue in a designed ternary complex.

**Why it works (the rigid-body question):** targets are *not* given input coordinates — the whole complex (binder + protein + ligand) is co-folded de novo each epoch (`model.py:get_distogram` builds `z` from sequence/MSA/Pairformer only; full mode's diffusion sampler generates all coordinates from noise). So the relative pose between two targets is a free, model-predicted quantity, not a fixed input. **Caveat:** only the binder logits are optimized (target grads zeroed, `boltzdesign_utils.py:~1302`), so a target↔target restraint is satisfied only by changing the *binder* so the trunk co-locates the two targets — the binder must act as the scaffold/glue. This is the ternary-complex regime; if the binder has no leverage over that pair the gradient is weak.

Two flavors (both keyed off the same `--atom_pairs` spec, wired exactly like the motif losses):
- **Distogram restraint** (`--atom_pair_distogram_loss`, default 1.0) from the trunk distogram, symmetrized over the token pair. **Fast-mode safe** (no diffusion). Token resolution: a **ligand atom = its own token (exact)**; a **polymer residue = one token at pseudo-Cβ** (so "near a sidechain" is a Cβ proxy on the polymer side). Always active when `--atom_pairs` is set. The **formulation is selectable** via `--atom_pair_distogram_loss_type`:
  - `prob` (default, original): `−log P(dist ∈ [lo,hi])` — the in-window probability **mass**. Can be driven down by parking a little mass in-window while the bulk of the distribution sits far away; **its gradient also collapses once the pair is far** (`+1e-8` clamp + softmax saturation → ~0 gradient by ~20 Å), so it stops pulling exactly when you need it to — the usual reason a design "ignores" the restraint.
  - `expected`: flat-bottom `relu(lo−E[d])² + relu(E[d]−hi)²` on the **expected distance** `E[d] = Σ prob·mid_pts`, i.e. it optimizes the actual predicted distance (same form as the coord loss). The force **grows with distance** (large far, zero at the window edge) — the natural "pull closer" profile — but both `E[d]` and its gradient saturate at the ~24.5 Å ceiling, so it weakens at the far end.
  - `contact`: BoltzDesign1's own categorical contact loss (`_get_con_loss`, `binary=False`) on the pair with contact cutoff `= hi` (`lo` ignored). A `−log`-type, numerically-stable, monotonic **attractive** objective — gradient is large when far and tapers as probability enters the window; it is the mechanism BoltzDesign1 already uses to *form* binder–target contacts, so its gradient stays strong across the whole resolved range (doesn't die at ~20 Å like `prob`). **Recommended for pulling a pair that starts far apart into contact.**
  - **Ceiling (all three):** the distogram spans ~2–22 Å with one catch-all last bin (center ~24.5 Å) and can't distinguish 30 Å from 100 Å; once the trunk is confident the pair is past ~24 Å the softmax is a near-delta and any distogram-loss gradient vanishes. For a force that keeps growing with distance with **no cap**, use the **coord loss** (`‖x_a−x_b‖` is unbounded) in full mode — see below.
- **Coord flat-bottom** (`--atom_pair_coords_loss`, default 1.0): zero inside `[lo,hi]`, squared violation outside, on the true Euclidean distance over `sample_atom_coords` (single formulation — no `_type` choice). **Atom-level on both sides** (any named atom, incl. true sidechain atoms via `@ATOM`). Active only with `--distogram_only False`; carries a gradient only with `--attach_coords True` (otherwise reported but inert, mirroring `rg_loss`/`motif_coords_loss`). Pair with `--deterministic_sampler True` for stable backprop.

**Debugging the restraint (animation folder output):** every run with `--atom_pairs` set writes, per design iteration, `animation/<input>_atom_pair_itr<N>_length<L>.png` and a matching `.csv`. The PNG has two panels: per-pair **distance vs. logged step** (distogram expected distance `E[d]` solid, argmax-bin `mode` dotted, and the coord distance dashed in full mode; the target `[lo,hi]` window is shaded) and per-pair **distogram loss vs. step**. The CSV has one row per `(step, pair)` with `lo, hi, exp_dist, mode_dist, p_window, distogram_loss_<method>, coord_dist, coord_loss` — the `distogram_loss` column header carries the active `--atom_pair_distogram_loss_type` (e.g. `distogram_loss_expected`) so its unit is unambiguous (expected → Å², prob/contact → nats). The same per-iteration atom-pair aggregates are also appended to the loss-folder aux CSV as `atom_pair_distogram_loss_<method>` and `atom_pair_coords_loss`, so all additive scalar components live in one place; `plot_final_aux_losses.py` reads that method suffix to group the distogram loss correctly (expected with the Å-scale losses after a √, prob/contact with the dimensionless ones). These diagnostics are logged **regardless of `--atom_pair_distogram_loss_type`** (the distance/`p_window` come from the same softmax the loss reads), so on the default `prob` method you can still see directly whether the predicted distance is actually moving into the window. Implemented by `get_atom_pair_distogram_loss(..., return_stats=True)` / `get_atom_pair_coords_loss(..., return_stats=True)` feeding per-pair series into `loss_component_history` under `atom_pair|<kind>|<label>` keys, plotted/dumped in `run_boltz_design`.

**Spec grammar** (`--atom_pairs`): pairs separated by `;`, four comma-separated fields each — `epA, epB, lo, hi`. Endpoint = `CHAIN:SEL[@ATOM]`:
- `SEL` is a **1-indexed residue position within the chain's input sequence** for polymer chains (note: the design's sequence position, *not* the original PDB author number — that numbering is lost when the target is re-parsed from the YAML), or an **atom name** for ligand chains.
- `@ATOM` (optional, polymer only) pins a specific named atom for the **coord** loss (the distogram loss stays at the residue's Cβ token regardless).
- `lo`/`hi` = distance window in Å.

```bash
# Heme FE within 0–6 Å of target-protein residue 145 (Cβ in the distogram; pin a
# sidechain atom for the coord term). Combine with --target_types (heme is a
# target ligand) and motif scaffolding for the full peroxidase objective.
# --atom_pair_distogram_loss_type contact uses BoltzDesign1's own contact loss:
# a robust attractive gradient that stays strong when the pair is far (the default
# `prob` term's gradient dies by ~20 A). Use `expected` for a force that grows
# with distance up to the ~24.5 A ceiling. For an uncapped long-range pull, add
# the coord loss in full mode (see the atom-precise example below).
python boltzdesign.py --target_name complex --pdb_path ref.pdb \
  --target_types "protein small_molecule" --pdb_target_ids B --target_mols HEM \
  --atom_pairs "C:FE, B:145, 0, 6" --atom_pair_distogram_loss 1.0 \
  --atom_pair_distogram_loss_type contact

# Atom-precise (full mode): heme FE 1.8–2.6 Å from a His NE2 on the target
python boltzdesign.py ... --atom_pairs "C:FE, B:145@NE2, 1.8, 2.6" \
  --distogram_only False --attach_coords True --deterministic_sampler True \
  --use_heun True --num_sampling_steps 3 --atom_pair_coords_loss 1.0
```

Plumbing (mirrors the `--motif_*` pattern; inert/byte-for-byte unchanged unless `--atom_pairs` is set):
- `boltzdesign/boltzdesign_utils.py` — `parse_atom_pairs_spec` (pure parse), `resolve_atom_pairs` (→ token & atom indices via `entity_id`/`structure.chains`/`structure.residues`; ligand atom name→token, polymer respos→token, Cβ `atom_disto` or `@ATOM` for coords), `get_atom_pair_distogram_loss(..., method=, return_stats=)`, `get_atom_pair_coords_loss(..., return_stats=)`; `atom_pairs` + `atom_pair_distogram_loss_type` params + a setup block in `boltz_hallucination` (resolved once; loss closures read `atom_pair_specs`/`atom_pair_distogram_loss_type` from enclosing scope); both loss-weight keys added to the `loss_scales` default dict + backfill. Per-pair diagnostics (`exp_dist`/`mode_dist`/`p_window`/per-pair loss, plus coord distance in full mode) go into `loss_component_history` under `atom_pair|<kind>|<label>` keys; `run_boltz_design` reads those keys to write the per-iteration `animation/<input>_atom_pair_itr<N>_length<L>.{png,csv}` (own try/except, warn-only).
- `boltzdesign.py` — `--atom_pairs` / `--atom_pair_distogram_loss` / `--atom_pair_coords_loss` / `--atom_pair_distogram_loss_type` argparse flags; `atom_pairs` + `atom_pair_distogram_loss_type` added to `advanced_params` (so they reach the run config → `**filtered_config` → `boltz_hallucination`); both weights added to the `run_boltz_design_step` `loss_scales`.

### Motif scaffolding (ColabDesign `partial` protocol)

Design a binder that binds the target **and** retains a given structural motif (e.g. a catalytic triad or a cofactor-binding site), so the binder is itself an enzyme / small-molecule binder. Note: **BindCraft does not have this** — it is pure de-novo binder design (`mk_afdesign_model(protocol="binder")` + MPNN interface fixing). Motif scaffolding lives in the upstream ColabDesign AFDesign `partial` protocol; this is a faithful port of its core supervised loss (`af/loss.py::get_dgram_loss` / `_loss_partial`).

Two complementary supervised motif losses (both active when `--motif_pdb` is set):

1. **Distogram CE** (`--motif_distogram_loss`, default 1.0): categorical cross-entropy between the predicted `pdistogram` restricted to motif token pairs and a one-hot target built from the reference structure's pseudo-Cβ coordinates (CB, CA for Gly/missing). Boltz1's distogram is 64 bins with edges `linspace(2,22,63)` and binning `(d > edges).sum(-1)` (`boltz/.../confidence.py:81,295`), identical to ColabDesign's discretization, so the port is exact. Runs in **fast mode** (`distogram_only=True`) — no diffusion/coords required. Bin-resolution limited (~0.32 Å bin width); rotation/reflection-invariant; constrains only pairwise Cβ distances, not orientation, sidechain rotamers, or motif-vs-binder pose.

2. **Backbone coord RMSD (+ optional ligand carry-along)** (`--motif_coords_loss`, default 1.0): Decoupled-alignment Kabsch RMSD between predicted motif positions (from `dict_out['sample_atom_coords']`) and reference. Alignment is fit on the **protein motif backbone N, CA, C, CB only** (~4·M points, much better conditioned than the sampler's internal SVD) and the same rigid transform is then **applied to any predicted ligand atoms named by `--motif_ligand_residues`**; the combined RMSD over backbone + ligand is the optimized scalar (per-component `motif_bb_rmsd` / `motif_lig_rmsd` also logged for diagnostics). The decoupling — rather than a joint Kabsch over backbone + ligand — is the design choice that makes this work for enzyme/cofactor scaffolding (hemoprotein etc.): a joint fit would let a ~40-atom heme dominate alignment and absorb its own placement error, defeating the very objective. Backbone closes the gaps the distance-only distogram CE can't see — crucially motif **orientation** (the 6D analog), plus sub-Å geometry and rigid-body float. Requires `--distogram_only False` for coords to exist, and `--attach_coords True` to carry a gradient (otherwise reported but inert, mirroring `rg_loss`). Pair with `--deterministic_sampler True` for stable backprop. **Predicted ligand lookup:** the designed system is scanned for a residue whose **resname** matches the motif-PDB ligand resname (CCD code) — robust to chain-ID drift between the motif PDB and the YAML-built design. Atom-name matches between motif PDB and the designed CCD; atoms present on one side only are silently skipped (with a coverage print at startup).

3. **Sidechain FAPE** (`--motif_fape_loss`, default 1.0): AlphaFold frame-aligned point error restricted to motif residues — backbone frames (N, CA, C) per residue, every motif sidechain heavy atom expressed in every motif frame, compared in local-frame coords to the reference, clamped at 10 Å. This is the genuinely-missing piece relative to the trRosetta 6D path's sidechain supervision (the AF KSI run used sidechain FAPE here). Same gates as the coord RMSD (`--distogram_only False`, gradient needs `--attach_coords True`, pair with `--deterministic_sampler True`). **Atom-set caveat:** because the binder is built as UNK (`N,CA,C,O,CB,CG` only), the FAPE matches just the shared sidechain heavy atoms — by default **CB,CG** (sidechain-base orientation). A full-rotamer FAPE would require building the binder with the reference residue types at the motif positions (not done at present). Inert/skipped if no shared sidechain atoms (e.g. an all-Gly motif).

**Diagnostics (loss folder output):** whenever motif scaffolding is active, every design iteration writes `loss/<input>_motif_loss_itr<N>_length<L>.png` and a matching `.csv` (alongside the other loss-history plots). The PNG has one panel per active motif loss — motif distogram CE always; motif backbone RMSD (Å) and motif sidechain FAPE (Å) only in full mode (and FAPE only when shared sidechain atoms exist) — each plotted vs. logged step. The CSV has one row per logged step with a column per active loss (full-mode-only series are blank where they end). All read the `motif_distogram_loss` / `motif_coords_loss` / `motif_fape_loss` series recorded in `loss_component_history`; emitted in `run_boltz_design` under its own try/except (warn-only). `plot_final_aux_losses.py` reads the motif CSV and places `motif_coords_loss` + `motif_fape_loss` in the Å-scale mean-bar group (3a) and `motif_distogram_loss` in the dimensionless group (3b).

This is the all-atom payoff: the same model that scores the protein–protein interface also keeps the scaffolded motif intact, and if the cofactor/intermediate is added as a ligand entity in the input YAML, the motif residues' geometry to that ligand is preserved through the normal contact + structure machinery (the basis for designing PTM enzymes and molecular-glue systems).

CLI (all inert unless `--motif_pdb` is set; non-motif runs are byte-for-byte unchanged):
```bash
# Distogram-only (fast, default; cheap but bin-limited)
python boltzdesign.py --target_name 7v11 --target_type small_molecule --target_mols OQO \
  --motif_pdb /path/ref.pdb --motif_residues "A57,A102,A195" \
  --motif_binder_positions "30,75,110" --fix_motif_seq True \
  --motif_distogram_loss 1.0 \
  --length_min 130 --length_max 160

# Distogram + backbone-RMSD + sidechain-FAPE (slow, gradient through sampler — tight motif geometry)
python boltzdesign.py --target_name 7v11 --target_type small_molecule --target_mols OQO \
  --motif_pdb /path/ref.pdb --motif_residues "A57,A102,A195" \
  --motif_binder_positions "30,75,110" --fix_motif_seq True \
  --motif_distogram_loss 1.0 --motif_coords_loss 1.0 --motif_fape_loss 1.0 \
  --distogram_only False --attach_coords True --deterministic_sampler True \
  --use_heun True --num_sampling_steps 3 \
  --length_min 130 --length_max 160

# Hemoprotein scaffolding: catalytic triad on chain A + heme (HEM) on chain B
# of the motif PDB, carried along by the rigid transform Kabsch-fit on the
# motif backbone. Combined backbone+heme RMSD is the optimized motif_coords_loss.
# The heme must also be in the designed system as a target ligand
# (--target_mols HEM), and the binder must have leverage over it (give it a
# pLDDT/contact lever via --target_plddt_chains or --num_inter_contacts).
python boltzdesign.py --target_name peroxidase --pdb_path /path/ref.pdb \
  --target_types "small_molecule" --target_mols HEM \
  --motif_pdb /path/ref.pdb \
  --motif_residues "A57,A102,A195" --motif_binder_positions "30,75,110" \
  --motif_ligand_residues "B1" --fix_motif_seq True \
  --motif_distogram_loss 1.0 --motif_coords_loss 1.0 --motif_fape_loss 1.0 \
  --distogram_only False --attach_coords True --deterministic_sampler True \
  --use_heun True --num_sampling_steps 3 \
  --length_min 150 --length_max 180

# Multi-chain motif PDB: a catalytic motif spread across chains A and B of the
# reference; --motif_chain has been removed -- prefix the chain on every entry.
python boltzdesign.py --target_name complex --pdb_path /path/ref.pdb \
  --motif_pdb /path/ref.pdb \
  --motif_residues "A57,B102,B195" --motif_binder_positions "30,75,110" \
  --motif_distogram_loss 1.0 --length_min 130 --length_max 160
```
- `--motif_residues` — chain-prefixed motif residue selection, e.g. `"A57,A102,A195"` (single chain) or `"A10-14,B57,C195"` (multi-chain motif PDB). Each comma-separated token is CHAIN+RESNUM (author/PDB) with optional ranges. **Bare residue numbers are not accepted** — every entry must prefix its chain explicitly. **Protein motif residues only**: count must match `--motif_binder_positions` 1-to-1. Ligand residues go in `--motif_ligand_residues` instead.
- `--motif_binder_positions` — 1-indexed binder positions the motif maps to (same count/order); default is the N-terminal contiguous block. Positions must be `< length`, so set `--length_min` high enough.
- `--motif_ligand_residues` (optional) — comma-separated chain-prefixed entries `CHAIN+RESNUM` (author/PDB) naming **ligand residues in the motif PDB** to be carried by the backbone-only Kabsch transform: `"B1"` for chain B residue 1; `"B1,C401"` for multiple. Does NOT consume `--motif_binder_positions` slots (the ligand lives in its own target chain in the designed system, added via `--target_mols`). Predicted ligand is found in the designed system by **resname match** (e.g. `HEM`), then atoms matched by **name**; chain IDs can differ between motif PDB and the YAML build. **Requires the target ligand to be a CCD ligand** (resname = CCD code, canonical atom names) — i.e. `--target_mols` must be a CCD code like `HEM`, not a non-CCD analog that falls back to a SMILES-built `LIG`; see "Ligand identifier handling — SMILES vs CCD" above. Inactive when empty; backward-compat byte-identical.
- `--fix_motif_seq True` (default) — pins those binder residues to the reference sequence: their `res_type` is overwritten with the reference one-hot every step (constant write, so it survives the soft/hard/omit machinery — including residues like Cys that BoltzDesign1 otherwise excludes from design) and their `res_type_logits.grad` is zeroed before `norm_seq_grad` so they neither move nor skew the gradient normalization of the free positions. `False` = scaffold geometry only, sequence stays designable. Nonstandard reference residues fall back to geometry-only automatically.

Plumbing chain (mirrors the `--rg_loss`/`--attach_coords` pattern):
- `boltzdesign/boltzdesign_utils.py` — `parse_motif_residue_spec` (chain-prefixed `[(chain, resnum), ...]`; `parse_residue_spec` is retained for `--motif_binder_positions` which is always one chain), `extract_motif_coords` and `extract_motif_residue_atoms` now take a `[(chain, resnum), ...]` list and cache per-chain lookups (so multi-chain motifs don't re-walk the structure), `parse_motif_ligand_spec` + `extract_motif_ligand_atoms` (chain-prefixed ligand residues with all heavy atoms, no CA required), `get_motif_target_distogram`, `get_motif_distogram_loss`, `kabsch_align`, `get_motif_coords_loss(..., lig_pred_idx, lig_ref_xyz, return_stats=)` — decoupled alignment with optional ligand carry-along, `_rigid_frames` + `get_motif_fape_loss` (sidechain FAPE). Setup block in `boltz_hallucination` resolves predicted protein atoms by name (`_find_named_atom` against `structure.residues`) and builds `motif_token_idx`, `motif_target_onehot`, `motif_bb_pred_idx`/`motif_bb_ref_xyz`, `motif_frame_pred_idx`/`motif_fape_sc_pred_idx`/`motif_fape_loc_ref` (+ `motif_fape_active`), and — when `--motif_ligand_residues` is set — predicted ligand atoms via **resname-match scan** over `structure.chains/residues`, populating `motif_lig_pred_idx`/`motif_lig_ref_xyz` (None otherwise, byte-identical fallback). Per-component `bb_rmsd`/`lig_rmsd` recorded under `motif_bb_rmsd`/`motif_lig_rmsd` keys in `loss_component_history`.
- `boltzdesign.py` — argparse flags (`--motif_distogram_loss`, `--motif_coords_loss`, `--motif_fape_loss`, `--motif_ligand_residues`), `advanced_params` injection (only when explicitly passed), all three weight keys added to `run_boltz_design_step` `loss_scales`.
- `run_boltz_design` passes the config keys straight through via `**filtered_config`; active in both the `pre_run` warm-up and the main run. It also writes the per-iteration motif-loss plot/CSV to the loss folder (reads `motif_distogram_loss`/`motif_coords_loss`/`motif_fape_loss` from `loss_component_history`).

### Shared Kabsch helper (`_kabsch_transform`)

A single SVD core in `boltzdesign_utils.py` underlies both the **motif coords loss** (`get_motif_coords_loss` via `kabsch_align`) and the **COM loss** (`add_com_loss`). The helper returns `(R, src_mean, dst_mean)` such that `aligned = (src − src_mean) @ R.T + dst_mean`. Both call sites run the SVD under `torch.no_grad()` (off-graph) and apply the detached R/t through the linear formula, which keeps `pred`'s gradient flowing through the apply but never through the SVD.

**Why off-graph is the right shape for both losses:**
- Kabsch is the closed-form **inner optimum** of `min_{R,t} ||(pred − t1) @ R.T − ref||²`. By the envelope theorem, `d(RMSD)/d(pred)` evaluated with R/t held constant *at their optimal values* equals the gradient with R/t propagated through — because at the optimum `∂RMSD/∂R = ∂RMSD/∂t = 0`. So differentiable Kabsch buys nothing in expectation; it only buys numerical risk (SVD backward is pathological near degenerate singular values — same instability the `--deterministic_sampler` flag avoids in the diffusion sampler's `weighted_rigid_align`).
- For COM loss specifically, off-graph fit *also* gives the physically correct gradient: the binder should move toward where the ORI is in the current target frame, not "rearrange the target's prediction so the ORI happens to land near wherever the binder is."
- For the motif coords loss with `--motif_ligand_residues`, the ligand RMSD is computed under a **backbone-best** alignment (not the ligand's own Kabsch optimum). With off-graph R/t, the backbone gradient is unchanged (envelope theorem at the backbone optimum), while the ligand RMSD no longer gets to push backbone atoms around through R to find a "more ligand-favorable" alignment frame. Backbone is pressured by backbone_ref, ligand by ligand_ref under the backbone-derived alignment — the decoupled story the motif-coords-loss docstring describes.

The motif FAPE loss (`get_motif_fape_loss`) doesn't use Kabsch — it builds per-residue frames from N/CA/C and compares atoms in local frames — so it's unaffected.

### Sequence redesign (LigandMPNN / ProteinMPNN)

After optimization, `run_ligandmpnn_redesign()` in `ligandmpnn_utils.py`:
- Fixes interface residues (< `cutoff` Å from target, default 4 Å)
- Redesigns non-interface residues
- Config in `LigandMPNN/run_ligandmpnn_logits_config.yaml`

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

### Per-epoch intermediate structures (`--save_intermediate_structures`)

When the diffusion sampler runs during optimization (i.e. `--distogram_only False`, or `--save_trajectory True`), `dict_out['sample_atom_coords']` is materialized whether or not `--attach_coords True` is set — that flag only controls **gradient flow** through the coords, not whether the coords exist. `--save_intermediate_structures True` writes those coords (truncated to `structure.atoms['coords'].shape[0]`) plus the per-token pLDDT to `<output>/intermediate_structures/<input>_itr<N>_length<L>/<stage>_epoch<NNNN>.pdb` via `boltz.data.write.pdb.to_pdb`. Save/restore around `structure.atoms` so per-epoch dumps don't pollute the structure object used by downstream consumers.

Implementation lives in `get_model_loss` inside `boltz_hallucination`; each `design()` call site is tagged with a `stage_name` so the filenames are self-explaining. Dead in fast mode (no coords produced).

## Tuning tips (from README)

- Binder too unstructured → increase `num_intra_contacts` (e.g. 2→4)
- No target interaction → increase `num_inter_contacts` (e.g. 2→4)
- Too many helices → set `helix_loss_max=-0.3`, `helix_loss_min=-0.6`
- No interface features with `recycling_steps=0` → try `recycling_steps=1`
