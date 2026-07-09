#!/usr/bin/env python3
"""Cross-folder comparison of final-fold design METRICS.

The metrics analogue of ``compare_loss_folders.py``: instead of the per-iteration
loss CSVs, it reads the standardized final-fold metrics JSON written for every
design (``compute_final_metrics`` -> ``metrics_<name>_model_0.json``) and, for
EVERY metric, writes one figure comparing the run folders side by side -- one box
per folder (median / quartiles / whiskers over the run's design iterations) with
each design's value overlaid as a jittered point and ``n`` annotated. Unlike the
reported losses, these come from the FINAL full 200-step fold and use fixed
per-target-type-default contact settings, so they are comparable across runs that
tuned --num_inter_contacts / --i_con_loss / the sampler / etc.

Metrics file location (per design, picked by ``--fold``):
  holo: ``<folder>/results_final/boltz_results_<name>/predictions/<name>/metrics_<name>_model_0.json``
  apo:  ``<folder>/results_final_apo/...`` (binder-alone; reduced metric set)

The nested JSON (confidence / geometry / standardized_loss, plus per-target and
per-restraint sub-dicts) is flattened to dotted scalar columns, e.g.
``confidence.binder_plddt``, ``confidence.target_plddt.B``,
``geometry.min_interface_distance.B``, ``geometry.atom_pairs.<label>.distance_struct``,
``standardized_loss.i_con_loss``. A column present in only some folders just shows
fewer bars.

Outputs (into ``--out-dir``, default ``./metric_comparison``):
  * ``metric_<col>.png`` per metric column;
  * ``metric_comparison_summary.csv`` (metric, folder, n, median, q1, q3, mean, std, min, max).

Pooling replicate runs: join folders with ':' in a single token to merge their
designs into ONE box, labeled by the FIRST folder's basename.

Usage:
    python compare_metrics_folders.py outputs/runA outputs/runB [outputs/runC ...]
    python compare_metrics_folders.py outputs/runA outputs/runB --fold apo -o /tmp/cmp
    python compare_metrics_folders.py outputs/runA_s1:outputs/runA_s2 outputs/runB \
        --labels "runA (pooled)" "runB"
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

# Reuse the generic pooling / plotting / summary engine from the loss comparator;
# only the per-folder data layer (metrics JSON instead of loss CSVs) differs.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_loss_folders import (  # noqa: E402
    collect_group, group_label, plot_compare, quartiles, series_for,
)

# Keys that are config/descriptive, not metrics, or constant-per-design.
_SKIP_KEYS = {"units", "settings", "interface_contact_cutoff_A", "rg_threshold"}
# Per-restraint entry fields that are the window bounds / flags, not the value.
_RESTRAINT_SKIP = {"label", "lo", "hi", "in_window"}


def _flatten(obj, prefix, out):
    """Recursively collect numeric leaves into ``out[dotted_key] = float``.

    dict -> recurse (skipping _SKIP_KEYS); list of labeled restraint dicts
    (atom_pairs / atom_angles) -> one column per label x numeric field; bools and
    strings ignored; NaN dropped."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        v = float(obj)
        if not math.isnan(v):
            out[prefix.rstrip(".")] = v
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _SKIP_KEYS:
                continue
            _flatten(v, f"{prefix}{k}.", out)
        return
    if isinstance(obj, list):
        for entry in obj:
            if isinstance(entry, dict) and "label" in entry:
                lbl = str(entry["label"])
                for k, v in entry.items():
                    if k in _RESTRAINT_SKIP:
                        continue
                    _flatten(v, f"{prefix}{lbl}.{k}.", out)
        return
    # strings / None / other: ignore


def flatten_metrics(metrics):
    """Flatten one metrics JSON dict to {dotted_col: float}."""
    out = {}
    _flatten(metrics, "", out)
    return out


def _parse_itr(path):
    """Design index from a metrics filename (..._results_itr<N>_length<L>...)."""
    m = re.search(r"itr(\d+)_length", os.path.basename(path))
    return int(m.group(1)) if m else None


def collect_folder(folder, fold="holo"):
    """All final-fold metrics for one run: ({design_idx: {col: val}}, col order).

    `folder` is a run folder (or its results_final / results_final_apo dir). The
    matching ``metrics_*_model_0.json`` files are flattened to dotted columns.
    Raises SystemExit (caught by collect_group) when none are found."""
    sub = "results_final" if fold == "holo" else "results_final_apo"
    # Accept either the run folder or a *_final dir passed directly.
    roots = [os.path.join(folder, sub), folder]
    files = []
    for root in roots:
        files = glob.glob(os.path.join(
            root, "boltz_results_*", "predictions", "*", "metrics_*_model_0.json"))
        if files:
            break
    if not files:
        raise SystemExit(f"no {fold} metrics JSON under {folder!r}")

    per_itr, cols, fallback = {}, [], 0
    for fp in sorted(files):
        itr = _parse_itr(fp)
        if itr is None or itr in per_itr:
            itr = 10_000 + fallback   # keep every design even on a name clash
            fallback += 1
        try:
            with open(fp) as f:
                flat = flatten_metrics(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[skip]   {fp}: {e}")
            continue
        per_itr[itr] = flat
        for c in flat:
            if c not in cols:
                cols.append(c)
    if not per_itr:
        raise SystemExit(f"no parseable {fold} metrics JSON under {folder!r}")
    return per_itr, cols


def _safe(name):
    """Filesystem-safe metric column name (labels can carry ':', '@', ',', ...)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


# --- Offline backfill for historical runs (no model) -------------------------
# For designs that predate the live metrics, rebuild compute_final_metrics' inputs
# from the run's saved files: the `output` dict (CIF coords + plddt/pae npz +
# confidence JSON) and `best_batch` (entity_id / atom_to_token built DIRECTLY from
# the CIF, so they are self-consistent with the coords by construction -- no
# re-featurization, hence robust to a historical run whose atom layout differs
# from a fresh featurization). compute_final_metrics is called with
# boltz_model=None and best_structure=None: its distogram block fails its guard
# (standardized con/i_con/helix + distogram-expected distances need the trunk,
# which Boltz does not persist) and com/atom_pair/atom_angle are off, so neither
# is touched. Every confidence + geometry metric is recovered. No GPU, no CCD.
# Pure post-run analysis, so it lives here, not in the pipeline module; the only
# heavy dep (torch) is imported lazily, on --backfill.

def _parse_cif_atoms(path):
    """Ordered atom records from a Boltz mmCIF: list of
    {record, atom, chain, res, xyz}, resolving columns by `_atom_site.<name>`."""
    if not os.path.exists(path):
        return None
    names, rows, reading = [], [], False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("_atom_site."):
                names.append(s.split(".")[-1])
            elif names and (s.startswith("ATOM") or s.startswith("HETATM")):
                reading = True
                rows.append(line.split())
            elif reading and (not s or s[0] in "#_" or s.startswith("loop_")):
                break
    idx = {n: i for i, n in enumerate(names)}
    need = ["group_PDB", "label_atom_id", "label_asym_id", "label_seq_id",
            "Cartn_x", "Cartn_y", "Cartn_z"]
    if not rows or any(n not in idx for n in need):
        return None
    atoms = []
    for r in rows:
        atoms.append({
            "record": r[idx["group_PDB"]],
            "atom": r[idx["label_atom_id"]],
            "chain": r[idx["label_asym_id"]],
            "res": r[idx["label_seq_id"]],
            "xyz": [float(r[idx["Cartn_x"]]), float(r[idx["Cartn_y"]]),
                    float(r[idx["Cartn_z"]])],
        })
    return atoms


def run_backfill(folders, fold):
    """Write metrics_*.json into every design folder (under each run in `folders`,
    de-duplicating ':'-pooled tokens) that lacks one, computed from saved files."""
    import torch  # heavy dep only when actually backfilling
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "boltzdesign"))
    from loss_functions import chain_to_number_ as chain_to_number
    from final_metrics import compute_final_metrics, save_metrics_json

    def build_batch_and_coords(atoms):
        """CIF atoms -> ({entity_id [1,T], atom_to_token [1,N,T]}, coords [1,N,3],
        chains). Tokens follow Boltz order: one per polymer residue (ATOM, grouped
        by chain+res), one per ligand/metal atom (HETATM)."""
        tok_chain, atom_tok, cur, prev = [], [], -1, None
        for a in atoms:
            if a["record"] == "ATOM":
                key = (a["chain"], a["res"])
                if key != prev:
                    cur += 1
                    tok_chain.append(a["chain"])
                    prev = key
            else:                                   # HETATM: 1 token per atom
                cur += 1
                tok_chain.append(a["chain"])
                prev = None
            atom_tok.append(cur)
        n_tok, n_atom = len(tok_chain), len(atoms)
        entity = torch.tensor([[chain_to_number.get(c, 99) for c in tok_chain]],
                              dtype=torch.long)
        a2t = torch.zeros(1, n_atom, n_tok)
        a2t[0, torch.arange(n_atom), torch.tensor(atom_tok)] = 1.0
        coords = torch.tensor([[a["xyz"] for a in atoms]], dtype=torch.float32)
        return {"entity_id": entity, "atom_to_token": a2t}, coords, sorted(set(tok_chain))

    def build_output(pred_dir, name, coords, n_token):
        """compute_final_metrics-shaped `output`: coords (CIF) + scalars (json) +
        per-token plddt/pae (npz, kept only when their token count matches)."""
        out = {"coords": coords}
        cj = os.path.join(pred_dir, f"confidence_{name}_model_0.json")
        if os.path.exists(cj):
            with open(cj) as f:
                c = json.load(f)
            for k in ("complex_plddt", "complex_iplddt", "ptm", "iptm", "ligand_iptm",
                      "protein_iptm", "complex_pde", "complex_ipde", "confidence_score"):
                if isinstance(c.get(k), (int, float)):
                    out[k] = torch.tensor(float(c[k]))
            pci = c.get("pair_chains_iptm")
            if isinstance(pci, dict):
                try:
                    out["pair_chains_iptm"] = {
                        int(k1): {int(k2): torch.tensor(float(v2)) for k2, v2 in d.items()}
                        for k1, d in pci.items()}
                except (ValueError, TypeError):
                    pass
        import numpy as np
        for key, fname in (("plddt", f"plddt_{name}_model_0.npz"),
                           ("pae", f"pae_{name}_model_0.npz")):
            fp = os.path.join(pred_dir, fname)
            if os.path.exists(fp):
                t = torch.as_tensor(np.load(fp)[key], dtype=torch.float32)
                while t.dim() < (2 if key == "plddt" else 3):
                    t = t.unsqueeze(0)
                if t.shape[-1] == n_token:
                    out[key] = t
        return out

    def backfill_folder(run_folder):
        sub = "results_final" if fold == "holo" else "results_final_apo"
        final_dir = os.path.join(run_folder, sub)
        if not os.path.isdir(final_dir):
            print(f"[backfill] no {sub}/ under {run_folder!r}")
            return
        binder_chain = "A"
        cfg_path = os.path.join(run_folder, "results_final", "config.yaml")
        if os.path.exists(cfg_path):
            try:
                import yaml
                with open(cfg_path) as f:
                    binder_chain = (yaml.safe_load(f) or {}).get("binder_chain", "A")
            except Exception:
                pass
        written = skipped = 0
        for br in sorted(glob.glob(os.path.join(final_dir, "boltz_results_*"))):
            for pred_dir in sorted(glob.glob(os.path.join(br, "predictions", "*"))):
                if not os.path.isdir(pred_dir):
                    continue
                name = os.path.basename(pred_dir)
                if os.path.exists(os.path.join(pred_dir, f"metrics_{name}_model_0.json")):
                    continue
                atoms = _parse_cif_atoms(os.path.join(pred_dir, f"{name}_model_0.cif"))
                if not atoms:
                    print(f"[backfill] {name}: no/unreadable CIF; skip")
                    skipped += 1
                    continue
                try:
                    batch, coords, chains = build_batch_and_coords(atoms)
                    n_token = batch["entity_id"].shape[-1]
                    output = build_output(pred_dir, name, coords, n_token)
                    n_binder = int((batch["entity_id"] == chain_to_number[binder_chain]).sum())
                    tcids = ([c for c in chains if c != binder_chain]
                             if fold == "holo" else [])
                    metrics = compute_final_metrics(
                        None, output, batch, None, binder_chain=binder_chain,
                        target_chain_ids=tcids, length=n_binder, atom_pairs="",
                        atom_angles="", com_loss_weight=0.0, pdb_path="",
                        metric_config=None)
                    metrics["_backfilled"] = True   # standardized losses omitted (no model)
                    save_metrics_json(final_dir, metrics, name, 0)
                    written += 1
                except Exception as e:
                    print(f"[backfill] {name} failed: {type(e).__name__}: {e}")
                    skipped += 1
        print(f"[backfill] {run_folder} ({fold}): wrote {written}, skipped {skipped}")

    seen = set()
    for spec in folders:
        for sub in (s.strip() for s in spec.split(":")):
            if sub and sub not in seen:
                seen.add(sub)
                backfill_folder(sub)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+",
                    help="Run folders to compare; one box per token. Join with ':' "
                         "(A:B) to POOL into one box labeled by A's basename.")
    ap.add_argument("--fold", choices=["holo", "apo"], default="holo",
                    help="Which fold's metrics to read (default: holo complex).")
    ap.add_argument("--backfill", action="store_true",
                    help="First compute & write metrics_*.json (confidence + geometry "
                         "only, no model) for any design lacking one, from the saved "
                         "CIF/npz/confidence files (needs torch). Standardized "
                         "con/i_con/helix are left out (they need the trunk distogram, "
                         "which isn't saved).")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Display labels, one per token (default: first folder's basename).")
    ap.add_argument("-o", "--out-dir", default="metric_comparison",
                    help="Where to write the comparison figures (default: ./metric_comparison).")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.folders):
        raise SystemExit(
            f"--labels has {len(args.labels)} entries but {len(args.folders)} folders given")
    labels = args.labels or [group_label(f) for f in args.folders]

    def collect_fn(sub):
        return collect_folder(sub, fold=args.fold)

    if args.backfill:
        run_backfill(args.folders, args.fold)

    os.makedirs(args.out_dir, exist_ok=True)
    for _stale in glob.glob(os.path.join(args.out_dir, "metric_*.png")):
        os.remove(_stale)

    data, metric_order = {}, []
    for spec, label in zip(args.folders, labels):
        try:
            per_itr, cols = collect_group(spec, collect_fn=collect_fn)
        except SystemExit as e:
            print(f"[skip] {label}: {e}")
            continue
        if not per_itr:
            print(f"[skip] {label}: no parseable metrics JSON")
            continue
        data[label] = per_itr
        for c in cols:
            if c not in metric_order:
                metric_order.append(c)
        print(f"[load] {label}: {len(per_itr)} designs, {len(cols)} metric columns")

    present = [l for l in labels if l in data]
    if not present:
        raise SystemExit("No folders with parseable metrics JSON to compare.")
    if len(present) < 2:
        print("[warn] only one folder has data -- figures show a single bar each.")

    ylabel = f"final-fold metric ({args.fold}, per design)"
    summary_rows = []
    for metric in metric_order:
        folder_vals = [(l, series_for(data[l], metric)) for l in present]
        for label, vals in folder_vals:
            if vals:
                q1, q3 = quartiles(vals)
                summary_rows.append((
                    metric, label, len(vals), statistics.median(vals), q1, q3,
                    statistics.mean(vals),
                    statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    min(vals), max(vals)))
        out_png = os.path.join(args.out_dir, f"metric_{_safe(metric)}.png")
        if plot_compare(metric, folder_vals, out_png, ylabel=ylabel):
            print(f"[plot] {out_png}")

    out_csv = os.path.join(args.out_dir, "metric_comparison_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "folder", "n", "median", "q1", "q3",
                    "mean", "std", "min", "max"])
        for metric, label, n, med, q1, q3, mean, std, lo, hi in summary_rows:
            w.writerow([metric, label, n, f"{med:.4f}", f"{q1:.4f}", f"{q3:.4f}",
                        f"{mean:.4f}", f"{std:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    print(f"[summary] {out_csv}")


if __name__ == "__main__":
    main()
