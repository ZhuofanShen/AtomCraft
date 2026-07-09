#!/usr/bin/env python3
"""Cross-folder comparison of final-step design losses.

Takes several BoltzDesign1 output run folders and, for EVERY loss component,
writes one figure that compares the folders side by side: one box per folder
(median / quartiles / whiskers over the run's design iterations) with each
iteration's final-step value overlaid as a jittered point and ``n`` annotated.
This shows the full distribution of final losses per config -- robust to
outliers -- making it easy to see how e.g. final rg_loss differs between two
sampler configs / hyperparameter sweeps.

"Final-step value" and per-folder parsing are shared with
``plot_final_aux_losses.py`` (same per-iteration category CSVs):
  * ``<input>_distogram_loss_history_itr<N>_length<L>.csv``
  * ``<input>_confidence_loss_history_itr<N>_length<L>.csv``
  * ``<input>_coords_loss_history_itr<N>_length<L>.csv``
  * ``<input>_loss_history_itr<N>_length<L>.csv``   (total/intra/inter contact)

Each loss column gets its OWN figure (raw value -- no unit transform -- so the
y-scale is consistent within a figure). A column present in only some folders
just shows fewer bars. The atom-pair distogram column keeps its method suffix
(``atom_pair_distogram_loss_<method>``), so two runs are only compared when they
used the same method (prob vs expected are different units -> different figures).

Outputs (into ``--out-dir``, default ``./loss_comparison``):
  * ``compare_<loss>.png`` per loss column;
  * ``loss_comparison_summary.csv`` -- tidy
    (loss, folder, n, median, q1, q3, mean, std, min, max).

Pooling replicate runs: join folders with ``:`` in a single token to merge all
their iterations into ONE box (more data points), labeled by the FIRST folder's
basename -- e.g. ``outputs/runA_seed1:outputs/runA_seed2`` is one box.

Usage:
    python compare_loss_folders.py outputs/runA outputs/runB [outputs/runC ...]
    python compare_loss_folders.py outputs/runA_a:outputs/runA_b outputs/runB \
        --labels "runA (pooled)" "runB" -o /tmp/compare
"""
import argparse
import csv
import glob
import os
import random
import numpy as np
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the per-iteration CSV parsing from the single-folder plotter.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_final_aux_losses import (  # noqa: E402
    CATEGORY_INFIXES, CONFIDENCE_GLOB, COORDS_GLOB, DISTOGRAM_GLOB, MAIN_GLOB,
    collect, find_loss_dir, merge_per_itr,
)


def collect_folder(folder):
    """All final-step losses for one run folder: ({itr: {col: val}}, col order).

    Merges the distogram/confidence/coords category CSVs and the main
    loss-history CSV (total/intra/inter); column names are disjoint across these,
    so the merge never clobbers.
    """
    loss_dir = find_loss_dir(folder)
    parts = [
        collect(loss_dir, DISTOGRAM_GLOB),
        collect(loss_dir, CONFIDENCE_GLOB),
        collect(loss_dir, COORDS_GLOB),
        collect(loss_dir, MAIN_GLOB, exclude_infixes=CATEGORY_INFIXES),
    ]
    per_itr = merge_per_itr(*[p for p, _ in parts])
    cols = []
    for _, cats in parts:
        cols += [c for c in cats if c not in cols]
    return per_itr, cols


def group_label(spec):
    """Auto-label for one CLI token: basename of its FIRST ':'-joined folder."""
    return os.path.basename(spec.split(":")[0].rstrip("/"))


def collect_group(spec, collect_fn=None):
    """Collect one CLI token, which may pool several folders joined by ':'.

    ``A:B`` pools the two runs into ONE comparison box: each sub-folder is
    collected separately, then merged into a single per-itr map with DISJOINT
    itr keys (the second run's iterations are offset past the first's) so every
    iteration of every pooled run survives as its own data point -- a plain
    itr-merge would clobber same-numbered iterations. Returns ``(per_itr, cols)``
    just like ``collect_folder`` so the caller is unchanged. The box's label is
    the basename of the FIRST sub-folder (see ``group_label``). Sub-folders with
    no parseable loss CSVs are skipped with a warning; if none parse, raises
    SystemExit so the whole token is dropped (same as a single bad folder).

    `collect_fn` overrides the per-folder collector (default `collect_folder`),
    so callers with a different data layer (e.g. the final-fold metrics JSONs)
    can reuse this pooling/offset logic unchanged.
    """
    collect_fn = collect_fn or collect_folder
    subs = [s for s in spec.split(":") if s]
    combined, cols, offset, kept = {}, [], 0, 0
    for sub in subs:
        try:
            per_itr, sub_cols = collect_fn(sub)
        except SystemExit as e:
            print(f"[skip]   pooled sub-folder {sub!r}: {e}")
            continue
        if not per_itr:
            print(f"[skip]   pooled sub-folder {sub!r}: no parseable loss CSVs")
            continue
        for itr in sorted(per_itr):
            combined[offset + itr] = per_itr[itr]
        offset += max(per_itr) + 1
        for c in sub_cols:
            if c not in cols:
                cols.append(c)
        kept += 1
        if len(subs) > 1:
            print(f"[debug]   pooled {sub!r}: +{len(per_itr)} iters "
                  f"(group total now {len(combined)})")
    if kept == 0:
        raise SystemExit(f"no parseable inputs in any of {subs}")
    return combined, cols


def series_for(per_itr, loss):
    """Sorted-by-itr list of a loss column's final values present in per_itr."""
    return [per_itr[i][loss] for i in sorted(per_itr) if loss in per_itr[i]]


def quartiles(v):
    """(q1, q3) of v; falls back to (min, max)/the lone value for tiny series."""
    if len(v) >= 2:
        q1, _q2, q3 = statistics.quantiles(v, n=4)
        return q1, q3
    return v[0], v[0]


def plot_compare(loss, folder_vals, out_png, ylabel=None):
    """One figure: a box per folder (median/quartiles) + jittered itr points.

    folder_vals: list of (label, [values]) in CLI order; folders with no data
    for this loss are dropped. `ylabel` overrides the default y-axis caption so
    non-loss callers (e.g. final-fold metrics) can label the axis correctly.
    """
    cols = [(lbl, v) for lbl, v in folder_vals if v]
    if not cols:
        return False
    labels = [l for l, _ in cols]
    data = [v for _, v in cols]
    positions = list(range(len(cols)))
    print(f"[debug] plot_compare({loss}): {len(cols)} bars, labels={labels}")
    if len(labels) != len(set(labels)):
        rep = sorted({l for l in labels if labels.count(l) > 1})
        print(f"[debug]   !! REPEATED bar label(s) {rep} -> a box is drawn twice "
              f"with the SAME data (x-jitter reseeded per box, so it looks different)")

    fig, ax = plt.subplots(figsize=(max(4.0, 1.9 * len(cols) + 1.0), 5.0))
    # ax.boxplot(
    #     data, positions=positions, widths=0.55, showfliers=False,
    #     patch_artist=True,
    #     boxprops=dict(facecolor="#AEC7E8", edgecolor="#4C72B0", linewidth=1.2),
    #     medianprops=dict(color="black", linewidth=1.8),
    #     whiskerprops=dict(color="#4C72B0", linewidth=1.2),
    #     capprops=dict(color="#4C72B0", linewidth=1.2),
    #     zorder=2)
    rng = random.Random(0)
    for i, v in enumerate(data):
        xs = [i + rng.uniform(-0.16, 0.16) for _ in v]
        ax.scatter(xs, v, color="#DD8452", edgecolor="black", linewidth=0.3,
                   s=22, alpha=0.8, zorder=3)
        # xs = [i + rng.uniform(-0.08, 0.08) for _ in v]
        # ax.scatter(
        #     xs,
        #     v,
        #     color="black",
        #     s=10,
        #     alpha=0.4,
        #     zorder=5,
        # )
    fig, ax = plt.subplots(figsize=(max(4.0, 1.9 * len(cols) + 1.0), 5.0))

    # Violin plots
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Style violins
    for body in parts["bodies"]:
        body.set_facecolor("#AEC7E8")
        body.set_edgecolor("#4C72B0")
        body.set_alpha(0.8)
        body.set_linewidth(1.2)

    # Add median + IQR overlay
    for i, v in enumerate(data):
        q1, med, q3 = np.percentile(v, [25, 50, 75])

        # IQR bar
        ax.vlines(
            i, q1, q3,
            color="black",
            linewidth=4,
            zorder=3
        )

        # Median line
        ax.hlines(
            med,
            i - 0.12,
            i + 0.12,
            color="white",
            linewidth=2.5,
            zorder=4
        )

        ax.annotate(
            f"n={len(v)}",
            (i, max(v)),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color="black",
        )
        ax.annotate(f"n={len(v)}", (i, max(v)), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, color="black")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel or "final-step loss (per design iteration)")
    ax.set_title(f"{loss} -- per-folder comparison")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+",
                    help="Output run folders (or their loss/ folders) to compare; "
                         "one box per token. Join folders with ':' (A:B) to POOL "
                         "them into one box whose label is A's basename.")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Display labels, one per token (default: first folder's basename)")
    ap.add_argument("-o", "--out-dir", default="loss_comparison",
                    help="Where to write the comparison figures (default: ./loss_comparison)")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.folders):
        raise SystemExit(
            f"--labels has {len(args.labels)} entries but {len(args.folders)} folders given")
    labels = args.labels or [group_label(f) for f in args.folders]

    print(f"[debug] {len(args.folders)} folder arg(s); folder -> label:")
    for f, l in zip(args.folders, labels):
        print(f"[debug]   {f!r}  ->  {l!r}")
    dups = sorted({l for l in labels if labels.count(l) > 1})
    if dups:
        print(f"[debug] !! DUPLICATE labels {dups}: distinct paths share a basename, "
              f"so they collide in the data dict (last one wins) and the dropped "
              f"folder's basename is missing from the plot.")

    os.makedirs(args.out_dir, exist_ok=True)
    # Clear our own previous comparison figures so a changed loss set never
    # leaves orphaned figures behind (scoped to the compare_ prefix).
    for _stale in glob.glob(os.path.join(args.out_dir, "compare_*.png")):
        os.remove(_stale)

    data, loss_order = {}, []
    for spec, label in zip(args.folders, labels):
        try:
            per_itr, cols = collect_group(spec)
        except SystemExit as e:
            print(f"[skip] {label}: {e}")
            continue
        if not per_itr:
            print(f"[skip] {label}: no parseable loss CSVs")
            continue
        if label in data:
            print(f"[debug] !! OVERWRITE: label {label!r} already loaded from an "
                  f"earlier token; this token ({spec!r}) replaces its data.")
        data[label] = per_itr
        for c in cols:
            if c not in loss_order:
                loss_order.append(c)
        print(f"[load] {label}: {len(per_itr)} iters, {len(cols)} loss columns")

    present_labels = [l for l in labels if l in data]
    print(f"[debug] data dict keys (unique labels actually loaded): {list(data.keys())}")
    print(f"[debug] present_labels (one bar each, in this order): {present_labels}")
    if len(present_labels) != len(set(present_labels)):
        print(f"[debug] !! present_labels has repeats -> every figure will draw a "
              f"duplicated bar reading from the same data dict entry.")
    if len(present_labels) < 1:
        raise SystemExit("No folders with parseable loss CSVs to compare.")
    if len(present_labels) < 2:
        print("[warn] only one folder has data -- figures show a single bar each.")

    summary_rows = []
    for loss in loss_order:
        folder_vals = [(l, series_for(data[l], loss)) for l in present_labels]
        for label, vals in folder_vals:
            if vals:
                q1, q3 = quartiles(vals)
                summary_rows.append((
                    loss, label, len(vals), statistics.median(vals), q1, q3,
                    statistics.mean(vals),
                    statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    min(vals), max(vals)))
        out_png = os.path.join(args.out_dir, f"compare_{loss}.png")
        if plot_compare(loss, folder_vals, out_png):
            print(f"[plot] {out_png}")

    out_csv = os.path.join(args.out_dir, "loss_comparison_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["loss", "folder", "n", "median", "q1", "q3",
                    "mean", "std", "min", "max"])
        for loss, label, n, med, q1, q3, mean, std, lo, hi in summary_rows:
            w.writerow([loss, label, n, f"{med:.4f}", f"{q1:.4f}", f"{q3:.4f}",
                        f"{mean:.4f}", f"{std:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    print(f"[summary] {out_csv}")


if __name__ == "__main__":
    main()
