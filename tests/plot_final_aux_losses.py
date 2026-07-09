#!/usr/bin/env python3
"""Bar plots of final-step losses per design iteration.

BoltzDesign1 writes, per design iteration, into ``<output>/loss/`` (one figure +
matching CSV per category; only loss series with data this run are emitted):
  * ``<input>_distogram_loss_history_itr<N>_length<L>.csv`` -- con_loss,
    i_con_loss, helix_loss, motif_distogram_loss, atom_pair_distogram_loss
    (trunk-distogram losses). When --atom_pairs is set the atom-pair distogram
    column is named ``atom_pair_distogram_loss_<method>`` (method =
    prob|expected|contact), which fixes its unit and hence how it is grouped.
  * ``<input>_confidence_loss_history_itr<N>_length<L>.csv`` -- plddt_loss,
    pae_loss, i_pae_loss, target_plddt_loss (confidence head, full mode only).
  * ``<input>_coords_loss_history_itr<N>_length<L>.csv`` -- rg_loss, com_loss,
    motif_coords_loss, motif_bb_rmsd, motif_lig_rmsd, motif_fape_loss,
    atom_pair_coords_loss (sample-coords losses, in Angstrom).
  * ``<input>_loss_history_itr<N>_length<L>.csv`` -- total_loss only (the contact
    losses are in the distogram CSV above as con_loss / i_con_loss).

Takes the last logged step of every iteration (the converged loss for that
design) and writes, into ``<output>/final_loss_plots/`` (a sibling of ``loss/``):
  * one bar chart per component: final loss vs. iteration index;
  * grouped "mean over iterations" bar figures (mean height, every itr as an
    overlaid point, +/-1 std error bars):
      1. plddt_loss, target_plddt_loss
      2. pae_loss, i_pae_loss
      3 (contact / distogram): con_loss, i_con_loss,
         helix_loss, motif_distogram_loss, atom_pair_distogram_loss. NB the last
         two are bin-center MSE in A^2 (or nats for prob/contact atom-pair
         methods), so this group mixes units with the dimensionless contact CEs.
      4 (Angstrom): rg_loss, motif_coords_loss, motif_fape_loss,
         atom_pair_coords_loss
      5. total_loss (the summed objective, on its own)
  * ``final_losses_by_itr.csv`` / ``main_losses_by_itr.csv`` (the plotted data).

Usage:
    python plot_final_aux_losses.py outputs/<run_folder>
    python plot_final_aux_losses.py outputs/<run_folder>/loss -o /tmp/plots
"""
import argparse
import csv
import glob
import os
import random
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-category loss CSVs (the current BoltzDesign1 layout).
DISTOGRAM_GLOB = "*_distogram_loss_history_itr*_length*.csv"
CONFIDENCE_GLOB = "*_confidence_loss_history_itr*_length*.csv"
COORDS_GLOB = "*_coords_loss_history_itr*_length*.csv"
# Total/intra/inter contact. NOTE: this glob also matches the three category
# CSVs above (they all end in ``_loss_history_itr...``); the category infixes
# are excluded when collecting the main file (see CATEGORY_INFIXES).
MAIN_GLOB = "*_loss_history_itr*_length*.csv"
CATEGORY_INFIXES = ("_distogram_loss_history_", "_confidence_loss_history_",
                    "_coords_loss_history_")
ITR_RE = re.compile(r"_itr(\d+)_length(\d+)\.csv$")

IDENT = lambda v: v


def find_loss_dir(path):
    """Accept either an output run folder or its loss/ folder directly."""
    cand = os.path.join(path, "loss")
    for d in (cand, path):
        if glob.glob(os.path.join(d, MAIN_GLOB)):
            return d
    raise SystemExit(f"No loss CSVs found under {path!r} or {cand!r}")


def last_value(values):
    """Last non-empty float in a column (longer/shorter series are blank-filled)."""
    for v in reversed(values):
        if v != "":
            return float(v)
    return None


def read_final_losses(csv_path):
    """Map each loss column -> its final-step value for one iteration CSV."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = {h: [] for h in header}
        for row in reader:
            for h, v in zip(header, row):
                cols[h].append(v)
    finals = {}
    for h in header:
        if h == "step":
            continue
        fv = last_value(cols[h])
        if fv is not None:
            finals[h] = fv
    return finals


def collect(loss_dir, glob_pat, exclude_infixes=()):
    """itr -> {column: final value} over CSVs matching glob_pat (+ category order).

    exclude_infixes: skip files whose basename contains any of these substrings
    (used to keep the broad MAIN_GLOB from also picking up the category CSVs).
    """
    per_itr, categories = {}, []
    for path in glob.glob(os.path.join(loss_dir, glob_pat)):
        base = os.path.basename(path)
        if any(s in base for s in exclude_infixes):
            continue
        m = ITR_RE.search(base)
        if not m:
            continue
        finals = read_final_losses(path)
        per_itr.setdefault(int(m.group(1)), {}).update(finals)
        for k in finals:
            if k not in categories:
                categories.append(k)
    return per_itr, categories


def merge_per_itr(*dicts):
    out = {}
    for d in dicts:
        for itr, vals in d.items():
            out.setdefault(itr, {}).update(vals)
    return out


def write_summary(per_itr, categories, out_csv):
    itrs = sorted(per_itr)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["itr"] + categories)
        for itr in itrs:
            w.writerow([itr] + [
                f"{per_itr[itr][c]:.4f}" if c in per_itr[itr] else ""
                for c in categories])
    print(f"[summary] {len(itrs)} iterations -> {out_csv}")


def plot_mean_bar(per_itr, items, title, out_png):
    """One bar per present item: mean over itrs, itr points overlaid, +/-1 std.

    items: list of (column_key, display_label, transform_fn).
    """
    itrs = sorted(per_itr)
    cols = []  # (label, transformed values)
    for key, label, fn in items:
        vals = [fn(per_itr[i][key]) for i in itrs if key in per_itr[i]]
        vals = [v for v in vals if v == v]  # drop NaN
        if vals:
            cols.append((label, vals))
    if not cols:
        return
    labels = [l for l, _ in cols]
    means = [statistics.mean(v) for _, v in cols]
    stds = [statistics.stdev(v) if len(v) > 1 else 0.0 for _, v in cols]

    fig, ax = plt.subplots(figsize=(max(4.0, 2.0 * len(cols)), 5.0))
    ax.bar(range(len(cols)), means, yerr=stds, capsize=6, color="#4C72B0",
           zorder=2, error_kw=dict(ecolor="black", elinewidth=1.2))
    rng = random.Random(0)
    for i, (_, v) in enumerate(cols):
        xs = [i + rng.uniform(-0.18, 0.18) for _ in v]
        ax.scatter(xs, v, color="#DD8452", edgecolor="black",
                   linewidth=0.3, s=22, alpha=0.8, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(f"final-step loss (mean over {len(itrs)} itrs, +/-1 std)")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[plot] {out_png}")


def plot_per_itr(per_itr, categories, out_dir):
    """One bar chart per category: final-step loss vs. iteration index."""
    itrs = sorted(per_itr)
    for cat in categories:
        xs = [i for i in itrs if cat in per_itr[i]]
        ys = [per_itr[i][cat] for i in xs]
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(xs)), 4.5))
        ax.bar(range(len(xs)), ys, color="#4C72B0", zorder=2)
        ax.plot(range(len(xs)), ys, "o", color="#DD8452", markersize=4, zorder=3)
        ax.axhline(0, color="black", linewidth=0.8, zorder=1)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs],
                           rotation=90 if len(xs) > 20 else 0, fontsize=8)
        ax.set_xlabel("Design iteration (itr)")
        ax.set_ylabel(f"final-step {cat}")
        ax.set_title(f"Final {cat} per iteration")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"final_{cat}_by_itr.png")
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[plot] {out_png}")


def find_apd(categories):
    """Locate the atom_pair_distogram_loss_<method> column and its method."""
    for c in categories:
        if c.startswith("atom_pair_distogram_loss_"):
            return c, c.rsplit("_", 1)[-1]
    return None, None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "output_folder",
        help="Output run folder (or its loss/ folder) holding the loss CSVs")
    ap.add_argument(
        "-o", "--out-dir", default=None,
        help="Where to write plots (default: <base>/final_loss_plots, beside loss/)")
    args = ap.parse_args()

    loss_dir = find_loss_dir(args.output_folder)
    base = os.path.dirname(loss_dir.rstrip("/"))   # parent of loss/  -> run folder
    out_dir = args.out_dir or os.path.join(base, "final_loss_plots")
    os.makedirs(out_dir, exist_ok=True)
    # Clear our own previously-written mean bars so renumbering/regrouping never
    # leaves orphaned figures from an earlier run (scoped to the mean_ prefix).
    for _stale in glob.glob(os.path.join(out_dir, "mean_*.png")):
        os.remove(_stale)

    disto_per_itr, disto_cats = collect(loss_dir, DISTOGRAM_GLOB)
    conf_per_itr, conf_cats = collect(loss_dir, CONFIDENCE_GLOB)
    coords_per_itr, coords_cats = collect(loss_dir, COORDS_GLOB)
    main_per_itr, main_cats = collect(loss_dir, MAIN_GLOB,
                                      exclude_infixes=CATEGORY_INFIXES)
    if not (disto_per_itr or conf_per_itr or coords_per_itr or main_per_itr):
        raise SystemExit(f"No parseable loss CSVs in {loss_dir!r}")

    # Components 1-3 come from the distogram/confidence/coords category CSVs
    # (merged per itr); motif_* and atom_pair_* columns ride along inside them.
    comp_per_itr = merge_per_itr(disto_per_itr, conf_per_itr, coords_per_itr)
    comp_cats = []
    for cats in (disto_cats, conf_cats, coords_cats):
        comp_cats += [c for c in cats if c not in comp_cats]

    if comp_per_itr:
        write_summary(comp_per_itr, comp_cats,
                      os.path.join(out_dir, "final_losses_by_itr.csv"))
        plot_per_itr(comp_per_itr, comp_cats, out_dir)
    if main_per_itr:
        write_summary(main_per_itr, main_cats,
                      os.path.join(out_dir, "main_losses_by_itr.csv"))

    # Grouped mean bars. group_3's contacts/helix/atom-pair-distogram come from
    # the category CSVs and group_5's total_loss from the main loss-history CSV,
    # so merge both per-itr maps before plotting (column names are disjoint, so
    # no clobbering).
    mean_per_itr = merge_per_itr(comp_per_itr, main_per_itr)
    if mean_per_itr:
        apd_key, _apd_method = find_apd(comp_cats)  # method no longer affects grouping

        # 1-2: same-unit confidence groups.
        plot_mean_bar(mean_per_itr,
                      [("plddt_loss", "plddt_loss", IDENT),
                       ("target_plddt_loss", "target_plddt_loss", IDENT)],
                      "pLDDT losses", os.path.join(out_dir, "mean_1_plddt.png"))
        plot_mean_bar(mean_per_itr,
                      [("pae_loss", "pae_loss", IDENT),
                       ("i_pae_loss", "i_pae_loss", IDENT)],
                      "PAE losses", os.path.join(out_dir, "mean_2_pae.png"))

        # 3: contact + distogram losses (all trunk-distogram terms). helix is
        # the i->i+3 distogram contact CE; motif_distogram and atom_pair_distogram
        # are added raw (bin-center MSE in A^2, or nats for prob/contact
        # atom-pair methods), so this group mixes units with the dimensionless
        # contact CEs.
        group_3 = [("con_loss", "con_loss", IDENT),
                   ("i_con_loss", "i_con_loss", IDENT),
                   ("helix_loss", "helix_loss", IDENT),
                   ("motif_distogram_loss", "motif_distogram_loss", IDENT)]
        if apd_key:
            group_3.append((apd_key, apd_key, IDENT))
        plot_mean_bar(mean_per_itr, group_3,
                      "Contact & distogram losses (contacts / helix / motif & atom-pair distogram)",
                      os.path.join(out_dir, "mean_3_contact.png"))

        # 4: linear-Angstrom sample-coords losses (all already in A: RMSD-form
        # rg / motif / FAPE and the RMS-violation atom-pair restraint).
        group_4 = [("rg_loss", "rg_loss", IDENT),
                   ("motif_coords_loss", "motif_coords_loss", IDENT),
                   ("motif_fape_loss", "motif_fape_loss", IDENT),
                   ("atom_pair_coords_loss", "atom_pair_coords_loss", IDENT)]
        plot_mean_bar(mean_per_itr, group_4,
                      "Angstrom-scale losses (rg / RMSD / atom-pair RMS violation)",
                      os.path.join(out_dir, "mean_4_angstrom.png"))

        # 5: the summed objective, on its own.
        plot_mean_bar(mean_per_itr,
                      [("total_loss", "total_loss", IDENT)],
                      "Total loss", os.path.join(out_dir, "mean_5_total.png"))


if __name__ == "__main__":
    main()
