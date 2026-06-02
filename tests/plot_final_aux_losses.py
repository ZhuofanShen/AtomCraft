#!/usr/bin/env python3
"""Bar plots of final-step losses per design iteration.

BoltzDesign1 writes, per design iteration, into ``<output>/loss/``:
  * ``<input>_aux_loss_history_itr<N>_length<L>.csv`` -- ``step`` +
    helix/plddt/pae/rg/target_plddt + (when --atom_pairs is set) the atom-pair
    aggregates. The atom-pair distogram column is named
    ``atom_pair_distogram_loss_<method>`` (method = prob|expected|contact),
    which fixes its unit and hence how it is grouped below.
  * ``<input>_motif_loss_itr<N>_length<L>.csv`` -- motif_distogram_loss /
    motif_coords_loss (when --motif_pdb is set).
  * ``<input>_loss_history_itr<N>_length<L>.csv`` -- total/intra/inter contact.

Takes the last logged step of every iteration (the converged loss for that
design) and writes, into ``<output>/final_loss_plots/`` (a sibling of ``loss/``):
  * one bar chart per component: final loss vs. iteration index;
  * grouped "mean over iterations" bar figures (mean height, every itr as an
    overlaid point, +/-1 std error bars):
      1. plddt_loss, target_plddt_loss
      2. pae_loss, i_pae_loss
      3a (Angstrom): rg_loss, motif_coords_loss, sqrt(atom_pair_coords_loss),
         and sqrt(atom_pair_distogram_loss) when its method is 'expected'
      3b (dimensionless / cross-entropy): helix_loss, motif_distogram_loss, and
         atom_pair_distogram_loss when its method is 'prob'/'contact'
      4. total_loss, intra_contact_loss, inter_contact_loss
  * ``final_losses_by_itr.csv`` / ``main_losses_by_itr.csv`` (the plotted data).

Usage:
    python plot_final_aux_losses.py outputs/<run_folder>
    python plot_final_aux_losses.py outputs/<run_folder>/loss -o /tmp/plots
"""
import argparse
import csv
import glob
import math
import os
import random
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUX_GLOB = "*_aux_loss_history_itr*_length*.csv"
MAIN_GLOB = "*_loss_history_itr*_length*.csv"      # also matches aux; filtered below
MOTIF_GLOB = "*_motif_loss_itr*_length*.csv"
ITR_RE = re.compile(r"_itr(\d+)_length(\d+)\.csv$")

IDENT = lambda v: v
SQRT = lambda v: math.sqrt(v) if v >= 0 else float("nan")  # squared losses are >=0


def find_loss_dir(path):
    """Accept either an output run folder or its loss/ folder directly."""
    cand = os.path.join(path, "loss")
    for d in (cand, path):
        if (glob.glob(os.path.join(d, AUX_GLOB))
                or glob.glob(os.path.join(d, MAIN_GLOB))
                or glob.glob(os.path.join(d, MOTIF_GLOB))):
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


def collect(loss_dir, glob_pat, exclude_aux=False):
    """itr -> {column: final value} over CSVs matching glob_pat (+ category order)."""
    per_itr, categories = {}, []
    for path in glob.glob(os.path.join(loss_dir, glob_pat)):
        base = os.path.basename(path)
        if exclude_aux and "_aux_loss_history_" in base:
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

    aux_per_itr, aux_cats = collect(loss_dir, AUX_GLOB)
    motif_per_itr, motif_cats = collect(loss_dir, MOTIF_GLOB)
    main_per_itr, main_cats = collect(loss_dir, MAIN_GLOB, exclude_aux=True)
    if not (aux_per_itr or motif_per_itr or main_per_itr):
        raise SystemExit(f"No parseable loss CSVs in {loss_dir!r}")

    # Components 1-4 come from aux + motif (merged per itr).
    comp_per_itr = merge_per_itr(aux_per_itr, motif_per_itr)
    comp_cats = aux_cats + [c for c in motif_cats if c not in aux_cats]

    if comp_per_itr:
        write_summary(comp_per_itr, comp_cats,
                      os.path.join(out_dir, "final_losses_by_itr.csv"))
        plot_per_itr(comp_per_itr, comp_cats, out_dir)

        apd_key, apd_method = find_apd(comp_cats)

        # 1-2: same-unit aux groups.
        plot_mean_bar(comp_per_itr,
                      [("plddt_loss", "plddt_loss", IDENT),
                       ("target_plddt_loss", "target_plddt_loss", IDENT)],
                      "pLDDT losses", os.path.join(out_dir, "mean_1_plddt.png"))
        plot_mean_bar(comp_per_itr,
                      [("pae_loss", "pae_loss", IDENT),
                       ("i_pae_loss", "i_pae_loss", IDENT)],
                      "PAE losses", os.path.join(out_dir, "mean_2_pae.png"))

        # 3a: linear-Angstrom losses (squared ones are sqrt'd to A).
        group_a = [("rg_loss", "rg_loss", IDENT),
                   ("motif_coords_loss", "motif_coords_loss", IDENT),
                   ("motif_fape_loss", "motif_fape_loss", IDENT),
                   ("atom_pair_coords_loss", "sqrt(atom_pair_coords_loss)", SQRT)]
        if apd_key and apd_method == "expected":
            group_a.append((apd_key, f"sqrt({apd_key})", SQRT))
        plot_mean_bar(comp_per_itr, group_a,
                      "Angstrom-scale losses (rg / RMSD / sqrt squared-violations)",
                      os.path.join(out_dir, "mean_3a_angstrom.png"))

        # 3b: dimensionless cross-entropy / -log losses (helix is the i->i+3
        # distogram contact CE, same unit as the motif/atom-pair distogram CEs).
        group_b = [("helix_loss", "helix_loss", IDENT),
                   ("motif_distogram_loss", "motif_distogram_loss", IDENT)]
        if apd_key and apd_method in ("prob", "contact"):
            group_b.append((apd_key, apd_key, IDENT))
        plot_mean_bar(comp_per_itr, group_b,
                      "Dimensionless distogram/contact losses (cross-entropy / -log)",
                      os.path.join(out_dir, "mean_3b_dimensionless.png"))

    # 5: total & contact losses from the main loss-history CSV.
    if main_per_itr:
        write_summary(main_per_itr, main_cats,
                      os.path.join(out_dir, "main_losses_by_itr.csv"))
        plot_mean_bar(main_per_itr,
                      [("total_loss", "total_loss", IDENT),
                       ("intra_contact_loss", "intra_contact_loss", IDENT),
                       ("inter_contact_loss", "inter_contact_loss", IDENT)],
                      "Total & contact losses",
                      os.path.join(out_dir, "mean_4_total_contact.png"))


if __name__ == "__main__":
    main()
