import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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

# Loss-history categorization for the per-iteration plots. Each tuple is
# (series_key in loss_component_history, plot label, color). Series with no
# data this run are silently skipped so a single-target / fast-mode run
# doesn't waste panel space on inert losses.
_LOSS_GROUPS = {
    'distogram': [
        ('con_loss',                 'Intra-Contact Loss',        '#ff3366'),
        ('i_con_loss',               'Inter-Contact Loss',        '#3366ff'),
        ('inter_target_con_loss',    'Inter-Target Contact Loss', '#ff66cc'),
        ('helix_loss',               'Helix Loss',                '#ffaa00'),
        ('motif_distogram_loss',     'Motif Distogram Loss',      '#aa66ff'),
        ('atom_pair_distogram_loss', 'Atom-Pair Distogram Loss',  '#66ffcc'),
    ],
    'confidence': [
        ('plddt_loss',        'pLDDT Loss',              '#00ff99'),
        ('pae_loss',          'PAE Loss',                '#3366ff'),
        ('i_pae_loss',        'Interface PAE Loss',      '#ff3366'),
        ('inter_target_pae_loss', 'Inter-Target PAE Loss', '#cc66ff'),
        ('target_plddt_loss', 'Target pLDDT Loss',       '#66ffcc'),
    ],
    'coords': [
        ('rg_loss',               'Rg Loss',                '#cc66ff'),
        ('com_loss',              'COM Loss',               '#ff9933'),
        ('motif_coords_loss',     'Motif Coords RMSD',      '#aa66ff'),
        ('motif_bb_rmsd',         'Motif BB RMSD',          '#66aaff'),
        ('motif_lig_rmsd',        'Motif Ligand RMSD',      '#ffaa66'),
        ('motif_fape_loss',       'Motif FAPE Loss',        '#cc99ff'),
        ('atom_pair_coords_loss', 'Atom-Pair Coords Loss',  '#66ffcc'),
        ('atom_angle_coords_loss','Atom-Angle Coords Loss', '#ffcc66'),
    ],
}


def _plot_loss_group(history, group_specs, png_path, csv_path, suptitle,
                     csv_col_overrides=None, augment=None):
    """Plot one category as a horizontal row of subplots (per-loss y-axis, NOT
    shared) into a single figure, and write a matching CSV.

    history          : dict mapping series_key -> list of floats (per logged step)
    group_specs      : list of (series_key, label, color); skipped if no data.
    csv_col_overrides: optional {series_key: csv_column_name} (used to rename
                       atom_pair_distogram_loss to include the active method).
    augment          : optional {series_key: fn(ax)} called after that subplot's
                       aggregate line, to overlay extra diagnostics (used for the
                       atom-pair per-pair distance traces on a twin y-axis). Best-
                       effort: an augment exception never aborts the loss figure.
    """
    import matplotlib.pyplot as _plt
    present = [(k, lbl, c) for k, lbl, c in group_specs if history.get(k)]
    if not present:
        return False
    augment = augment or {}
    _plt.style.use('dark_background')
    n = len(present)
    fig, axes = _plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor('#1C1C1C')
    for ax, (k, lbl, c) in zip(axes, present):
        ax.plot(history[k], color=c, linewidth=2)
        ax.set_xlabel('Logged Steps', fontsize=12)
        ax.set_ylabel(lbl, fontsize=12)
        ax.set_title(lbl, fontsize=12, pad=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        if augment.get(k) is not None:
            try:
                augment[k](ax)
            except Exception:
                pass
    fig.suptitle(suptitle, fontsize=14, color='white')
    _plt.tight_layout(pad=3.0, rect=(0, 0, 1, 0.96))
    _plt.savefig(png_path, facecolor='#1C1C1C', edgecolor='none',
                 bbox_inches='tight', dpi=300)
    _plt.close(fig)
    # Matching CSV (one row per logged step; shorter series blank-padded).
    overrides = csv_col_overrides or {}
    n_steps = max(len(history[k]) for k, _, _ in present)
    with open(csv_path, 'w', newline='') as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(['step'] + [overrides.get(k, k) for k, _, _ in present])
        for t in range(n_steps):
            row = [t]
            for k, _, _ in present:
                s = history[k]
                row.append(f'{s[t]:.4f}' if t < len(s) else '')
            w.writerow(row)
    return True


def _atom_pair_subplot_augments(history):
    """Build ``(distogram_augment, coords_augment)`` overlays for the atom-pair
    loss subplots, or ``(None, None)`` if no ``--atom_pairs`` diagnostics were
    logged this run.

    Each returned ``fn(ax)`` draws -- on a twin y-axis of the given Atom-Pair
    *Loss subplot -- the per-pair distance vs. logged step with each pair's
    target window shaded, folding the old stand-alone atom-pair figure into the
    category figures. The distogram overlay uses the expected distance E[d]
    (solid) + argmax-bin "mode" (dotted); the coords overlay uses the full-mode
    coord distance. Per-pair series live in ``history`` under the
    ``atom_pair|<kind>|<label>`` keys written in ``get_model_loss``.
    """
    import re as _re
    import matplotlib.pyplot as _plt
    ap_keys = [k for k in history if k.startswith('atom_pair|dist|')]
    if not ap_keys:
        return None, None
    labels = [k.split('atom_pair|dist|', 1)[1] for k in ap_keys]

    def _win(lab):
        m = _re.search(r'\[([-\d.]+),\s*([-\d.]+)\]A\s*$', lab)
        return (float(m.group(1)), float(m.group(2))) if m else (None, None)

    def _series(kind, lab):
        return history.get(f'atom_pair|{kind}|{lab}', [])

    def _make(dist_kind, mode_kind=None):
        def _aug(ax):
            tw = ax.twinx()
            cmap = _plt.get_cmap('tab10')
            drew = False
            for i, lab in enumerate(labels):
                c = cmap(i % 10)
                short = lab.split(' [')[0]
                d = _series(dist_kind, lab)
                if d:
                    tw.plot(d, color=c, linewidth=1.6, linestyle='--',
                            alpha=0.9, label=short)
                    drew = True
                if mode_kind:
                    md = _series(mode_kind, lab)
                    if md:
                        tw.plot(md, color=c, linewidth=1.0, linestyle=':',
                                alpha=0.6)
                lo, hi = _win(lab)
                if lo is not None:
                    tw.axhspan(lo, hi, color=c, alpha=0.10)
            tw.set_ylabel('pair distance (A, dashed)', fontsize=9)
            if drew:
                tw.legend(fontsize=6, loc='upper right', framealpha=0.3)
        return _aug

    return _make('dist', 'mode'), _make('cdist')


def _atom_angle_subplot_augments(history):
    """Build a coords augment ``fn(ax)`` for the Atom-Angle Coords Loss subplot,
    or ``None`` if no ``--atom_angles`` diagnostics were logged this run.

    The returned ``fn(ax)`` draws -- on a twin y-axis of the Atom-Angle Coords
    Loss subplot -- each angle's value (degrees) vs. logged step with its target
    window shaded, folding the per-angle trace into the coords category figure.
    Per-angle series live in ``history`` under ``atom_angle|cang|<label>``.
    """
    import re as _re
    import matplotlib.pyplot as _plt
    aa_keys = [k for k in history if k.startswith('atom_angle|cang|')]
    if not aa_keys:
        return None
    labels = [k.split('atom_angle|cang|', 1)[1] for k in aa_keys]

    def _win(lab):
        m = _re.search(r'\[([-\d.]+),\s*([-\d.]+)\]deg\s*$', lab)
        return (float(m.group(1)), float(m.group(2))) if m else (None, None)

    def _aug(ax):
        tw = ax.twinx()
        cmap = _plt.get_cmap('tab10')
        drew = False
        for i, lab in enumerate(labels):
            c = cmap(i % 10)
            short = lab.split(' [')[0]
            ang = history.get(f'atom_angle|cang|{lab}', [])
            if ang:
                tw.plot(ang, color=c, linewidth=1.6, linestyle='--',
                        alpha=0.9, label=short)
                drew = True
            lo, hi = _win(lab)
            if lo is not None:
                tw.axhspan(lo, hi, color=c, alpha=0.10)
        tw.set_ylabel('angle (deg, dashed)', fontsize=9)
        if drew:
            tw.legend(fontsize=6, loc='upper right', framealpha=0.3)
    return _aug


def visualize_results(plots):
    # Plot distogram predictions
    if plots:
        num_plots = len(plots)
        num_rows = (num_plots + 5) // 6
        fig, axs = plt.subplots(num_rows, 6, figsize=(15, num_rows * 2.5))
        
        if num_rows == 1:
            axs = axs.reshape(1, -1)

        for i, plot_data in enumerate(plots):
            row, col = i // 6, i % 6
            axs[row, col].imshow(plot_data)
            axs[row, col].set_title(f'Epoch {i + 1}')
            axs[row, col].axis('off')

        # Hide unused subplots
        for j in range(num_plots, num_rows * 6):
            axs[j // 6, j % 6].axis('off')

        plt.tight_layout()
        plt.show()
        plots.clear()


def plot_total_loss_history(loss_component_history, loss_history, loss_dir, save_filename):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig.patch.set_facecolor('#1C1C1C')
    ax.plot(loss_component_history.get('total_loss', loss_history),
            color='#00ff99', linewidth=2)
    ax.set_xlabel('Logged Steps', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Total Loss History', fontsize=14, pad=15)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout(pad=3.0)
    if loss_dir:
        plt.savefig(
            os.path.join(loss_dir, save_filename),
            facecolor='#1C1C1C', edgecolor='none', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)


def visualize_training_history(best_batch, loss_history, sequence_history, distogram_history, length, binder_chain='A', save_dir=None, save_filename=None, plddt_history=None, pae_history=None):
    """
    Visualize training history including distogram, sequence, pLDDT, and PAE
    evolution animations.
    Args:
        loss_history (list): List of loss values over training
        sequence_history (list): List of sequence probability matrices over training
        distogram_history (list): List of distogram matrices over training
        length (int): Length of sequence to visualize
        save_dir (str): Directory to save visualizations
        plddt_history (list|None): Per-epoch per-token pLDDT vectors (full mode
            only; None/empty -> pLDDT animation skipped).
        pae_history (list|None): Per-epoch symmetric PAE matrices in A (full
            mode only; None/empty -> PAE animation skipped).
    """

    mask = (best_batch['entity_id']==chain_to_number_[binder_chain]).squeeze(0).detach().cpu().numpy()
    sequence_history = [seq[mask] for seq in sequence_history]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)


    def create_distogram_animation():
        plt.style.use('default')  # Use default white background style
        fig, ax = plt.subplots(figsize=(6,6))
        distogram_2d = distogram_history[0]
        im = ax.imshow(distogram_2d)
    
        plt.colorbar(im, ax=ax)
        ax.set_title('Distogram Evolution')

        def update(frame):
            distogram_2d = distogram_history[frame]
            im.set_data(distogram_2d)
            ax.set_title(f'Distogram Epoch {frame + 1}')
            return im,

        ani = FuncAnimation(fig, update, frames=len(distogram_history), interval=200)
        if save_dir:
            ani.save(os.path.join(save_dir, f'{save_filename}_distogram_evolution.gif'), writer='pillow')
        plt.close()
        return ani

    # Create sequence evolution animation
    def create_sequence_animation():
        plt.style.use('default')  # Use default white background style
        fig, ax = plt.subplots(figsize=(12,3.5))
        im = ax.imshow(sequence_history[0].T, vmin=0, vmax=1, cmap='Blues', aspect='auto', alpha=0.8)
        plt.colorbar(im, ax=ax)
        ax.set_yticks(np.arange(20))
        ax.set_yticklabels(list('ARNDCQEGHILKMFPSTWYV'))
        ax.set_title('Sequence Evolution')

        def update(frame):
            im.set_data(sequence_history[frame].T)
            ax.set_title(f'Sequence Epoch {frame + 1}')
            return im,

        ani = FuncAnimation(fig, update, frames=len(sequence_history), interval=200)
        if save_dir:
            ani.save(os.path.join(save_dir, f'{save_filename}_sequence_evolution.gif'), writer='pillow')
        plt.close()
        return ani

    # Token-level chain boundaries (indices where entity_id changes), used to
    # draw chain borders on the PAE map and separate binder/target on pLDDT,
    # mirroring the AF3-server PAE plot.
    entity_ids = best_batch['entity_id'].squeeze(0).detach().cpu().numpy()
    chain_boundaries = [i for i in range(1, len(entity_ids))
                        if entity_ids[i] != entity_ids[i - 1]]

    # pLDDT evolution: line chart of per-token pLDDT vs token (residue) index.
    def create_plddt_animation():
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 3.5))
        n = len(plddt_history[0])
        x = np.arange(n)
        (line,) = ax.plot(x, plddt_history[0], color='#1f77b4', linewidth=1.5)
        ax.set_xlim(0, max(n - 1, 1))
        ax.set_ylim(0, 1)
        ax.set_xlabel('Token (residue) index')
        ax.set_ylabel('pLDDT')
        for b in chain_boundaries:
            ax.axvline(b - 0.5, color='black', lw=1, linestyle='--', alpha=0.6)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title('pLDDT Evolution')

        def update(frame):
            line.set_ydata(plddt_history[frame])
            ax.set_title(f'pLDDT Epoch {frame + 1}')
            return line,

        ani = FuncAnimation(fig, update, frames=len(plddt_history), interval=200)
        if save_dir:
            ani.save(os.path.join(save_dir, f'{save_filename}_plddt_evolution.gif'), writer='pillow')
        plt.close()
        return ani

    # PAE evolution: AF3-server-style heatmap (Greens_r, 0-30 A, chain borders).
    # The full symmetric PAE already contains both the intra (PAE) and
    # interface (iPAE) blocks, so a single map shows both.
    def create_pae_animation():
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(pae_history[0], cmap='Greens_r', vmin=0, vmax=30, origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='PAE (Å)')
        for b in chain_boundaries:
            ax.axhline(b - 0.5, color='black', lw=1)
            ax.axvline(b - 0.5, color='black', lw=1)
        ax.set_xlabel('Aligned residue')
        ax.set_ylabel('Aligned residue')
        ax.set_title('Predicted Aligned Error (PAE)')

        def update(frame):
            im.set_data(pae_history[frame])
            ax.set_title(f'PAE Epoch {frame + 1}')
            return im,

        ani = FuncAnimation(fig, update, frames=len(pae_history), interval=200)
        if save_dir:
            ani.save(os.path.join(save_dir, f'{save_filename}_pae_evolution.gif'), writer='pillow')
        plt.close()
        return ani

    # Create and save animations
    distogram_ani = create_distogram_animation()
    sequence_ani = create_sequence_animation()
    # pLDDT/PAE are full-mode-only -> skip cleanly if no frames were captured.
    plddt_ani = create_plddt_animation() if plddt_history else None
    pae_ani = create_pae_animation() if pae_history else None

    return distogram_ani, sequence_ani, plddt_ani, pae_ani
