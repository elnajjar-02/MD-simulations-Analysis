import os
import glob
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# COLOR PALETTES FOR MULTI-COMPLEX / MULTI-METRIC
# ==============================================================================
PALETTES = {
    "nature": [
        "#1F449C",  # Deep Blue
        "#E84A5F",  # Carmine Red
        "#2A9D8F",  # Teal Green
        "#DDA15E",  # Ochre / Gold
        "#9B5DE5",  # Purple
        "#F15BB5",  # Magenta
        "#00BBF9",  # Sky Blue
        "#38B000",  # Forest Green
    ],
    "classic": [
        "#1F77B4",  # Muted Blue
        "#FF7F0E",  # Safety Orange
        "#2CA02C",  # Cooked Asparagus Green
        "#D62728",  # Brick Red
        "#9467BD",  # Muted Purple
        "#8C564B",  # Chestnut Brown
        "#E377C2",  # Raspberry Pink
        "#7F7F7F",  # Middle Gray
    ],
    "muted": [
        "#4C72B0",  # Slate Blue
        "#C44E52",  # Muted Red
        "#55A868",  # Soft Green
        "#8172B3",  # Lavender
        "#CCB974",  # Straw Yellow
        "#64B5CD",  # Powder Blue
        "#495057",  # Charcoal
        "#E27C7C",  # Rose
    ],
    "dark": [
        "#005F73",  # Deep Cyan
        "#AE2012",  # Dark Crimson
        "#0A9396",  # Teal
        "#EE9B00",  # Amber
        "#94D2BD",  # Mint
        "#CA6702",  # Rust
        "#9B2226",  # Maroon
        "#1D3557",  # Oxford Blue
    ]
}

CURRENT_PALETTE = PALETTES["nature"]

# ==============================================================================
# GLOBAL PUBLICATION STYLING
# ==============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 600,
    "savefig.bbox": "tight"
})


# ==============================================================================
# DATA PROCESSING UTILITIES
# ==============================================================================
def read_xvg(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('@', '#')):
                continue
            try:
                row = [float(val) for val in line.split()]
                if row:
                    data.append(row)
            except ValueError:
                continue
    return np.array(data)


def smooth_series(y, window=5):
    if window <= 1 or len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode="same")


def process_slice_and_smooth(x, y, time_range=None, rezero=True, window=1):
    if time_range is not None:
        t_min, t_max = time_range
        mask = (x >= t_min) & (x <= t_max)
        x = x[mask]
        y = y[mask] if y.ndim == 1 else y[mask, :]
        if rezero and len(x) > 0:
            x = x - x[0]

    if window > 1:
        if y.ndim == 1:
            y_smooth = smooth_series(y, window)
        else:
            y_smooth = np.zeros_like(y)
            for i in range(y.shape[1]):
                y_smooth[:, i] = smooth_series(y[:, i], window)
        return x, y_smooth, y
    return x, y, None


def apply_grid_and_box(ax, panel_label=None):
    ax.grid(True, linestyle="--", alpha=0.6, color="#D5D5D5", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.1)

    if panel_label:
        ax.text(
            -0.12, 1.06, panel_label,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="right"
        )


def find_matching_file(files, keyword):
    for f in files:
        base = os.path.basename(f).lower()
        if base.startswith(keyword):
            return f
    return None


# ==============================================================================
# MULTI-COMPLEX PLOTTING FUNCTIONS
# ==============================================================================
def plot_rmsd(ax, systems_data, protein_title, time_range=None, window=1, panel_label=None):
    apply_grid_and_box(ax, panel_label)
    x_max = 0

    for idx, sys in enumerate(systems_data):
        color = CURRENT_PALETTE[idx % len(CURRENT_PALETTE)]
        t = sys["data"][:, 0]
        rmsd = sys["data"][:, 1] * 10.0  # nm -> Å
        t, rmsd_smooth, rmsd_raw = process_slice_and_smooth(t, rmsd, time_range, window=window)

        if len(systems_data) == 1 and rmsd_raw is not None:
            ax.plot(t, rmsd_raw, color=color, alpha=0.25, lw=1.0)

        ax.plot(t, rmsd_smooth, color=color, lw=1.8, label=sys["label"])
        if len(t) > 0:
            x_max = max(x_max, t[-1])

    title = f"Root Mean Square Deviation\n{protein_title}" if protein_title else "Root Mean Square Deviation"
    ax.set_title(title)
    ax.set_xlabel("Time (ns)" if not time_range else "Time (ns, re-zeroed)")
    ax.set_ylabel("RMSD (Å)")
    ax.set_xlim(0, x_max if x_max > 0 else 100)
    ax.set_ylim(0, 7)
    ax.legend(loc="best", frameon=True, edgecolor="#CCCCCC", facecolor="white")


def plot_rmsf(ax, systems_data, protein_title, window=1, panel_label=None):
    apply_grid_and_box(ax, panel_label)
    res_max = 0

    for idx, sys in enumerate(systems_data):
        color = CURRENT_PALETTE[idx % len(CURRENT_PALETTE)]
        res = sys["data"][:, 0]
        rmsf = sys["data"][:, 1] * 10.0  # nm -> Å
        _, rmsf_smooth, _ = process_slice_and_smooth(res, rmsf, window=window)

        ax.plot(res, rmsf_smooth, color=color, lw=1.6, label=sys["label"])
        if len(systems_data) == 1:
            ax.fill_between(res, rmsf_smooth, color=color, alpha=0.18)

        if len(res) > 0:
            res_max = max(res_max, res[-1])

    title = f"Residue Flexibility (RMSF)\n{protein_title}" if protein_title else "Residue Flexibility (RMSF)"
    ax.set_title(title)
    ax.set_xlabel("Residue Number")
    ax.set_ylabel(r"C$\alpha$ RMSF (Å)")
    if res_max > 0:
        ax.set_xlim(left=550, right=res_max)
    ax.legend(loc="upper right", frameon=True, edgecolor="#CCCCCC", facecolor="white")


def plot_hbond(ax, systems_data, protein_title, time_range=None, window=1, panel_label=None):
    apply_grid_and_box(ax, panel_label)
    x_max = 0
    y_max = 3

    for idx, sys in enumerate(systems_data):
        color = CURRENT_PALETTE[idx % len(CURRENT_PALETTE)]
        t = sys["data"][:, 0]
        hb = sys["data"][:, 1]
        t, hb, _ = process_slice_and_smooth(t, hb, time_range=time_range, window=1)

        marker = "o" if len(systems_data) <= 2 else None
        ax.plot(t, hb, color=color, marker=marker, markersize=3.5, lw=1.4, label=sys["label"], alpha=0.85)

        if len(t) > 0:
            x_max = max(x_max, t[-1])
        if len(hb) > 0:
            y_max = max(y_max, max(hb))

    title = f"Hydrogen Bonds\n{protein_title}" if protein_title else "Hydrogen Bonds"
    ax.set_title(title)
    ax.set_xlabel("Time (ns)" if not time_range else "Time (ns, re-zeroed)")
    ax.set_ylabel("Number of H-Bonds")
    ax.set_ylim(-0.3, y_max + 0.8)
    ax.set_xlim(0, x_max if x_max > 0 else 100)
    ax.legend(loc="upper right", frameon=True, edgecolor="#CCCCCC", facecolor="white")


def plot_ie(ax, systems_data, protein_title, time_range=None, window=1, panel_label=None):
    apply_grid_and_box(ax, panel_label)
    x_max = 0

    if len(systems_data) == 1:
        # Full breakdown (Coul, LJ, Total) for single complex
        sys = systems_data[0]
        data = sys["data"]
        t_ns = data[:, 0] / 1000.0 if data[-1, 0] > 500 else data[:, 0]
        coul = data[:, 1]
        lj = data[:, 2]
        total = coul + lj

        stacked = np.column_stack((coul, lj, total))
        t, y_smooth, _ = process_slice_and_smooth(t_ns, stacked, time_range, window=window)

        labels = [f"Coul-SR ({sys['label']})", f"LJ-SR ({sys['label']})", f"Total Energy ({sys['label']})"]
        colors = ["#E84A5F", "#DDA15E", "#1F449C"]

        for i in range(3):
            lw = 2.0 if i == 2 else 1.2
            style = "-" if i == 2 else "--"
            ax.plot(t, y_smooth[:, i], label=labels[i], color=colors[i], lw=lw, linestyle=style)
        if len(t) > 0:
            x_max = t[-1]
    else:
        # Overlay Total Interaction Energies for multiple complexes
        for idx, sys in enumerate(systems_data):
            color = CURRENT_PALETTE[idx % len(CURRENT_PALETTE)]
            data = sys["data"]
            t_ns = data[:, 0] / 1000.0 if data[-1, 0] > 500 else data[:, 0]
            total = data[:, 1] + data[:, 2]
            t, total_smooth, _ = process_slice_and_smooth(t_ns, total, time_range, window=window)

            ax.plot(t, total_smooth, color=color, lw=1.8, label=f"Total: {sys['label']}")
            if len(t) > 0:
                x_max = max(x_max, t[-1])

    ax.axhline(0, color="black", linestyle=":", lw=0.9, alpha=0.7)
    title = f"Interaction Energy\n{protein_title}" if protein_title else "Interaction Energy"
    ax.set_title(title)
    ax.set_xlabel("Time (ns)" if not time_range else "Time (ns, re-zeroed)")
    ax.set_ylabel(r"Energy (kJ $\cdot$ mol$^{-1}$)")
    ax.set_xlim(0, x_max if x_max > 0 else 100)
    ax.legend(loc="best", frameon=True, edgecolor="#CCCCCC", facecolor="white")


def plot_sasa(ax, systems_data, protein_title, time_range=None, window=1, panel_label=None):
    apply_grid_and_box(ax, panel_label)
    x_max = 0

    for idx, sys in enumerate(systems_data):
        color = CURRENT_PALETTE[idx % len(CURRENT_PALETTE)]
        t = sys["data"][:, 0]
        sasa = sys["data"][:, 1]
        t, sasa_smooth, sasa_raw = process_slice_and_smooth(t, sasa, time_range, window=window)

        if len(systems_data) == 1 and sasa_raw is not None:
            ax.plot(t, sasa_raw, color=color, alpha=0.25, lw=1.0)

        ax.plot(t, sasa_smooth, color=color, lw=1.8, label=sys["label"])
        if len(t) > 0:
            x_max = max(x_max, t[-1])

    title = f"Solvent Accessible Surface Area (SASA)\n{protein_title}" if protein_title else "Solvent Accessible Surface Area (SASA)"
    ax.set_title(title)
    ax.set_xlabel("Time (ns)" if not time_range else "Time (ns, re-zeroed)")
    ax.set_ylabel(r"Area (nm$^2$)")
    ax.set_xlim(0, x_max if x_max > 0 else 100)
    ax.legend(loc="upper right", frameon=True, edgecolor="#CCCCCC", facecolor="white")


def plot_gyrate(ax, systems_data, protein_title, time_range=None, window=1, panel_label=None):
    apply_grid_and_box(ax, panel_label)
    x_max = 0

    for idx, sys in enumerate(systems_data):
        color = CURRENT_PALETTE[idx % len(CURRENT_PALETTE)]
        t = sys["data"][:, 0]
        rg = sys["data"][:, 1]
        t, rg_smooth, rg_raw = process_slice_and_smooth(t, rg, time_range, window=window)

        if len(systems_data) == 1 and rg_raw is not None:
            ax.plot(t, rg_raw, color=color, alpha=0.25, lw=1.0)

        ax.plot(t, rg_smooth, color=color, lw=1.8, label=sys["label"])
        if len(t) > 0:
            x_max = max(x_max, t[-1])

    title = f"Radius of Gyration ($R_g$)\n{protein_title}" if protein_title else "Radius of Gyration ($R_g$)"
    ax.set_title(title)
    ax.set_xlabel("Time (ns)" if not time_range else "Time (ns, re-zeroed)")
    ax.set_ylabel(r"$R_g$ (nm)")
    ax.set_xlim(0, x_max if x_max > 0 else 100)
    ax.legend(loc="upper right", frameon=True, edgecolor="#CCCCCC", facecolor="white")


DISPATCH = {
    "rmsd": plot_rmsd,
    "rmsf": plot_rmsf,
    "ie": plot_ie,
    "sasa": plot_sasa,
    "hbond": plot_hbond,
    "gyrate": plot_gyrate
}


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    global CURRENT_PALETTE

    parser = argparse.ArgumentParser(description="Publication MD figures with multi-complex overlay and grid themes.")
    parser.add_argument("-d", "--dirs", "--dir", nargs="+", default=["."],
                        help="One or more directories containing .xvg files (e.g., -d sim1 sim2 sim3)")
    parser.add_argument("-pdb", "--pdb", nargs="+", default=["Complex"],
                        help="PDB ID(s) for the runs (e.g., -pdb 134D 13FF 2DF2)")
    parser.add_argument("-l", "--ligand", nargs="+", default=[""],
                        help="Ligand name(s) (e.g., -l 16e or -l 16e 9h 12b)")
    parser.add_argument("-p", "--protein", type=str, default="",
                        help="Common Protein name (e.g., VEGFR-2 or c-Met)")
    parser.add_argument("-s", "--slice", nargs=2, type=float, default=None, metavar=('START', 'END'),
                        help="Trajectory slice in ns (e.g., --slice 10 100)")
    parser.add_argument("-w", "--window", type=int, default=5,
                        help="Smoothing window size (use 1 for raw data)")
    parser.add_argument("-o", "--outdir", type=str, default="plots",
                        help="Directory to save generated figures")
    parser.add_argument("-c", "--combine", nargs="+",
                        default=["rmsd", "rmsf", "ie", "sasa", "hbond", "gyrate"],
                        choices=["rmsd", "rmsf", "ie", "sasa", "hbond", "gyrate"],
                        help="Metrics to combine in the multi-panel plot")
    parser.add_argument("-g", "--grid", nargs=2, type=int, default=None, metavar=('ROWS', 'COLS'),
                        help="Define custom grid layout: rows cols (e.g., --grid 2 3)")
    parser.add_argument("--palette", type=str, default="nature",
                        choices=["nature", "classic", "muted", "dark"],
                        help="Color palette style")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    CURRENT_PALETTE = PALETTES[args.palette]

    # Normalize system metadata lengths to match number of directories
    dirs = args.dirs
    n_sims = len(dirs)

    # Pad or broadcast pdbs and ligands
    pdbs = args.pdb
    if len(pdbs) == 1 and n_sims > 1:
        pdbs = pdbs * n_sims
    elif len(pdbs) < n_sims:
        pdbs += [f"Run-{i+1}" for i in range(len(pdbs), n_sims)]

    ligands = args.ligand
    if len(ligands) == 1 and n_sims > 1 and ligands[0] != "":
        ligands = ligands * n_sims
    elif len(ligands) < n_sims:
        ligands += [""] * (n_sims - len(ligands))

    # Construct clean legend labels per complex
    complex_labels = []
    for p, lig in zip(pdbs, ligands):
        if lig:
            complex_labels.append(f"PDB: {p} ({lig})")
        else:
            complex_labels.append(f"PDB: {p}" if p != "Complex" else "System")

    # Index files across all directories
    metric_dataset = {}
    for metric in DISPATCH.keys():
        metric_dataset[metric] = []
        for i, d in enumerate(dirs):
            xvg_files = glob.glob(os.path.join(d, "*.xvg"))
            match = find_matching_file(xvg_files, metric)
            if match:
                data = read_xvg(match)
                if len(data) > 0:
                    metric_dataset[metric].append({
                        "label": complex_labels[i],
                        "data": data
                    })

    protein_title = args.protein.strip()

    # 1. SAVE INDIVIDUAL FIGURES
    for metric, systems_data in metric_dataset.items():
        if not systems_data:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        func = DISPATCH[metric]

        if metric == "rmsf":
            func(ax, systems_data, protein_title, window=args.window, panel_label="A")
        else:
            func(ax, systems_data, protein_title, time_range=args.slice, window=args.window, panel_label="A")

        pdb_tag = "_".join(pdbs[:3]) + (f"_and_{n_sims-3}_more" if n_sims > 3 else "")
        out_name = f"{metric}_{pdb_tag}_{args.palette}.png".replace(" ", "_")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, out_name))
        plt.close(fig)
        print(f"[✓] Saved individual figure: {out_name}")

    # 2. SAVE COMBINED MULTI-PANEL FIGURE
    selected_metrics = [m for m in args.combine if metric_dataset.get(m)]
    n_plots = len(selected_metrics)

    if n_plots == 0:
        print("[!] Error: None of the selected metrics had data across the directories.")
        return

    # Determine Grid Size
    if args.grid is not None:
        n_rows, n_cols = args.grid
        if n_rows * n_cols < n_plots:
            n_rows = math.ceil(n_plots / n_cols)
    else:
        if n_plots <= 3:
            n_rows, n_cols = 1, n_plots
        elif n_plots == 4:
            n_rows, n_cols = 2, 2
        else:
            n_cols = 3
            n_rows = math.ceil(n_plots / 3)

    figsize = (5.5 * n_cols, 4.3 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.array(axes).flatten()

    panel_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

    for idx, metric in enumerate(selected_metrics):
        ax = axes_flat[idx]
        letter = panel_letters[idx]
        func = DISPATCH[metric]
        systems_data = metric_dataset[metric]

        if metric == "rmsf":
            func(ax, systems_data, protein_title, window=args.window, panel_label=letter)
        else:
            func(ax, systems_data, protein_title, time_range=args.slice, window=args.window, panel_label=letter)

    for extra_idx in range(n_plots, len(axes_flat)):
        fig.delaxes(axes_flat[extra_idx])

    combined_name = f"combined_{args.palette}_{n_rows}x{n_cols}_{'_'.join(selected_metrics)}.png".replace(" ", "_")
    plt.tight_layout(pad=2.2)
    plt.savefig(os.path.join(args.outdir, combined_name))
    plt.close(fig)
    print(f"[✓] Saved combined panel figure ({n_rows}x{n_cols}): {combined_name}")


if __name__ == "__main__":
    main()
