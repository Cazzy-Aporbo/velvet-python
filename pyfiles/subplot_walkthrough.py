"""
subplot_walkthrough.py

Subplot Interface in Matplotlib 
I start with simple subplots, then layer on correlation heatmaps, and finish
with a compact Seaborn tour. I explain choices in plain language and keep
every visual labeled, readable, and consistent.

CITATIONS (documentation I reference while coding):
- Matplotlib subplots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
- Matplotlib imshow (for heatmaps & backgrounds): https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
- Matplotlib event handling (for simple interactivity): https://matplotlib.org/stable/users/explain/event_handling.html
- Pandas corr: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
- Seaborn user guide: https://seaborn.pydata.org/introduction.html
- Seaborn relplot: https://seaborn.pydata.org/generated/seaborn.relplot.html
- Seaborn lineplot: https://seaborn.pydata.org/generated/seaborn.lineplot.html
"""

# ===== Standard Library =====
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

# ===== Third-party =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Seaborn is used in the second half; I keep it optional if you prefer pure Matplotlib.
try:
    import seaborn as sns
    _seaborn_available_cazzy = True
except Exception:
    _seaborn_available_cazzy = False


# ------------------------------------------------------------------------------
# A) Pastel ombré theme (once), and a tiny utility for gradient backgrounds
# ------------------------------------------------------------------------------
def ombre_palette_core_cazzy() -> Dict[str, str]:
    """My house palette for the lesson: pastel pink → lavender → mint → blue."""
    return {
        "ombre_pink_hex": "#FFD6E8",
        "ombre_lavender_hex": "#E6CCFF",
        "ombre_mint_hex": "#D4FFE4",
        "ombre_blue_hex": "#A6D8FF",
        "ombre_purple_text_hex": "#6B5B95",
        "ombre_grey_text_hex": "#444444",
        "ombre_border_hex": "#EADFF7",
    }


def ombre_apply_rcparams_cazzy() -> None:
    """Set Matplotlib defaults for a quiet, readable pastel aesthetic."""
    pastel = ombre_palette_core_cazzy()
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "axes.edgecolor": pastel["ombre_border_hex"],
        "axes.labelcolor": pastel["ombre_grey_text_hex"],
        "axes.titlecolor": pastel["ombre_purple_text_hex"],
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.color": "#CFC9E8",
        "xtick.color": pastel["ombre_grey_text_hex"],
        "ytick.color": pastel["ombre_grey_text_hex"],
        "font.family": "DejaVu Sans",
        "legend.frameon": False,
        "savefig.bbox": "tight"
    })


def ombre_axes_background_gradient_cazzy(ax_handle, color_sequence: Tuple[str, str, str, str]) -> None:
    """
    Paint a subtle left→right gradient behind the plotting area using imshow.
    This stays *behind* data (zorder low) and keeps labels readable.
    """
    # Build a 1x256 gradient from left→right
    gradient_canvas = np.linspace(0, 1, 256)[None, :]
    # Create a 256×4 RGBA from the requested colors
    # We'll fade from color A to B to C to D by blending quarters.
    from matplotlib.colors import to_rgba, to_rgb
    cA, cB, cC, cD = map(to_rgb, color_sequence)

    # compute 256 colors via piecewise linear blend
    def blend_col(c0, c1, w):
        return (1 - w) * np.array(c0) + w * np.array(c1)

    palette_256 = np.zeros((256, 3))
    # first 128 blend A→B, next 64 blend B→C, last 64 blend C→D
    for i in range(256):
        if i < 128:
            w = i / 127.0
            palette_256[i] = blend_col(cA, cB, w)
        elif i < 192:
            w = (i - 128) / 63.0
            palette_256[i] = blend_col(cB, cC, w)
        else:
            w = (i - 192) / 63.0
            palette_256[i] = blend_col(cC, cD, w)

    # create an image (1 x 256 x 3) expanded in Y so it fills the axes
    gradient_img = np.repeat(gradient_canvas, repeats=5, axis=0)
    ax_handle.imshow(
        gradient_img,
        aspect="auto",
        cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
            "ombre_linear_cazzy", palette_256
        ),
        interpolation="bilinear",
        zorder=0,
        extent=[*ax_handle.get_xlim(), *ax_handle.get_ylim()],
        origin="lower",
        alpha=0.35,
    )
    # After plotting data, we’ll re-tighten the extent to current limits.


# ------------------------------------------------------------------------------
# B) Subplot interface — beginner to intermediate (multiple panels in one figure)
# ------------------------------------------------------------------------------
def subplot_quadrants_ensemble_cazzy() -> None:
    """
    I build a 2×2 grid:
      [0,0] line: sum of two sine waves,
      [0,1] scatter: samples from the sum (with a tiny hover),
      [1,0] line: the individual components with legend,
      [1,1] unused → removed for a clean layout.

    Everything is labeled; background has a quiet gradient.
    """
    ombre_apply_rcparams_cazzy()
    pastel = ombre_palette_core_cazzy()

    # Data construction (names deliberately unique and descriptive)
    gridline_theta_bridge = np.linspace(0, 2 * np.pi, 200)
    signal_primary_rosa = np.sin(gridline_theta_bridge)
    signal_harmonic_violet = np.sin(2 * gridline_theta_bridge)
    signal_mixture_mint = signal_primary_rosa + signal_harmonic_violet

    # Figure + axes array
    figure_tile_symphony, axes_matrix_tiles = plt.subplots(2, 2, figsize=(11, 7))
    # --- Panel A: Sum line ---
    axes_matrix_tiles[0, 0].plot(
        gridline_theta_bridge, signal_mixture_mint,
        color=pastel["ombre_purple_text_hex"], linewidth=2.0, label="y₁ + y₂"
    )
    axes_matrix_tiles[0, 0].set_title("Sum of Sine Waves (Line)", loc="left")
    axes_matrix_tiles[0, 0].set_xlabel("Angle (radians)")
    axes_matrix_tiles[0, 0].set_ylabel("Amplitude")
    axes_matrix_tiles[0, 0].legend(loc="upper right")

    ombre_axes_background_gradient_cazzy(
        axes_matrix_tiles[0, 0],
        (pastel["ombre_pink_hex"], pastel["ombre_lavender_hex"], pastel["ombre_mint_hex"], pastel["ombre_blue_hex"])
    )

    # --- Panel B: Scatter + hover annotation (simple built-in event) ---
    scatter_points = axes_matrix_tiles[0, 1].scatter(
        gridline_theta_bridge, signal_mixture_mint,
        s=18, c=pastel["ombre_blue_hex"], edgecolors="white", linewidths=0.5
    )
    axes_matrix_tiles[0, 1].set_title("Sum of Sine Waves (Scatter)", loc="left")
    axes_matrix_tiles[0, 1].set_xlabel("Angle (radians)")
    axes_matrix_tiles[0, 1].set_ylabel("Amplitude")

    ombre_axes_background_gradient_cazzy(
        axes_matrix_tiles[0, 1],
        (pastel["ombre_lavender_hex"], pastel["ombre_mint_hex"], pastel["ombre_blue_hex"], pastel["ombre_pink_hex"])
    )

    # Lightweight hover: annotate nearest point under cursor.
    hover_note_widget_cazzy = axes_matrix_tiles[0, 1].annotate(
        "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.25", fc=pastel["ombre_mint_hex"], ec=pastel["ombre_border_hex"], alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=pastel["ombre_purple_text_hex"]),
        fontsize=9, color=pastel["ombre_grey_text_hex"], visible=False
    )

    def on_move_show_value_cazzy(evt):
        if evt.inaxes != axes_matrix_tiles[0, 1]:
            hover_note_widget_cazzy.set_visible(False)
            figure_tile_symphony.canvas.draw_idle()
            return
        # Find nearest point (simple search; fine for 200 points)
        xdata = gridline_theta_bridge
        ydata = signal_mixture_mint
        if xdata.size == 0:
            return
        nearest_idx = np.argmin(np.abs(xdata - evt.xdata))
        nearest_x, nearest_y = xdata[nearest_idx], ydata[nearest_idx]
        hover_note_widget_cazzy.xy = (nearest_x, nearest_y)
        hover_note_widget_cazzy.set_text(f"x={nearest_x:.2f}\ny={nearest_y:.2f}")
        hover_note_widget_cazzy.set_visible(True)
        figure_tile_symphony.canvas.draw_idle()

    figure_tile_symphony.canvas.mpl_connect("motion_notify_event", on_move_show_value_cazzy)

    # --- Panel C: Components with legend ---
    axes_matrix_tiles[1, 0].plot(
        gridline_theta_bridge, signal_primary_rosa,
        color=pastel["ombre_pink_hex"], linewidth=2.0, label="y₁ = sin(x)"
    )
    axes_matrix_tiles[1, 0].plot(
        gridline_theta_bridge, signal_harmonic_violet,
        color=pastel["ombre_lavender_hex"], linewidth=2.0, label="y₂ = sin(2x)"
    )
    axes_matrix_tiles[1, 0].set_title("Individual Sine Waves", loc="left")
    axes_matrix_tiles[1, 0].set_xlabel("Angle (radians)")
    axes_matrix_tiles[1, 0].set_ylabel("Amplitude")
    axes_matrix_tiles[1, 0].legend(loc="upper right")

    ombre_axes_background_gradient_cazzy(
        axes_matrix_tiles[1, 0],
        (pastel["ombre_blue_hex"], pastel["ombre_pink_hex"], pastel["ombre_lavender_hex"], pastel["ombre_mint_hex"])
    )

    # --- Panel D: removed for clean layout ---
    figure_tile_symphony.delaxes(axes_matrix_tiles[1, 1])

    figure_tile_symphony.suptitle("Matplotlib Subplot Interface — A Pastel Ombré Tour", color=pastel["ombre_purple_text_hex"])
    figure_tile_symphony.tight_layout()
    figure_tile_symphony.savefig("subplot_quadrants_pastel_demo.png")
    plt.close(figure_tile_symphony)
    print("[saved] subplot_quadrants_pastel_demo.png")


# ------------------------------------------------------------------------------
# C) Correlation heatmap (Matplotlib): abalone dataset from UCI (with fallback)
# ------------------------------------------------------------------------------
def correlation_abalone_exhibit_cazzy() -> None:
    """
    I calculate a correlation matrix and visualize it as a heatmap.
    If the UCI URL is not reachable, I synthesize a small numeric table with
    similar column names so the example always runs.
    """
    ombre_apply_rcparams_cazzy()
    pastel = ombre_palette_core_cazzy()

    # Attempt to fetch abalone dataset from UCI repository
    data_url_reference_cazzy = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    headers_reference_cazzy = ["Sex", "Length", "Diameter", "Height",
                               "WholeWeight", "ShuckedWeight", "VisceraWeight", "ShellWeight", "Rings"]

    try:
        frame_abalone_inspect_iris = pd.read_csv(data_url_reference_cazzy, names=headers_reference_cazzy)
    except Exception:
        # Fallback synthetic numeric frame if offline
        rng_seed_temp_cazzy = np.random.default_rng(7)
        frame_abalone_inspect_iris = pd.DataFrame({
            "Sex": rng_seed_temp_cazzy.choice(list("MFI"), size=300),
            "Length": rng_seed_temp_cazzy.uniform(0.1, 0.8, 300),
            "Diameter": rng_seed_temp_cazzy.uniform(0.1, 0.7, 300),
            "Height": rng_seed_temp_cazzy.uniform(0.01, 0.3, 300),
            "WholeWeight": rng_seed_temp_cazzy.uniform(0.1, 2.5, 300),
            "ShuckedWeight": rng_seed_temp_cazzy.uniform(0.05, 1.8, 300),
            "VisceraWeight": rng_seed_temp_cazzy.uniform(0.02, 1.0, 300),
            "ShellWeight": rng_seed_temp_cazzy.uniform(0.05, 1.2, 300),
            "Rings": rng_seed_temp_cazzy.integers(1, 30, 300)
        })

    # Keep numeric columns for correlation
    numeric_view_cazzy = frame_abalone_inspect_iris.select_dtypes(include=[np.number])
    matrix_correlation_shells = numeric_view_cazzy.corr()

    fig_corr_canvas_cazzy, ax_corr_view_cazzy = plt.subplots(figsize=(10, 8))
    im = ax_corr_view_cazzy.imshow(matrix_correlation_shells.values,
                                   cmap="PuRd",  # purple-red reads well with our theme
                                   interpolation="nearest")
    cbar = fig_corr_canvas_cazzy.colorbar(im, ax=ax_corr_view_cazzy, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", color=pastel["ombre_grey_text_hex"])

    ax_corr_view_cazzy.set_xticks(range(len(matrix_correlation_shells.columns)))
    ax_corr_view_cazzy.set_xticklabels(matrix_correlation_shells.columns, rotation=45, ha="right")
    ax_corr_view_cazzy.set_yticks(range(len(matrix_correlation_shells.columns)))
    ax_corr_view_cazzy.set_yticklabels(matrix_correlation_shells.columns)

    ax_corr_view_cazzy.set_title("Abalone Feature Correlations — Pastel Ombré Heatmap", loc="left")
    ax_corr_view_cazzy.set_xlabel("Features")
    ax_corr_view_cazzy.set_ylabel("Features")

    # Subtle background gradient
    ax_corr_view_cazzy.set_xlim(-0.5, len(matrix_correlation_shells.columns)-0.5)
    ax_corr_view_cazzy.set_ylim(len(matrix_correlation_shells.columns)-0.5, -0.5)
    ombre_axes_background_gradient_cazzy(
        ax_corr_view_cazzy,
        (pastel["ombre_pink_hex"], pastel["ombre_lavender_hex"], pastel["ombre_mint_hex"], pastel["ombre_blue_hex"])
    )

    fig_corr_canvas_cazzy.tight_layout()
    fig_corr_canvas_cazzy.savefig("correlation_pastel_heatmap.png")
    plt.close(fig_corr_canvas_cazzy)
    print("[saved] correlation_pastel_heatmap.png")


# ------------------------------------------------------------------------------
# D) Seaborn introduction (with graceful fallbacks), pastel consistent
# ------------------------------------------------------------------------------
def seaborn_relplot_gallery_cazzy() -> None:
    """
    I show a few compact Seaborn patterns: relplot for scatter, hue for categories,
    and simple time series. If seaborn or internet is unavailable, I synthesize data.
    """
    ombre_apply_rcparams_cazzy()
    pastel = ombre_palette_core_cazzy()

    if not _seaborn_available_cazzy:
        print("[info] seaborn not available; skipping seaborn demos.")
        return

    sns.set_theme(style="whitegrid")
    # A custom pastel palette for seaborn plots
    sns.set_palette(sns.color_palette([
        pastel["ombre_pink_hex"],
        pastel["ombre_lavender_hex"],
        pastel["ombre_mint_hex"],
        pastel["ombre_blue_hex"]
    ]))

    # --- Tips dataset relplot: total_bill vs tip ---
    try:
        dataset_tips_cazzy = sns.load_dataset("tips")
    except Exception:
        rng_tips_cazzy = np.random.default_rng(12)
        dataset_tips_cazzy = pd.DataFrame({
            "total_bill": rng_tips_cazzy.uniform(3, 60, 244),
            "tip": rng_tips_cazzy.uniform(1, 12, 244),
            "smoker": rng_tips_cazzy.choice(["Yes", "No"], size=244),
            "size": rng_tips_cazzy.integers(1, 6, 244)
        })

    g_scatter_basic_cazzy = sns.relplot(
        x="total_bill", y="tip", data=dataset_tips_cazzy, height=5, aspect=1.2
    )
    g_scatter_basic_cazzy.set_axis_labels("Total Bill", "Tip")
    g_scatter_basic_cazzy.fig.suptitle("Seaborn relplot — Basic Relationship (Pastel)", color=pastel["ombre_purple_text_hex"])
    g_scatter_basic_cazzy.fig.savefig("seaborn_relplot_basic_pastel.png", dpi=120)
    plt.close(g_scatter_basic_cazzy.fig)
    print("[saved] seaborn_relplot_basic_pastel.png")

    # --- Hue/style exploration: smoker as hue ---
    g_scatter_hue_cazzy = sns.relplot(
        x="total_bill", y="tip", hue="smoker", style="smoker",
        data=dataset_tips_cazzy, height=5, aspect=1.2
    )
    g_scatter_hue_cazzy.set_axis_labels("Total Bill", "Tip")
    g_scatter_hue_cazzy.fig.suptitle("Seaborn relplot — Hue & Style by Smoker (Pastel)", color=pastel["ombre_purple_text_hex"])
    g_scatter_hue_cazzy.fig.savefig("seaborn_relplot_hue_style_pastel.png", dpi=120)
    plt.close(g_scatter_hue_cazzy.fig)
    print("[saved] seaborn_relplot_hue_style_pastel.png")

    # --- Enhanced with size (party size) ---
    g_scatter_size_cazzy = sns.relplot(
        x="total_bill", y="tip", hue="smoker", style="smoker", size="size",
        data=dataset_tips_cazzy, height=5, aspect=1.2
    )
    g_scatter_size_cazzy.set_axis_labels("Total Bill", "Tip")
    g_scatter_size_cazzy.fig.suptitle("Seaborn relplot — Hue/Style/Size (Pastel)", color=pastel["ombre_purple_text_hex"])
    g_scatter_size_cazzy.fig.savefig("seaborn_relplot_hue_style_size_pastel.png", dpi=120)
    plt.close(g_scatter_size_cazzy.fig)
    print("[saved] seaborn_relplot_hue_style_size_pastel.png")

    # --- fmri time series ---
    try:
        dataset_fmri_cazzy = sns.load_dataset("fmri")
        g_fmri_line_cazzy = sns.relplot(
            x="timepoint", y="signal", hue="event", style="event", kind="line",
            data=dataset_fmri_cazzy, height=5, aspect=1.4
        )
        g_fmri_line_cazzy.set_axis_labels("Timepoint", "Signal")
        g_fmri_line_cazzy.fig.suptitle("Seaborn Lineplot — fmri (Pastel)", color=pastel["ombre_purple_text_hex"])
        g_fmri_line_cazzy.fig.savefig("seaborn_fmri_line_pastel.png", dpi=120)
        plt.close(g_fmri_line_cazzy.fig)
        print("[saved] seaborn_fmri_line_pastel.png")
    except Exception:
        # If fmri download fails, synthesize a simple multi-line time series
        synthetic_time_cazzy = np.arange(0, 60)
        synthetic_signal_A = np.sin(synthetic_time_cazzy / 6.0) + 0.05*np.random.randn(60)
        synthetic_signal_B = np.cos(synthetic_time_cazzy / 8.0) + 0.05*np.random.randn(60)
        fig_ts_fallback, ax_ts_fallback = plt.subplots(figsize=(10, 5))
        ax_ts_fallback.plot(synthetic_time_cazzy, synthetic_signal_A, label="event=A")
        ax_ts_fallback.plot(synthetic_time_cazzy, synthetic_signal_B, label="event=B")
        ax_ts_fallback.set_title("Synthetic Time Series (fmri fallback) — Pastel", loc="left")
        ax_ts_fallback.set_xlabel("Time")
        ax_ts_fallback.set_ylabel("Signal")
        ax_ts_fallback.legend()
        fig_ts_fallback.savefig("seaborn_fmri_fallback_lines_pastel.png", dpi=120)
        plt.close(fig_ts_fallback)
        print("[saved] seaborn_fmri_fallback_lines_pastel.png")

    # --- Autoformatted dates example ---
    # I generate a local time series and show how Matplotlib handles the tick formatting.
    date_index_construct_cazzy = pd.date_range("2022-01-01", "2022-04-01", freq="1D")
    value_random_walk_cazzy = np.cumsum(np.random.default_rng(11).normal(0, 1, len(date_index_construct_cazzy)))
    frame_time_autofmt_cazzy = pd.DataFrame({"date": date_index_construct_cazzy, "value": value_random_walk_cazzy})

    fig_auto_date_cazzy, ax_auto_date_cazzy = plt.subplots(figsize=(10, 5))
    if _seaborn_available_cazzy:
        sns.lineplot(x="date", y="value", data=frame_time_autofmt_cazzy, ax=ax_auto_date_cazzy)
    else:
        ax_auto_date_cazzy.plot(frame_time_autofmt_cazzy["date"], frame_time_autofmt_cazzy["value"])

    ax_auto_date_cazzy.set_title("Time Series with Autoformatted Dates — Pastel", loc="left")
    ax_auto_date_cazzy.set_xlabel("Date")
    ax_auto_date_cazzy.set_ylabel("Value")
    fig_auto_date_cazzy.autofmt_xdate()
    fig_auto_date_cazzy.savefig("seaborn_time_autodates_pastel.png", dpi=120)
    plt.close(fig_auto_date_cazzy)
    print("[saved] seaborn_time_autodates_pastel.png")


# ------------------------------------------------------------------------------
# Main runner — I keep it explicit, so you can open each saved PNG in order.
# ------------------------------------------------------------------------------
def main_driver_cazzy() -> None:
    subplot_quadrants_ensemble_cazzy()
    correlation_abalone_exhibit_cazzy()
    seaborn_relplot_gallery_cazzy()
    print("\nAll figures saved. Open the PNGs in this folder to review the outputs.\n")


if __name__ == "__main__":
    main_driver_cazzy()
