# -*- coding: utf-8 -*-
"""
Matplotlib Masterclass — Interactive How-To (advanced, aesthetic, didactic)
Author: Cazzy Aporbo  •  made September 16, 2025

What this does
--------------
• Builds a synthetic dataset and lets you explore it via interactive widgets:
  plot type (line / scatter / bar / hist / box / heatmap), style (primary / mono / pastel ombré),
  grid/legend/spines toggles, title editor, and data sliders (freq, noise, n).
• Demonstrates expert Matplotlib craft: rcParams hygiene, style contexts, ombré line coloring,
  inset zoom, secondary axis, minor ticks, formatters, pick events (click points), annotations,
  layout with constrained_layout, high-DPI export to PNG/SVG/PDF.
• Side-panel “Notebook Notes” explaining which core Matplotlib calls you’re exercising (mapped
  to your bullet list).

How to run
----------
$ pip install matplotlib numpy
$ python matplotlib_masterclass_interactive.py

Tip: If you’re in a headless environment, the script falls back to saving figures.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, TextBox, RangeSlider
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

# --------------------------- Global knobs (safe defaults) ---------------------------

DEFAULT_DPI = 140
FIGSIZE = (10.5, 7.2)
SEED = 7

# High-DPI sanity
mpl.rcParams.update({
    "figure.dpi": DEFAULT_DPI,
    "savefig.dpi": 300,
    "axes.titleweight": "semibold",
    "axes.labelweight": "regular",
    "axes.grid": False,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "font.size": 11,
})

# Style palettes (primary, mono, pastel ombré)
PALETTES = {
    "primary": dict(line="#1f77b4", scatter="#d62728", bar="#2ca02c",
                    heatmap="viridis", patch="#9467bd"),
    "mono":    dict(line="#111111", scatter="#222222", bar="#333333",
                    heatmap="Greys",   patch="#000000"),
    "pastel":  dict(line="#a1c9f4", scatter="#ffb3e6", bar="#cdeccd",
                    heatmap="magma",   patch="#f9c7d0"),
}

# --------------------------- Data model --------------------------------------------

@dataclass
class DataBundle:
    x: np.ndarray           # 1D x
    y: np.ndarray           # 1D y (signal + noise)
    cat_labels: np.ndarray  # categories for bar/pie demo
    cat_vals: np.ndarray    # values per category
    grid_Z: np.ndarray      # 2D field for imshow/heatmap
    grid_extent: Tuple[float, float, float, float]

def synthesize(n: int = 300, f_hz: float = 1.25, noise: float = 0.35, seed: int = SEED) -> DataBundle:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    trend = 0.15 * x
    y_clean = np.sin(2*np.pi*f_hz * x) + trend
    y = y_clean + rng.normal(0, noise, size=n)

    # categories (for bar/box/hist demos)
    cats = np.array(["Alpha", "Beta", "Gamma", "Delta", "Epsilon"])
    cat_vals = (np.abs(rng.normal(1.0, 0.4, size=cats.size)) * 10).round(1)

    # 2D field: two smooth bumps + noise
    gx = np.linspace(-3, 3, 160)
    gy = np.linspace(-2, 2, 120)
    X, Y = np.meshgrid(gx, gy)
    Z = (np.exp(-((X-1.0)**2 + (Y-0.6)**2))
         + 0.75*np.exp(-((X+1.2)**2 + (Y+0.3)**2))
         + 0.08*rng.standard_normal(X.shape))
    extent = (gx.min(), gx.max(), gy.min(), gy.max())

    return DataBundle(x=x, y=y, cat_labels=cats, cat_vals=cat_vals, grid_Z=Z, grid_extent=extent)

# --------------------------- Helper: ombré line -------------------------------------

def draw_ombre_line(ax, x, y, cmap_name="viridis", linewidth=2.4):
    """Color a line by progression using a LineCollection (advanced aesthetic)."""
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=plt.get_cmap(cmap_name), linewidth=linewidth, alpha=0.95)
    t = np.linspace(0, 1, len(x)-1)
    lc.set_array(t)
    ax.add_collection(lc)
    ax.plot([], [])  # keep autoscale happy
    ax.set_xlim(x.min(), x.max())
    ypad = 0.05*(y.max()-y.min() if y.ptp()>0 else 1)
    ax.set_ylim(y.min()-ypad, y.max()+ypad)
    return lc

# --------------------------- Interactive application -------------------------------

class MPLPlayground:
    def __init__(self):
        self.data = synthesize()
        self.style_key = "primary"
        self.plot_mode = "line"
        self.show_grid = True
        self.show_legend = True
        self.show_spines = True

        # Figure with right-hand control strip
        self.fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
        self.gs = self.fig.add_gridspec(ncols=4, nrows=3, width_ratios=[12, 0.6, 4.2, 0.6],
                                        height_ratios=[10, 1.6, 1.2])
        self.ax = self.fig.add_subplot(self.gs[0, 0])
        self.ax_inset = self.ax.inset_axes([0.6, 0.58, 0.35, 0.35])
        self.ax_right = self.fig.add_subplot(self.gs[0, 2])
        self.fig.canvas.manager.set_window_title("Matplotlib Masterclass — Cazzy A.")

        # Widgets row
        self._build_widgets()

        # Teaching notes
        self._init_notes()

        # First draw
        self._redraw_full()

        # Pick events (click to annotate a nearest point)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    # ----------------------- UI construction -----------------------------------

    def _build_widgets(self):
        # Sliders under the main plot
        ax_freq = self.fig.add_subplot(self.gs[1, 0])
        ax_noise = self.fig.add_subplot(self.gs[2, 0])
        ax_n = self.fig.add_subplot(self.gs[2, 2])

        self.s_freq = Slider(ax_freq, "Frequency", 0.2, 3.0, valinit=1.25, valstep=0.05)
        self.s_noise = Slider(ax_noise, "Noise σ", 0.0, 1.2, valinit=0.35, valstep=0.05)
        self.s_n = Slider(ax_n, "N points", 50, 1000, valinit=300, valstep=10)

        self.s_freq.on_changed(self._on_data_change)
        self.s_noise.on_changed(self._on_data_change)
        self.s_n.on_changed(self._on_data_change)

        # Right controls: radios + checks + text + export buttons
        self.ax_right.axis("off")
        r1 = self.ax_right.inset_axes([0.05, 0.70, 0.9, 0.26])
        r2 = self.ax_right.inset_axes([0.05, 0.44, 0.9, 0.22])
        c1 = self.ax_right.inset_axes([0.05, 0.29, 0.9, 0.12])
        t1 = self.ax_right.inset_axes([0.05, 0.18, 0.9, 0.08])
        b_row = self.ax_right.inset_axes([0.05, 0.04, 0.9, 0.10])

        self.r_plot = RadioButtons(r1, ("line", "scatter", "bar", "hist", "box", "heatmap"))
        self.r_style = RadioButtons(r2, ("primary", "mono", "pastel"))
        self.chk = CheckButtons(c1, ("grid", "legend", "spines"), (True, True, True))
        self.txt = TextBox(t1, "Title", initial="Matplotlib Masterclass — interactive")

        self.r_plot.on_clicked(self._on_plot_mode)
        self.r_style.on_clicked(self._on_style)
        self.chk.on_clicked(self._on_checks)
        self.txt.on_submit(self._on_title)

        # Export buttons
        b_row.axis("off")
        bx_png = b_row.inset_axes([0.00, 0.00, 0.30, 1.00])
        bx_svg = b_row.inset_axes([0.35, 0.00, 0.30, 1.00])
        bx_pdf = b_row.inset_axes([0.70, 0.00, 0.30, 1.00])
        self.btn_png = Button(bx_png, "Save PNG")
        self.btn_svg = Button(bx_svg, "Save SVG")
        self.btn_pdf = Button(bx_pdf, "Save PDF")
        self.btn_png.on_clicked(lambda evt: self._export("png"))
        self.btn_svg.on_clicked(lambda evt: self._export("svg"))
        self.btn_pdf.on_clicked(lambda evt: self._export("pdf"))

    def _init_notes(self):
        self.notes = self.fig.add_subplot(self.gs[1:, 2])
        self.notes.axis("off")
        self._update_notes()

    # ----------------------- Event handlers -----------------------------------

    def _on_data_change(self, _):
        n = int(self.s_n.val)
        f = float(self.s_freq.val)
        s = float(self.s_noise.val)
        self.data = synthesize(n=n, f_hz=f, noise=s, seed=SEED)
        self._redraw_plot()

    def _on_plot_mode(self, label):
        self.plot_mode = label
        self._redraw_plot()

    def _on_style(self, label):
        self.style_key = label
        self._redraw_full()

    def _on_checks(self, _label):
        self.show_grid, self.show_legend, self.show_spines = self.chk.get_status()
        self._redraw_plot()

    def _on_title(self, text):
        self.ax.set_title(text)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        # Only annotate when clicking inside main axes on scatter/line
        if event.inaxes != self.ax or self.plot_mode not in ("scatter", "line"):
            return
        # Find nearest point (quick and robust)
        x, y = self.data.x, self.data.y
        d2 = (x - event.xdata)**2 + (y - event.ydata)**2
        i = int(np.argmin(d2))
        self._annotate_point(i)

    # ----------------------- Rendering ----------------------------------------

    def _apply_style(self):
        # Clean rcParams that affect colors/axes each redraw
        mpl.rcParams.update({
            "axes.grid": self.show_grid,
            "axes.spines.right": self.show_spines,
            "axes.spines.top": self.show_spines,
            "axes.spines.left": self.show_spines,
            "axes.spines.bottom": self.show_spines,
        })

    def _clear_axes(self):
        self.ax.clear()
        self.ax_inset.clear()

    def _base_axes_decor(self):
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.xaxis.set_minor_locator(AutoMinorLocator())
        self.ax.yaxis.set_minor_locator(AutoMinorLocator())
        self.ax.grid(self.show_grid, which="major", alpha=0.25)
        self.ax.grid(self.show_grid, which="minor", alpha=0.12)
        # Secondary y to show normalized scale (pedagogical)
        try:
            self.ax.secondary_yaxis("right",
                functions=(lambda v: (v - np.nanmean(self.data.y)) / (np.nanstd(self.data.y) + 1e-9),
                           lambda v: v * (np.nanstd(self.data.y) + 1e-9) + np.nanmean(self.data.y))
            ).set_ylabel("z-score")
        except Exception:
            pass

    def _draw_line(self, pal):
        # Ombré line
        lc = draw_ombre_line(self.ax, self.data.x, self.data.y, cmap_name=pal["heatmap"])
        self.ax.set_title("Line plot with ombré coloring (LineCollection)")
        self._add_inset_zoom()
        return [lc]

    def _draw_scatter(self, pal):
        sc = self.ax.scatter(self.data.x, self.data.y, s=28, alpha=0.85,
                             edgecolor="white", linewidth=0.6, c=pal["scatter"])
        self.ax.set_title("Scatter with outlier highlighting (click a point)")
        self._highlight_outliers(sc, pal)
        self._add_inset_zoom()
        return [sc]

    def _draw_bar(self, pal):
        cats = self.data.cat_labels
        vals = self.data.cat_vals
        bars = self.ax.bar(cats, vals, color=pal["bar"], alpha=0.9)
        self.ax.set_ylabel("value")
        self.ax.set_title("Bar chart (categorical)")
        # Annotate bars
        for b in bars:
            self.ax.annotate(f"{b.get_height():.1f}",
                             xy=(b.get_x()+b.get_width()/2, b.get_height()),
                             xytext=(0, 5), textcoords="offset points",
                             ha="center", va="bottom",
                             path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        return list(bars)

    def _draw_hist(self, pal):
        n, bins, patches = self.ax.hist(self.data.y, bins=30, alpha=0.9, color=pal["bar"], edgecolor="white", linewidth=0.5)
        self.ax.set_title("Histogram with density curve")
        # Density curve
        ys = self.data.y
        xs = np.linspace(np.nanmin(ys), np.nanmax(ys), 400)
        mu, sigma = float(np.nanmean(ys)), float(np.nanstd(ys) + 1e-9)
        pdf = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(xs-mu)**2/(2*sigma**2))
        scale = np.trapz(n, bins[:-1]) / np.trapz(pdf, xs)
        self.ax.plot(xs, pdf*scale, lw=2.5, color=pal["line"])
        self.ax.axvline(mu, ls="--", lw=1.5, color=pal["line"], alpha=0.8, label="mean")
        return list(patches)

    def _draw_box(self, pal):
        b = self.ax.boxplot([self.data.y], vert=True, patch_artist=True,
                            boxprops=dict(facecolor=pal["bar"], alpha=0.75),
                            medianprops=dict(color="white", linewidth=1.6))
        self.ax.set_xticklabels(["y distribution"])
        self.ax.set_title("Box plot")
        return [b["boxes"][0]]

    def _draw_heatmap(self, pal):
        im = self.ax.imshow(self.data.grid_Z, cmap=pal["heatmap"],
                            extent=self.data.grid_extent, origin="lower", aspect="auto")
        self.ax.set_title("Heatmap (imshow)")
        cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("intensity")
        return [im]

    def _add_inset_zoom(self):
        # Zoom into the region of maximum variation
        y = self.data.y
        x = self.data.x
        if x.size < 12:
            return
        i0 = max(int(np.argmax(np.abs(np.gradient(y)))) - 15, 0)
        i1 = min(i0 + 50, len(x)-1)
        self.ax_inset.plot(x[i0:i1], y[i0:i1], lw=2.0, color="#444444")
        self.ax_inset.set_title("Inset zoom", fontsize=9)
        self.ax_inset.grid(self.show_grid, alpha=0.2)
        # Rectangle on main axes
        rect = Rectangle((x[i0], min(y[i0:i1])), x[i1]-x[i0], max(y[i0:i1])-min(y[i0:i1]),
                         fill=False, ec="#999999", lw=1.0, ls="--", alpha=0.8)
        self.ax.add_patch(rect)

    def _highlight_outliers(self, sc, pal):
        y = self.data.y
        z = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-9)
        mask = np.abs(z) > 2.25
        if mask.any():
            self.ax.scatter(self.data.x[mask], self.data.y[mask], s=90,
                            facecolor="none", edgecolor=pal["patch"], linewidth=1.8,
                            label="outliers")

    def _annotate_point(self, i):
        x, y = float(self.data.x[i]), float(self.data.y[i])
        txt = self.ax.annotate(f"({x:.2f}, {y:.2f})", xy=(x, y),
                               xytext=(10, 12), textcoords="offset points",
                               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#333333", alpha=0.9),
                               arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])
        self.fig.canvas.draw_idle()

    def _redraw_full(self):
        self._apply_style()
        self._redraw_plot()
        self._update_notes()

    def _redraw_plot(self):
        pal = PALETTES[self.style_key]
        self._clear_axes()
        artists = []
        if self.plot_mode == "line":
            artists = self._draw_line(pal)
        elif self.plot_mode == "scatter":
            artists = self._draw_scatter(pal)
        elif self.plot_mode == "bar":
            artists = self._draw_bar(pal)
        elif self.plot_mode == "hist":
            artists = self._draw_hist(pal)
        elif self.plot_mode == "box":
            artists = self._draw_box(pal)
        elif self.plot_mode == "heatmap":
            artists = self._draw_heatmap(pal)

        self._base_axes_decor()
        if self.show_legend:
            self.ax.legend(loc="best")
        self.fig.canvas.draw_idle()

    def _export(self, ext: str):
        fname = f"mpl_masterclass_export.{ext}"
        self.fig.savefig(fname, bbox_inches="tight", metadata={
            "Title": "Matplotlib Masterclass — Cazzy Aporbo",
            "Author": "Cazzy Aporbo",
            "Subject": f"Interactive {self.plot_mode} demo in {self.style_key} style"
        })
        print(f"[saved] {fname}")

    def _update_notes(self):
        self.notes.cla()
        self.notes.axis("off")
        lines = [
            "Notebook Notes (mapping to your API list):",
            " • plt.figure / plt.subplots  → figure & axes; constrained_layout",
            " • ax.plot / ax.scatter / ax.bar / ax.hist / ax.boxplot / ax.imshow",
            " • ax.set_xlabel / ax.set_ylabel / ax.set_title / ax.legend / ax.grid",
            " • ax.set_xlim / set_ylim / tick formatters / minor ticks",
            " • widgets: Slider, RadioButtons, CheckButtons, TextBox, Button",
            " • advanced: LineCollection (ombré), inset_axes, secondary_yaxis",
            " • savefig(..., dpi, bbox_inches='tight', metadata=...)",
            " • rcParams hygiene; style palettes; pick events for annotations",
        ]
        self.notes.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace")

# --------------------------- Entrypoint --------------------------------------------

def main():
    # Headless safety: fall back to Agg and just render files
    headless = False
    try:
        plt.get_current_fig_manager()
    except Exception:
        headless = True
        mpl.use("Agg")

    app = MPLPlayground()
    if headless:
        # Render one of each then exit
        for mode in ("line", "scatter", "bar", "hist", "box", "heatmap"):
            app.plot_mode = mode
            app._redraw_plot()
            app._export(f"{mode}.png")
        print("[headless] Saved examples for all modes. Bye.")
        return 0
    else:
        plt.show()
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
