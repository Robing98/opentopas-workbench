#!/usr/bin/env python3
"""Figures for the ripple filter kernel: filter shape -> thickness distribution, and the convolution itself.

  python3 rifi_shapes_figure.py shapes [--h 4] [--period 1.5] [--spr 1.16] [--out form_kern]
      four filter shapes of equal height (1D grooves, 2D pyramids, 2D pins with uniform distribution, 2D pins shaped
      for a Gaussian kernel): height map of 2 x 2 cells, cross-section, and kernel K(t) with mean, sigma, and
      modulation power. Writes form_kern.png and form_kern.pdf.
  python3 rifi_shapes_figure.py convolution WITHOUT WITH [--slices 8] [--out faltung]
      depth dose without filter, its shifted and weighted copies, their sum, and the curve with filter (TOPAS csv;
      the kernel is fitted as box first). Writes faltung.png and faltung.pdf.
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rifi_kernel_fit as kf  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def de(x, nd=2):
    """Number with a decimal comma (German text)."""
    return f"{x:.{nd}f}".replace(".", ",")


def comma_ticks(*axes):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for a in axes: a.xaxis.set_major_formatter(f); a.yaxis.set_major_formatter(f)


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 8.5, "axes.titlecolor": INK,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": False, "savefig.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def shape_height(kind, u, h, sigma):
    """Height above the base for the distance u (0 = pin axis, 1 = cell border) of each shape."""
    if kind == "ridge": return h * (1 - u)
    if kind == "pyramid": return h * (1 - u)
    if kind == "uniform": return h * (1 - u ** 2)
    from scipy.stats import truncnorm
    a_, b_ = (0 - h / 2) / sigma, (h / 2) / sigma
    return truncnorm.ppf(np.clip(1 - u ** 2, 1e-12, 1 - 1e-12), a_, b_, loc=h / 2, scale=sigma)


def shapes_figure(a):
    style()
    h, p, spr = a.h, a.period, a.spr
    sig = h / 5
    cols = [("ridge", "1D-Rillen (Dreieck)", 1), ("pyramid", "2D-Pyramiden", 2), ("uniform", "2D-Stifte, gleichverteilt", 2),
            ("gauss", "2D-Stifte für Gauß-Kern", 2)]
    fig, ax = plt.subplots(3, 4, figsize=(7.2, 5.9), gridspec_kw={"height_ratios": [1.15, 0.8, 1.0]})
    n = 240; x = (np.arange(2 * n) + 0.5) / n * p            # 2 x 2 cells
    X, Y = np.meshgrid(x, x)
    cx, cy = (X % p) - p / 2, (Y % p) - p / 2
    kmax = 0
    for j, (kind, title, dim) in enumerate(cols):
        u = np.abs(cx) / (p / 2) if dim == 1 else np.maximum(np.abs(cx), np.abs(cy)) / (p / 2)
        H = shape_height(kind, np.clip(u, 0, 1), h, sig)
        im = ax[0, j].imshow(H, cmap=BLUES, vmin=0, vmax=h, origin="lower", extent=[0, 2 * p, 0, 2 * p], interpolation="bilinear")
        ax[0, j].set_title(title, pad=6); ax[0, j].set_xticks([0, p, 2 * p]); ax[0, j].set_yticks([0, p, 2 * p])
        ax[0, j].tick_params(labelsize=7)
        if j == 0: ax[0, j].set_ylabel("y [mm]")
        ax[0, j].set_xlabel("x [mm]")
        uu = np.linspace(-1, 1, 801)
        ax[1, j].fill_between((uu + 1) * p / 2, 0, shape_height(kind, np.abs(uu), h, sig), color="#9ec5f4", lw=0)
        ax[1, j].plot((uu + 1) * p / 2, shape_height(kind, np.abs(uu), h, sig), color=SERIES[0], lw=1.6)
        ax[1, j].set_xlim(0, p); ax[1, j].set_ylim(0, h * 1.08); ax[1, j].set_xlabel("x durch die Zellmitte [mm]")
        if j == 0: ax[1, j].set_ylabel("Höhe [mm PMMA]")
        s = kf.profile_samples("pyramid" if kind == "ridge" else kind, h, 0.0, sigma=sig, dim=dim) * spr
        t, w, m, sd = kf.hist_kernel(s, step=0.1)
        ax[2, j].bar(t, w / 0.1, width=0.09, color=SERIES[0], lw=0)
        ax[2, j].plot([m, m], [0, (w / 0.1).max() * 1.08], color=INK2, lw=1)
        ax[2, j].text(0.97, 0.97, f"t̄ = {de(m)} mm\nσ_t = {de(sd)} mm\nσ_t²/t̄ = {de(sd ** 2 / m)} mm", transform=ax[2, j].transAxes,
                      ha="right", va="top", fontsize=7, color=INK, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))
        ax[2, j].set_xlim(0, h * spr * 1.02); ax[2, j].set_xlabel("Dicke t [mm Wasser]")
        ax[2, j].grid(axis="y"); ax[2, j].set_axisbelow(True)
        if j == 0: ax[2, j].set_ylabel("Anteil pro mm")
        kmax = max(kmax, (w / 0.1).max())
    for j in range(4): ax[2, j].set_ylim(0, kmax * 1.75)
    comma_ticks(*ax.ravel())
    fig.suptitle(f"Filterform und Dickenverteilung: Höhe {de(h, 1)} mm PMMA, Periode {de(p, 1)} mm "
                 f"(Wasseräquivalent: PMMA × {de(spr)})", fontsize=9, color=INK)
    fig.subplots_adjust(left=0.08, right=0.89, top=0.9, bottom=0.08, hspace=0.62, wspace=0.42)
    b = ax[0, 3].get_position()
    cax = fig.add_axes([0.905, b.y0, 0.012, b.height])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Höhe [mm PMMA]", color=INK2); cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    for ext in ("png", "pdf"): fig.savefig(f"{a.out}.{ext}", dpi=220)
    print(f"-> {a.out}.png, {a.out}.pdf")


def convolution_figure(a):
    style()
    z0, d0, _ = kf.read_curve(a.without); z1, d1, i1 = kf.read_curve(a.with_)
    d0, d1 = d0 / d0.max(), d1 / d1.max(); d0s = kf.smooth(d0)
    m0, m1 = kf.metrics(z0, d0s), kf.metrics(z1, kf.smooth(d1))
    p, e, rms = kf.fit_param(z0, d0s, z1, d1, "box", [1.0, max(m0["R80"] - m1["R80"] - 0.3, 0.1), 1.0])
    A, tm, sg = p; hw = np.sqrt(3) * sg
    ts = tm - hw + 2 * hw * (np.arange(a.slices) + 0.5) / a.slices
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    lo, hi = m0["peak"] - 22, m0["R05"] + 4
    sel0 = (z0 >= lo - 6) & (z0 <= hi)
    for k, t in enumerate(ts):
        ax.plot(z0[sel0], A / a.slices * kf.shifted(z0[sel0], z0, d0s, t), color="#86b6ef", lw=0.9, label="verschobene Kopien, je 1/%d" % a.slices if k == 0 else None)
    ax.plot(z0[sel0], d0s[sel0], color=MUTED, lw=1.4, label="ohne RiFi")
    zz = z1[(z1 >= lo) & (z1 <= hi)]
    ax.plot(zz, kf.model(zz, z0, d0s, A, tm, sg, "box"), color=SERIES[0], lw=2, label=f"Summe = Faltung (Kern t̄ {de(tm)} mm, σ_t {de(sg)} mm)")
    sel1 = (z1 >= lo) & (z1 <= hi)
    ax.plot(z1[sel1][::4], d1[sel1][::4], "o", ms=2.6, color=INK, label="mit RiFi (TOPAS)")
    E = i1.get("energy")
    ax.set_xlim(lo, hi); ax.set_ylim(0, 1.08); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.set_xlabel("Tiefe in Wasser [mm]"); ax.set_ylabel("Dosis (relativ zum Maximum)")
    ax.set_title((f"Protonen {de(E)} MeV: " if E else "") + "Tiefendosis mit RiFi als Mittel verschobener Kopien", color=INK)
    comma_ticks(ax)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(f"{a.out}.{ext}", dpi=220)
    print(f"-> {a.out}.png, {a.out}.pdf (fit rms {rms:.2f} %)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shapes"); s.add_argument("--h", type=float, default=4.0); s.add_argument("--period", type=float, default=1.5)
    s.add_argument("--spr", type=float, default=1.16); s.add_argument("--out", default="form_kern")
    c = sub.add_parser("convolution"); c.add_argument("without"); c.add_argument("with_", metavar="with")
    c.add_argument("--slices", type=int, default=8); c.add_argument("--out", default="faltung")
    a = ap.parse_args()
    shapes_figure(a) if a.cmd == "shapes" else convolution_figure(a)


if __name__ == "__main__":
    main()
