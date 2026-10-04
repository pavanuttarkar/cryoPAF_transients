#!/usr/bin/env python3
"""
Interactive FREDDA (FDMT) candidate plotter.

FREDDA writes candidate text files (*.cand / *.fil.cand) with columns:
    S/N, sampno, secs_from_start, boxcar, idt, dm, beamno, mjd
Header ('#') lines are parsed if they name the columns; otherwise that
default order is assumed.

Usage:
    python fredda_plotter.py candidates.fil.cand [more files...]
    python fredda_plotter.py --demo

Controls:
    left click        inspect the brightest candidate under the cursor
    right-drag        box-select candidates (time / DM)
    toolbar           normal zoom / pan (left click inspects only when no tool is active)
    sliders           min S/N and DM range
    radio / checks    pick a beam, toggle boxcar widths
    keys              e = export selected (or all visible) to CSV
                      x = clear selection
                      a = reset all filters
                      l = toggle log DM axis (matplotlib default)

Requires matplotlib >= 3.5 and numpy.
"""
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import (Button, CheckButtons, RadioButtons,
                                RangeSlider, RectangleSelector, Slider)

DEFAULT_COLS = ["snr", "sampno", "secs", "boxcar", "idt", "dm", "beam", "mjd"]
MAX_RADIO = 12  # max beams / boxcars to offer as widgets


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _parse_header(line):
    """Map a '# S/N, sampno, secs from file start, ...' header to column names."""
    toks = [t.strip().lower() for t in line.lstrip("#").split(",")]
    names = []
    for t in toks:
        if "s/n" in t or "snr" in t:
            names.append("snr")
        elif "sampno" in t:
            names.append("sampno")
        elif "secs" in t:
            names.append("secs")
        elif "boxcar" in t:
            names.append("boxcar")
        elif "idt" in t:
            names.append("idt")
        elif t.startswith("dm"):
            names.append("dm")
        elif "beam" in t:
            names.append("beam")
        elif "mjd" in t:
            names.append("mjd")
        else:
            return None
    return names if {"snr", "dm"} <= set(names) else None


def load_candidates(paths, tsamp=None):
    tables = []
    for p in paths:
        names = None
        with open(p) as f:
            for line in f:
                if line.startswith("#"):
                    names = _parse_header(line) or names
                elif line.strip():
                    break
        arr = np.loadtxt(p, comments="#", ndmin=2)
        if arr.size == 0:
            print(f"warning: no candidates in {p}", file=sys.stderr)
            continue
        ncol = arr.shape[1]
        if names is None or len(names) != ncol:
            names = DEFAULT_COLS[:ncol] + [f"col{i}" for i in range(len(DEFAULT_COLS), ncol)]
        tables.append({n: arr[:, i] for i, n in enumerate(names)})
    if not tables:
        sys.exit("No candidates loaded.")
    keys = [k for k in tables[0] if all(k in t for t in tables)]
    d = {k: np.concatenate([t[k] for t in tables]) for k in keys}
    return finalize(d, tsamp)


def finalize(d, tsamp=None):
    n = len(d["snr"])
    if "secs" not in d:
        if "sampno" in d and tsamp:
            d["secs"] = d["sampno"] * tsamp
        elif "sampno" not in d:
            sys.exit("Need a 'secs' or 'sampno' column for the time axis.")
    d.setdefault("boxcar", np.zeros(n))
    d.setdefault("beam", np.zeros(n))
    order = np.argsort(d["snr"], kind="stable")  # brightest drawn last
    return {k: v[order] for k, v in d.items()}


def make_demo(seed=0):
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in DEFAULT_COLS}

    def add(t, dm, snr, box, beam):
        n = len(t)
        cols["snr"].append(snr)
        cols["secs"].append(t)
        cols["sampno"].append((t / 1.2656e-3).astype(int))
        cols["boxcar"].append(box)
        cols["idt"].append((dm / 1.5).astype(int))
        cols["dm"].append(dm)
        cols["beam"].append(beam)
        cols["mjd"].append(59000 + t / 86400.0)

    n = 3000  # noise
    add(rng.uniform(0, 60, n), rng.uniform(0, 1000, n),
        6 + rng.exponential(0.6, n), rng.integers(0, 8, n), rng.integers(0, 4, n))
    n = 400  # RFI near DM 0
    add(rng.uniform(0, 60, n), np.abs(rng.normal(0, 4, n)),
        7 + rng.exponential(2.0, n), rng.integers(0, 4, n), rng.integers(0, 4, n))
    for t0, dm0, peak, beam in [(12.3, 350, 25, 2), (41.7, 520, 14, 1), (53.2, 120, 10, 3)]:
        n = 40
        dm = dm0 + rng.normal(0, 12, n)
        snr = np.maximum(6.1, peak * np.exp(-((dm - dm0) / 25) ** 2) + rng.normal(0, 0.3, n))
        add(t0 + rng.normal(0, 0.003, n), dm, snr, rng.integers(0, 4, n), np.full(n, beam))
    return finalize({k: np.concatenate(v).astype(float) for k, v in cols.items()})


# --------------------------------------------------------------------------
# Plotter
# --------------------------------------------------------------------------
class CandPlotter:
    def __init__(self, d, title="FREDDA candidates", min_snr=None, beam=None,
                 outfile="fredda_selected.csv"):
        self.d = d
        self.outfile = outfile
        self.t, self.dm, self.snr = d["secs"] if "secs" in d else d["sampno"], d["dm"], d["snr"]
        self.tlabel = "Time since file start (s)" if "secs" in d else "Sample number"
        self.boxcar = np.rint(d["boxcar"]).astype(int)
        self.beam = np.rint(d["beam"]).astype(int)
        self.beams = np.unique(self.beam)
        self.boxes = np.unique(self.boxcar)
        self.n = len(self.snr)
        self.vis = np.arange(self.n)
        self.selected = np.array([], dtype=int)
        self.cur = None
        self.detail = ""
        self.msg = ""
        self.beam_sel = beam
        self.min_snr0 = self.snr.min() if min_snr is None else min_snr

        self.smin, self.smax = self.snr.min(), self.snr.max()
        self.sclip = max(np.percentile(self.snr, 99.5), self.smin + 1e-6)
        self._build(title)
        self.update()

    # -- layout -------------------------------------------------------------
    def _build(self, title):
        fig = self.fig = plt.figure(figsize=(14, 8.5))
        fig.suptitle(title, fontsize=11)
        gs = GridSpec(2, 2, figure=fig, width_ratios=[5, 1.3], height_ratios=[1, 3.2],
                      left=0.17, right=0.97, top=0.93, bottom=0.20, hspace=0.12, wspace=0.05)
        self.ax_snr = fig.add_subplot(gs[0, 0])
        self.ax_info = fig.add_subplot(gs[0, 1])
        self.ax = fig.add_subplot(gs[1, 0])
        self.ax_dm = fig.add_subplot(gs[1, 1], sharey=self.ax)
        self.ax_info.axis("off")
        self.info_txt = self.ax_info.text(0, 1, "", va="top", ha="left",
                                          family="monospace", fontsize=8,
                                          transform=self.ax_info.transAxes)

        # main scatter
        norm = Normalize(vmin=self.smin, vmax=self.sclip)
        self.sc = self.ax.scatter([], [], c=[], s=[], cmap="viridis", norm=norm,
                                  alpha=0.85, edgecolors="none", picker=6)
        self.sel_sc = self.ax.scatter([], [], s=70, facecolors="none",
                                      edgecolors="red", linewidths=1.0, zorder=5)
        (self.hl,) = self.ax.plot([], [], "o", ms=16, mfc="none", mec="orange",
                                  mew=2, zorder=6)
        cax = self.ax.inset_axes([0.02, 0.94, 0.22, 0.025])
        cb = fig.colorbar(self.sc, cax=cax, orientation="horizontal")
        cb.set_label("S/N", fontsize=8, labelpad=-30)
        cax.tick_params(labelsize=7)
        self.ax.set_xlabel(self.tlabel)
        self.ax.set_ylabel("DM (pc cm$^{-3}$)")
        self.ax.grid(alpha=0.25)
        tp, dp = 0.02 * np.ptp(self.t) + 1e-9, 0.04 * np.ptp(self.dm) + 1e-9
        self.ax.set_xlim(self.t.min() - tp, self.t.max() + tp)
        self.ax.set_ylim(self.dm.min() - dp, self.dm.max() + dp)

        # histograms (StepPatch so they update without cla())
        self.snr_edges = np.linspace(self.smin, self.smax + 1e-6, 60)
        self.dm_edges = np.linspace(self.dm.min(), self.dm.max() + 1e-6, 80)
        z = np.full(len(self.snr_edges) - 1, 0.5)
        self.snr_st = self.ax_snr.stairs(z, self.snr_edges, baseline=0.5, fill=True,
                                         color="tab:blue", alpha=0.8)
        self.ax_snr.set_yscale("log")
        self.ax_snr.set_xlim(self.smin, self.smax)
        self.ax_snr.set_xlabel("S/N", fontsize=8)
        self.ax_snr.set_ylabel("N", fontsize=8)
        self.ax_snr.tick_params(labelsize=7)
        self.dm_st = self.ax_dm.stairs(np.zeros(len(self.dm_edges) - 1), self.dm_edges,
                                       orientation="horizontal", fill=True,
                                       color="tab:orange", alpha=0.8)
        self.ax_dm.tick_params(labelleft=False, labelsize=7)
        self.ax_dm.set_xlabel("N", fontsize=8)

        # widgets
        ax_s = fig.add_axes([0.30, 0.095, 0.45, 0.03])
        self.snr_slider = Slider(ax_s, "min S/N", self.smin, self.smax,
                                 valinit=min(max(self.min_snr0, self.smin), self.smax))
        ax_d = fig.add_axes([0.30, 0.045, 0.45, 0.03])
        lo, hi = self.dm.min(), self.dm.max()
        if hi <= lo:
            hi = lo + 1
        self.dm_slider = RangeSlider(ax_d, "DM range", lo, hi, valinit=(lo, hi))
        self.snr_slider.on_changed(self.update)
        self.dm_slider.on_changed(self.update)

        self.radio = self.check = None
        if 1 < len(self.beams) <= MAX_RADIO:
            ax_r = fig.add_axes([0.02, 0.58, 0.11, 0.30])
            ax_r.set_title("Beam", fontsize=9)
            labels = ["all"] + [str(b) for b in self.beams]
            active = 0 if self.beam_sel is None else labels.index(str(self.beam_sel))
            self.radio = RadioButtons(ax_r, labels, active=active)
            self.radio.on_clicked(self._on_beam)
        if 1 < len(self.boxes) <= MAX_RADIO:
            ax_c = fig.add_axes([0.02, 0.22, 0.11, 0.30])
            ax_c.set_title("Boxcar", fontsize=9)
            self.check = CheckButtons(ax_c, [str(b) for b in self.boxes],
                                      [True] * len(self.boxes))
            self.check.on_clicked(self.update)

        ax_b1 = fig.add_axes([0.80, 0.09, 0.08, 0.04])
        ax_b2 = fig.add_axes([0.89, 0.09, 0.08, 0.04])
        ax_b3 = fig.add_axes([0.80, 0.04, 0.17, 0.04])
        self.b_export = Button(ax_b1, "Export (e)")
        self.b_clear = Button(ax_b2, "Clear sel (x)")
        self.b_reset = Button(ax_b3, "Reset filters (a)")
        self.b_export.on_clicked(self.export)
        self.b_clear.on_clicked(self.clear_selection)
        self.b_reset.on_clicked(self.reset_filters)

        # events
        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=True, button=[3], minspanx=5,
            minspany=5, spancoords="pixels", interactive=False,
            props=dict(facecolor="red", edgecolor="red", alpha=0.2))
        fig.canvas.mpl_connect("pick_event", self._on_pick)
        fig.canvas.mpl_connect("key_press_event", self._on_key)

    # -- helpers ------------------------------------------------------------
    def _size(self, s):
        frac = np.clip((s - self.smin) / (self.sclip - self.smin + 1e-9), 0, 1)
        return 8 + 90 * frac ** 1.5

    def _active_boxcars(self):
        if self.check is None:
            return self.boxes
        return self.boxes[np.array(self.check.get_status(), dtype=bool)]

    def _toolbar_active(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        mode = getattr(tb, "mode", "")
        return bool(getattr(mode, "value", mode))

    def _on_beam(self, label):
        self.beam_sel = None if label == "all" else int(label)
        self.update()

    # -- core update --------------------------------------------------------
    def update(self, _=None):
        lo, hi = self.dm_slider.val
        m = (self.snr >= self.snr_slider.val) & (self.dm >= lo) & (self.dm <= hi)
        m &= np.isin(self.boxcar, self._active_boxcars())
        if self.beam_sel is not None:
            m &= self.beam == self.beam_sel
        v = self.vis = np.flatnonzero(m)
        self.selected = np.intersect1d(self.selected, v)
        if self.cur is not None and not m[self.cur]:
            self.cur, self.detail = None, ""
            self.hl.set_data([], [])

        self.sc.set_offsets(np.column_stack((self.t[v], self.dm[v])))
        self.sc.set_array(self.snr[v])
        self.sc.set_sizes(self._size(self.snr[v]))
        s = self.selected
        self.sel_sc.set_offsets(np.column_stack((self.t[s], self.dm[s])))

        hs, _e = np.histogram(self.snr[v], self.snr_edges)
        self.snr_st.set_data(np.maximum(hs, 0.5), self.snr_edges)
        self.ax_snr.set_ylim(0.5, max(hs.max(), 1) * 3)
        hd, _e = np.histogram(self.dm[v], self.dm_edges)
        self.dm_st.set_data(hd, self.dm_edges)
        self.ax_dm.set_xlim(0, max(hd.max(), 1) * 1.1)

        self._refresh_info()
        self.fig.canvas.draw_idle()

    def _refresh_info(self):
        v = self.vis
        lines = [f"showing {len(v)} / {self.n}", f"selected {len(self.selected)}"]
        if len(v):
            lines.append(f"max S/N  {self.snr[v].max():.1f}")
        if self.msg:
            lines += ["", self.msg]
        if self.detail:
            lines += ["", self.detail]
        self.info_txt.set_text("\n".join(lines))

    # -- interaction --------------------------------------------------------
    def _on_pick(self, event):
        if event.artist is not self.sc or event.mouseevent.button != 1:
            return
        if self._toolbar_active() or len(event.ind) == 0:
            return
        idx = self.vis[np.asarray(event.ind)]
        self.show_candidate(idx[np.argmax(self.snr[idx])])

    def show_candidate(self, i):
        self.cur = i
        self.hl.set_data([self.t[i]], [self.dm[i]])
        row = {k: v[i] for k, v in self.d.items()}
        self.detail = "\n".join(f"{k:>7s} {v:.6g}" for k, v in row.items())
        print("candidate:", ", ".join(f"{k}={v:.6g}" for k, v in row.items()))
        self._refresh_info()
        self.fig.canvas.draw_idle()

    def _on_select(self, press, release):
        if self._toolbar_active() or press.xdata is None or release.xdata is None:
            return
        x0, x1 = sorted([press.xdata, release.xdata])
        y0, y1 = sorted([press.ydata, release.ydata])
        v = self.vis
        inside = ((self.t[v] >= x0) & (self.t[v] <= x1) &
                  (self.dm[v] >= y0) & (self.dm[v] <= y1))
        self.selected = v[inside]
        self.msg = ""
        if len(self.selected):
            i = self.selected[np.argmax(self.snr[self.selected])]
            print(f"selected {len(self.selected)} candidates; brightest: "
                  f"S/N={self.snr[i]:.2f} t={self.t[i]:.4f} DM={self.dm[i]:.2f}")
        self.update()

    def clear_selection(self, _=None):
        self.selected = np.array([], dtype=int)
        self.msg = ""
        self.update()

    def reset_filters(self, _=None):
        self.snr_slider.set_val(self.min_snr0)
        self.dm_slider.set_val((self.dm.min(), max(self.dm.max(), self.dm.min() + 1)))
        if self.radio is not None:
            self.radio.set_active(0)
        if self.check is not None:
            for i, on in enumerate(self.check.get_status()):
                if not on:
                    self.check.set_active(i)
        self.update()

    def export(self, _=None):
        idx = self.selected if len(self.selected) else self.vis
        if len(idx) == 0:
            self.msg = "nothing to export"
        else:
            keys = list(self.d)
            arr = np.column_stack([self.d[k][idx] for k in keys])
            np.savetxt(self.outfile, arr, delimiter=",", header=",".join(keys),
                       comments="", fmt="%.8g")
            self.msg = f"wrote {len(idx)} rows\n-> {self.outfile}"
            print(self.msg.replace("\n", " "))
        self._refresh_info()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "e":
            self.export()
        elif event.key == "x":
            self.clear_selection()
        elif event.key == "a":
            self.reset_filters()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="FREDDA candidate file(s)")
    ap.add_argument("--demo", action="store_true", help="use synthetic candidates")
    ap.add_argument("--min-snr", type=float, default=None, help="initial S/N threshold")
    ap.add_argument("--beam", type=int, default=None, help="initial beam filter")
    ap.add_argument("--tsamp", type=float, default=None,
                    help="sample time (s), used only if the file has no 'secs' column")
    ap.add_argument("-o", "--output", default="fredda_selected.csv",
                    help="CSV written by the export button / 'e' key")
    args = ap.parse_args()

    if args.demo:
        d, title = make_demo(), "FREDDA candidates (synthetic demo)"
    elif args.files:
        d, title = load_candidates(args.files, args.tsamp), ", ".join(args.files)
    else:
        ap.error("give candidate file(s) or --demo")

    # Keep a reference: matplotlib only weak-references bound-method callbacks,
    # so if the object is garbage-collected the sliders/buttons stop responding.
    plotter = CandPlotter(d, title=title, min_snr=args.min_snr, beam=args.beam,
                          outfile=args.output)
    plt.show()
    del plotter


if __name__ == "__main__":
    main()
