import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from header import CAND_DTYPE
from fil_veiwer import read_fil

K_DM = 4.148808e-3                                          # s GHz^2 pc^-1 cm^3
CLUSTER_DTYPE = np.dtype(CAND_DTYPE.descr + [("cluster", "i8")])

def chan_delays(freqs_ghz, dm):
    """Dispersion delay [s] of each channel relative to the top of the band."""
    return K_DM * dm * (freqs_ghz ** -2 - freqs_ghz.max() ** -2)


def plot_filterbank(cand, hdr, fil_path, outpath, pad=0.1):
    """Dispersed + dedispersed dynamic spectrum around one candidate."""
    dt = hdr["tsamp"]
    freqs = (hdr["fch1"] + hdr["foff"] * np.arange(hdr["nchans"])) * 1e-3       # GHz
    delays = chan_delays(freqs, cand["dm"])
    shifts = np.round(delays / dt).astype(int)
    npad, maxs = int(pad / dt), shifts.max()

    start = max(int(cand["sampno"]) - npad, 0)
    stop = int(cand["sampno"]) + maxs + npad
    _, raw = read_fil(fil_path, nsamples=stop - start, start=start)             # only this window
    win = raw[:, 0, :].astype(float)                                            # [t, f], first IF
    win = (win - np.median(win, axis=0)) / (win.std(axis=0) + 1e-9)             # per-channel norm
    if hdr["foff"] < 0:                                                         # low freq at bottom
        win, freqs, shifts = win[:, ::-1], freqs[::-1], shifts[::-1]

    nout = win.shape[0] - maxs
    dd = np.stack([win[s:s + nout, c] for c, s in enumerate(shifts)], axis=1)   # dedispersed
    ts = np.convolve(dd.sum(axis=1), np.ones(int(cand["boxcar"])), "same")      # boxcar-smoothed

    t0 = start * dt
    ext = lambda n: [t0, t0 + n * dt, freqs[0], freqs[-1]]
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 3, 1.5]})
    ax[2].sharex(ax[1])                                                         # dedispersed panels share time axis
    ax[0].imshow(win.T, aspect="auto", origin="lower", extent=ext(win.shape[0]), cmap="viridis")
    ax[0].plot(cand["sampno"] * dt + delays[::-1] if hdr["foff"] < 0 else cand["sampno"] * dt + delays,
               freqs, "r--", lw=0.8, alpha=0.8)                                 # expected sweep
    ax[0].set_title(f"cluster {cand['cluster']}  DM={cand['dm']:.2f}  S/N={cand['sn']:.1f}  "
                    f"boxcar={cand['boxcar']}  (red: expected sweep)")
    ax[1].imshow(dd.T, aspect="auto", origin="lower", extent=ext(nout), cmap="viridis")
    ax[2].plot(t0 + np.arange(nout) * dt, ts, "k", lw=0.8)
    ax[0].set_ylabel("Freq [GHz]"); ax[1].set_ylabel("Freq [GHz]")
    ax[1].text(0.01, 0.92, "dedispersed", transform=ax[1].transAxes, color="w")
    ax[0].set_xlabel("Time from file start [s]"); ax[2].set_xlabel("Time from file start [s]"); ax[2].set_ylabel("Sum")
    fig.tight_layout(); fig.savefig(outpath, dpi=110); plt.close(fig)


def plot_butterfly(members, best, outpath):
    """Candidates of one cluster in (time, DM), then S/N vs DM. Shows the bow-tie shape."""
    dtms = (members["secs"] - best["secs"]) * 1e3
    
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    sc = ax[0].scatter(dtms, members["dm"], c=members["sn"], s=6 * members["boxcar"],
                       cmap="plasma", alpha=0.7, edgecolor="none")
    ax[0].plot(0, best["dm"], "k*", ms=14, mfc="none")
    ax[0].set_xlabel("Time from best candidate [ms]"); ax[0].set_ylabel(r"DM [pc cm$^{-3}$]")
    fig.colorbar(sc, ax=ax[0], label="S/N")
    ax[1].scatter(members["dm"], members["sn"], c=members["boxcar"], cmap="viridis", s=12)
    ax[1].axvline(best["dm"], color="k", ls=":")
    ax[1].set_xlabel(r"DM [pc cm$^{-3}$]"); ax[1].set_ylabel("S/N")
    fig.colorbar(ax[1].collections[0], ax=ax[1], label="boxcar [samples]")
    fig.suptitle(f"cluster {best['cluster']}: {len(members)} candidates (marker size ~ boxcar)")
    fig.tight_layout(); fig.savefig(outpath, dpi=110); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Butterfly and filterbank plots for clustered candidates.")
    p.add_argument("-f", "--file", required=True, help="clustered candidates (FREDDA columns + cluster id)")
    p.add_argument("--fil", help="SIGPROC filterbank file (omit to skip filterbank plots)")
    p.add_argument("-o", "--outdir", default="plots")
    p.add_argument("--top", type=int, default=10, help="plot the N brightest clusters (default 10)")
    p.add_argument("--pad", type=float, default=0.1, help="padding around the burst in s (default 0.1)")
    args = p.parse_args()

    cands = np.loadtxt(args.file, dtype=CLUSTER_DTYPE)
    cands = cands[cands["cluster"] >= 0]                                        # drop noise
    os.makedirs(args.outdir, exist_ok=True)
    hdr = read_fil(args.fil, nsamples=0)[0] if args.fil else None             # header only

    best = {c: np.where(cands["cluster"] == c)[0][np.argmax(cands["sn"][cands["cluster"] == c])]
            for c in np.unique(cands["cluster"])}
    order = sorted(best, key=lambda c: -cands["sn"][best[c]])[:args.top]
    for c in order:
        b = cands[best[c]]
        plot_butterfly(cands[cands["cluster"] == c], b, f"{args.outdir}/butterfly_{c}.png")
        if hdr:
            plot_filterbank(b, hdr, args.fil, f"{args.outdir}/fil_{c}.png", args.pad)
        print(f"cluster {c}: S/N {b['sn']:.1f}  t {b['secs']:.3f}  DM {b['dm']:.2f}")
