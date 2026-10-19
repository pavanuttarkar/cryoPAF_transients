import argparse
import math
import numpy as np
from numba import njit
from src.header import *

UNVISITED = -2
NOISE     = -1


def load_fredda(path, dt):
    # columns: S/N, sampno, secs, boxcar, idt, dm, beamno, mjd
    fredda_cands = np.atleast_1d(np.loadtxt(path, dtype=CAND_DTYPE))
    ok   = fredda_cands['idt'] > 0
    disp = dt / np.median(fredda_cands['dm'][ok] / fredda_cands['idt'][ok])   # sweep [s per DM unit]
    return fredda_cands, disp


@njit(cache=True)
def _region_query(i, t, y, w, eps, lim, buf):
    """Fill buf with all j where hypot(dt, dy) <= eps * max(w_i, w_j). Self is included.
    t is sorted, so only a time window of +-lim needs to be scanned."""
    n = t.shape[0]
    k = 0
    j = i
    while j >= 0 and t[i] - t[j] <= lim:                    # scan backwards (includes i itself)
        dx = t[i] - t[j]
        dy = y[i] - y[j]
        s  = w[i] if w[i] > w[j] else w[j]
        if math.sqrt(dx * dx + dy * dy) <= eps * s:
            buf[k] = j
            k += 1
        j -= 1
    j = i + 1
    while j < n and t[j] - t[i] <= lim:                     # scan forwards
        dx = t[j] - t[i]
        dy = y[i] - y[j]
        s  = w[i] if w[i] > w[j] else w[j]
        if math.sqrt(dx * dx + dy * dy) <= eps * s:
            buf[k] = j
            k += 1
        j += 1
    return k


@njit(cache=True)
def _dbscan(t, y, w, eps, min_samples):
    """DBSCAN with the pair metric hypot(dt, dy) / max(w_i, w_j). t must be sorted."""
    n      = t.shape[0]
    labels = np.full(n, UNVISITED, dtype=np.int64)
    buf    = np.empty(n, dtype=np.int64)
    stack  = np.empty(n, dtype=np.int64)
    lim    = eps * w.max()                                  # no pair can be further apart in time
    c      = 0

    for i in range(n):
        if labels[i] != UNVISITED:
            continue
        k = _region_query(i, t, y, w, eps, lim, buf)
        if k < min_samples:                                 # self counts, same as sklearn
            labels[i] = NOISE
            continue

        labels[i] = c
        sp = 0
        for m in range(k):
            q = buf[m]
            if labels[q] == UNVISITED:
                stack[sp] = q
                sp += 1
                labels[q] = c
            elif labels[q] == NOISE:                        # border point, joins but is not expanded
                labels[q] = c

        while sp > 0:
            sp -= 1
            p  = stack[sp]
            k2 = _region_query(p, t, y, w, eps, lim, buf)
            if k2 >= min_samples:                           # p is a core point: expand
                for m in range(k2):
                    q = buf[m]
                    if labels[q] == UNVISITED:
                        stack[sp] = q
                        sp += 1
                        labels[q] = c
                    elif labels[q] == NOISE:
                        labels[q] = c
        c += 1
    return labels


def cluster_fdmt(fredda_cands, dt, disp, eps=1.0, min_samples=2):
    secs = np.ascontiguousarray(fredda_cands['secs'], dtype=np.float64)
    y    = np.ascontiguousarray(fredda_cands['dm'] * disp, dtype=np.float64)   # DM as delay [s]
    w    = np.maximum(fredda_cands['boxcar'] * dt, dt).astype(np.float64)

    order = np.argsort(secs, kind="stable")                 # sorted by time for the window search
    lab_s = _dbscan(secs[order], y[order], w[order], float(eps), int(min_samples))

    labels = np.empty_like(lab_s)
    labels[order] = lab_s                                   # back to the input order
    return labels


def cluster_mb(path, dt, eps=1.0, min_samples=2):
    """Cluster each beam independently. Returns (cands, labels, beams).
    Cluster ids are unique across beams, -1 means noise."""
    cands, disp = load_fredda(path, dt)
    beams  = np.unique(cands['beamno'])
    labels = np.full(len(cands), -1, dtype=np.int64)
    offset = 0

    for b in beams:
        idx = np.flatnonzero(cands['beamno'] == b)
        if len(idx) < min_samples:
            continue
        lab  = cluster_fdmt(cands[idx], dt, disp, eps, min_samples)
        good = lab >= 0
        if good.any():
            labels[idx[good]] = lab[good] + offset
            offset += lab.max() + 1

    return cands, labels, beams


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Cluster FREDDA/FDMT candidates (per beam, numba).")
    p.add_argument("-f", 
                    "--file", 
                    required=True,
                    help="candidate file (S/N sampno secs boxcar idt dm beam mjd)")
    p.add_argument("--dt", 
                    type=float, 
                    required=True,
                    help="sample time in milliseconds, e.g. 1.28")
    p.add_argument("--eps", 
                    type=float, 
                    default=1.0,
                    help="DBSCAN eps in boxcar widths (default 1)")
    p.add_argument("--min-samples", 
                    type=int, 
                    default=2,
                    help="DBSCAN min_samples (default 2)")
    p.add_argument("-o", 
                    "--outfile", 
                    type=str, 
                    default=2,
                    help="Clustered outputfile")
    args = p.parse_args()

    fredda_cands, labels, beams = cluster_mb(args.file,
                                             dt=args.dt * 1e-3,
                                             eps=args.eps,
                                             min_samples=args.min_samples)

    for b in beams:
        lab = labels[fredda_cands['beamno'] == b]
        print(f"beam {b}: {len(lab)} candidates -> {len(np.unique(lab[lab >= 0]))} clusters, "
              f"{(lab == -1).sum()} noise")
    print(f"total: {len(labels)} candidates -> {len(np.unique(labels[labels >= 0]))} clusters, "
          f"{(labels == -1).sum()} noise")

    valid = np.flatnonzero(labels >= 0)
    order = valid[np.lexsort((fredda_cands['sn'][valid], labels[valid]))]
    uniq, first, counts = np.unique(labels[order], return_index=True, return_counts=True)
    best = order[first + counts - 1]                        # highest S/N per cluster

    np.savetxt(f"{args.outfile}_clustered.txt",
               np.column_stack([fredda_cands['sn'][best],
                                fredda_cands['sampno'][best],
                                fredda_cands['secs'][best],
                                fredda_cands['boxcar'][best],
                                fredda_cands['idt'][best],
                                fredda_cands['dm'][best],
                                fredda_cands['beamno'][best],
                                fredda_cands['mjd'][best],
                                uniq,
                                counts]),
               fmt=["%.2f", "%d", "%.4f", "%d", "%d", "%.2f", "%d", "%.6f", "%d", "%d"],
               header="sn  sampno  secs  boxcar  idt  dm  beamno  mjd  cluster  ncands")

    noise = np.flatnonzero(labels < 0)
    np.savetxt(f"{args.outfile}_noise_candidates.txt",
               np.column_stack([noise,
                                fredda_cands['sn'][noise],
                                fredda_cands['secs'][noise],
                                fredda_cands['dm'][noise],
                                fredda_cands['boxcar'][noise],
                                fredda_cands['beamno'][noise]]),
               fmt=["%d", "%.2f", "%.4f", "%.2f", "%d", "%d"],
               header="index  sn  secs  dm  boxcar  beamno")
