import argparse
import numpy as np
from header import *
from sklearn.cluster import DBSCAN
 
def load_fredda(path, dt):
    # columns: S/N, sampno, secs, boxcar, idt, dm, beamno, mjd
    fredda_cands = np.loadtxt(path, dtype=CAND_DTYPE)
    disp = dt / np.median(fredda_cands['dm'][fredda_cands['idt'] > 0] / (fredda_cands['idt'][fredda_cands['idt'] > 0]))            # sweep [s per DM unit]
    return fredda_cands, disp
 
def cluster_fdmt(fredda_cands, 
                 dt, 
                 disp, 
                 eps=1.0, 
                 min_samples=2):

    w       = np.maximum(fredda_cands['boxcar'] * dt, dt)
    sig     = np.maximum(w[:, None], w[None, :])                          
    dist    = np.hypot((fredda_cands['secs'][:, None] - fredda_cands['secs'][None, :]) / sig,
                    (fredda_cands['dm'][:, None] - fredda_cands['dm'][None, :]) * disp / sig)

    return DBSCAN(eps=eps, 
                min_samples=min_samples, 
                metric="precomputed").fit_predict(dist)
"""



def normalise_fdmt(fredda_cands, dt, disp, sigma=None):
    # sigma: fixed tolerance in seconds (default: one sample)
    if sigma is None:
        sigma = dt
    return np.column_stack([fredda_cands['secs'] / sigma,
                            fredda_cands['dm'] * disp / sigma])


def cluster_fdmt(fredda_cands,
                       dt,
                       disp,
                       eps=1.0,
                       min_samples=2,
                       sigma=None):
    X = normalise_fdmt(fredda_cands, dt, disp, sigma)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
""" 
if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Cluster FREDDA/FDMT candidates.")
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
    args = p.parse_args()
 
    fredda_cands, disp      = load_fredda(args.file, args.dt*1e-3)
    labels                  = cluster_fdmt(fredda_cands,
                                        args.dt*1e-3, 
                                        disp, 
                                        args.eps, 
                                        args.min_samples)
                
    print(len(fredda_cands['secs']), "candidates ->", labels.max() + 1, "clusters,", (labels == -1).sum(), "noise")

    valid = np.flatnonzero(labels >= 0)                                        # ignore noise (-1)
    order = valid[np.lexsort((fredda_cands['sn'][valid], labels[valid]))]      

    # last row of each label block = highest S/N in that cluster
    uniq, first, counts = np.unique(labels[order], return_index=True, return_counts=True)
    best = order[first + counts - 1]

    np.savetxt("best_per_cluster.txt",
               np.column_stack([
                                fredda_cands['sn'][best],
                                fredda_cands['sampno'][best],
                                fredda_cands['secs'][best],
                                fredda_cands['boxcar'][best],
                                fredda_cands['idt'][best],
                                fredda_cands['dm'][best],
                                fredda_cands['beamno'][best],
                                fredda_cands['mjd'][best],
                                uniq,]),
               fmt="%s",#["%d", "%.2f", "%.4f", "%.2f", "%.0f", "%d"],
               header="cluster  sn  secs  dm  boxcar  n")
    """
    ("sn",     "f8"),
    ("sampno", "i8"),
    ("secs",   "f8"),
    ("boxcar", "i8"),
    ("idt",    "i8"),
    ("dm",     "f8"),
    ("beamno", "i8"),
    ("mjd",    "f8"),
    ("cls",    "f8"),
    """
    noise = np.flatnonzero(labels < 0)                                         # noise points (-1)

    np.savetxt("noise_candidates.txt",
               np.column_stack([noise,
                                fredda_cands['sn'][noise],
                                fredda_cands['secs'][noise],
                                fredda_cands['dm'][noise],
                                fredda_cands['boxcar'][noise]]),
               fmt=["%d", "%.2f", "%.4f", "%.2f", "%.0f"],
               header="index  sn  secs  dm  boxcar")
