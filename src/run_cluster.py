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

    for k in range(labels.max() + 1):                                 # best candidate per cluster
        idx = np.where(labels == k)[0]
        b = idx[np.argmax(fredda_cands['sn'][idx])]
        print(f"S/N {fredda_cands['sn'][b]:.2f}  t {fredda_cands['secs'][b]:.4f}  "
              f"DM {fredda_cands['dm'][b]:.2f}  box {fredda_cands['boxcar'][b]:.0f}  n {len(idx)}")
 

