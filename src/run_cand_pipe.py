import argparse
from src.run_cluster_numba import * 
from src.pulsar_crossmatch import read_beam_positions, \
                              attach_beam_coords, \
                              crossmatch, \
                              load_local_catalog, \
                              save_csv




def main():
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
    p.add_argument("-o",                                                            
                    "--outfile",                                                    
                    type=str,                                                       
                    default=2,                                                      
                    help="Clustered outputfile")                                    

    p.add_argument("--candidates-have-radec",
                    action="store_true",
                    help="Set if candidate file already contains ra/dec columns "
                        "instead of beamno")
    p.add_argument("--beams", 
                    default=None,
                    help="Path to beam position file (beamno, ra, dec). "
                        "Required unless --candidates-have-radec")
    p.add_argument("--catalog", 
                    default=None,
                    help="Path to local known-pulsar catalog CSV (name, ra, dec, dm)")
    p.add_argument("--use-atnf", 
                    action="store_true",
                    help="Fetch known-pulsar catalog live from ATNF via psrqpy")
    p.add_argument("--radius", 
                    type=float, 
                    default=0.5,
                    help="Positional match radius in degrees (default: 0.5)")
    p.add_argument("--dm-tol", 
                    type=float, 
                    default=0.1,
                    dest="dm_tol",
                    help="Fractional DM tolerance, e.g. 0.1 = 10%% (default: 0.1)")
    p.add_argument("--dm-abs-tol", 
                    type=float, 
                    default=None, 
                    dest="dm_abs_tol",
                   help="Absolute DM tolerance in % (overrides --dm-tol if set)")
    p.add_argument("--out",
                    default="matches.csv",
                    help="Output CSV path for matches (default: matches.csv)")
    p.add_argument("--out_unq",
                    default="unique.csv",
                    help="Output CSV path for unique sources (default: unique.csv)")
    p.add_argument("--all", 
                    action="store_true",
                    help="Write all candidates with nearest-match info, "
                        "not just confirmed matches")
    p.add_argument("--npy_file", 
                    default=1,
                    help="Use npy files? Default=1")





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




    #Converting to clustered dtype for pulsar 
    cands       =       np.empty((len(fredda_cands['sn'][best])), dtype=CAND_DTYPE_CLS)
    fields = ['sn', 'sampno', 'secs', 'boxcar', 'idt', 'dm', 'beamno', 'mjd']

    for name in fields:
        cands[name] = fredda_cands[name][best]
    cands['cls']        =       uniq
    cands['ncands']     =       counts

    # --- candidates ---                                                                          
    beams = read_beam_positions(args.beams)                                                   
    cands = attach_beam_coords(cands, beams)                                                  
                                                                                                  
    print(f"[INFO] Loaded {len(cands)} candidates")                                               
                                                                                                  
    # --- catalog ---                                                                             
    if args.catalog:                                                                              
        catalog = load_local_catalog(args.catalog, args.npy_file)                                 
    else:                                                                                         
        sys.exit("Provide --catalog <file>")                                                      
                                                                                                  
    print(f"[INFO] Catalog has {len(catalog)} known pulsars")                                     
                                                                                                  
    # --- cross-match ---                                                                         
    result = crossmatch(                                                                          
        cands,                                                                                    
        catalog,                                                                                  
        radius_deg=args.radius,                                                                   
        dm_tol_frac=args.dm_tol,                                                                  
        dm_tol_abs=args.dm_abs_tol,                                                               
    )                                                                                             
                                                                                                  
    n_match = int(result["is_match"].sum())                                                       
    print(f"[INFO] {n_match} / {len(result)} candidates matched a known pulsar "                  
          f"(radius<={args.radius} deg, dm_tol={args.dm_abs_tol or args.dm_tol})")                
                                                                                                  
    out = result if args.all else result[result["is_match"]]                                      
    unq = result if args.all else result[np.where(result["is_match"]==False)[0]]                  
    save_csv(f"{args.file}_matches.txt", out)                                                                       
    save_csv(f"{args.file}_unique.txt", unq)                                                                   
    print(f"[INFO] Wrote {len(out)} rows to {args.out}")                                          
                                                                                                  






 
if __name__ == "__main__":
    main()


