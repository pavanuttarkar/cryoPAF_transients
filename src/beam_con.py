from utils import beam_coincidence 

def main():
    """
    Wrapper code for beam coincidencing. 
    """        
    import argparse
    
    p = argparse.ArgumentParser(description="Beam coincidence for clustered candidates")
    p.add_argument("-f", 
                    "--file", 
                    help="one clustered file from all beams")

    p.add_argument("--dm-tol", 
                    type=float, 
                    default=1.0, 
                    help="DM tolerance in pc/cc, default=1 DMunit")
    p.add_argument("--time_tol", 
                    type=float, 
                    default=10e-3,
                    help="time tolerance in ms, default=10ms")
    p.add_argument("-o", 
                    "--output", 
                    default="coincidence.cand", 
                    help="output file")
    args        =    p.parse_args()     


    #def beam_coincidence(fname,
    #                 dm_tol,
    #                 sec_tol,
    #                 nbeams=None,
    #                 snr_field="sn"):
   
 
    out_cands   =       beam_coincidence(fname=args.file, 
                                        dm_tol=args.dm_tol, 
                                        sec_tol=args.time_tol,
                                        
                                        )


if __name__ == "__main__":
    main()

    
    
