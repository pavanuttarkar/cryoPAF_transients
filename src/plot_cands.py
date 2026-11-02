from src.utils import plot_cands, push_png
import argparse


def main():
    p = argparse.ArgumentParser(description="Cluster FREDDA/FDMT candidates.")
    p.add_argument("-f",
                    "--file",
                    required=True,
                    help="Beam con candidate file to plot data")
    p.add_argument("--dt",
                    type=float,
                    required=True,
                    help="sample time in milliseconds, e.g. 1.28")
    p.add_argument("-o",
                    "--out_dir",
                    type=str,
                    default="plots",
                    help="Output directory (default `plots`)")
    p.add_argument("-w",
                    "--window",
                    type=int,
                    default=1024,
                    help="window length")
    p.add_argument("-fil",
                    "--filterbank",
                    type=str,
                    required=True,
                    help="Filterbank file, provide any beam filterbank file")
    p.add_argument("-s",
                    "--slack",
                    type=int,
                    default=0,
                    help="Send candidates to `slack`")


    args = p.parse_args()


    cand_file   =       plot_cands(args.file, 
                                args.filterbank, 
                                args.window, 
                                time_res=args.dt, 
                                out_dir=args.out_dir, 
                                slack=args.slack)

if __name__ == "__main__":
    main()
