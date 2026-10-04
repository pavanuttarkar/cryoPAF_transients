import numpy as np
from utils import *


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Script to dump npy files for faster read")
    parser.add_argument("-i",
                        "--input", 
                        default="known_pulsars.csv",
                        help="Path to the input file")
    parser.add_argument("-o", 
                        "--output",
                        help="Path to the output file", 
                        default="known_pulsars")

    args = parser.parse_args()
    dump_npy(args.input, args.output)
    return None 



if __name__ == "__main__":
    main()
