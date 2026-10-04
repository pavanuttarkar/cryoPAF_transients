from utils import read_fil



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-f", 
                    "--file",
                    required=True,
                    help="Path to filterbank")



    return p.parse_args()


def main() -> None:
    
    args = parse_args()
    read_fil(args.file)
    
    return None



if __name__ == "__main__":
    main()
