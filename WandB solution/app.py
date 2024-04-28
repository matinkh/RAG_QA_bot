import argparse


if __name__ == "__main__":
    # python app.py --folder=<FOLDER_ADDRESS>

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("folder", type=str,
                            help="Address to the location where context files are stored. Nested folders will be read too.")
