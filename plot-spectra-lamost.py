import spectra_lamost
import argparse
from pathlib import Path

ROOT_PATH = Path("Spectra/")

# Read the file
parser = argparse.ArgumentParser(
    description="""Make a spectras""")

parser.add_argument("fileLamost", type=str,
                    default="teste-program",
                    help="Name of file, taken the prefix")


cmd_args = parser.parse_args()
file_spec = cmd_args.fileLamost + ".fits"

spectra_lamost.spectra(file_spec)
