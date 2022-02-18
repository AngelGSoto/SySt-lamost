from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("source", type=str,
                    default=" teste-program",
                    help="Name of catalog, taken the prefix ")

cmd_args = parser.parse_args()

file_ = cmd_args.source + ".fits"

hdu_list = fits.open(file_, memmapper=True)
tab = hdu_list[1].data

n = len(tab["ra"])
sep = np.linspace(2.0, 2.0, num=n)
ra = tab["ra"]
dec = tab["dec"]
table = Table([ra, dec, sep], names=('ra', 'dec', 'radius'), meta={'name': 'first table'})


# Save the file

asciifile = file_.replace(".fits", 
                  "-coorLamost.dat")

table.write(asciifile, format="ascii.commented_header", delimiter=',')
