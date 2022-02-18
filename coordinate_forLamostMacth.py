'''
This is a simply script to make table with the format of Lamost for cross-match.
'''
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

file_ = cmd_args.source + ".dat"

#datadir = "../"
try:
    tab = Table.read(file_, format="ascii")
except FileNotFoundError:
    file_ = cmd_args.source + ".csv"
    tab = pd.read_csv(file_)


n = len(tab["RAJ2000"])
sep = np.linspace(2.0, 2.0, num=n)
ra = tab["RAJ2000"]
dec = tab["DEJ2000"]
table = Table([ra, dec, sep], names=('ra', 'dec', 'radius'), meta={'name': 'first table'})


# Save the file
if file_.endswith(".dat"):
    asciifile = file_.replace(".dat", 
                  "-coorLamost.dat")
    table.write(asciifile, format="ascii.commented_header", delimiter=',')
else:
    asciifile = file_.replace(".csv", 
                  "-coorLamost.dat")
    table.write(asciifile, format="ascii.commented_header", delimiter=',')

