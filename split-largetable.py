import pandas as pd
from astropy.table import Table, vstack
import numpy as np
import argparse
from astropy.io import fits

parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("source", type=str,
                    default=" teste-program",
                    help="Name of catalog, taken the prefix ")

cmd_args = parser.parse_args()

file_ = cmd_args.source + ".fits"

hdu_list = fits.open(file_, memmapper=True)
tab = hdu_list[1].data
table = Table(tab)

colums = ["ra", "dec"]
df = (table[colums].to_pandas())
n = int(len(df) / 500000.) + 1

df_ = [] # list
j = 0 # counter
chunk = 500000
for i in range(n):
    j += 1  
    df_.append(df.iloc[chunk*i:chunk*j])

for a in range(n):
    n_ = len(df_[a]["ra"])
    sep = np.linspace(2.0, 2.0, num=n_)
    df_[a]["radius"] = sep
    Table.from_pandas(df_[a]).write("2mass-13J14-"+str(a)+".dat", format='ascii.commented_header',  delimiter=',', overwrite=True)
