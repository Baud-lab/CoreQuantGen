#!/usr/bin/env python3

import pickle
from pathlib import Path
import gzip
import argparse as argp

parser = argp.ArgumentParser(formatter_class=argp.RawTextHelpFormatter)
# Optional arguments
parser.add_argument('-i','--input', help = 'list of files ending in with *.plk.gz', required=True)
parser.add_argument('-o','--output', help = 'path to output result, ending in .pkl.gz', required=True)

args = vars(parser.parse_args())

# Directory containing your pickle files
#main_dir = Path("/users/abaud/htonnele/git/me/core_VD/code/pysrc/test/VD/bivariate/noBatch_20cages/pruned_dosages_include_DGE_IGE_IEE_cageEffect/")
#pattern = "*.pkl.gz"

# Recursively find all matching files
#files = [str(path) for path in main_dir.rglob(pattern)]

#files = args['input'].split(",")

# Open the file in read mode
with open(args['input'], "r") as f:
    #files = f.readlines()  # Read all lines into a list
    files = [file.strip() for file in f.readlines()]

# Initialize an empty dictionary to store the combined data
all_pickles = {}

# Iterate through all pickle files in the directory
for f in files:
    print(f)
    assert f.endswith(".pkl.gz"), "file not gzip pickle"
    # Load the dictionary from the pickle file
    with gzip.open(f, "rb") as file:
        data = pickle.load(file)
        # Merge the data into the combined dictionary
        #combined_dict.update(data['vc_covs'])
        all_pickles[Path(f).name] = data

# Save the combined dictionary to a new pickle file
#output_file = "/users/abaud/htonnele/git/me/core_VD/code/pysrc/test/VD/bivariate/noBatch_20cages/pruned_dosages_include_DGE_IGE_IEE_cageEffect_all.pkl.gz"  # Name of the output file
output_file = args['output']
with gzip.open(output_file, "wb") as f:
     pickle.dump(all_pickles, f)
     print("Files saved to", f)

print(f"Combined dictionary saved to {output_file}")
