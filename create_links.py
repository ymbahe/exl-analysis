"""Create symlinks for full snapshots."""

import numpy as np
from astropy.io import ascii
from pdb import set_trace
import os
import argparse
import glob

print("Parsing input arguments...")
parser = argparse.ArgumentParser(description="Parse input parameters.")
parser.add_argument('sim', type=int, help='Simulation index to analyse')
parser.add_argument('file', help='Output list file name')
parser.add_argument('--type', type=int, help='Type code for full snapshots (default=1)',
                    default=1)
parser.add_argument('--names', help='Prefix of existing outputs (default: "output")',
                    default='output')
parser.add_argument('--linknames', help='Prefix of links (default: "eagle")',
                    default='eagle')
args = parser.parse_args()

dirs = glob.glob(f'/cosma7/data/dp004/dc-bahe1/EXL/ID{args.sim}*/')
if len(dirs) != 1:
    print(f"Could not unambiguously find directory for simulation {args.sim}!")
    set_trace()
wdir = dirs[0]
if not wdir.endswith('/'):
    wdir = wdir + '/'

print(f"Processing simulation {wdir}...")

# Load output codes from ASCII file
if not os.path.isfile(args.file):
    print("It looks like the specified file '{args.file}' does not exist...")
    set_trace()
output_table = ascii.read(args.file, format='no_header')
f = open(args.file, "r")
line1 = f.readline()
f.close()

if line1.startswith('# Redshift'):
    print("Output file in redshift units")
else:
    set_trace()

zred = np.array(output_table['col1'])
types = np.array(output_table['col2'])

ind_snaps = np.nonzero(types == args.type)[0]
print(f"Found {len(ind_snaps)} snapshots (type {args.type})...")

for iisnap, isnap in enumerate(ind_snaps):
    filename = wdir + args.names + f'_{isnap:04d}.hdf5'
    linkname = wdir + args.linknames + f'_{iisnap:04d}.hdf5'
    if os.path.exists(linkname):
        print(f" ... link '{linkname}' already exists, skipping it.")
    else:
        if os.path.isfile(filename):
            print(f" ... creating link for snap {iisnap} [output {isnap}]...")
            os.symlink(filename, linkname)
        else:
            print(f" ... output does not exist, skipping it.")


print("Done!")


