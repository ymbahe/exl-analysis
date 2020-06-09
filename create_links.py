"""Create symlinks for full snapshots."""

import numpy as np
from astropy.io import ascii
from pdb import set_trace
import os
import argparse
import glob
import local

def main():

    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    parser.add_argument('sims', nargs='+',
                    help='Simulation inde(x/ices) or name(s) to analyse')
    parser.add_argument('list_file', help='Output list file name')
    parser.add_argument('--base_dir', help='Simulation base directory',
                        default=local.BASE_DIR)
    parser.add_argument('--snapshot_type',
                        help='Type code for full snapshots (default: 1)',
                        default='1')
    parser.add_argument('--output_names',
                        help='Prefix of existing outputs (default: "output")',
                        default='output')
    parser.add_argument('--link_names', help='Prefix of links (default: "eagle")',
                        default='eagle')

    args = parser.parse_args()

    if args.sims[0].lower() == 'all':
        args.sims = local.get_all_sims(args.base_dir)
        have_full_sim_dir = True
    else:
        have_full_sim_dir = False

    for isim in args.sims:
        process_sim(args, isim, have_full_sim_dir)


def process_sim(args, isim, have_full_sim_dir):
    """Process one individual simulation."""

    if have_full_sim_dir:
        wdir = isim
    else:
        wdir = local.get_sim_dir(args.base_dir, isim)

    print(f"Processing simulation {wdir}...")

    # Check if output directory exists, create if needed
    if not os.path.isdir(os.path.dirname(f'{wdir}{args.link_names}')):
        os.makedirs(os.path.dirname(f'{wdir}{args.link_names}'))
    
    # Load output codes from ASCII file
    if not os.path.isfile(args.list_file):
        print(f"It looks like the specified file '{args.list_file}' "
              f"does not exist...")
        set_trace()

    output_table = ascii.read(args.list_file, format='no_header')
    f = open(args.list_file, "r")
    line1 = f.readline()
    f.close()

    if line1.startswith('# Redshift'):
        print("Output file in redshift units")
    else:
        set_trace()

    zred = np.array(output_table['col1'])
    types = np.array(output_table['col2'])

    ind_snaps = np.nonzero(types == args.snapshot_type)[0]
    print(f"Found {len(ind_snaps)} snapshots (type {args.snapshot_type})...")

    for iisnap, isnap in enumerate(ind_snaps):
        create_link(iisnap, isnap, wdir, args)

    print("   ...done!")

    
def create_link(iisnap, isnap, wdir, args):
    """Create link for one specific snapshot."""

    file_name = f'{wdir}{args.output_names}_{isnap:04d}.hdf5'
    link_name = f'{wdir}{args.link_names}_{iisnap:04d}.hdf5'

    if os.path.exists(link_name):
        print(f" ... link '{link_name}' already exists, skipping it.")
    else:
        if os.path.isfile(file_name):
            print(f" ... creating link for snap {iisnap} [output {isnap}]...")
            os.symlink(file_name, link_name)
        else:
            print(f" ... output does not exist, skipping it.")


if __name__ == '__main__':
    main()
