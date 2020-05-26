"""Extract BH evolution data for specified simulation(s)."""

import numpy as np
import os
import hydrangea.hdf5 as hd
import glob
import argparse
from pdb import set_trace

def main():
    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    parser.add_argument('--sims', type=int,
                        help='Simulation indices to process.', nargs='+')
    parser.add_argument('--outfile', help='Name of output catalogue.',
                        default='bh_growth.hdf5')
    args = parser.parse_args()

    n_sims = len(args.sims)
    n_snaps = len(args.snapshots)
    
    args.bh_fields = ['SubgridMasses']
                      
    print(f"Processing {n_sims} simulations")
    if n_sims == 0:
        print("")
        set_trace()
    
    for isim in args.sims:
        process_sim(isim, args)

        
def process_sim(isim, args):
    """Process one individual simulation"""

    wdirs = glob.glob(f'/cosma7/data/dp004/dc-bahe1/EXL/ID{isim}*/')
    if len(wdirs) != 1:
        set_trace()
    wdir = wdirs[0]
    print(f"Analysing simulation {wdir}...")

    bfile = wdir + 'black_hole_data.hdf5'

    # Copy out the desired fields
    for field in args.bh_fields:
        data = hd.read_data(bfile, field)
        if data is None:
            print(f"Could not load data set '{field}'!")
            set_trace()

        hd.write_data(args.outfile, f'ID{isim}/{field}')

    # Copy out metadata fields
    times = hd.read_data(bfile, 'Times')
    redshifts = hd.read_data(bfile, 'Redshifts')
    first_indices = hd.read_data(bfile, 'FirstIndices')

    hd.write_data(args.outfile, f'ID{isim}/Times', times)
    hd.write_data(args.outfile, f'ID{isim}/Redshifts', redshifts)
    hd.write_data(args.outfile, f'ID{isim}/FirstIndices', first_indices)
        
        
if __name__ == '__main__':
    main()
    

