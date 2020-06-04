"""Extract BH evolution data for specified simulation(s)."""

import numpy as np
import os
import hydrangea.hdf5 as hd
import glob
import argparse
from pdb import set_trace
import local

def main():
    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    parser.add_argument('--sims', type=int,
                        help='Simulation indices to process.', nargs='+')
    parser.add_argument('--outfile', help='Name of output catalogue.',
                        default='bh_growth.hdf5')
    args = parser.parse_args()

    n_sims = len(args.sims)
    
    args.bh_fields = ['SubgridMasses']
                      
    print(f"Processing {n_sims} simulations")
    if n_sims == 0:
        print("")
        set_trace()

    args.outfile = f'{local.BASE_DIR}/{args.outfile}'
    print(args.outfile)
        
    for isim in args.sims:
        process_sim(isim, args)

    
def process_sim(isim, args):
    """Process one individual simulation"""

    wdirs = glob.glob(f'{local.BASE_DIR}/ID{isim}*/')
    if len(wdirs) != 1:
        set_trace()
    wdir = wdirs[0]
    print(f"Analysing simulation {wdir}...")

    bfile = wdir + 'black_hole_data.hdf5'
    if not os.path.isfile(bfile):
        print(f"Could not find BH data file for simulation {isim}.")
        return
        
    # Copy out the desired fields
    for field in args.bh_fields:
        data = hd.read_data(bfile, field)
        if data is None:
            print(f"Could not load data set '{field}'!")
            set_trace()

        hd.write_data(args.outfile, f'ID{isim}/{field}', data)

    # Copy out metadata fields
    times = hd.read_data(bfile, 'Times')
    redshifts = hd.read_data(bfile, 'Redshifts')
    first_indices = hd.read_data(bfile, 'FirstIndices')
    vr_haloes = hd.read_data(bfile, 'Haloes')
    vr_halo_mstar = hd.read_data(bfile, 'Halo_MStar')
    vr_halo_sfr = hd.read_data(bfile, 'Halo_SFR')
    vr_halo_m200c = hd.read_data(bfile, 'Halo_M200c')
    vr_halo_types = hd.read_data(bfile, 'HaloTypes')
    vr_halo_flag = hd.read_data(bfile, 'Flag_MostMassiveInHalo')
    
    hd.write_data(args.outfile, f'ID{isim}/Times', times)
    hd.write_data(args.outfile, f'ID{isim}/Redshifts', redshifts)
    hd.write_data(args.outfile, f'ID{isim}/FirstIndices', first_indices)

    hd.write_data(args.outfile, f'ID{isim}/Haloes', vr_haloes)
    hd.write_data(args.outfile, f'ID{isim}/Halo_MStar', vr_halo_mstar)
    hd.write_data(args.outfile, f'ID{isim}/Halo_SFR', vr_halo_sfr)
    hd.write_data(args.outfile, f'ID{isim}/Halo_M200c', vr_halo_m200c)
    hd.write_data(args.outfile, f'ID{isim}/Halo_Types', vr_halo_types)
    hd.write_data(args.outfile, f'ID{isim}/Halo_FlagMostMassiveBH', vr_halo_flag)
    
            
if __name__ == '__main__':
    main()
    

