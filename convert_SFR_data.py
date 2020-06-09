"""Convert SFR data into HDF5 format."""

import numpy as np
from astropy.io import ascii
import hydrangea.hdf5 as hd
import sys
import glob
from pdb import set_trace
import time
import argparse
import local
import os

def main():

    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    parser.add_argument('sims', help='Simulation index (or indices) or names to analyse',
                        nargs='+')
    parser.add_argument('--basedir', help='Base directory of simulations '
                        '(default: [LOCAL])',
                        default=local.BASE_DIR)
    parser.add_argument('--outfile', help='File to store output in (default: '
        '"SFR.hdf5")', default='SFR.hdf5')
    args = parser.parse_args()

    if args.sims[0].lower() == 'all':
        args.sims = glob.glob(f'{args.basedir}/*')
        is_full_path = True
    else:
        is_full_path = False
        
    for sim_id_str in args.sims:

        if is_full_path:
            wdir = sim_id_str
        else:
            try:
                sim_id = int(sim_id_str)
                sim_dirs = glob.glob(f'{args.basedir}/ID{sim_id}*/')
                if len(sim_dirs) != 1:
                    print(f"No (unique) match for sim ID {sim_id}..")
                    set_trace()
                wdir = sim_dirs[0]

            except:
                sim_id = sim_id_str
                wdir = f'{args.basedir}/{sim_id}/'

        if not os.path.isdir(wdir):
            continue
                
        sfr_file = wdir + '/SFR.txt'
        out_file = wdir + '/' + args.outfile

        if not os.path.isdir(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        
        if os.path.isfile(sfr_file):
            convert_sfr(sfr_file, out_file)
                
def convert_sfr(ascii_file, hdf5_file, sim_type='swift', unit_sfr=1.022690e-2):

    #hdf5_file = '.'.join(ascii_file.split('.')[:-1]) + '.hdf5'

    stime = time.time()
    print(f"Reading file '{ascii_file}'...", end='', flush=True)
    sfrdata = ascii.read(ascii_file)    
    print(f"done (took {(time.time()-stime):.3f} sec.)")

    if sim_type.lower() == 'swift':
        
        aexp = np.array(sfrdata['col3']) 
        zred = np.array(sfrdata['col4'])
        sfr = np.array(sfrdata['col8']) * unit_sfr

    else:  # Gadget-style SFR log
        aexp = np.array(sfrdata['col1'])
        zred = 1/aexp - 1
        sfr = np.array(sfrdata['col3'])

    hd.write_data(hdf5_file, 'aExp', aexp)
    hd.write_data(hdf5_file, 'Redshift', zred)
    hd.write_data(hdf5_file, 'SFR', sfr)

if __name__ == '__main__':
    main()
