"""Convert SFR data into HDF5 format."""

import numpy as np
from astropy.io import ascii
import hydrangea.hdf5 as hd
import sys
import glob
from pdb import set_trace
import time

def main():

    for sim_id_str in sys.argv[1:]:
        sim_id = int(sim_id_str)
        sim_dir = glob.glob(f'ID{sim_id}*/')

        if len(sim_dir) != 1:
            print(f"No (unique) match for sim ID {sim_id}..")
            set_trace()
            
        sfr_file = sim_dir[0] + '/SFR.txt'
        convert_sfr(sfr_file)
                
def convert_sfr(ascii_file, sim_type='swift', unit_sfr=1.022690e-2):

    hdf5_file = '.'.join(ascii_file.split('.')[:-1]) + '.hdf5'

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
