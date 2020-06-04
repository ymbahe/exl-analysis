"""Extract VR data for specified simulations and snapshots"""

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
    parser.add_argument('--sims', type=int, help='Simulation indices to process.',
                        nargs='+')
    parser.add_argument('--snapshots', type=int, help='Snapshots to process.',
                        nargs='+')
    parser.add_argument('--outfile', help='Name of output catalogue.',
                        default='vr_catalogue.hdf5')
    args = parser.parse_args()

    n_sims = len(args.sims)
    n_snaps = len(args.snapshots)

    args.extract_bh = True
    args.extract_vr = True
    
    args.bh_fields = ['SubgridMasses', 'FormationScaleFactors', 'CumulativeNumberSeeds',
                      'LastHighEddingtonFractionScaleFactors',
                      'LastMajorMergerScaleFactors', 'LastMinorMergerScaleFactors',
                      'TotalAccretedMasses']

    args.vr_fields = [('hostHaloID', 'HostID'),
                      ('Structuretype', 'HaloType'),
                      ('Mass_tot', 'Mass'),
                      ('Mass_200crit', 'M200c'),
                      ('SFR_gas', 'SFR'),
                      ('M_bh', 'MBH_Dynamical'),
                      ('Aperture_SubgridMasses_aperture_total_solar_mass_bh_5_kpc', 'MBH_Subgrid_5kpc'),
                      ('Aperture_SubgridMasses_aperture_total_solar_mass_bh_100_kpc', 'MBH_Subgrid_100kpc'),
                      ('M_star_30kpc', 'MStar_30kpc'),
                      ('SubgridMasses_max_solar_mass_bh', 'MBH_Subgrid_max'),
                      ('M_star', 'MStar'),
                      #('R_HalfMass_star', 'RHalfStar')] 
                      ('Projected_aperture_1_rhalfmass_star_30_kpc', 'RHalfStar')]
                      
    print(f"Processing {n_sims} simulations and {n_snaps} snapshots.")
    if n_sims == 0 or n_snaps == 0:
        set_trace()
    
    for isim in args.sims:
        process_sim(isim, args)

        
def process_sim(isim, args):
    """Process one individual simulation"""

    wdirs = glob.glob(f'{local.BASE_DIR}/ID{isim}*/')
    if len(wdirs) != 1:
        set_trace()
    wdir = wdirs[0]
    print(f"Analysing simulation {wdir}...")

    for isnap in args.snapshots:
        process_snap(isim, wdir, isnap, args)

        
def process_snap(isim, wdir, isnap, args):
    """Process one specific snapshot."""

    snapfile = wdir + f'eagle_{isnap:04d}.hdf5'
    vrfile = wdir + f'vr/halos_{isnap:04d}.properties'

    if args.extract_bh and os.path.isfile(snapfile):
        extract_bh_data(snapfile, isim, isnap, args)

    if args.extract_vr and os.path.isfile(vrfile):
        extract_vr_data(vrfile, isim, isnap, args)

    if os.path.isfile(snapfile):
        zred = hd.read_attribute(snapfile, 'Header', 'Redshift')[0]
        hd.write_attribute(args.outfile, 'Header', 'Redshift', zred)

def extract_bh_data(snapfile, isim, isnap, args):
    """Extract BH data from one particular file"""

    pre = f'ID{isim}/S{isnap}/BH/'
    zred = hd.read_attribute(snapfile, 'Header', 'Redshift')[0]
    hd.write_attribute(args.outfile, f'ID{isim}/S{isnap}', 'Redshift', zred)
    
    for field in args.bh_fields:
        data = hd.read_data(snapfile, 'PartType5/' + field)
        if data is not None:
            hd.write_data(args.outfile, pre + '/' + field, data)

            
def extract_vr_data(vrfile, isim, isnap, args):
    """Extract VR data from one particular file"""

    pre = f'ID{isim}/S{isnap}/'
    for field in args.vr_fields:
        data = hd.read_data(vrfile, field[0])
        if data is None:
            print(f"Could not find field '{field[0]}' in VR file '{vrfile}'!")
            set_trace()
        hd.write_data(args.outfile, f'{pre}/{field[1]}', data)
    
        
if __name__ == '__main__':
    main()
    

