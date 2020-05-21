import numpy as np
from pdb import set_trace
import argparse
import glob
import hydrangea.hdf5 as hd

print("Parsing input arguments...")
parser = argparse.ArgumentParser(description="Parse input parameters.")
parser.add_argument('sim', type=int, help='Simulation index to analyse')
parser.add_argument('snapshot', type=int, help='Snapshot to analyse')
parser.add_argument('--mrange', type=float, nargs='+', help='Subgrid mass range',
                    default=[5.0, 100.0])
parser.add_argument('--ftrange', type=float, nargs='+', help='Formation aexp range',
                    default=[0, 1])
args = parser.parse_args()


dirs = glob.glob(f'/cosma7/data/dp004/dc-bahe1/EXL/ID{args.sim}*/')
if len(dirs) != 1:
    print(f"Could not unambiguously find directory for simulation {args.sim}!")
    set_trace()
datafile = dirs[0] + f'/output_{args.snapshot:04d}.hdf5'
print(f"Analysing output file {datafile}...")

bh_ids = hd.read_data(datafile, 'PartType5/ParticleIDs')
if bh_ids is None:
    print("No BHs found in output snapshot!")
else:
    bh_msg = np.log10(hd.read_data(datafile, 'PartType5/SubgridMasses') * 1e10)
    bh_ft = hd.read_data(datafile, 'PartType5/FormationScaleFactors')

    ind = np.nonzero((bh_msg >= args.mrange[0]) &
                     (bh_msg <= args.mrange[1]) &
                     (bh_ft >= args.ftrange[0]) &
                     (bh_ft <= args.ftrange[1]))[0]

    print(f"There are {len(ind)} BHs satisfying criteria.")

    for ibh in ind:
        print(f"ID {bh_ids[ibh]}: m_sg={bh_msg[ibh]:.3f}, a_form={bh_ft[ibh]:.3f}")
