"""Find possible matches for a given BH index in another simulation."""

import numpy as np
import hydrangea.hdf5 as hd
import glob
import os
import argparse
from pdb import set_trace

print("Parsing input arguments...")
parser = argparse.ArgumentParser(description="Parse input parameters.")
parser.add_argument('sim1', type=int, help='Simulation of reference BH')
parser.add_argument('sim2', type=int, help='Simulation in which to find matches')
parser.add_argument('snapshot', type=int, help='Snapshot of BH')
parser.add_argument('index', type=int, help='Index of ref BH in specified snap.',
                    default=None, nargs='?')
parser.add_argument('--id', type=int, help='ID of BH in reference simulation')
parser.add_argument('--different_snaps', action='store_true')
parser.add_argument('--name1', default='eagle')
parser.add_argument('--name2', default='eagle')


args = parser.parse_args()

dirs1 = glob.glob(f'/cosma7/data/dp004/dc-bahe1/EXL/ID{args.sim1}*/')
dirs2 = glob.glob(f'/cosma7/data/dp004/dc-bahe1/EXL/ID{args.sim2}*/')
if len(dirs1) != 1 or len(dirs2) != 1:
    set_trace()
wdir1 = dirs1[0]
wdir2 = dirs2[0]
print(f"Analysing simulation {wdir1}")
print(f"Matching into simulation {wdir2}")

# Look up info about target BH in REF simulation and snapshot 
datafile_ref = wdir1 + f'/{args.name1}_{args.snapshot:04d}.hdf5'

if args.index is not None:
    bh_id = hd.read_data(datafile_ref, 'PartType5/ParticleIDs', read_index=args.index)
else:
    if args.id is None:
        print("Well, we need either an index or an ID!")
        set_trace()
    bh_ids = hd.read_data(datafile_ref, 'PartType5/ParticleIDs')
    args.index = np.nonzero(bh_ids == args.id)[0]
    if len(args.index) != 1:
        print(f"Could not unambiguously find BH ID {args.id}...")
        set_trace()
    args.index = args.index[0]
    bh_id = args.id
    
bh_ft = hd.read_data(datafile_ref, 'PartType5/FormationScaleFactors',
                     read_index=args.index)
bh_pos = hd.read_data(datafile_ref, 'PartType5/Coordinates',
                      read_index=args.index)
aexp_ref = hd.read_attribute(datafile_ref, 'Header', 'Scale-factor')[0]

print(f"BH ID {bh_id} formed at aexp={bh_ft:.4f}")
print(f"BH position at snap {args.snapshot}: ", bh_pos)

# Set the formation time range of possible BH matches
search_aexp = np.array([bh_ft-0.1, bh_ft+0.1])
search_aexp = np.clip(search_aexp, 0, min(1, aexp_ref))

datafile_end = None
for isnap in range(args.snapshot+1):
    datafile_end = wdir1 + f'/{args.name1}_{isnap:04d}.hdf5'

    if not os.path.isfile(datafile_end):
        continue

    aexp_factor = hd.read_attribute(datafile_end, 'Header', 'Scale-factor')[0]
    if aexp_factor >= search_aexp[1]:
        print(f"First snapshot after BH formation interval is {isnap}")
        print(f"Aexp = {aexp_factor:.4f}")
        break

if datafile_end is None:
    print("Something went wrong in snapshot lookup...")
    set_trace()

# Load ref BH position in end snapshot
bh_ids_end = hd.read_data(datafile_end, 'PartType5/ParticleIDs')

ind_targ = np.nonzero(bh_ids_end == bh_id)[0]
if len(ind_targ) != 1:
    print("Something went wrong -- could not find ref BH in end snap.")
    set_trace()
    
bh_pos_end = hd.read_data(datafile_end, 'PartType5/Coordinates', read_index=ind_targ[0])

# Now load BHs in COMPARISON sim in same end snapshot

if args.different_snaps:
    datafile_comp = None
    for isnap2 in range(10000):
        datafile_comp = wdir2 + f'/{args.name2}_{isnap2:04d}.hdf5'
        if not os.path.isfile(datafile_comp):
            continue

        aexp_factor = hd.read_attribute(datafile_comp, 'Header', 'Scale-factor')[0]
        if aexp_factor >= search_aexp[1]:
            print(f"First snapshot after BH formation interval in sim 2 is {isnap2}")
            print(f"Aexp = {aexp_factor:.4f}")
            break

    if datafile_comp is None:
        print("Something went wrong in snapshot lookup...")
        set_trace()
else:
    datafile_comp = wdir2 + f'/{args.name2}_{isnap:04d}.hdf5'

bh_ids_comp = hd.read_data(datafile_comp, 'PartType5/ParticleIDs')
bh_fts_comp = hd.read_data(datafile_comp, 'PartType5/FormationScaleFactors')
bh_pos_comp = hd.read_data(datafile_comp, 'PartType5/Coordinates')

ind_cand = np.nonzero((bh_fts_comp >= search_aexp[0]) &
                      (bh_fts_comp <= search_aexp[1]) &
                      (np.max(np.abs(bh_pos_comp-bh_pos_end[None, :]), axis=1) < 1.0))[0]

print(f"There are {len(ind_cand)} candidate matches.")

sorter = np.argsort(bh_fts_comp[ind_cand])
for icand in ind_cand[sorter]:
    dpos = bh_pos_comp[icand, :] - bh_pos_end
    print(f"Cand {icand}: ID={bh_ids_comp[icand]}, ft={bh_fts_comp[icand]:.3f}, "
          f"dx={dpos[0]:.3f}, dy={dpos[1]:.3f}, dz={dpos[2]:.3f}")

if len(ind_cand) == 0:
    set_trace()

