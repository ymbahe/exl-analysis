"""Extract the black hole information from outputs into a single file."""

import numpy as np
import hydrangea.hdf5 as hd
import hydrangea.crossref as hx
from pdb import set_trace
import argparse
import glob
import time

def main():
    """Main program"""

    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    parser.add_argument('sim', type=int, help='Simulation index to analyse')
    parser.add_argument('--name', help='Name prefix of simulation outputs '
        '(default: "output")', default='output')
    parser.add_argument('--maxsnap', type=int,
        help='Maximum number of outputs (default: 3000)', default=3000)
    parser.add_argument('--basedir', help='Base directory of simulations',
        default='/cosma7/data/dp004/dc-bahe1/EXL/')
    parser.add_argument('--outfile', help='File to store output in (default: '
        '"black_hole_data.hdf5"')
    args = parser.parse_args()

    # Sanity checks on input arguments
    if not args.basedir.endwith('/'):
        args.basedir = args.basedir + '/'

    args.firstsnap = None
    args.lastsnap = None

    # Construct simulation directory to process
    dirs = glob.glob(args.basedir + f'ID{args.sim}*/')
    if len(dirs) != 1:
        print(f"Could not unambiguously find directory for simulation "
              f"{args.sim}!")
        set_trace()
    args.wdir = dirs[0]
    if not args.wdir.endswith('/'):
        args.wdir = args.wdir + '/'

    # Find total number of black holes and assign their black-IDs
    bpart_ids = find_black_ids(args)

    if args.firstsnap is None:
        print("Did not find any black holes, aborting.")
        return

    # Set up output arrays
    output_dict, comment_dict = setup_output(args)

    # Loop through all snapshots and fill output arrays
    for iisnap, isnap in enumerate(range(args.firstsnap, args.lastsnap+1)):
        process_output(iisnap, isnap, output_dict, bpart_ids, args)

    # Write output HDF5 file
    write_output_file(output_dict, comment_dict, bpart_ids, args)


def find_black_ids(args):
    """Get the IDs of all black holes that ever existed in a simulation."""

    # List holding all black hole particle IDs, starts with zero elements.
    particle_ids = np.zeros(0, dtype=int)

    for isnap in range(args.maxsnap+1):
        snapfile = args.wdir + args.name + f'_{isnap:04d}.hdf5'
        if not os.path.isfile(snapfile):
            continue

        # Load IDs of all black holes existing in current output
        bpart_ids = hd.read_data(snapfile, 'PartType5/ParticleIDs')
        if bpart_ids is None:
            print(f"Did not find any black holes in output {isnap}...")
            continue
        else:
            # Update first/last-snap-with-BHs tracker
            args.lastsnap = isnap
            if args.firstsnap is None:
                args.firstsnap = isnap

        # Check which of these are new to the club
        if len(particle_ids) == 0:
            ind_new = np.arange(len(bpart_ids))
        else:
            bhids, ind_old = hx.find_id_indices(bpart_ids, particle_ids)
            ind_new = np.nonzero(bhids < 0)[0]
            print(f"Found {len(ind_new)} new BHs in output {isnap}.")

        # Subscribe all new members
        if len(ind_new) > 0:
            particle_ids = np.concatenate((particle_ids, bpart_ids[ind_new]))

    # Done looping through outputs, report what we caught
    args.num_bhs = len(particle_ids)
    if args.firstsnap is not None:
        args.num_bh_snaps = args.lastsnap - args.firstsnap + 1
    else:
        args.num_bh_snaps = 0
    print(f"Found a total of {args.num_bhs} black holes in "
          f"{args.num_bh_snaps} snapshots.")
    return particle_ids


def setup_output(args):
    """Set up a dict of arrays to hold the various black hole data."""

    # Get the names of all existing BH data sets
    snapfile = args.wdir + args.name + f'_{args.firstsnap:04d}.hdf5'
    bh_datasets = hd.list_datasets(snapfile, 'PartType5')
    print(f"There are {len(bh_datasets)} BH data sets...")

    # Starting from empty dict, add one array for each data set (except IDs)
    output_dict = {}
    comment_dict = {}
    for dset in bh_datasets:

        # We don't need to load particle IDs, have these already
        if dset == 'ParticleIDs':
            continue
        
        # For simplicity, read the data set in to get its shape/type
        data = hd.read_data(snapfile, f'PartType5/{dset}')
        comment = hd.read_attribute(snapfile, f'PartType5/{dset}',
            'Description')
        if data is None:
            print(f"Strange -- could not read BH data set {dset}?!")
            set_trace()
        outshape = list(data.shape)
        outshape[0] = args.num_bhs
        outshape.append(args.num_bh_snaps)
        array = np.zeros(tuple(outshape), data.dtype) + np.nan

        # Add array to overall dict
        output_dict[dset] = array
        comment_dict[dset] = comment

    print("... finished creating output arrays.")
    return output_dict, comment_dict       


def process_output(iisnap, isnap, output_dict, bpart_ids, args):
    """Transcribe black hole data from one simulation output file.

    Parameters:
    -----------
    iisnap : int
        Index of currently processed output in collective array.
    isnap : int
        Simulation index of currently processed output.
    output_dict : dict of ndarrays
        Dictionary containing arrays to be filled with data.
    bpart_ids : ndarray
        The IDs of black holes to fill into output lists.
    args : dict of values
        Configuration parameters.
    """    
    print(f"Transcribing BH data for snapshot {isnap}...")
    stime = time.time()

    snapfile = args.wdir + args.name + f'_{isnap:04d}.hdf5'

    # Get the names of all data sets to transcribe
    dataset_list = list(output_dict.keys())

    # Load IDs of particles in current output snapshot:
    bpart_ids_curr = hd.read_data(snapfile, 'PartType5/ParticleIDs')

    # Convert them to 'Black-IDs', i.e. their index in the output list
    bh_ids, ind_matched = hx.find_id_indices(bpart_ids_curr, bpart_ids)
    if len(ind_matched) != len(bpart_ids_curr):
        print(f"Why can't we match all BHs from output {isnap}?!?")
        set_trace()

    # Go through all to-transcribe data sets and copy them out
    for dset in dataset_list:

        # Make sure that the output data set has the expected shape
        if output_dict[dset].shape[0] != len(bpart_ids):
            print(f"Inconsistent shape of BH output array '{dset}'.")
            set_trace()
        
        # Load the data, make sure this actually worked
        data = hd.read_data(snapfile, 'PartType5/' + dset)
        if data is None:
            print(f"Oh my goodness, why can we now not find data set "
                  f"'{dset}' for black holes in output {isnap}?")
            set_trace()

        output_dict[dset][bh_ids, ..., iisnap] = data

    print(f"... finished in {time.time() - stime} sec.")


def write_output_file(output_dict, comment_dict, bpart_ids, args):
    """Write the completed arrays to an HDF5 file."""
    print(f"Writing output file '{args.wdir + args.outfile}...'")

    dataset_list = list(output_dict.keys())

    hd.write_data(args.wdir + args.outfile, bpart_ids)
    for dset in dataset_list:
        hd.write_data(args.wdir + args.outfile, dset, output_dict[dset],
            comment=comment_dict[dset])

    print("...done!")


# Execute main program
if __name__ == "__main__":
    main()