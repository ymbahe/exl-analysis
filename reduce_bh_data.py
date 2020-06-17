"""Extract the black hole information from outputs into a single file."""

import numpy as np
import hydrangea.hdf5 as hd
import hydrangea.crossref as hx
from pdb import set_trace
import argparse
import glob
import time
import os
import local
import xltools as xl
import h5py as h5

def main():
    """Main program"""

    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    parser.add_argument('sims', help='Simulation inde(x/ices) or names to analyse',
                        nargs='+')
    parser.add_argument('--snap_name', help='Name prefix of simulation outputs '
        '(default: "output")', default='output')
    parser.add_argument('--max_snap', type=int,
        help='Maximum number of outputs (default: 3000)', default=3000)
    parser.add_argument('--base_dir', help='Base directory of simulations '
                                           f'(default: {local.BASE_DIR})',
                        default=local.BASE_DIR)
    parser.add_argument('--full_dir', action='store_true')
    parser.add_argument('--out_file', help='File to store output in (default: '
        '"black_hole_data.hdf5")', default='black_hole_data.hdf5')
    parser.add_argument('--include', help='Only include the listed data sets',
                        nargs='+')
    parser.add_argument('--exclude', help='Exclude the listed data sets',
                        nargs='+')
    parser.add_argument('--vr_snap', default=36, type=int,
                        help='Link to VR catalogue in this snapshot '
                        '(default: 36). Set to -1 to disable VR linking.')
    parser.add_argument('--vr_file', default='vr',
                        help='Base name of VR catalogue to use (default: "vr")')
    parser.add_argument('--out_dir')
    args = parser.parse_args()

    
    # Sanity checks on input arguments
    if not args.base_dir.endswith('/'):
        args.base_dir = args.base_dir + '/'

    if args.vr_snap < 0:
        args.vr_snap = None
        
    if args.sims[0].lower() == 'all':
        args.sims = xl.get_all_sims(args.base_dir)
        have_full_sim_dir = True
    elif args.full_dir:
        have_full_sim_dir = True
    else:
        have_full_sim_dir = False

    args.combined_vr = True
        
    for isim in args.sims:
        process_sim(args, isim, have_full_sim_dir)

        
def process_sim(args, isim, have_full_sim_dir):
        
    args.first_snap = None
    args.last_snap = None
        
    if have_full_sim_dir:
        args.wdir = isim
    else:
        args.wdir = xl.get_sim_dir(args.base_dir, isim)

    if args.out_dir is None:
        args.out_dir = args.wdir
            
    # Find total number of black holes and assign their black-IDs
    bpart_ids, bpart_first_output = find_black_ids(args)

    if args.first_snap is None:
        print("Did not find any black holes, aborting.")
        return

    # Set up output arrays
    output_dict, comment_dict = setup_output(args)

    # For efficiency, create a reverse list of BH IDs (if possible)
    if max(bpart_ids) < 1e10:
        bpart_rev_ids = hx.ReverseList(bpart_ids, assume_positive=True)
        use_rev_list = True
    else:
        use_rev_list = False

    # Retrieve header info from first valid snapshot
    collect_header_data(args)

    # Loop through all snapshots and fill output arrays
    for iisnap, isnap in enumerate(range(args.first_snap, args.last_snap+1)):
        if use_rev_list:
            process_output(iisnap, isnap, output_dict, bpart_ids, args, bpart_rev_ids=bpart_rev_ids)
        else:
            process_output(iisnap, isnap, output_dict, bpart_ids, args, bpart_rev_ids=None)

    # Connect black holes to z = 0 galaxies
    # For this, exclude BHs that are not BHs at the linking snapshot.

    if args.vr_snap is not None:
        get_vr_props(args)
        bh_vr_snap = np.argmin(np.abs(args.redshifts - args.vr_zred))
        print(f"VR snap corresponds to BH output index {bh_vr_snap}.")
        bpart_ids_mod = np.copy(bpart_ids)
        ind_nobh = np.nonzero(output_dict['SubgridMasses']
                                     [:, bh_vr_snap] *.0 != 0)[0]
        bpart_ids_mod[ind_nobh] = -1
        gal_props = connect_to_galaxies(bpart_ids_mod, args)
            
        # Finish galaxy-based analysis
        if gal_props is not None:
            finish_galaxy_analysis(output_dict, gal_props, args)

    else:
        gal_props = None
            
    # Write output HDF5 file
    write_output_file(output_dict, comment_dict, bpart_ids, bpart_first_output,
                          gal_props, args)


def get_vr_props(args):
    """Construct VR file names and redshift for matching."""

    if args.vr_snap is None:
         return

    if args.combined_vr:
        args.vr_particles = args.wdir + f'{args.vr_file}_{args.vr_snap:04d}_particles.hdf5'
        args.vr_outfile = args.wdir + f'{args.vr_file}_{args.vr_snap:04d}.hdf5'
    else:
        print("Please transcribe VR catalogue...")
        set_trace()

    aexp = float(hd.read_attribute(args.vr_outfile, 'SimulationInfo',
                                   'ScaleFactor'))
    args.vr_zred = 1/aexp - 1
    
    
def find_black_ids(args):
    """Get the IDs of all black holes that ever existed in a simulation."""

    # List holding all black hole particle IDs, starts with zero elements.
    particle_ids_set = set()

    # We only use the above set for efficient finding of new members. Actual
    # "membership list" is kept in a separate array, so we can keep it aligned
    # with the list of first snapshots.
    particle_ids = np.zeros(0, dtype=int)
    first_snaps = np.zeros(0, dtype=int)
    
    for isnap in range(args.max_snap+1):
        snapfile = args.wdir + args.snap_name + f'_{isnap:04d}.hdf5'
        if not os.path.isfile(snapfile):
            continue

        # Load IDs of all black holes existing in current output
        bpart_ids = hd.read_data(snapfile, 'PartType5/ParticleIDs')
        if bpart_ids is None:
            print(f"Did not find any black holes in output {isnap}...")
            continue
        else:
            #print(f"Processing output {isnap}...")
            bpart_ids = bpart_ids.astype(int)
            # Update first/last-snap-with-BHs tracker
            args.last_snap = isnap
            if args.first_snap is None:
                args.first_snap = isnap

        # Check which of these are new to the club
        if len(particle_ids_set) == 0:
            ind_new = np.arange(len(bpart_ids))
        else:
            status = [_id in particle_ids_set for _id in bpart_ids]
            ind_new = [i for i, val in enumerate(status) if not val]
            #ind_new = np.nonzero(status is False)[0]
            #    bhids, ind_old = hx.find_id_indices(bpart_ids, particle_ids)
            #ind_new = np.nonzero(bhids < 0)[0]

        print(f"Found {len(ind_new)} new BHs in output {isnap} (out of "
              f"{len(bpart_ids)}).")
                
        # Subscribe all new members
        if len(ind_new) > 0:
            for inew in ind_new:
                particle_ids_set.add(bpart_ids[inew])

            particle_ids = np.concatenate((particle_ids, bpart_ids[ind_new]))
            first_snaps = np.concatenate(
                (first_snaps, np.zeros(len(ind_new), dtype=int) + isnap))
            
    # Done looping through outputs, report what we caught
    particle_ids_from_set = np.array(list(particle_ids_set))

    if len(particle_ids_from_set) != len(particle_ids):
        print("Inconsistent number of caught BHs!")
        set_trace()
    if np.max(np.abs(np.sort(particle_ids_from_set)
                     - np.sort(particle_ids))) > 0:
        print("Inconsistent IDs of caught BHs!")
        set_trace()

    args.num_bhs = len(particle_ids)
    if args.first_snap is not None:
        args.num_bh_snaps = args.last_snap - args.first_snap + 1
        first_snaps -= args.first_snap
    else:
        args.num_bh_snaps = 0
    print(f"Found a total of {args.num_bhs} black holes in "
          f"{args.num_bh_snaps} snapshots.")

    return particle_ids, first_snaps


def connect_to_galaxies(bpart_ids, args):
    """Connect black holes to galaxies at z = 0."""

    if args.vr_snap is None:
        print("Skipping galaxy linking on your request...")
        return

    if args.combined_vr:
        args.vr_particles = args.wdir + f'{args.vr_file}_{args.vr_snap:04d}_particles.hdf5'
        args.vr_outfile = args.wdir + f'{args.vr_file}_{args.vr_snap:04d}.hdf5'
    else:
        print("Please transcribe VR catalogue...")
        set_trace()

    aexp = float(hd.read_attribute(args.vr_outfile, 'SimulationInfo',
                                   'ScaleFactor'))
    args.vr_zred = 1/aexp - 1
    args.vr_aexp = aexp

    print(f"Connecting to VR snapshot {args.vr_snap} at redshift "
          f"{args.vr_zred}...")
    
    # Load VR particle IDs
    vr_ids = hd.read_data(args.vr_particles, 'Haloes/IDs')
    vr_nums = hd.read_data(args.vr_particles, 'Haloes/Numbers')
    vr_offsets = hd.read_data(args.vr_particles, 'Haloes/Offsets')

    # Locate 'our' BHs in the VR ID list
    print("Locating BHs in VR list...")
    stime = time.time()
    ind_in_vr, found_in_vr = hx.find_id_indices(bpart_ids, vr_ids)
    print(f"... took {(time.time() - stime):.3f} sec., located "
          f"{len(found_in_vr)} "
          f"/ {len(bpart_ids)} BHs in VR list "
          f"({len(found_in_vr)/len(bpart_ids)*100:.3f}%).")
    
    # Now convert VR index to halo
    bh_halo = np.zeros(len(bpart_ids), dtype=int)-1
    halo_guess = np.searchsorted(vr_offsets, ind_in_vr[found_in_vr],
                                 side='right')-1
    ind_good = np.nonzero(ind_in_vr[found_in_vr] <
                          (vr_offsets[halo_guess] + vr_nums[halo_guess]))[0]
    bh_halo[found_in_vr[ind_good]] = halo_guess[ind_good]

    print(f"... could match {len(ind_good)} / {len(bpart_ids)} BHs to haloes. "
          f"({len(ind_good)/len(bpart_ids)*100:.3f}%).")

    gal_props = {'halo': bh_halo}
    
    # Add a few key properties of the haloes, for convenience
    ind_in_halo = found_in_vr[ind_good]

    vr_mstar = hd.read_data(args.vr_outfile, 'ApertureMeasurements/30kpc/Stars/Masses')
    vr_sfr = hd.read_data(args.vr_outfile, 'ApertureMeasurements/30kpc/SFR/')
    vr_m200c = hd.read_data(args.vr_outfile, 'M200crit')
    vr_haloTypes = hd.read_data(args.vr_outfile, 'StructureTypes')
    
    gal_props['MStar'] = np.zeros(len(bpart_ids))
    gal_props['SFR'] = np.zeros(len(bpart_ids))
    gal_props['M200'] = np.zeros(len(bpart_ids))
    gal_props['HaloTypes'] = np.zeros(len(bpart_ids), dtype=int)
    
    gal_props['MStar'][ind_in_halo] = vr_mstar[bh_halo[ind_in_halo]]
    gal_props['SFR'][ind_in_halo] = vr_sfr[bh_halo[ind_in_halo]]
    gal_props['M200'][ind_in_halo] = vr_m200c[bh_halo[ind_in_halo]]
    gal_props['HaloTypes'][ind_in_halo] = vr_haloTypes[bh_halo[ind_in_halo]]

    return gal_props


def finish_galaxy_analysis(output_dict, gal_props, args):
    """Finish calculation of galaxy-based quantities, now that we have
       loaded all the BH masses."""

    ind_for_vr = np.argmin(np.abs(args.redshifts - args.vr_zred))
    print("Finding most massive BHs per galaxy at z = "
          f"{args.redshifts[ind_for_vr]:.3f} "
          f"(VR at z = {args.vr_zred:.3f})...")

    stime = time.time()
    
    haloes_unique = np.unique(gal_props['halo'])
    flag_most_massive = np.zeros(len(gal_props['halo']), dtype=int)
    
    for ihalo in haloes_unique:
        if ihalo < 0: continue  # Don't care about out-of-halo BHs
        ind_in_this = np.nonzero(gal_props['halo'] == ihalo)[0]
        msg_thishalo = output_dict['SubgridMasses'][ind_in_this, ind_for_vr]
        try:
            max_in_this = np.nanargmax(msg_thishalo)
        except ValueError:
            print("Something fishy is going on here.")
            set_trace()
            
        flag_most_massive[ind_in_this[max_in_this]] = 1

    print("Finished finding most massive BHs per galaxy, took "
          f"{(time.time() - stime):.3f} sec.")

    gal_props['flag_most_massive_bh'] = flag_most_massive

    
def setup_output(args):
    """Set up a dict of arrays to hold the various black hole data."""

    # Get the names of all existing BH data sets
    snapfile = args.wdir + args.snap_name + f'_{args.first_snap:04d}.hdf5'
    bh_datasets = hd.list_datasets(snapfile, 'PartType5')
    print(f"There are {len(bh_datasets)} BH data sets...")

    # Starting from empty dict, add one array for each data set (except IDs)
    output_dict = {}
    comment_dict = {}

    for dset in bh_datasets:

        # We don't need to load particle IDs, have these already
        if dset == 'ParticleIDs':
            continue

        if args.include is not None and dset not in args.include:
            continue
        if args.exclude is not None and dset in args.exclude:
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

    args.times = np.zeros(args.num_bh_snaps)
    args.redshifts = np.zeros(args.num_bh_snaps)

    return output_dict, comment_dict       


def process_output(iisnap, isnap, output_dict, bpart_ids, args, bpart_rev_ids=None):
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
    rev : bool, optional
        If True, assume that bpart_ids is actually the reverse list of
        BH IDs.
    """
    if iisnap % 50 == 0:
        print(f"Transcribing BH data for snapshot {isnap}...")
        stime = time.time()

    snapfile = args.wdir + args.snap_name + f'_{isnap:04d}.hdf5'

    # Get the names of all data sets to transcribe
    dataset_list = list(output_dict.keys())

    cstime = time.time()
    # Load IDs of particles in current output snapshot:
    bpart_ids_curr = hd.read_data(snapfile, 'PartType5/ParticleIDs')

    # Convert them to 'Black-IDs', i.e. their index in the output list
    if bpart_rev_ids is not None:
        rstime = time.time()
        bh_ids = bpart_rev_ids.query(bpart_ids_curr)
        #print(f"Querying {isnap} took {(time.time()-rstime):.3f} sec.")
        ind_matched = np.nonzero(bh_ids >= 0)[0]
        rstime = time.time()
        #print(f"Checking {isnap} took {(time.time()-rstime):.3f} sec.")
    else:
        fstime = time.time()
        bh_ids, ind_matched = hx.find_id_indices(bpart_ids_curr, bpart_ids)
        print(f"FII {isnap} took {(time.time()-fstime):.3f} sec.")
        
    if len(ind_matched) != len(bpart_ids_curr):
        print(f"Why can't we match all BHs from output {isnap}?!?")
        set_trace()
    cetime = time.time()

    if iisnap % 50 == 0:
        print(f"... lookup took {cetime - cstime:.3f} sec.")
    
        
    # Load the time and redshift of current output
    redshift = hd.read_attribute(snapfile, 'Header', 'Redshift')[0]
    sim_time = hd.read_attribute(snapfile, 'Header', 'Time')[0]
    utime = hd.read_attribute(snapfile, 'Units', 'Unit time in cgs (U_t)')[0]
    utime /= (3600.0 * 24 * 365.24 * 1e9)  # Convert from sec to Gyr
    sim_time *= utime

    args.times[iisnap] = sim_time
    args.redshifts[iisnap] = redshift
            
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

    if iisnap % 50 == 0:
        print(f"... finished in {time.time() - stime:.3f} sec.")


def collect_header_data(args):
    """Copy relevant Header/code data sections to output file."""
    snap_file = args.wdir + args.snap_name + f'_{args.first_snap:04d}.hdf5'
    out_file = args.out_dir + args.out_file 

    f_snap = h5.File(snap_file, 'r')
    f_out = h5.File(out_file, 'w')

    for igrp in ['Header', 'Code', 'Cosmology', 'GravityScheme', 'HydroScheme',
                 'InternalCodeUnits', 'Parameters', 'PhysicalConstants',
                 'Policy', 'StarsScheme', 'SubgridScheme', 'Units',
                 'UnusedParameters']:
        f_snap.copy(igrp, f_out)

    f_snap.close()
    f_out.close()


def write_output_file(output_dict, comment_dict, bpart_ids,
                      bpart_first_outputs, gal_props, args):
    """Write the completed arrays to an HDF5 file."""
    print(f"Writing output file '{args.out_dir + args.out_file}...'")

    dataset_list = list(output_dict.keys())

    hd.write_data(args.out_dir + args.out_file, 'ParticleIDs', bpart_ids)
    hd.write_data(args.out_dir + args.out_file, 'FirstIndices', bpart_first_outputs)
    hd.write_data(args.out_dir + args.out_file, 'Redshifts', args.redshifts)
    hd.write_data(args.out_dir + args.out_file, 'Times', args.times)

    if gal_props is not None:
    
        hd.write_data(args.out_dir + args.out_file, 'Haloes', gal_props['halo'],
                  comment='Index of the velociraptor halo containing each '
                          f'black hole at redshift {args.vr_zred:.3f}.')
        hd.write_attribute(args.out_dir + args.out_file, 'Haloes',
                           'VR_Snapshot', args.vr_snap)
        hd.write_attribute(args.out_dir + args.out_file, 'Haloes',
                           'VR_Redshift', args.vr_zred)
        hd.write_attribute(args.out_dir + args.out_file, 'Haloes',
                           'VR_ScaleFactor', args.vr_aexp)

        hd.write_data(args.out_dir + args.out_file, 'Halo_MStar', gal_props['MStar'],
                  comment='Stellar mass (< 30kpc) of the halo containing '
                          f'the black holes at redshift {args.vr_zred:.3f} '
                          '[M_sun].')
        hd.write_data(args.out_dir + args.out_file, 'Halo_SFR', gal_props['SFR'],
                  comment='Star formation rates (< 30kpc) of the halo '
                          'containing the black holes at redshift '
                          f'{args.vr_zred:.3f} [M_sun/yr].')
        hd.write_data(args.out_dir + args.out_file, 'Halo_M200c', gal_props['M200'],
                  comment='Halo virial masses (M200c) of the halo containing '
                          'the black holes at redshift {args.vr_zred:.3f} '
                          '[M_sun].')
        hd.write_data(args.out_dir + args.out_file, 'HaloTypes', gal_props['HaloTypes'],
                  comment='Types of the haloes containing the black holes at '
                          'redshift {args.vr_zred:.3f}. Central haloes have '
                          'a value of 10.')
        hd.write_data(args.out_dir + args.out_file, 'Flag_MostMassiveInHalo',
                  gal_props['flag_most_massive_bh'],
                  comment='1 if this is the most massive black hole in its '
                          f'halo at redshift {args.vr_zred}, 0 otherwise.')
    
    for dset in dataset_list:
        hd.write_data(args.out_dir + args.out_file, dset, output_dict[dset],
            comment=comment_dict[dset])

    print("...done!")


# Execute main program
if __name__ == "__main__":
    main()
