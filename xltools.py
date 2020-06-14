"""Collection of routines for XL analysis."""

import numpy as np
import hydrangea.hdf5 as hd
import hydrangea.crossref as hx
import glob
import os
import operator
from pdb import set_trace

# Define a dict for string lookup of comparison operators
ops = {
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
}


def connect_to_galaxies(bpart_ids, wdir, vr_snap, combined_vr=True):
    """Connect black holes to galaxies at a specified redshift.

	Parameters
	----------
	bpart_ids : ndarray(int)
	    The IDs of the black hole particle to match.
	wdir : string
		Directory of the simulation to work with
    vr_snap : int or None
        Snapshot of the VR catalogue to connect to (None for no matching)
	combined_vr : bool, optional
		Flag for using transcribed VR, must currently be True.

	Returns
	-------
	gal_props : dict
	    Dict with the results of the VR matching:
	    - Haloes --> VR halo indices for each BH, -1 if unmatched
	    - MStar --> Stellar mass of the host halo [M_Sun] within 30 pkpc,
	                np.nan if not matched
	    - SFR --> SFR of the host halo [M_Sun/yr] within 30 pkpc, np.nan
	              if not matched
	    - M200c --> M200c of the host halo [M_Sun], np.nan if not matched
	    - HaloTypes --> Type index of the host halo, -1 if not matched
		- Redshift --> Redshift of VR catalogue
    """
    num_bhs = len(bpart_ids)
    gal_props = {}

    if args.vr_snap is None:
        print("Skipping galaxy linking on your request...")
        return

    if combined_vr:
        vr_particle_file = (f'{args.wdir}{args.vr_file}_'
        	                f'{args.vr_snap:04d}_particles.hdf5')
        vr_main_file = f'{args.wdir}{args.vr_file}_{args.vr_snap:04d}.hdf5'
    else:
        print("Please transcribe VR catalogue...")
        set_trace()

    # Also abort if VR catalogue could not be found
    if ((not os.path.isfile(vr_particle_file)) or
    	(not os.path.isfile(vr_main_file))):
        print(f"VR catalogue for snapshot {vr_snap} does not exist...")
        return None

    # Find redshift of VR catalogue
    aexp = float(hd.read_attribute(vr_main_file, 'SimulationInfo',
    	                           'ScaleFactor'))
    gal_props['Redshift'] = 1/aexp - 1
    print(f"Connecting to VR snapshot {vr_snap} at redshift "
    	  f"{gal_props['Redshift']}...")
    
    # Load VR particle IDs
    vr_ids = hd.read_data(vr_particle_file, 'Haloes/IDs')
    vr_nums = hd.read_data(vr_particle_file, 'Haloes/Numbers')
    vr_offsets = hd.read_data(vr_particle_file, 'Haloes/Offsets')

    # Locate 'our' BHs in the VR ID list
    print("Locating BHs in VR list...")
    stime = time.time()
    ind_in_vr, found_in_vr = hx.find_id_indices(bpart_ids, vr_ids)
    print(f"... took {(time.time() - stime):.3f} sec., located "
          f"{len(found_in_vr)} "
          f"/ {num_bhs} BHs in VR list ({len(found_in_vr)/num_bhs*100:.3f}%).")
    
    # Now convert VR particle indices to halo indices
    halo_guess = np.searchsorted(vr_offsets, ind_in_vr[found_in_vr],
                                 side='right') - 1
    ind_good = np.nonzero(ind_in_vr[found_in_vr] <
                          (vr_offsets[halo_guess] + vr_nums[halo_guess]))[0]

    vr_halo = np.zeros(num_bhs, dtype=int) - 1
    vr_halo[found_in_vr[ind_good]] = halo_guess[ind_good]
    print(f"... could match {len(ind_good)} / {num_bhs} BHs to haloes. "
          f"({len(ind_good)/num_bhs*100:.3f}%).")

    # Store result in (new) dict
    gal_props = {'Haloes': vr_halo}
    
    # Add a few key properties of the haloes, for convenience
    ind_in_halo = found_in_vr[ind_good]

    vr_mstar = hd.read_data(args.vr_main_file,
    	                    'ApertureMeasurements/30kpc/Stars/Masses')
    vr_sfr = hd.read_data(args.vr_main_file, 'ApertureMeasurements/30kpc/SFR/')
    vr_m200c = hd.read_data(args.vr_main_file, 'M200crit')
    vr_haloTypes = hd.read_data(args.vr_main_file, 'StructureTypes')
    
    gal_props['MStar'] = np.zeros(num_bhs) + np.nan
    gal_props['SFR'] = np.zeros(num_bhs) + np.nan
    gal_props['M200c'] = np.zeros(num_bhs) + np.nan
    gal_props['HaloTypes'] = np.zeros(num_bhs, dtype=int) - 1
    
    gal_props['MStar'][ind_in_halo] = vr_mstar[vr_halo[ind_in_halo]]
    gal_props['SFR'][ind_in_halo] = vr_sfr[vr_halo[ind_in_halo]]
    gal_props['M200c'][ind_in_halo] = vr_m200c[vr_halo[ind_in_halo]]
    gal_props['HaloTypes'][ind_in_halo] = vr_haloTypes[vr_halo[ind_in_halo]]

    return gal_props


def get_sim_dir(base_dir, isim):
    """Construct the directory of a simulation.

    Parameters
    ----------
    base_dir : str
        The base directory of the simulation family
    isim : str or int
        The simulation index (if beginning with 'ID[xxx]') or name.
        It is assumed that the simulation lives within [base_dir].

    Returns
    -------
    sim_dir : str
        The full directory of the specified simulation, including a trailing
        forward slash.
    """

    if isinstance(isim, int):
        # Case A: simulations labelled ID[xxx]
        dirs = glob.glob(f'{base_dir}/ID{isim}*/')

        # Make sure nothing stupid has happened
        if len(dirs) != 1:
            print(f"Could not unambiguously find directory for simulation "
            	  f"{args.sim}!")
            set_trace()
        wdir = dirs[0]

    else:
        # Case B: simulations labelled with string name
        wdir = f'{base_dir}/{isim}'
        
    if not wdir.endswith('/'):
        wdir = wdir + '/'

    return wdir


def get_all_sims(base_dir, base_name=None):
    """Find all simulations in a directory.

	Parameters
	----------
	base_dir : str
	    The directory in which to search for simulation runs.
	base_name : str or None
	    A string with which all simulations start. If None (default),
	    all subdirectories are returned. NOT CURRENTLY IMPLEMENTED.

	Returns
	-------
	simulations : list of str
	    A list containing the (full) paths of all matching subdirectories

    """
	return [f.path for f in os.scandir(base_dir) if f.is_dir()]


def swift_Planck_cosmology():
	"""Construct a flat FLRW cosmology that is, emprically, close to what
	   Swift uses as its 'Planck13' cosmology."""
	H0 = 67.79
	h = H0/100
	Oc0 = 0.1187/h**2
	Ob0 = 0.02214/h**2
	Om0 = Oc0 + Ob0
	return FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0)


def lookup_bh_data(bh_data_file, bh_props_list, selection_list=None):
    """Load info from BH file into arrays and find target BHs."""

    bh_data = {}

    # Load the required data from the evolution tables
    num_bhs = None
    for idata in bh_props_list:
        data = hd.read_data(bh_data_file, idata)
        bh_data[idata] = data
        num_bhs = shape(data)[0]

    # Now find the list of BHs we are interested in (based on z=0 props)

    if selection_list is None:
    	sel = None

    else:
    	if num_bhs is None:
    		# Did not load any properties, so cannot select anything. Doh.
    		print("Cannot select any BHs, because we did not load any data.")
    		set_trace()

    	sel = np.arange(num_bhs, dtype=int)
    	for selector in selection_list:
    		
    		# Apply requested comparison operation
    		cmp = ops[selector[1]]
    		ind_subsel = np.nonzero(cmp(bh_data[selector[0]][sel],
    			                        selector[2]))[0]

    		# Adjust selection list
    		sel = sel[ind_subsel]

	    # Sort BHs by index
	    sel = np.sort(sel)
	    print(f"There are {len(sel)} BHs in selection list.")

    return bh_data, sel
