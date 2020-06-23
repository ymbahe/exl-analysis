"""Collection of routines for XL analysis."""

import numpy as np
import hydrangea.hdf5 as hd
import hydrangea.crossref as hx
import glob
import os
import operator
from pdb import set_trace
from astropy.cosmology import FlatLambdaCDM
import time

# Set up Matplotlib
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.use('pdf')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'][0] = 'palatino'
mpl.rc('text', usetex=True)
import matplotlib.pyplot as plt

# Define a dict for string lookup of comparison operators
ops = {
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
}


def connect_to_galaxies(bpart_ids, vr_file, combined_vr=True,
                        extra_props=None):
    """Connect black holes to galaxies at a specified redshift.

    Parameters
    ----------
    bpart_ids : ndarray(int)
        The IDs of the black hole particle to match.
    vr_file : string
        VR file to connect to (None for no matching)
    combined_vr : bool, optional
        Flag for using transcribed VR, must currently be True.
    extra_props : list of tuples, optional
        Extra properties to be retrieved from the VR catalogue. If None
        (default), none are extracted. Entries must be of the form
        ('VR_dataset', 'Output_name').

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
        - [other properties as given in extra_props dict]
        - Redshift --> Redshift of VR catalogue
    """

    num_bhs = len(bpart_ids)
    gal_props = {}

    if vr_file is None:
        print("Skipping galaxy linking on your request...")
        return

    # Look up BH IDs in VR
    gal_props['Haloes'], gal_props['ScaleFactor'], gal_props['Redshift'] = (
        connect_ids_to_vr(bpart_ids, vr_file, require=False))
    
    # Add a few key properties of the haloes, for convenience
    ind_in_halo = np.nonzero(gal_props['Haloes'] >= 0)[0]
    vr_halo = gal_props['Haloes']
    
    vr_main_file = f'{vr_file}.hdf5'
    vr_mstar = hd.read_data(vr_main_file,
                            'ApertureMeasurements/30kpc/Stars/Masses')
    vr_sfr = hd.read_data(vr_main_file, 'ApertureMeasurements/30kpc/SFR/')
    vr_m200c = hd.read_data(vr_main_file, 'M200crit')
    vr_haloTypes = hd.read_data(vr_main_file, 'StructureTypes')
    
    gal_props['MStar'] = np.zeros(num_bhs) + np.nan
    gal_props['SFR'] = np.zeros(num_bhs) + np.nan
    gal_props['M200c'] = np.zeros(num_bhs) + np.nan
    gal_props['HaloTypes'] = np.zeros(num_bhs, dtype=int) - 1
    
    gal_props['MStar'][ind_in_halo] = vr_mstar[vr_halo[ind_in_halo]]
    gal_props['SFR'][ind_in_halo] = vr_sfr[vr_halo[ind_in_halo]]
    gal_props['M200c'][ind_in_halo] = vr_m200c[vr_halo[ind_in_halo]]
    gal_props['HaloTypes'][ind_in_halo] = vr_haloTypes[vr_halo[ind_in_halo]]

    if extra_props is not None:
        for iextra in extra_props:
            print(f"Extracting extra quantity '{iextra[0]}' --> "
                  f"'{iextra[1]}'")
            vr_quant = hd.read_data(vr_main_file, iextra[0])
            gal_props[iextra[1]] = np.zeros(num_bhs) + np.nan
            gal_props[iextra[1]][ind_in_halo] = vr_quant[vr_halo[ind_in_halo]]

    return gal_props


def connect_ids_to_vr(ids, vr_file, require=False):
    """Core function to find the VR halo of a set of IDs."""

    num_ids = len(ids)
    
    vr_particle_file = f'{vr_file}_particles.hdf5'
    vr_main_file = f'{vr_file}.hdf5'
    
    # Abort if VR catalogue could not be found
    if ((not os.path.isfile(vr_particle_file)) or
        (not os.path.isfile(vr_main_file))):
        print(f"VR catalogue {vr_file} does not exist...")
        if require:
            set_trace()
        return None

    # Find redshift of VR catalogue
    aexp = float(hd.read_attribute(vr_main_file, 'SimulationInfo',
                                   'ScaleFactor'))
    zred = 1/aexp - 1
    print(f"Connecting to VR catalogue {vr_file} at redshift {zred}...")
    
    # Load VR particle IDs
    vr_ids = hd.read_data(vr_particle_file, 'Haloes/IDs')
    vr_offsets = hd.read_data(vr_particle_file, 'Haloes/Offsets')

    # Locate 'our' IDss in the VR ID list
    print("Locating IDs in VR list...")
    stime = time.time()
    ind_in_vr, found_in_vr = hx.find_id_indices(ids, vr_ids)
    print(f"... took {(time.time() - stime):.3f} sec., located "
          f"{len(found_in_vr)} "
          f"/ {num_ids} IDs in VR list ({len(found_in_vr)/num_ids*100:.3f}%).")
    
    # Now convert VR particle indices to halo indices
    halo_guess = np.searchsorted(vr_offsets, ind_in_vr[found_in_vr],
                                 side='right') - 1
    ind_good = np.nonzero(ind_in_vr[found_in_vr] < vr_offsets[halo_guess+1])[0]

    vr_halo = np.zeros(num_ids, dtype=int) - 1
    vr_halo[found_in_vr[ind_good]] = halo_guess[ind_good]
    print(f"... could match {len(ind_good)} / {num_ids} IDs to haloes. "
          f"({len(ind_good)/num_ids*100:.3f}%).")

    return vr_halo, aexp, zred
    

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

    try:
        # Case A: simulations labelled ID[xxx]
        isim_int = int(isim)
        dirs = glob.glob(f'{base_dir}/ID{isim}*/')

        # Make sure nothing stupid has happened
        if len(dirs) != 1:
            print(f"Could not unambiguously find directory for simulation "
                  f"ID{isim}!")
            set_trace()
        wdir = dirs[0]

    except ValueError:
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


def lookup_bh_data(bh_data_file, bh_props_list, selection_list=None,
                   bh_list=None):
    """Load info from BH file into arrays and find target BHs."""

    bh_data = {}

    # Make sure all selectors are part of the bh_props_list
    if selection_list is not None:
        for isel in selection_list:
            if isel[0] not in bh_props_list:
                print(f"Selector '{isel[0]}' was not in requested BH props "
                      f"list, adding it now...")
                bh_props_list.append(isel[0])
    
    # Load the required data from the evolution tables
    num_bhs = None
    for idata in bh_props_list:
        data = hd.read_data(bh_data_file, idata)

        # May not get all data, such as VR links 
        if data is not None:
            bh_data[idata] = data
            num_bhs = data.shape[0]

    # Now find the list of BHs we are interested in (based on z=0 props)

    if bh_list is not None:
        sel = bh_list
    elif selection_list is None:
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
            try:
                selector_data = bh_data[selector[0]][sel]
                if len(selector_data.shape) == 1:
                    ind_subsel = np.nonzero(cmp(bh_data[selector[0]][sel],
                                                selector[2]))[0]
                elif len(selector_data.shape) == 2:
                    ind_subsel = np.nonzero(cmp(selector_data[:, selector[3]],
                                                selector[2]))
                else:
                    print("3+ dimensional selector arrays not yet handled.")
                    set_trace()
                    
            except KeyError:
                print(f"Could not locate selector '{selector[0]}' in BH data "
                      "file.")
                if 'Halo' in selector[0]:
                    print("Assuming that halo linking was not done, skipping "
                          "this comparison.")
                    continue
                else:
                    set_trace()
                
            # Adjust selection list
            sel = sel[ind_subsel]

        # Sort BHs by index
        sel = np.sort(sel)
        print(f"There are {len(sel)} BHs in selection list.")

    return bh_data, sel


def legend_item(ax, xr, y, text, alpha=1, ls='-', color='black',
                text_side='right', fontsize=10):
    """Add one legend item to the axes ax."""

    tr = ax.get_xlim()
    yr = ax.get_ylim()

    plt.plot(tr[0] + (tr[1]-tr[0]) * np.array(xr),
             yr[0] + (yr[1]-yr[0]) * np.array([y, y]),
             linestyle=ls, color=color)

    if text_side == 'right':
        text_x, text_ha = np.max(xr) + 0.02, 'left'
    else:
        text_x, text_ha = np.min(xr) -0.02, 'right'

    plt.text(tr[0] + (tr[1]-tr[0]) * text_x,
             yr[0] + (yr[1]-yr[0]) * y,
             text, va='center', ha=text_ha, alpha=alpha, color=color,
             fontsize=fontsize)


def legend_text(ax, x, y, text, alpha=1, color='black', fontsize=8):
    """Add one legend text-only item to the axes ax."""

    tr = ax.get_xlim()
    yr = ax.get_ylim()

    plt.text(tr[0] + (tr[1]-tr[0]) * x,
             yr[0] + (yr[1]-yr[0]) * y,
             text, va='bottom', ha='left', alpha=alpha, color=color,
             fontsize=fontsize)


def plot_cumdist(quantities, weights=None, indices=None,
                 norm=True, log=False, **kwargs):
    """Plot a cumulative distribution."""

    if indices is None:
        indices = np.arange(len(quantities))
    
    sorter = np.argsort(quantities[indices])
    xquant = quantities[indices[sorter]]
    
    if weights is None:
        yquant = np.arange(len(indices)) + 1
        norm_val = len(indices)
    else:
        yquant = np.cumsum(weights[indices[sorter]])
        norm_val = np.sum(weights[indices])

    if norm:
        yquant /= norm_val

    if log:
        yquant = np.log10(yquant)

    plt.plot(xquant, yquant, **kwargs)
