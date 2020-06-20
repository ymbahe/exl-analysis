"""Plot the stellar birth densities."""

import numpy as np
import os
import argparse
from pdb import set_trace

import local
import hydrangea.hdf5 as hd
import xltools as xl

# Set up Matplotlib
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.use('pdf')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'][0] = 'palatino'
mpl.rc('text', usetex=True)
import matplotlib.pyplot as plt

cvec = ['red', 'seagreen', 'royalblue']
ap_list = [5.0, 15.0, 1e5]
ap_min = [0.0, 5.0, 0.0]

def main():
    """Main program loop."""

    # Parse input parameters
    argvals = None
    args = get_args(argvals)

    # Deal with request for all-sim-batch
    if args.sims[0].lower() == 'all':

        if args.bh_bid is not None:
            print("It does not make sense to analyse the same BH BIDs "
                  "in different simulations.")
            return

        args.sims = xl.get_all_sims(args.base_dir)
        args.have_full_sim_dir = True
    else:
        args.have_full_sim_dir = False

    print(f"Processing {len(args.sims)} simulations")

    for isim in args.sims:
        process_sim(isim, args)


def get_args(argv=None):
    """Parse input arguments."""

    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")

    parser.add_argument('sims', nargs='+',
                        help='Simulation index/name to analyse ("all" to '
                             'process all simulations in base_dir.')
    parser.add_argument('--base_dir', default=local.BASE_DIR,
                        help='Base directory of the simulation, default: '
                             f'{local.BASE_DIR}')
    parser.add_argument('--bh_file', default='black_hole_data.hdf5',
                        help='Name of the file containing the BH data, '
                             'default: "black_hole_data.hdf5"')
    parser.add_argument('--plot_prefix',
                        default='gallery/stellar_birth_densities',
                        help='Prefix of output files, default: '
                             '"gallery/stellar_birth_densities".')
    parser.add_argument('--snapshots', type=int, nargs='+',
                        help='Snapshots to analyse.')
    parser.add_argument('--snap_name', default='snapshot',
                        help='Snapshot name prefix (default: "snapshot").')
    parser.add_argument('--vr_name', default='vr',
                        help='Name prefix of (transcribed) VR catalogue; '
                             'default: "vr".')
    
    parser.add_argument('--bh_bid', type=int, nargs='+',
                        help='BID(s) of black hole to analyse. If not '
                             'specified, the selection will be made '
                             'based on the other parameters below.')
    parser.add_argument('--bh_mass_range', type=float, nargs='+',
                        help='Min and max BH mass for selection, at target z '
                             '[M_Sun], default: no selection.')
    parser.add_argument('--bh_selection_redshift', type=float, default=0.0,
                        help='Redshift at which to make BH mass selection '
                             '(default: 0.0)')
    parser.add_argument('--include_subdominant_bhs', action='store_true',
                        help='Only show black holes that are the most massive '
                             'in their galaxy, at the linked snapshot.')
    parser.add_argument('--halo_mstar_range', default=[3e10, 4e11], type=float,
                        help='Min and max stellar mass of host galaxy '
                             '[M_Sun], default: 3e10, 4e11.',
                        nargs='+')
    parser.add_argument('--halo_m200_range', default=[0, 1e16], type=float,
                        help='Min and max halo mass of host galaxy '
                             '[M_Sun], default: 0, 1e16.',
                        nargs='+')
    parser.add_argument('--include_satellites', action='store_true',
                        help='Only show BHs in central haloes, not satellites.')

    return parser.parse_args(argv)


def process_sim(isim, args):
    """Process one individual simulation."""

    if args.have_full_sim_dir:
        args.wdir = isim
    else:
        args.wdir = xl.get_sim_dir(args.base_dir, isim)
    print(f"Analysing simulation {args.wdir}...")

    # Name of the input catalogue, containing all the data to plot
    args.catloc = f'{args.wdir}{args.bh_file}'
    bh_props_list = ['Haloes']
    
    # Select BHs in this sim
    if args.bh_bid is None:

        # Find BHs we are intereste in, load data
        select_list = [
            ["Halo_MStar", '>=', args.halo_mstar_range[0]],
            ["Halo_MStar", '<', args.halo_mstar_range[1]],
            ["Halo_M200c", '>=', args.halo_m200_range[0]],
            ["Halo_M200c", '<', args.halo_m200_range[1]]
        ]
        if not args.include_subdominant_bhs:
            select_list.append(['Flag_MostMassiveInHalo', '==', 1])
        if not args.include_satellites:
            select_list.append(['HaloTypes', '==', 10])

        if args.bh_mass_range is not None:
            zreds = hd.read_data(args.wdir + args.bh_file, 'Redshifts')
            best_index = np.argmin(np.abs(zreds - args.bh_selection_redshift))
            print(f"Best index for redshift {args.bh_selection_redshift} is "
                  f"{best_index}.")
            
            # Subgrid masses are in 10^10 M_sun, so need to adjust selection
            # range
            select_list.append(
                ['SubgridMasses', '>=', args.bh_mass_range[0]/1e10, best_index])
            select_list.append(
                ['SubgridMasses', '<=', args.bh_mass_range[1]/1e10, best_index])        

    else:
        select_list = None

    bh_file = args.wdir + args.bh_file
    bh_data, bh_list = xl.lookup_bh_data(bh_file, bh_props_list, select_list)
    
    # Overwrite selection with input, if specific BID(s) provided
    if args.bh_bid is not None:
        bh_list = args.bh_bid    

    for isnap in args.snapshots:
        process_snapshot(args, bh_list, bh_data, isim, isnap)


def process_snapshot(args, bh_list, bh_data, isim, isnap):
    """Process one individual snapshot for one simulation."""

    star_birth_densities, star_initial_masses = (
        get_birth_densities(args, isnap))
    star_haloes, star_radii, aexp = get_star_haloes(args, isnap)

    for iibh, ibh in enumerate(bh_list):
        process_bh(args, ibh, bh_data, isim, isnap, star_birth_densities,
                   star_initial_masses, star_radii, star_haloes, aexp)


def get_birth_densities(args, isnap):
    """Get the birth densities of all stars in a given snapshot."""
    snap_file = f'{args.wdir}{args.snap_name}_{isnap:04d}.hdf5'
    birth_densities = hd.read_data(snap_file, 'PartType4/BirthDensities')
    masses = hd.read_data(snap_file, 'PartType4/InitialMasses')
    
    # Get conversion factor to n_H [cm^-3]
    m_p = 1.673e-24  # Proton mass in g
    X_H = 0.752 # Hydrogen mass fraction (primordial)
    rho_to_cgs_factor = hd.read_attribute(
        snap_file, 'PartType4/BirthDensities',
        'Conversion factor to physical CGS (including cosmological '
        'corrections)')
    rho_to_nH_cgs = (X_H / m_p) * (rho_to_cgs_factor)
        
    # Convert the densities to the "critical density" (see Crain+15)
    n_crit = 10.0 * (1.81)**(-1/2)  # in n_H [cm^-3]

    return np.log10(birth_densities * rho_to_nH_cgs / n_crit), masses
        

def get_star_haloes(args, isnap):
    """Get the halo ID of all stars in a given snapshot."""
    snap_file = f'{args.wdir}{args.snap_name}_{isnap:04d}.hdf5'
    star_ids = hd.read_data(snap_file, 'PartType4/ParticleIDs')
    star_coordinates = hd.read_data(snap_file, 'PartType4/Coordinates')
    star_coordinates *= (
        hd.read_attribute(snap_file, 'Header', 'Scale-factor')[0])
    
    vr_file = f'{args.wdir}{args.vr_name}_{isnap:04d}'
    star_haloes, aexp, zred = (
        xl.connect_ids_to_vr(star_ids, vr_file, require=True))

    # Get radii of stars
    halo_centres = hd.read_data(f'{vr_file}.hdf5',
                                'MinimumPotential/Coordinates')
    star_radii = np.linalg.norm(star_coordinates
                                - halo_centres[star_haloes, :], axis=1) * 1e3
    ind_not_in_halo = np.nonzero(star_haloes < 0)[0]
    star_radii[ind_not_in_halo] = -1
    
    args.aexp = aexp
    args.zred = zred
    return star_haloes, star_radii, aexp


def process_bh(args, ibh, bh_data, isim, isnap, star_birth_densities,
               star_masses, star_radii, star_haloes, aexp):
    """Make the plot for one BH/galaxy, at one snapshot."""
    
    fig = plt.figure(figsize=(4.5, 4.0))
    ax = plt.gca()
    ax.set_xlim((-3, 3))
    ax.set_ylim((0, 1))
    ax.set_xlabel(r'$\log_{10}(n_\mathrm{H}\,/\,n_\mathrm{H,\,tc})$')
    ax.set_ylabel(r'$f_\mathrm{Stellar\,\,initial\,\,mass} (< n_\mathrm{H})$')

    # Plot distribution of all stars
    plot_cumdist(star_birth_densities, star_masses, None,
                 color='grey')

    xl.legend_item(ax, [0.03, 0.1], 0.95, 'Full simulation', color='grey')

    add_eagle(ax, 'Ref-L0025N0376', aexp, linestyle='--', legend_y = 0.6)
    add_eagle(ax, 'FBconst-L0025N0376', aexp, linestyle=':', legend_y = 0.53)
    
    # Plot distribution only for current BH/galaxy, in set of apertures
    ind_in_halo = np.nonzero(star_haloes == bh_data['Haloes'][ibh])[0]

    for iiap, iap in enumerate(ap_list):
        iap_min = ap_min[iiap]
        subind_in_ap = np.nonzero((star_radii[ind_in_halo] < iap) &
                                  (star_radii[ind_in_halo] >= iap_min))[0]
        plot_cumdist(star_birth_densities, star_masses,
                     ind_in_halo[subind_in_ap],
                     color=cvec[iiap])

        if iiap < 2:
            legend_text = f'{iap_min} ' r'$< r <$' f' {iap} kpc'
        else:
            legend_text = 'Whole galaxy'
        xl.legend_item(ax, [0.03, 0.1], 0.86-iiap*0.07, legend_text,
                       color=cvec[iiap])
        
    # Save figure
    plt.subplots_adjust(left=0.15, right=0.96, bottom=0.15, top=0.95)
    plt.show
    plotloc = f'{args.wdir}{args.plot_prefix}_BH-BID-{ibh}_snap-{isnap}.png'
    if not os.path.isdir(os.path.dirname(plotloc)):
        os.makedirs(os.path.dirname(plotloc))
    plt.savefig(plotloc, dpi=200)
    plt.close('all')


def add_eagle(ax, eagle_name, aexp, linestyle='-', legend_y=None):
    """Add distribution from an EAGLE simulation."""

    eagle_aexps = hd.read_data('./comparison_data/EAGLE_birth_densities.hdf5',
                               f'{eagle_name}/ScaleFactors')
    isnap_eagle = np.argmin(np.abs(aexp - eagle_aexps))
    print(f"Adding EAGLE snap {isnap_eagle}...")
    eagle_x = hd.read_data('./comparison_data/EAGLE_birth_densities.hdf5',
                           f'{eagle_name}/S{isnap_eagle}/nH_by_nCrit')
    eagle_y = hd.read_data('./comparison_data/EAGLE_birth_densities.hdf5',
                           f'{eagle_name}/S{isnap_eagle}/CumulativeMassFraction')

    plt.plot(eagle_x, eagle_y, color='black', linestyle=linestyle)

    if legend_y is not None:
        xl.legend_item(ax, [0.03, 0.1], legend_y, eagle_name, color='black',
                       ls=linestyle, fontsize=9)

    
def plot_cumdist(quantities, weights=None, indices=None,
                 **kwargs):
    """Plot a cumulative distribution."""

    if indices is None:
        indices = np.arange(len(quantities))
    
    sorter = np.argsort(quantities[indices])
    xquant = quantities[indices[sorter]]

    if weights is None:
        yquant = np.arange(len(indices)) + 1 / len(indices)
    else:
        yquant = np.cumsum(weights[indices[sorter]]) / np.sum(weights[indices])

    plt.plot(xquant, yquant, **kwargs)

    
        
if __name__ == '__main__':
    main()
