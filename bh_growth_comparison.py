"""Plot simple black hole growth tracks for a collection of BHs."""

import hydrangea.hdf5 as hd
import numpy as np
from plot_routines import plot_average_profile
import local
import xltools as xl
from astropy.io import ascii
import argparse
import os

from pdb import set_trace

# Set up Matplotlib
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.use('pdf')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'][0] = 'palatino'
mpl.rc('text', usetex=True)
import matplotlib.pyplot as plt

# Cosmology instance, for redshift <--> time conversion
cosmo = xl.swift_Planck_cosmology()

bh_props_list = ['SubgridMasses', 'Times']

def main():

    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    parser.add_argument('sim', help='Simulation index/name to analyse')
    parser.add_argument('--bh_bid', type=int,
                        help='BID of the black hole to highlight.')
    parser.add_argument('--base_dir', default=local.BASE_DIR,
                        help='Base directory of the simulation, default: '
                             f'{local.BASE_DIR}')
    parser.add_argument('--bh_file', default='black_hole_data.hdf5',
                        help='Name of the file containing the BH data, '
                             'default: "black_hole_data.hdf5"')
    parser.add_argument('--plot_prefix', default='gallery/bh_growth_tracks',
                        help='Prefix of output files, default: '
                             '"gallery/bh_growth_tracks')
    parser.add_argument('--plot_mass_range', type=float, nargs='+',
                        help='Min and max mass of plot range, default: '
                             '5.0 8.5', default=[5.0, 8.5])
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
    parser.add_argument('--include_satellites', action='store_true',
                        help='Only show BHs in central haloes, not satellites.')
    parser.add_argument('--show_target_only', action='store_true',
                        help='Show only the target BH, not possible others '
                             'that also match the selection criteria.')
    parser.add_argument('--show_median', action='store_true',
                        help='Add the median for all selected BHs.')
    parser.add_argument('--summary_only', action='store_true',
                        help='Only generate summary plot, not ones for '
                             'individual BHs.')
    parser.add_argument('--alpha_others', type=float, default=0.25,
                        help='Alpha value for non-target BHs, default: 0.25')

    args = parser.parse_args()

    args.wdir = xl.get_sim_dir(args.base_dir, args.sim)

    # Name of the input catalogue, containing all the data to plot
    args.catloc = f'{args.wdir}{args.bh_file}'

    # Find BHs we are intereste in, load data
    select_list = [
        ["Halo_MStar", '>=', args.halo_mstar_range[0]],
        ["Halo_MStar", '<', args.halo_mstar_range[1]],
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
        
        # Subgrid masses are in 10^10 M_sun, so need to adjust selection range
        select_list.append(
            ['SubgridMasses', '>=', args.bh_mass_range[0]/1e10, best_index])
        select_list.append(
            ['SubgridMasses', '<=', args.bh_mass_range[1]/1e10, best_index])        
        
    bh_data, bh_list = xl.lookup_bh_data(args.wdir + args.bh_file,
                                         bh_props_list, select_list)

    generate_track_image(args, bh_data, bh_list)

    if args.bh_bid is None:
        # No target specified: process all selected BHs
        for iibh, ibh in enumerate(bh_list):
            generate_track_image(args, bh_data, bh_list, iibh)
    else:
        # Only plot the selected BH.
        iibh = np.nonzero(bh_list == args.bh_bid)[0]
        if len(iibh) == 1:
            generate_track_image(args, bh_data, bh_list, iibh[0])
        else:
            print(f"Could not (unambiguously) find BH-BID {args.bh_bid} "
                  f"in BH list.")

    print("Done!")

    
def generate_track_image(args, bh_data, bh_list, iibh=None):
    """Generate the track image, optionally highlighting one particular BH.""" 

    fig = plt.figure(figsize=(5.0, 4.5))

    xr = [0, cosmo.age(0).value]   # Time axis range
    xlabel = r'Time [Gyr]'

    yr = args.plot_mass_range
    ylabel = (r'$\log_{10}\,(M_\mathrm{BH,\, subgrid}\,/\,'
              r'\mathrm{M}_\odot)$')

    ax = plt.gca()
    ax.set_xlim(xr)
    ax.set_ylim(yr)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Second axis on top for redshifts */
    ax1b = get_redshift_axis(ax, xr, yr)

    times = bh_data['Times']
    log_masses = np.log10(bh_data['SubgridMasses']) + 10.0

    if 'Halo_MStar' in bh_data and bh_data['Halo_MStar'] is not None:
        halo_logmstar = np.log10(bh_data['Halo_MStar'])
    else:
        halo_logmstar = None
        
    # Plot on the individual BH tracks                        
    for iixbh, ixbh in enumerate(bh_list):
        if halo_logmstar is not None:
            col = plt.cm.viridis(
                (halo_logmstar[ixbh] - np.log10(args.halo_mstar_range[0]))
                / np.log10(args.halo_mstar_range[1] / args.halo_mstar_range[0])
                )
        else:
            col = 'black'
                    
        if iibh is not None and iixbh == iibh:
            plot_alpha = 1.0
        else:
            plot_alpha = args.alpha_others
                    
        plt.plot(times, log_masses[ixbh, :], linewidth = 1.0,
                 linestyle = '-', alpha=plot_alpha, color=col)

        if args.show_median:
            plot_average_profile(times, log_masses[sel, :].T, color=col,
                                 linestyle='-',
                                 uncertainty=True,
                                 scatter=True,
                                 percent=50)    
                
    # Save image
    plt.subplots_adjust(left = 0.15, right = 0.95, bottom = 0.15, top = 0.88,
                        wspace=0, hspace=0.35)
    plt.show
    if iibh is None:
        plotloc = f'{args.wdir}{args.plot_prefix}_BH-all.png'
    else:
        plotloc = f'{args.wdir}{args.plot_prefix}_BH-BID-{bh_list[iibh]}.png'       

    if not os.path.isdir(os.path.dirname(plotloc)):
        os.makedirs(os.path.dirname(plotloc))
    plt.savefig(plotloc, dpi = 200, transparent = False)
    plt.close('all')
    
    
def get_redshift_axis(ax, xr, yr):
    """Create a second axis on the top bar for redshift labelling"""

    ax1b = ax.twiny()
    zreds = np.array([10.0, 3.0, 2.0, 1.0, 0.5, 
                      0.2, 0.1, 0.0])
    zredtimes = np.array([cosmo.age(_z).value for _z in zreds])
    ind_good = np.nonzero((zredtimes >= xr[0]) &
                          (zredtimes <= xr[1]))[0]
    
    ax1b.set_xticks(zredtimes[ind_good])
    zreds_good = zreds[ind_good]
    ax1b.set_xticklabels([f'{_zt:.1f}' for _zt in zreds_good])
    ax1b.set_xlabel('Redshift')
    ax1b.set_xlim(xr)
    ax1b.set_ylim(yr)

    return ax1b


if __name__ == '__main__':
    main()       
                                 
                                  
