"""Plot M_star vs. M200, colour-coded by sSFR."""

import numpy as np
from pdb import set_trace
import hydrangea.hdf5 as hd
import os
import local
import xltools as xl
import argparse

from reduce_bh_data import connect_to_galaxies

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.use('pdf')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'][0] = 'palatino'
mpl.rc('text', usetex=True)

import matplotlib.pyplot as plt

# Define general settings

simdir = f'{local.BASE_DIR}/ID179_JE25/'
sim = 179
snapshot = 24

black_file = simdir + 'black_hole_data.hdf5'
bh_maxflag = hd.read_data(black_file, 'Flag_MostMassiveInHalo')
bh_mstar = hd.read_data(black_file, 'Halo_MStar')
bh_m200 = hd.read_data(black_file, 'Halo_M200c')
bh_sfr = hd.read_data(black_file, 'Halo_SFR')
bh_halotype = hd.read_data(black_file, 'HaloTypes')
bh_ids = hd.read_data(black_file, 'ParticleIDs')

bh_logssfr = np.log10(bh_sfr) - np.log10(bh_mstar)

bh_list = np.nonzero((bh_maxflag == 1) &
                     (bh_mstar >= 3e10) &
                     (bh_halotype == 10))[0]

print(f"There are {len(bh_list)} BHs in selection list.")

print(f"Min/max log M200 = {np.log10(np.min(bh_m200[bh_list]))}, "
      f"{np.log10(np.max(bh_m200[bh_list]))}")

def main():

    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    parser.add_argument('sims', nargs='+',
                        help='Simulation inde(x/ices) or names to '
                             'analyse ("all" for all in base_dir).')
    parser.add_argument('--snap_name', default='output',
                        help='Name prefix of simulation outputs '
                        '(default: "output")'.)
    parser.add_argument('--snapshots', type=int,
                        help='Snapshots for which to create plots.')
    parser.add_argument('--base_dir', default=local.BASE_DIR,
                        help='Base directory of simulations (default: '
                             f'{local.BASE_DIR}).')
    parser.add_argument('--vr_file', default="vr",
                        help='File of (transcribed) VR catalogue '
                             '(default: vr).')
    parser.add_argument('--bh_file', default="black_hole_data.hdf5",
                        help='Combine BH data file, default: '
                             'black_hole_data.hdf5')
    parser.add_argument('--plot_prefix', default='gallery/bh_growth_tracks',
                        help='Prefix of output files, default: '
                             '"gallery/vr_plots')
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
                             '[M_Sun], default: 3e10, 4e11.')
    parser.add_argument('--include_satellites', action='store_true',
                        help='Only show BHs in central haloes, not satellites.')
    parser.add_argument('--show_target_only', action='store_true',
                        help='Show only the target BH, not possible others '
                             'that also match the selection criteria.')
    parser.add_argument('--show_median', action='store_true',
                        help='Add the median for all selected BHs.')
    parser.add_argument('--alpha_others', type=float, default=1.0,
                        help='Alpha value for non-target BHs, default: 1.0')
    args = parser.parse_args()

    # Resolve (possible) placeholder for all simulations
    if args.sims[0].lower() == 'all':
        args.sims = xl.get_all_sims(args.base_dir)
        have_full_sim_dir = True
    else:
        have_full_sim_dir = False

    # Process each simulation in turn        
    for isim in args.sims:
        process_sim(args, isim, have_full_sim_dir)

    args.vrsnap = snapshot
    vr_data = connect_to_galaxies(bh_ids[bh_list], args)
    
    for ibh in bh_list:
        make_plot_mstar_m200(ibh, args, vr_data)


def process_sim(args, isim, have_full_sim_dir):
    """Generate the images for one particular simulation."""

    args.wdir = xl.get_sim_dir(args.base_dir, isim)

    # Name of the input BH data catalogue
    args.catloc = f'{args.wdir}{args.bh_file}'

    # Find BHs we are intereste in, load data (of internal VR match)
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

    # Go through snapshots
    for iisnap, isnap in enumerate(args.snapshots):

        # Need to explicitly connect BHs to VR catalogue from this snap
        vr_data = xl.connect_to_galaxies(bh_data['ParticleIDs'][bh_list],
                                         args.wdir, isnap)

        # Make plots for each BH, and "general" overview plot
        generate_vr_plots(args, vr_data, isnap)
        for iibh, ibh in enumerate(bh_list):
            generate_vr_plots(args, vr_data, isnap, iibh, ibh)

        print("Done!")


def generate_vr_plots(args, vr_data, isnap, iibh=None, ibh=None):
    """Driver function to create all plots for one snap and target BH.

    Parameters
    ----------
    args : object
        Configuration parameters read from the arg parser
    vr_data : dict
        Relevant VR catalogue data for all BHs to be plotted
    isnap : int
        Snapshot index of this plot; only relevant for output naming.
    iibh : int, optional
        BH index of target BH in vr_data. If None (default), no BH is 
        highlighted especially.
    ibh : int, optional
        Full BH-ID of target BH (only used for output naming).
    """

    


def make_plot_mstar_m200(ibh, args, vr_data):
    """Make plot for individual BH."""

    plotloc_bh = simdir + f'gallery/mstar-m200-ssfr_bh-bid-{ibh}_snap-{snapshot}.png'
    
    fig = plt.figure(figsize=(5.5, 4.5))

    ax = fig.add_axes([0.15, 0.15, 0.67, 0.8])
    
    alpha = np.zeros(len(bh_list)) + 0.2
    ind_this = np.nonzero(bh_list == ibh)[0][0]
    alpha[ind_this] = 1

    for iibh, iibh_id in enumerate(bh_list):

        if iibh_id == ibh:
            alpha, s, edgecolor = 1.0, 50.0, 'red'
        else:
            alpha, s, edgecolor = 1.0, 15.0, 'none'

        logm200 = np.log10(vr_data['M200'][iibh])
        logmstar = np.log10(vr_data['MStar'][iibh])
        logssfr = np.log10(vr_data['SFR'][iibh]) - logmstar
            
        sc=plt.scatter([logm200], [logmstar],
                       c=[logssfr], cmap=plt.cm.viridis,
                       vmin=-12.5+0.6*args.vr_zred, vmax=-10.0+0.6*args.vr_zred,
                       alpha=alpha, s=s,
                       edgecolor=edgecolor)

        if iibh_id == ibh:
            sc_keep = sc
        
    ax = plt.gca()
    ax.set_xlim((11.7, 13.3))
    ax.set_ylim((10.4, 11.3))

    ax.set_xlabel(r'$\log_{10} (M_\mathrm{200c}\,/\,\mathrm{M}_\odot)$')
    ax.set_ylabel(r'$\log_{10} (M_\mathrm{star}\,/\,\mathrm{M}_\odot)$')

    ax2 = fig.add_axes([0.82, 0.15, 0.04, 0.8])
    ax2.set_yticks([])
    ax2.set_xticks([])
    cbar = plt.colorbar(sc_keep, cax=ax2, orientation = 'vertical')

    fig.text(0.99, 0.5, r'$\log_{10}\,(\mathrm{sSFR}\,/\,\mathrm{yr}^{-1})$',
             color = 'black', rotation = 90.0,
             fontsize = 10, ha = 'right', va = 'center')

    #cb = plt.colorbar(sc_keep, label = r'$\log_{10}\,(\mathrm{sSFR}\,/\,\mathrm{yr}^{-1})$')
    
    #plt.subplots_adjust(left = 0.15, right = 0.95, bottom = 0.15, top = 0.88,
    #                wspace=0, hspace=0.35)
    plt.show
    plt.savefig(plotloc_bh, dpi = 200, transparent = False)


if __name__ == '__main__':
    main()
