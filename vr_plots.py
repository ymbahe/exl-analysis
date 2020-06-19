"""Plot M_star vs. M200, colour-coded by sSFR."""

import numpy as np
from pdb import set_trace
import hydrangea.hdf5 as hd
import os
import local
import xltools as xl
from plot_routines import plot_distribution
import argparse

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.use('pdf')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'][0] = 'palatino'
mpl.rc('text', usetex=True)

import matplotlib.pyplot as plt

# Define general settings
plot_list = [('M200c', 'MStar', 'sSFR'),
             ('M200c', 'MBH', 'MStar'),
             ('M200c', 'Size', 'MStar')]

ax_labels = {'M200c': r'$\log_{10}\,(M_\mathrm{200c}\,/\,\mathrm{M}_\odot)$',
             'MStar': r'$\log_{10}\,(M_\mathrm{star}\,/\,\mathrm{M}_\odot)$',
             'MBH': r'$\log_{10}\,(m_\mathrm{BH}\,/\,\mathrm{M}_\odot)$',
             'Size': r'Stellar $R_{1/2}$ ($< 30$ kpc, Proj. 1) [kpc]',
             'sSFR': r'$\log_{10}\,(\mathrm{sSFR}\,/\,\mathrm{yr}^{-1})$'}

plot_ranges = {'M200c': [11.7, 13.5],
               'MStar': [9.0, 11.5],
               'MBH': [5.0, 8.5],
               'Size': [0.0, 12.0],
               'sSFR': [-12.0, -9.5]}

def main():

    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    parser.add_argument('sims', nargs='+',
                        help='Simulation inde(x/ices) or names to '
                             'analyse ("all" for all in base_dir).')
    parser.add_argument('--snap_name', default='output',
                        help='Name prefix of simulation outputs '
                        '(default: "output").')
    parser.add_argument('--snapshots', type=int, nargs='+',
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
    parser.add_argument('--plot_prefix', default='gallery/vr-plots',
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
                             '[M_Sun], default: 3e10 4e11.', nargs='+')
    parser.add_argument('--halo_m200_range', type=float, nargs='+',
                        help='Min and max M200 of host galaxy '
                             '[M_Sun], default: 0 1e16.', default=[0, 1e16])
    parser.add_argument('--include_satellites', action='store_true',
                        help='Only show BHs in central haloes, not satellites.')
    parser.add_argument('--show_target_only', action='store_true',
                        help='Show only the target BH, not possible others '
                             'that also match the selection criteria.')
    parser.add_argument('--show_median', action='store_true',
                        help='Add the median for all selected BHs.')
    parser.add_argument('--alpha_others', type=float, default=1.0,
                        help='Alpha value for non-target BHs, default: 1.0')
    parser.add_argument('--summary_only', action='store_true',
                        help='Only generate summary plots, not those '
                             'highlighting each invididual BH.')
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


def process_sim(args, isim, have_full_sim_dir):
    """Generate the images for one particular simulation."""

    if have_full_sim_dir:
        args.wdir = isim
    else:
        args.wdir = xl.get_sim_dir(args.base_dir, isim)

    # Name of the input BH data catalogue
    args.catloc = f'{args.wdir}{args.bh_file}'

    # Find BHs we are intereste in, load data (of internal VR match)
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
        
        # Subgrid masses are in 10^10 M_sun, so need to adjust selection range
        select_list.append(
            ['SubgridMasses', '>=', args.bh_mass_range[0]/1e10, best_index])
        select_list.append(
            ['SubgridMasses', '<=', args.bh_mass_range[1]/1e10, best_index])        
       
    bh_props_list = ['SubgridMasses', 'Redshifts', 'ParticleIDs']    
    bh_data, bh_list = xl.lookup_bh_data(args.wdir + args.bh_file,
                                         bh_props_list, select_list)

    if len(bh_list) == 0:
        print("No BHs selected, aborting.")
        return
    
    args.sim_pointdata_loc = args.wdir + args.plot_prefix + '.hdf5'
    hd.write_data(args.sim_pointdata_loc, 'BlackHoleBIDs', bh_list,
                  comment='BIDs of all BHs selected for this simulation.')

    # Go through snapshots
    for iisnap, isnap in enumerate(args.snapshots):

        # Need to explicitly connect BHs to VR catalogue from this snap
        vr_data = xl.connect_to_galaxies(
            bh_data['ParticleIDs'][bh_list],
            f'{args.wdir}{args.vr_file}_{isnap:04d}',
            extra_props=[('ApertureMeasurements/Projection1/30kpc/'
                          'Stars/HalfMassRadii', 'StellarHalfMassRad')])

        
        # Add subgrid mass of BHs themselves
        ind_bh_cat = np.argmin(np.abs(bh_data['Redshifts']
                                      - vr_data['Redshift']))
        print(f"Using BH masses from index {ind_bh_cat}")
        vr_data['BH_SubgridMasses'] = (
            bh_data['SubgridMasses'][bh_list, ind_bh_cat] * 1e10)

        # Make plots for each BH, and "general" overview plot
        generate_vr_plots(args, vr_data, isnap)

        if not args.summary_only:
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
    for iiplot, iplot in enumerate(plot_list):
        make_plot(args, vr_data, iplot, isnap, iibh, ibh)


def make_plot(args, vr_data, iplot, isnap, iibh, ibh):
    """Make one specific plot.

    Parameters
    ----------
    args : object
        Configuration parameters read from the arg parser
    vr_data : dict
        Relevant VR catalogue data for all BHs to be plotted
    iplot : int
        Tuple containing information about the specific plot to make.
        Content: [x-quant, y-quant, color-quant]
    isnap : int
        Snapshot index of this plot; only relevant for output naming.
    iibh : int, optional
        BH index of target BH in vr_data. If None (default), no BH is 
        highlighted especially.
    ibh : int, optional
        Full BH-ID of target BH (only used for output naming).
    """
    plotloc = (f'{args.wdir}{args.plot_prefix}_{iplot[0]}-{iplot[1]}-'
               f'{iplot[2]}_BH-{ibh}_snap-{isnap}.png')
    fig = plt.figure(figsize=(5.5, 4.5))

    # To enable the HTML link map, we must be able to reconstruct the
    # location of each point on the plot. This is easier with an explicitly
    # constructed axes frame ([x_off, y_off, x_width, y_width])
    ax = fig.add_axes([0.15, 0.15, 0.67, 0.8])

    xr, yr = plot_ranges[iplot[0]], plot_ranges[iplot[1]]
    ax.set_xlim(xr)
    ax.set_ylim(yr)
    ax.set_xlabel(ax_labels[iplot[0]])
    ax.set_ylabel(ax_labels[iplot[1]])
    vmin, vmax = plot_ranges[iplot[2]]

    # Extract relevant quantities
    xquant = get_vrquant(vr_data, iplot[0])
    yquant = get_vrquant(vr_data, iplot[1])
    cquant = get_vrquant(vr_data, iplot[2])

    xquant_plt = np.copy(xquant)
    yquant_plt = np.copy(yquant)

    if args.show_median:
        plot_distribution(xquant, yquant, xrange=xr, uncertainty=True,
                          scatter=False, plot_at_median=True, color='grey',
                          nbins=5, dashed_below=5)
    
    for iixbh in range(len(xquant)):
        
        # Special treatment for (halo of) to-be-highlighted BH
        if iixbh == iibh and iibh is not None:
            s, edgecolor = 50.0, 'red'
        else:
            s, edgecolor = 15.0, 'none'

        marker='o'
        
        ixquant, iyquant = xquant[iixbh], yquant[iixbh]
        if ((ixquant < xr[0] or ixquant > xr[1]) and
            (iyquant < yr[0] or iyquant > yr[1])):
            continue
        
        if ixquant < xr[0]:
            ixquant = xr[0] + (xr[1] - xr[0]) * 0.02
            marker = r'$\mathbf{\leftarrow}$'
            s *= 3
        elif ixquant > xr[1]:
            ixquant = xr[1] - (xr[1] - xr[0]) * 0.02
            marker = r'$\mathbf{\rightarrow}$'
            s *= 3
        if iyquant < yr[0]:
            iyquant = yr[0] + (yr[1] - yr[0]) * 0.02
            marker = r'$\mathbf{\downarrow}$'
            s *= 3
        elif iyquant > yr[1]:
            iyquant = yr[1] - (yr[1] - yr[0]) * 0.02
            marker = r'$\mathbf{\uparrow}$'
            s *= 3
            
        icquant = np.clip(cquant[iixbh], plot_ranges[iplot[2]][0],
                                         plot_ranges[iplot[2]][1])

        print(f'x={ixquant}, y={iyquant}, c={icquant}')
        sc = plt.scatter([ixquant], [iyquant], c=[icquant],
                         cmap=plt.cm.viridis, marker=marker,
                         vmin=vmin, vmax=vmax, s=s, edgecolor=edgecolor,
                         zorder=100)

        # Feed possible clipping changes back to array, for outputting.
        xquant_plt[iixbh], yquant_plt[iixbh] = ixquant, iyquant

        
    # Colour bar on the right-hand side
    ax2 = fig.add_axes([0.82, 0.15, 0.04, 0.8])
    ax2.set_yticks([])
    ax2.set_xticks([])
    cbar = plt.colorbar(sc, cax=ax2, orientation = 'vertical')
    fig.text(0.99, 0.5, ax_labels[iplot[2]],
             color='black', rotation=90.0, fontsize=10, ha='right',
             va='center')

    plt.show
    plt.savefig(plotloc, dpi = 200, transparent = False)
    plt.close('all')

    # Final bit: store normalised data points in HDF5 file
    imx = (xquant_plt - xr[0]) / (xr[1] - xr[0])
    imx = imx*0.67 + 0.15

    imy = (yquant_plt - yr[0]) / (yr[1] - yr[0])
    imy = (1 - (imy*0.8 + 0.15)) * (4.5/5.5)

    hd.write_data(args.sim_pointdata_loc, f'S{isnap}/{iplot[0]}-{iplot[1]}/xpt',
                  imx, comment='Normalised x coordinates of points.')
    hd.write_data(args.sim_pointdata_loc, f'S{isnap}/{iplot[0]}-{iplot[1]}/ypt',
                  imy, comment='Normalised y coordinates of points.')


def get_vrquant(vr_data, short_name):
    """Extract or compute a quantity from the VR catalogue."""
    if short_name == 'M200c':
        return np.log10(vr_data['M200c'])
    elif short_name == 'MStar':
        return np.log10(vr_data['MStar'])
    elif short_name == 'Size':
        return vr_data['StellarHalfMassRad']*1e3  # in kpc
    elif short_name == 'sSFR':
        log_mstar = np.log10(vr_data['MStar'])
        log_sfr = np.log10(vr_data['SFR'])
        return log_sfr - log_mstar
    elif short_name == 'MBH':
        return np.log10(vr_data['BH_SubgridMasses'])


if __name__ == '__main__':
    main()
