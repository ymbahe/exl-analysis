import numpy as np
import os
import argparse
from pdb import set_trace
from scipy.interpolate import interp1d
import astropy.units as u

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

# Conversion factor for accretion rates:
conv_mdot = 1.023045e-02
conv_mass = 1e10
density_to_nH = 1.989e40 / (3.0857e22)**3 / (1.67e-27 / ((1e-2)**3)) * 0.752

# Hard-coded (for now) behaviour switches
bh_props_list = ['Times', 'ParticleIDs',
                 'SubgridMasses', 'DynamicalMasses', 'AccretionRates',
                 'Velocities', 'GasRelativeVelocities',
                 'GasCircularVelocities', 'GasSoundSpeeds',
                 'GasVelocityDispersions', 'GasDensities', 'NumberOfTimeSteps',
                 'NumberOfRepositions', 'NumberOfRepositionAttempts',
                 'NumberOfSwallows', 'NumberOfMergers', 'ViscosityFactors',
                 'Haloes', 'NumberOfRepositionings', 'Redshifts']

panels = ['Masses', 'AccretionRate', 'Speeds', 'Density', 'StellarMass',
          'Repositions', 'RepositionFractions']

yranges = {'Masses': [4.8, 8.0],
           'AccretionRate': [-8.9, 0],
           'ViscosityFactor': [-4, 0.05],
           'Speeds': [-0.9, 3.0],
           'StellarMass': [0, 10.0],
           'StarFormationRate': [-1, 2.0],
           'Density': [4.0, 9.0],
           'Repositions': [0, 100],
           'RepositionFractions': [-0.02, 1.02]}

xlabel = r'Time [Gyr]'
ylabels = {'Masses': r'BH masses (log $m$ [M$_\odot$])',
           'AccretionRate': 'Accretion rates' '\n'
                             r'(log $\dot{m}$ [M$_\odot$/yr])',
           'ViscosityFactor': r'$\log_{10}$ Viscosity factor',
           'Speeds': r'Gas speeds (log $v$ [km/s])',
           'Density': r'Gas density (log $n_\mathrm{H}$ [cm$^{-3}$])',
           'StellarMass': r'Stellar initial mass [$10^{10}\,\mathrm{M}_\odot$]',
           'StarFormationRate': r'log$_{10}$ (SFR$_\mathrm{paleo.}$ '
                                r'[M$_\odot$ yr$^{-1}$])',
           'Repositions': r'Reposition numbers',
           'RepositionFractions': r'Reposition fractions'}

# Plot the absolute velocity of the BH (w.r.t. the simulation frame)?
plot_bvel = False

# Plot the circular velocity of the BH?
plot_bcvel = True

num_repos_name = 'NumberOfRepositions'

def main():

    global num_repos_name
    
    argvals = None
    args = get_args(argvals)

    # Follow immediate instructions from args
    mpl.rcParams['lines.linewidth'] = args.plot_linewidth

    if args.plot_mass_range is not None:
        yranges['Masses'] = args.plot_mass_range

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
    if args.use_old_names:
        num_repos_name = 'NumberOfRepositionings'
    
    for isim in args.sims:
        process_sim(isim, args)


def process_sim(isim, args):
    """Process one specific simulation."""

    if args.have_full_sim_dir:
        args.wdir = isim
    else:
        args.wdir = xl.get_sim_dir(args.base_dir, isim)
    print(f"Analysing simulation {args.wdir}...")

    # Name of the input catalogue, containing all the data to plot
    args.catloc = f'{args.wdir}{args.bh_file}'

    # Select BHs in this sim
    args.plotdata_file = f'{args.wdir}{args.vrplot_prefix}.hdf5'
    if os.path.isfile(args.plotdata_file):
        bh_list = hd.read_data(args.plotdata_file, 'BlackHoleBIDs')
        select_list = None
     
    elif args.bh_bid is not None:
        select_list = None
        bh_list = args.bh_bid 

    else:

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

        bh_list = None

    bh_file = args.wdir + args.bh_file
    bh_data, bh_list = xl.lookup_bh_data(bh_file, bh_props_list, select_list,
                                         bh_list)
    args.nsnap = len(bh_data['Times'])

    # Extract meta-data from Header
    bh_data['CodeBranch'] = hd.read_attribute(bh_file, 'Code', 'Git Branch')
    bh_data['CodeDate'] = hd.read_attribute(bh_file, 'Code', 'Git Date')
    bh_data['CodeRev'] = hd.read_attribute(bh_file, 'Code', 'Git Revision')
    bh_data['SimName'] = hd.read_attribute(bh_file, 'Header', 'RunName')

    # Look up stars data
    stars = Stars(args)
    
    for ibh in bh_list:
        process_bh(args, stars, bh_data, ibh, isim)

        
def process_bh(args, stars, bh_data, ibh, isim):
    """Process one BH from one simulation."""

    fig = plt.figure(figsize=(args.plot_width, 15))

    ind_good = np.nonzero((bh_data['Times'] >= 0) & 
                          (bh_data['SubgridMasses'][ibh, :] * 0 == 0) & 
                          (bh_data['SubgridMasses'][ibh, :] > 0))[0]

    t_range = [np.min(bh_data['Times'][ind_good]),
               np.max(bh_data['Times'][ind_good])]
    print("t_range: ", t_range)

    # Find times of mergers, swallows, repositionings
    event_times = find_event_times(bh_data, ibh)

    plot_config = {}
    plot_config['ibh'] = ibh
    plot_config['inds'] = ind_good
    plot_config['tr'] = t_range
    plot_config['events'] = event_times

    yranges['Repositions'][1] = np.nanmax(
        bh_data['NumberOfRepositionAttempts'][ibh, :])

    # Actual plots
    for iipanel, ipanel in enumerate(panels):
        plot_bh_panel(args, stars, bh_data, plot_config, iipanel)

    plt.subplots_adjust(left=0.2/(args.plot_width/9),
                    right=max(0.93/(args.plot_width/9), 0.93),
                    bottom=0.05, top=0.95, hspace=0, wspace=0)

    plt.show
    plotloc = f'{args.wdir}{args.plot_prefix}_BH-BID-{ibh}.png'
    if not os.path.isdir(os.path.dirname(plotloc)):
        os.makedirs(os.path.dirname(plotloc))
    plt.savefig(plotloc, dpi=200)
    plt.close('all')


def plot_bh_panel(args, stars, bh_data, plot_config, iipanel):
    """Plot one property panel."""

    panel = panels[iipanel]
    n_panels = len(panels)

    # Set up the panel axes
    ax = plt.subplot(n_panels, 1, iipanel+1)
    yrange = list(yranges[panel])
    ax.set_ylabel(ylabels[panel])
    if iipanel < n_panels - 1:
        ax.axes.get_xaxis().set_visible(False)
    else:
        ax.set_xlabel(xlabel)

    ax.set_xlim(plot_config['tr'])
    ax.set_ylim(yrange)  # May overwrite later
        
    # Draw on actual evolution trends
    inds = plot_config['inds']
    times = bh_data['Times'][inds]
    ibh = plot_config['ibh']

    # ---------------------------------------------------------------------

    if panel == 'Masses':
        plt.plot(times,
                 np.log10(bh_data['SubgridMasses'][ibh, inds] * conv_mass),
                 color='black')
        plt.plot(times, np.log10(bh_data['DynamicalMasses'][ibh, inds]
                                 * conv_mass),
                 color='black', linestyle=':')

        legend_item(ax, [0.03, 0.1], 0.94, 'Dynamical mass', alpha=0.5, ls=':')
        legend_item(ax, [0.03, 0.1], 0.87, 'Subgrid mass')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    elif panel == 'AccretionRate':
        plt.plot(times,
                 np.log10(bh_data['AccretionRates'][ibh, inds] * conv_mdot),
                 color = 'black')

        if bh_data['ViscosityFactors'] is not None:
            f_visc = bh_data['ViscosityFactors'][ibh, inds]
            axy = get_yaxis_viscosity(ax)
            plt.sca(axy)
            plt.plot(times, np.log10(f_visc), color='purple', linestyle=':')
            plt.sca(ax)

            f_visc_med = np.median(f_visc)
            f_visc_av = np.average(f_visc)
            f_visc_av_weighted = np.average(
                f_visc,
                weights=(bh_data['AccretionRates'][ibh, inds] / f_visc))
                                   
            legend_text(ax, 0.97, 0.05, r'Average $f_\mathrm{visc}$ = '
                        f'{f_visc_av:.3f} '
                        f'(weighted: {f_visc_av_weighted:.3f}), '
                        f'median = {f_visc_med:.3f}',
                        color='purple', ha='right')
            
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    elif panel == 'Speeds':
        vgas = np.log10(np.linalg.norm(bh_data['GasRelativeVelocities']
                                              [ibh, :, inds], axis=1))
        plt.plot(times, vgas, color='black')
        plt.plot(times, np.log10(bh_data['GasSoundSpeeds'][ibh, inds]
                                 / (1/(1+bh_data['Redshifts'][inds]))),
                 color='goldenrod', linestyle=':')

        legend_item(ax, [0.90, 0.97], 0.06, 'Gas sound speed', color='goldenrod',
                    ls=':', text_side='left')
        legend_item(ax, [0.90, 0.97], 0.2, 'Gas bulk speed', text_side='left')

        if plot_bvel:
            vbh = np.log10(np.linalg.norm(bh_data['Velocities'][ibh, :, inds],
                                          axis=1))
            plt.plot(times, vbh, color='black', linestyle = '--')
            legend_item(ax, [0.90, 0.97], 0.13, 'BH velocity',
                        color='cornflowerblue', ls='--', text_side='left')
        elif plot_bcvel:
            vcgas = np.log10(np.linalg.norm(bh_data['GasCircularVelocities']
                                                   [ibh, :, inds], axis=1))
            plt.plot(times, vcgas, color='cornflowerblue', linestyle='--')
            legend_item(ax, [0.90, 0.97], 0.13, 'Gas circular speed',
                        color='cornflowerblue', ls='--', text_side='left')
        elif bh_data['GasVelocityDispersions'] is not None:
            plt.plot(times, np.log10(bh_data['GasVelocityDispersion']
                                            [ibh, inds]),
                     color='cornflowerblue', linestyle='--')
            legend_item(ax, [0.90, 0.97], 0.13, 'Gas velocity dispersion',
                        color='cornflowerblue', ls='--', text_side='left')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    elif panel == 'Density':
        n_H = np.log10(bh_data['GasDensities'][ibh, inds] * density_to_nH
                       * (1 / (1 + bh_data['Redshifts'][inds]))**(-3))
        plt.plot(times, n_H, color='black', linestyle='-')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    elif panel == 'StellarMass' and 'Haloes' in bh_data:
        total_mass = plot_cumulative_stellar_mass(
            args, stars, bh_data['Haloes'][ibh], ax)

        yrange[1] = total_mass * 1.05
        ax.set_ylim(yrange)
        
        legend_item(ax, [0.9, 0.97], 0.06, r'r $<$ 5.0 kpc',
                    color='red', text_side='left')
        legend_item(ax, [0.9, 0.97], 0.14, r'r $<$ 15.0 kpc',
                    color='seagreen', text_side='left')
        legend_item(ax, [0.9, 0.97], 0.22, r'Whole galaxy',
                    color='royalblue', text_side='left')
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    elif panel == 'Repositions':
        plt.plot(times, bh_data[num_repos_name][ibh, inds],
                 color='black')
        plt.plot(times, bh_data['NumberOfRepositionAttempts'][ibh, inds],
                 color='black', linestyle=':')
        legend_item(ax, [0.90, 0.97], 0.06, 'Actual repositions',
                    text_side='left')
        legend_item(ax, [0.90, 0.97], 0.13, 'Attempted repositions',
                    ls=':', alpha=0.5, text_side='left')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    elif panel == 'RepositionFractions':
        frac_repos, frac_att = get_repositioning_fractions(args, plot_config,
                                                           bh_data)
        plt.plot(times, frac_repos[inds], color='black')
        plt.plot(times, frac_att[inds], color='black', linestyle=':')
        legend_item(ax, [0.90, 0.97], 0.06, 'Actual repositions',
                    text_side='left')
        legend_item(ax, [0.90, 0.97], 0.13, 'Attempted repositions',
                    ls=':', alpha=0.5, text_side='left')

    # ---------------------------------------------------------------------

    ax.set_xlim(plot_config['tr'])
    ax.set_ylim(yrange)  # May overwrite later
    
    # Mark times of repositions, swallows, mergers
    draw_time_lines(plot_config['events'], yrange)
    
    # Information that is not directly related to the individual quantities
    if iipanel == 0:
        axx = get_xaxis_redshift(ax)

        # Code info
        legend_text(ax, 0.02, 0.03,
            f"{bh_data['CodeBranch']} ({bh_data['CodeRev']}, "
            f"{bh_data['CodeDate']})")

    elif iipanel == 1:
        legend_text(ax, 0.03, 0.03,
            f"BH {ibh} [ID = {bh_data['ParticleIDs'][ibh]}]")

    elif iipanel == 2:
        sim_name = bh_data['SimName'].replace('_', '\_')
        legend_text(ax, 0.03, 0.03, f"{sim_name}")


def legend_item(ax, xr, y, text, alpha=1, ls='-', color='black',
                text_side='right'):
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
             text, va='center', ha=text_ha, alpha=alpha, color=color)


def legend_text(ax, x, y, text, **kwargs):
    """Add one legend text-only item to the axes ax."""

    tr = ax.get_xlim()
    yr = ax.get_ylim()

    plt.text(tr[0] + (tr[1]-tr[0]) * x, yr[0] + (yr[1]-yr[0]) * y,
             text, **kwargs)


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
    parser.add_argument('--plot_prefix', default='gallery/bh_evolution',
                        help='Prefix of output files, default: '
                             '"gallery/bh_evolution')
    parser.add_argument('--vrplot_prefix', default='gallery/vr_plots.hdf5',
                        help='Prefix of VR plots, default: '
                             '"gallery/vr_plots.hdf5"')
    parser.add_argument('--snap_name', default='snapshot',
                        help='Snapshot name prefix, default: "snapshot".')
    parser.add_argument('--vr_file', default='vr',
                        help='VR name prefix, default: "vr".')
    parser.add_argument('--use_old_names', action='store_true',
                        help='Use "old-style" output names.')
    
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

    parser.add_argument('--plot_mass_range', type=float, nargs='+',
                        help='Min and max (log) mass of plot range, default: '
                             '5.0 8.5', default=[5.0, 8.5])
    parser.add_argument('--plot_width', type=float, default=9.0,
                        help='Plot width in inches (default: 9.0)')
    parser.add_argument('--plot_linewidth', type=float, default=0.2,
                        help='Plot linewidth, default: 0.2')

    return parser.parse_args(argv)


def find_event_times(bh_data, ibh):
    """Find times of swallows, mergers, and repositionings."""

    # Find times of swallows, mergers, and first repositioning
    ind_swallow = np.nonzero((bh_data['NumberOfSwallows'][ibh, 1:] > 
                              bh_data['NumberOfSwallows'][ibh, :-1]) &
                             (bh_data['NumberOfSwallows'][ibh, 1:] > 0))[0]
    if len(ind_swallow) > 0:
        times_swallow = bh_data['Times'][ind_swallow]
    else:
        times_swallow = None
    print(f"Total of {len(ind_swallow)} gas swallows") 

    ind_merger = np.nonzero((bh_data['NumberOfMergers'][ibh, 1:] >
                             bh_data['NumberOfMergers'][ibh, :-1]) &
                            (bh_data['NumberOfMergers'][ibh, 1:] > 0))[0]
    if len(ind_merger) > 0:
        times_merger = bh_data['Times'][ind_merger]
    else:
        times_merger = None
    print(f"Total of {len(ind_merger)} BH mergers") 

    ind_repos = np.nonzero((bh_data[num_repos_name][ibh, 1:] >
                            bh_data[num_repos_name][ibh, :-1]) &
                           (bh_data[num_repos_name][ibh, 1:] > 0))[0]

    if len(ind_repos) > 0:
        times_repos = bh_data['Times'][ind_repos]
    else:
        times_repos = None
    
    event_times = {'Swallows': times_swallow,
                   'Mergers': times_merger,
                   'Repositionings': times_repos} 

    return event_times


def draw_time_lines(event_times, yr):
    """Plot lines of swallows and reposition times"""
    if event_times['Repositionings'] is not None:
        for iitime, itime in enumerate(event_times['Repositionings']):
            if iitime >= 10: break
            plt.plot([itime, itime], yr, color = 'black', alpha = 0.2,
                     zorder=-100)
    if event_times['Mergers'] is not None:
        for itime in event_times['Mergers']:
            plt.plot([itime, itime], yr, color='royalblue', alpha=0.2,
                     zorder=-100, linestyle=':')

    if event_times['Swallows'] is not None:
        for itime in event_times['Swallows']:
            plt.plot([itime, itime], yr, color='black', alpha=0.2, zorder=-100,
                     linestyle='--')


def get_yaxis_viscosity(ax):
    """Create a second y axis for viscosity factor."""
    ax2 = ax.twinx()

    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(yranges['ViscosityFactor'])
    ax2.set_ylabel(ylabels['ViscosityFactor'])

    ax2.spines['right'].set_color('purple')
    ax2.yaxis.label.set_color('purple')
    ax2.tick_params(axis='y', colors='purple')

    return ax2


def get_yaxis_sfr(ax):
    """Create a second y axis for star formation rates."""
    ax2 = ax.twinx()

    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(yranges['StarFormationRate'])
    ax2.set_ylabel(ylabels['StarFormationRate'])

    ax2.spines['right'].set_color('grey')
    ax2.yaxis.label.set_color('grey')
    ax2.tick_params(axis='y', colors='grey')

    return ax2


def get_xaxis_redshift(ax):
    """Create a second x axis for redshifts."""

    trange = ax.get_xlim()
    ax2 = ax.twiny()
    zreds = np.array([10.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.7, 0.5, 0.2,
                      0.1, 0.0])

    # Cosmology instance, for redshift <--> time conversion
    cosmo = xl.swift_Planck_cosmology()
    zred_times = np.array([cosmo.age(_z).value for _z in zreds])

    ind_good = np.nonzero((zred_times >= trange[0]) &
                          (zred_times <= trange[1]))[0]

    ax2.set_xticks(zred_times[ind_good])
    zreds_good = zreds[ind_good]
    ax2.set_xticklabels([f'{_zt:.1f}' for _zt in zreds_good])
    ax2.set_xlabel('Redshift')
    ax2.set_xlim(trange)
    ax2.set_ylim(ax.get_ylim())


def get_repositioning_fractions(args, plot_config, bh_data):
    """Compute the running repositioning fractions."""

    n_snap = args.nsnap
    inds = plot_config['inds']
    ibh = plot_config['ibh']

    # For each point in time, find the index range from (relative) -5 to +5
    # (clipped to the very first and last output)
    start_ind = np.clip(np.arange(n_snap) - 5, inds[0], inds[-1])
    end_ind = np.clip(np.arange(n_snap) + 5, inds[0], inds[-1])

    # Work out the fraction of repositions and reposition attempts in
    # these ranges of outputs
    num_steps = bh_data['NumberOfTimeSteps']
    num_repos = bh_data[num_repos_name]
    num_att = bh_data['NumberOfRepositionAttempts']
    delta_steps = num_steps[ibh, end_ind] - num_steps[ibh, start_ind]
    delta_repos = num_repos[ibh, end_ind] - num_repos[ibh, start_ind]
    delta_repos_att = num_att[ibh, end_ind] - num_att[ibh, start_ind]

    frac_repos = delta_repos / delta_steps
    frac_repos_att = delta_repos_att / delta_steps

    return frac_repos, frac_repos_att


def plot_cumulative_stellar_mass(args, stars, halo, ax):
    """Plot the cumulative stellar mass as a function of formation time."""

    # Cannot do this if we have not linked to VR
    if halo is None or not stars.is_set_up:
        return

    # Get stellar mass and birth time for this halo
    stellar_mass, stellar_time, stellar_radii = stars.query_halo(halo)

    xl.plot_cumdist(stellar_time, stellar_mass/1e10, norm=False, log=False,
                    color='royalblue')

    ax2 = get_yaxis_sfr(ax)    
    plot_rate(stellar_time, stellar_mass/1e9, xr = ax.get_xlim(),
              nbins=100, color='royalblue', linestyle=':', log=True)
    plt.sca(ax)
    
    cvec = ['red', 'seagreen']
    for iiap, iap in enumerate([5.0, 15.0]):
        ind = np.nonzero(stellar_radii < iap)[0]
        xl.plot_cumdist(stellar_time[ind], stellar_mass[ind]/1e10,
                        norm=False, log=False, color=cvec[iiap])
        plt.sca(ax2)
        plot_rate(stellar_time[ind], stellar_mass[ind]/1e9, xr = ax.get_xlim(),
                  nbins=100, color=cvec[iiap], linestyle=':', log=True)
        plt.sca(ax)
        
    return np.sum(stellar_mass)/1e10


def plot_rate(tquant, yquant, xr, nbins, log=False, **kwargs):
    """Plot the rate of change of a quantity yquant."""
    hist, edges = np.histogram(tquant, bins=nbins, range=xr, weights=yquant)
    delta_t = edges[1:] - edges[:-1]
    mid = edges[:-1] + delta_t / 2
    pquant = hist/delta_t
    if log:
        pquant = np.log10(pquant)
    plt.plot(mid, pquant, **kwargs)


class Stars:

    def __init__(self, args):
        """Load star data of entire simulation."""
    
        # Get birth time and mass of all stars in this halo.
        vr_snap = hd.read_attribute(f'{args.wdir}{args.bh_file}', 'Haloes',
                                    'VR_Snapshot')
        if vr_snap is None:
            self.is_set_up = False
            return

        snap_file = f'{args.wdir}{args.snap_name}_{vr_snap:04d}.hdf5'
        stellar_aexp = hd.read_data(snap_file, 'PartType4/BirthScaleFactors')
        self.birth_times = aexp_to_time(stellar_aexp)
        self.mass = hd.read_data(snap_file, 'PartType4/InitialMasses') * 1e10
        self.ids = hd.read_data(snap_file, 'PartType4/ParticleIDs')
        coordinates = hd.read_data(snap_file, 'PartType4/Coordinates')
        coordinates *= (
            hd.read_attribute(snap_file, 'Header', 'Scale-factor')[0])

        # Look up star IDs in VR
        vr_file = f'{args.wdir}{args.vr_file}_{vr_snap:04d}'
        self.haloes, vr_zred, vr_aexp = xl.connect_ids_to_vr(
            self.ids, vr_file, require=True)

        # Get radii of stars
        halo_centres = hd.read_data(f'{vr_file}.hdf5',
                                    'MinimumPotential/Coordinates')
        self.radii = np.linalg.norm(
            coordinates - halo_centres[self.haloes, :], axis=1) * 1e3
        ind_not_in_halo = np.nonzero(self.haloes < 0)[0]
        self.radii[ind_not_in_halo] = -1

        self.is_set_up = True
        
    def query_halo(self, halo):
        """Return star data for a particular halo."""
        ind = np.nonzero(self.haloes == halo)[0]
        return self.mass[ind], self.birth_times[ind], self.radii[ind]

    
def aexp_to_time(aexp):
    # Cosmology instance corresponding to Swift-Planck13
    cosmo = xl.swift_Planck_cosmology()

    # Direct lookup for small number of values
    if len(aexp) < 1000:
        return cosmo.age(1/aexp - 1).to(u.Gyr).value

    # For large input numbers, go via interpolation list
    aexps = np.linspace(aexp.min(), aexp.max(), 200)
    times = cosmo.age(1/aexps-1).to(u.Gyr).value
    csi_time = interp1d(aexps, times, kind='cubic')
    return csi_time(aexp)
    
    
if __name__ == "__main__":
    main()
