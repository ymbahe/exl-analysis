#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot black hole growth tracks.

The data have to be pre-extracted into an HDF5 file, with the structure
/ID{sim_id}/[data]
"""

import hydrangea.hdf5 as hd
import numpy as np
from plot_routines import plot_average_profile
from pdb import set_trace

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.use('pdf')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'][0] = 'palatino'
mpl.rc('text', usetex=True)

import matplotlib.pyplot as plt


from matplotlib import gridspec
from astropy.cosmology import Planck13
from matplotlib.ticker import MultipleLocator
from astropy.io import ascii
import glob
import local

from astropy.cosmology import FlatLambdaCDM
H0 = 67.79
h = H0/100
Oc0 = 0.1187/h**2
Ob0 = 0.02214/h**2
Om0 = Oc0 + Ob0
cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0)

import argparse
print("Parsing input arguments...")
parser = argparse.ArgumentParser(description="Parse input parameters.")
parser.add_argument('--bh_bid', type=int,
                    help='BID of the black hole to highlight.')
parser.add_argument('--simdir', help='Directory of simulation to analyse.')
parser.add_argument('--sim')
parser.add_argument('--basedir', default=f'{local.BASE_DIR}')
args = parser.parse_args()

# Construct simulation directory to process
dirs = glob.glob(args.basedir + f'ID{args.sim}*/')
if len(dirs) != 1:
    print(f"Could not unambiguously find directory for simulation "
        f"{args.sim}!")
    set_trace()
args.simdir = dirs[0]
if not args.simdir.endswith('/'):
    args.simdir = args.simdir + '/'


# Name of the input catalogue, containing all the data to plot
catloc = f'{args.simdir}/black_hole_data.hdf5'
plotloc_bh = f'{args.simdir}/gallery/bh_growth_tracks_BH-BID-{args.bh_bid}.png'

snap_list = None#'outputs_dt5myr_multi.txt'

# List the black hole selections to be plotted
# [formation redshift interval | mass interval | selection point | colour]
selection_list = [[(20.0, 0.0), (5.0, 8.5), 0.0, None],
         ]
ymin, ymax = None, 8.5

bh_min_mass = 5.0


most_massive_only = True
halo_mstar_range = [10.5, 11.4]
central_haloes_only = True

selection_ids = [args.bh_bid]
#selection_ids = [299, 1803, 2987, 3452, 3619, 1729, 3713]
selection_type = 'highlight'

offset_selection = True
show_median = False
show_uncertainty = False
show_scatter = False

show_individual = True
individual_alpha = 0.25
label_at = None #10.0 # None # 10.5

bh_props_list = ['SubgridMasses']

sim_list = [(args.sim, '', '', 2.0, '-')]

if snap_list is not None:
    snap_data = ascii.read(args.simdir + snap_list)
    snap_zred = np.array(snap_data['col1'])
    snap_type = np.array(snap_data['col2'])

    ind_fullsnap = np.nonzero(snap_type == 1)[0]
    zred_snaps = snap_zred[ind_fullsnap]
    times_snaps = cosmo.age(zred_snaps).value
else:
    times_snaps = []

# ------------------------
# Plot I: BH growth tracks
# ------------------------



fig = plt.figure(figsize=(5.0, 4.5))

xr = [0, 14.0]   # Time axis range
xlabel = r'Time [Gyr]'

for iibhl, ibhl in enumerate(bh_props_list):

    if ymin is None:
        ymin = bh_min_mass
    
    if ibhl == 'SubgridMasses':
        yr = [ymin, ymax]
        ylabel = (r'$\log_{10}\,(M_\mathrm{BH,\, subgrid}\,/\,'
                  r'\mathrm{M}_\odot)$')

    else:
        print("Currently, we only plot subgrid masses!")
        set_trace()

    ax = plt.gca()

    ax.set_xlim(xr)
    ax.set_ylim(yr)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

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

    for iisnap, istime in enumerate(times_snaps):
        if iisnap in [9, 18, 24, 25, 30, 33, 36]:
            lwfac = 3
        else:
            lwfac = 1
        plt.plot((istime, istime), yr, linestyle=':', color='grey',
                 linewidth=0.5*lwfac)
        plt.text(istime, yr[1]-(yr[1]-yr[0])*0.02, f'{iisnap}',
                 va='top', ha='center', fontsize=4, color='grey')

    for iisim, isim in enumerate(sim_list):
        
        pre = '' #f'ID{isim[0]}/'
        times = hd.read_data(catloc, pre + 'Times')
        redshifts = hd.read_data(catloc, pre + 'Redshifts')
        first_indices = hd.read_data(catloc, pre + 'FirstIndices')
        masses = hd.read_data(catloc, pre + ibhl)
        masses = np.log10(masses) + 10.0

        first_indices[:] = -1
        # Fix for first indices, while it's not working properly oob
        print("Fixing first indices...")
        for isnap in range(masses.shape[-1]):
            ind_this = np.nonzero((first_indices < 0) &
                                  (masses[:, isnap] * 0 == 0))[0]
            first_indices[ind_this] = isnap
        print("... done. Thank you for your patience.")

        halo_m200 = np.log10(hd.read_data(catloc, pre + 'Halo_M200c'))
        halo_sfr = hd.read_data(catloc, pre + 'Halo_SFR')
        halo_mstar = np.log10(hd.read_data(catloc, pre + 'Halo_MStar'))

        ax = plt.gca()

        if label_at is not None:
            label_ind = np.argmin(np.abs(times - label_at))

        # Go through each selection criterion separately
        for iisel, isel in enumerate(selection_list):

            if isel[0] is None:
                sel = np.arange(len(first_indices))
            else:
                sel = np.nonzero((redshifts[first_indices] > min(isel[0])) &
                                 (redshifts[first_indices] <= max(isel[0])))[0]
          
            if most_massive_only:
                flag_most_massive = hd.read_data(
                    catloc, pre + 'Flag_MostMassiveInHalo')
                ind_subsel = np.nonzero(flag_most_massive[sel] == 1)[0]
                sel = sel[ind_subsel]
                
            if halo_mstar_range is not None:
                ind_subsel = np.nonzero(
                    (halo_mstar[sel] > halo_mstar_range[0]) &
                    (halo_mstar[sel] <= halo_mstar_range[1]))[0]
                sel = sel[ind_subsel]
            
            if central_haloes_only:
                halo_type = hd.read_data(catloc, pre + 'HaloTypes')
                ind_subsel = np.nonzero(halo_type[sel] == 10)[0]
                sel = sel[ind_subsel]
            
            if isel[1] is not None:
                # Black holes in a given mass range at a target redshift
                # First find closest redshift to exact point
                ind_best = np.argmin(np.abs(redshifts - isel[2]))
                if offset_selection and isim[0] in [147, 148]: 
                    offset = -1.0
                else:
                    offset = 0.0
                ind_subsel = np.nonzero(
                    (masses[sel, ind_best] > isel[1][0]+offset) &
                    (masses[sel, ind_best] <= isel[1][1]+offset))[0]

                sel = sel[ind_subsel]
                
            if selection_ids is not None:
                if selection_type == 'highlight':
                    high_sel = np.array(selection_ids)
                else:
                    sel = np.array(selection_ids)
                    high_sel = []
            else:
                high_sel = []
                
            print(f"{len(sel)} BHs are in selection.")
            
            if isel[3] is not None:
                col = isel[3]
            else:
                col = isim[2]
            
            
            if show_individual:
                for ibh in sel:
                    col = plt.cm.viridis(
                        (halo_mstar[ibh] - halo_mstar_range[0])
                        / (halo_mstar_range[1] - halo_mstar_range[0])
                        )
                    #col = plt.cm.viridis(
                    #    (np.log10(redshifts[first_indices[ibh]])-0.5)/0.7)
                    
                    if ibh in high_sel:
                        plot_alpha = 1.0
                    else:
                        plot_alpha = individual_alpha
                    
                    
                    plt.plot(times, masses[ibh, :], linewidth = isim[3]/3,
                             linestyle = isim[4], alpha=plot_alpha,
                             color=col)
                    if label_at is not None:
                        plt.text(label_at, masses[ibh, label_ind], f'{ibh}',
                             fontsize=4,
                             va='top', ha='center', color=col, alpha=plot_alpha)

            if show_median:
                plot_average_profile(times, masses[sel, :].T, color=col,
                                     linestyle=isim[4],
                                     uncertainty=show_uncertainty,
                                     scatter=show_scatter,
                                     percent=75)
            
            #if isel[1] is not None:
            #    plt.plot(times[[ind_best, ind_best]],
            #             (isel[1][0]+offset, isel[1][1]+offset), color='black', linewidth=2)
            #    plt.plot(times[[ind_best, ind_best]],
            #             (isel[1][0]+offset, isel[1][1]+offset), color=col, linewidth=1,
            #             linestyle=isim[4])
                   
    plt.plot(xr, (5.1761, 5.1761), color='grey', linestyle=':', linewidth=0.2)
    plt.plot(xr, (5.204, 5.204), color='grey', linestyle=':', linewidth=0.2)
    plt.plot(xr, (5.398, 5.398), color='grey', linestyle=':', linewidth=0.2)
    plt.plot(xr, (6.061, 6.061), color='grey', linestyle=':', linewidth=0.2)
    
                
plt.subplots_adjust(left = 0.15, right = 0.95, bottom = 0.15, top = 0.88,
                    wspace=0, hspace=0.35)
plt.show
plt.savefig(plotloc_bh, dpi = 200, transparent = False)


print("Done!")
            
                                 
                                  
