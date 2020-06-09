"""Plot M_star vs. M200, colour-coded by sSFR."""

import numpy as np
from pdb import set_trace
import hydrangea.hdf5 as hd
import os
import local

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

    args = type('test', (object,), {})()
    args.wdir = simdir
    args.vrsnap = snapshot
    vr_data = connect_to_galaxies(bh_ids[bh_list], args)
    
    for ibh in bh_list:
        make_plot_mstar_m200(ibh, args, vr_data)


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
