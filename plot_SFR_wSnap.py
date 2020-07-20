#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:56:13 2019

@author: bahe
"""

import sys
import local
import xltools as xl
import argparse

# Set up Matplotlib
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.use('pdf')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'][0] = 'palatino'
mpl.rc('text', usetex=True)
import matplotlib.pyplot as plt


def main():

    """Main program"""

    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    
    parser.add_argument('mode', help='Comparison type')

    parser.add_argument('--snap_name', help='Name prefix of simulation outputs '
                        '(default: "output")', default='output')
    parser.add_argument('--max_snap', type=int,
        help='Maximum number of outputs (default: 3000)', default=3000)
    parser.add_argument('--base_dir', help='Base directory of simulations '
        '(default: [LOCAL])',
    default=f'{local.BASE_DIR}')
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
    parser.add_argument('--combined_vr', action='store_true')

    parser.add_argument('--out_dir')
    args = parser.parse_args()


if len(sys.argv) == 1:
    # Select 'mode': 'mergers', 'hostseed', 'colibre', 'upsilon', 'multiphase'
    mode = 'multiphase'
else:
    mode = sys.argv[1]
    
from astropy.io import ascii
import numpy as np
import hydrangea.hdf5 as hd
import os
import glob


from pdb import set_trace


# Load output redshifts
#output_list = f'{local.BASE_DIR}/output_list.txt'
#output_zred = np.array(ascii.read(output_list, format='no_header')['col1'])
#set_trace()

# EAGLE RUNS
sfrloc_e25ref = ''
sfrloc_e25noAGN = ''

def sfr_file(id):
    """Retrieve the SFR file for a specified run ID."""
    dirs = glob.glob(f'{local.BASE_DIR}/ID{id}*/')
    if len(dirs) != 1:
        print(f"Could not unambiguously find directory for simulation {id}!")
        set_trace()
    datafile = dirs[0] + f'/SFR.hdf5'
    if not os.path.isfile(datafile):
        datafile = dirs[0] + f'/SFR.txt'
        if not os.path.isfile(datafile):
            print(f"Could not find SFR logs for sim {id}, neither as HDF5 nor TXT.")
            set_trace()
    return datafile


# Define output files
outdir = f'{local.BASE_DIR}/SFR_comparisons/'
plotloc = outdir + 'SFR-' + mode + '.pdf'
    
def plot_sfr(loc, color, label, simtype='Swift', boxsize=25, linewidth=1.0, linestyle='-', alpha=1.0):

    print(f"Plotting SFR for {label}...")
    if not os.path.exists(loc):
        print(f"Could not find '{loc}'!")
        return

    if simtype == 'Swift':
        ls = '-'
    else:
        ls = ':'
    
    if loc.endswith('.hdf5'):
        aexp = hd.read_data(loc, 'aExp')
        sfr = hd.read_data(loc, 'SFR')/(boxsize**3)
    elif loc.endswith('.txt'):
        sfrdata = ascii.read(loc)
        if simtype == 'Swift':
            aexp = np.array(sfrdata['col3']) 
            sfr = np.array(sfrdata['col8'])/(boxsize**3)*1.022690e-2
        else:
            aexp = np.array(sfrdata['col1'])
            sfr = np.array(sfrdata['col3'])/(boxsize**3)
    else:
        print(f'Incompatible file ending "{loc}"')
        set_trace()
        
    ind_plot = np.linspace(0, len(aexp), 2000, endpoint=False).astype(int)
    plt.plot(aexp[ind_plot], sfr[ind_plot], color = color, linewidth=linewidth,
             label=label, linestyle=linestyle, alpha=alpha)
    print("... done!")
    

fig = plt.figure(figsize = (6.0, 4.5), dpi = None)

h = 0.6777
obsdir = f'{local.BASE_DIR}/SFR_comparisons/observational_data'

# SFR Observational data from Hopkins 2004                                                               
hcorr = np.log10(h) - np.log10(0.7) # h^-2 for SFR, h^-3 for volume                                            
(z, z_up, z_down, lgrho, lgrho_up, lgrho_down) = np.loadtxt(obsdir + "/sfr_hopkins2004_cor.dat"\
, unpack=True )
lgrho = lgrho + np.log10(0.6) + hcorr
obs1_a = 1./(1.+z)
obs1_rho = 10**lgrho
obs1_rho_err = np.array([obs1_rho-10**lgrho_down, 10**lgrho_up - obs1_rho])

# SFR Observational data from Karim 2011                                                                 
(z, rho, err_up, err_down) = np.loadtxt(obsdir + "/sfr_karim2011.dat", unpack=True )
obs2_a = 1./(1.+z)
obs2_rho = rho * 0.6777/0.7
obs2_rho_err = np.array([-err_down, err_up])

# SFR Observational data from Bouwens 2012                                                               
z, rhostar = np.loadtxt(obsdir + "/bouwens_2012_sfrd_no_dust.txt", unpack=True)
z, rhostar_dust = np.loadtxt(obsdir + "/bouwens_2012_sfrd_dustcorr.txt", unpack=True)
rhostar = (rhostar / 1.8) * 0.6777 / 0.7 #convert to Chabrier IMF from Salpeter and adjust for change in cosmology                                                                                               
rhostar_dust = (rhostar_dust / 1.8) * 0.6777 / 0.7  #convert to Chabrier IMF from Salpeter and adjust for change in cosmology                                                                                    
obs3_a = 1./ (1.+z)
obs3_rho = rhostar
obs3_rho_dust = rhostar_dust

# SFR Observational data from Rodighierio 2012                                                           
z, rhostar, err_m, err_p = np.loadtxt(obsdir + "/sfr_rodighiero2010.dat", unpack=True)
rhostar = (rhostar/1.65) * 0.6777 / 0.7 #convert to Chabrier IMF from Salpeter and adjust for change in cosmology                                                                                                
obs4_a = 1. / (1.+z)
obs4_rho = rhostar
obs4_rho_err = np.array([-err_m/1.65, err_p/1.65])

# SFR Observational data from Cucciati 2012                                                              
z, rhostar, err_m, err_p = np.loadtxt(obsdir + "/sfr_cucciati2011.dat", unpack=True)
rhostar = (rhostar - 2. * np.log10(0.7) + 2.*np.log10(0.6777))
obs5_a = 1. / (1+z)
obs5_rho = 10**rhostar / 1.65
obs5_rho_err = 10**np.array([rhostar + (err_m), rhostar + err_p]) / 1.65
obs5_rho_err[0] = -obs5_rho_err[0]  + obs5_rho
obs5_rho_err[1] = -obs5_rho + obs5_rho_err[1]

##################################################################################                       

# SFR from Madau & Dickinson fitting formula (z < 10 and assuming h=0.7)                                 
obs6_a = np.logspace(np.log10(1. / (1. + 10.)), 0, 100)
obs6_z = 1. / obs6_a - 1.
obs6_rho = 0.015 * ((1. + obs6_z)**2.7) / (1. + ((1. + obs6_z) / 2.9)**5.6)  # Msun / yr / Mpc^3         
obs6_rho /= 1.65 # Salpeter -> Chabrier correction                                                       

##################################################################################      

# Observational data                                                                                     
plt.plot(obs6_a, obs6_rho, 'k-', lw=0.5, alpha=0.3) 
     #label="${\\rm Madau~\\&~Dickinson~(2014)~~(h=0.7)}$")

plt.errorbar(obs4_a, obs4_rho, yerr=obs4_rho_err, fmt='s', mec='0.3', color='0.3', 
         markersize=4, markeredgewidth=0.5, linewidth=0.5, mfc='w') 
         #label="${\\rm Rodighiero~et~al.~(2010)~(24\\mu m)}$")
plt.errorbar(obs2_a, obs2_rho, yerr=obs2_rho_err, fmt='.', mec='0.3', color='0.3', 
         markersize=7, markeredgewidth=0.5, linewidth=0.5, mfc='w') 
         #label="${\\rm Karim~et~al.~(2011)~(radio)}$")
plt.errorbar(obs5_a, obs5_rho, yerr=obs5_rho_err, fmt='^', mec='0.3', color='0.3', 
         markersize=4, markeredgewidth=0.5, linewidth=0.5, mfc='w') 
         #label="${\\rm Cucciati~et~al.~(2012)~(FUV)}$")
plt.plot(obs3_a, obs3_rho_dust, 'd', mec='0.3', color='0.3', markersize=4, 
     markeredgewidth=0.5, linewidth=1, mfc='w') 
     #label="${\\rm Bouwens~et~al.~(2012)~(UV,~no~dust)}$")

"""
plot_sfr(sfrloc_e25ref, 'red', 'E25-REF', simtype='Gadget')
plot_sfr(sfrloc_e25noAGN, 'orange', 'E25-noAGN', simtype='Gadget')

plot_sfr(sfrloc_yse25_defWithAGN, 'royalblue', 'YB-Def-wAGN')
plot_sfr(sfrloc_yse25_minNoAGN, 'purple', 'YB-Min-noAGN')
plot_sfr(sfrloc_yse25_minWithAGN, 'fuchsia', 'YB-Min-withAGN')

plot_sfr(sfrloc_yse25_limWithAGN, 'limegreen', 'YB-Lim-withAGN')
plot_sfr(sfrloc_yse25_limNoAGN, 'seagreen', 'YB-Lim-noAGN')

#plot_sfr(sfrloc_se25_limiter, 'seagreen', 'MS-ADULim-noAGN')
#plot_sfr(sfrloc_se25_minimal, 'limegreen', 'MS-Min-noAGN')
"""

#plot_sfr(sfrloc_id6, 'red', 'ID6', boxsize = 12)
#plot_sfr(sfrloc_id7, 'blue', 'ID7', boxsize = 12)
#plot_sfr(sfrloc_id21, 'black', 'ID21_stdAGN')
#if mode != 'upsilon' and not mode.endswith('colibre') and not mode == 'bugtest-xl' and not mode in ['bugtest-colibre', 'angmomtest-xl', 'buglrtest-xl', 'seedtest-xl', 'newtest-xl', 'sn-fs-test']:
#    plot_sfr(sfr_file(22), 'gray', 'ID22_noAGN')
#plot_sfr(sfrloc_id23, 'gold', 'ID12\_f0p10')
#plot_sfr(sfrloc_id23, 'seagreen', 'ID23\_f0p05')
#plot_sfr(sfrloc_id24, 'royalblue', 'ID24\_mergerMod')
#plot_sfr(sfrloc_id25, 'purple', 'ID25\_multiPhaseMod')

#plot_sfr(sfrloc_id34, 'red', 'ID34\_EEE\_6', boxsize=6)
#plot_sfr(sfrloc_id37, 'blue', 'ID37\_CCC\_6', boxsize=6)

#plot_sfr(sfrloc_id43, 'blue', 'ID43\_modCS', boxsize=25)
#plot_sfr(sfrloc_id44, 'purple', 'ID44\_modCS\_MPB', boxsize=25)

if mode == 'mergers':
    plot_sfr(sfr_file(55), 'brown', 'ID55_bugfix', boxsize=25)
#plot_sfr(sfrloc_id56, 'orange', 'ID56_no-vel-cut', boxsize=25)

#if not mode.endswith('colibre') and not mode in ['bugtest-xl', 'bugtest-colibre', 'angmomtest-xl', 'buglrtest-xl', 'seedtest-xl', 'newtest-xl', 'sn-fs-test']:
#    plot_sfr(sfr_file(57), 'red', 'ID57_new-mergers', boxsize=25)

if mode == 'upsilon':
    plot_sfr(sfr_file(58), 'goldenrod', 'ID58_no-vel-cut_new-merg', boxsize=25)

#plot_sfr(sfrloc_id59, 'purple', 'ID59\_no-vel\_eps1', boxsize=25)

#plot_sfr(sfrloc_id61, 'purple', 'ID61_repos-all', boxsize=25)
#plot_sfr(sfrloc_id62, 'royalblue', 'ID62_repos-all_no-vel-cut', boxsize=25)
#plot_sfr(sfrloc_id63, 'skyblue', 'ID63_repos-all-active_no-vel-cut', boxsize=25)


#plot_sfr(sfrloc_id64, 'limegreen', 'ID64_no-vel-cut_seed-1e4', boxsize=25)
#plot_sfr(sfrloc_id65, 'seagreen', 'ID65_no-velcut_merg2_seed-1e4', boxsize=25)
#plot_sfr(sfrloc_id66, 'darkgreen', 'ID66_repos-AA_NV_seed-1e4', boxsize=25)
#plot_sfr(sfrloc_id67, 'turquoise', 'ID67_no-velcut_mhalo-1e11', boxsize=25)
#plot_sfr(sfrloc_id68, 'darkcyan', 'ID68_no-velcut_mhalo-1e11_seed-1e4', boxsize=25)

#plot_sfr(sfrloc_id70, 'palevioletred', 'ID70_C25_vellim30_badsoft', boxsize=25)
#plot_sfr(sfrloc_id71, 'violet', 'ID71_C25_std_badsoft', boxsize=25)

if mode == 'colibre':
    plot_sfr(sfrloc_id72, 'purple', 'ID72_C25_std', boxsize=25)
    plot_sfr(sfrloc_id73, 'magenta', 'ID73_C25_vellim30', boxsize=25)

if mode == 'hostseed':
    plot_sfr(sfrloc_id74, 'green', 'ID74_1p5e10', boxsize=25)
    plot_sfr(sfrloc_id75, 'orange', 'ID75_3e10', boxsize=25)
    plot_sfr(sfrloc_id76, 'goldenrod', 'ID76_6e10', boxsize=25)
    plot_sfr(sfrloc_id77, 'magenta', 'ID77_1p5e11', boxsize=25)

if mode == 'colibre':
    plot_sfr(sfrloc_id78, 'skyblue', 'ID78_CC12', boxsize=12.5)
    plot_sfr(sfrloc_id79, 'brown', 'ID79_C25_noAGN', boxsize=25)

if mode == 'upsilon':
    plot_sfr(sfrloc_id85, 'darkred', 'ID85_std')
    plot_sfr(sfrloc_id86, 'grey', 'ID86_noAGN')
    plot_sfr(sfrloc_id82, 'purple', 'ID82_ups1em5')
    plot_sfr(sfrloc_id83, 'royalblue', 'ID83_ups1em6')
    plot_sfr(sfrloc_id87, 'skyblue', 'ID87_ups1em7')
    plot_sfr(sfrloc_id88, 'aquamarine', 'ID88_ups1em8')
    plot_sfr(sfrloc_id89, 'seagreen', 'ID89_ups1em9')
    plot_sfr(sfrloc_id90, 'limegreen', 'ID90_no-repos')
    plot_sfr(sfrloc_id105, 'magenta', 'ID105_ups1em3')
    plot_sfr(sfrloc_id106, 'deeppink', 'ID106_ups1em4')
    plot_sfr(sfrloc_id107, 'red', 'ID107_use1em5')

if mode == 'upsilon-colibre':
    plot_sfr(sfrloc_id93, 'purple', 'ID93_ups1em5')
    plot_sfr(sfrloc_id94, 'royalblue', 'ID94_ups1em6')
    plot_sfr(sfrloc_id95, 'skyblue', 'ID95_ups1em7')
    plot_sfr(sfrloc_id96, 'seagreen', 'ID96_ups1em8')
    plot_sfr(sfrloc_id97, 'grey', 'ID97_no-repos')
    plot_sfr(sfrloc_id108, 'brown', 'ID108_no-agn')
    plot_sfr(sfrloc_id109, 'magenta', 'ID109_ups1em3')
    plot_sfr(sfrloc_id110, 'deeppink', 'ID110_ups1em4')

    
if mode == 'components-colibre':
    #plot_sfr(sfrloc_id72, 'purple', 'ID72_std')
    #plot_sfr(sfrloc_id79, 'brown', 'ID79_noAGN')
    plot_sfr(sfrloc_id94, 'seagreen', 'ID94_ups1em6')
    plot_sfr(sfrloc_id98, 'goldenrod', 'ID98_no-multi')
    plot_sfr(sfrloc_id99, 'red', 'ID99_no-subgrid-bondi')

if mode == 'multiphase-xl':
    plot_sfr(sfrloc_id100, 'royalblue', 'ID100_no-multiphase')
    plot_sfr(sfrloc_id101, 'goldenrod', 'ID101_with-multiphase')
    
if mode == 'sntest-colibre':
    plot_sfr(sfrloc_id72, 'grey', 'ID72_1e51_bh-old-repos')
    plot_sfr(sfrloc_id94, 'seagreen', 'ID94_2e51')
    plot_sfr(sfrloc_id102, 'goldenrod', 'ID102_3e51')

if mode == 'restest-colibre':
    plot_sfr(sfrloc_id94, 'grey', 'ID94_C25_2e51')
    plot_sfr(sfrloc_id103, 'royalblue', 'ID103_CC12_SN1e51', boxsize=12.5)
    plot_sfr(sfrloc_id104, 'goldenrod', 'ID104_C12_SN1e51', boxsize=12.5)

if mode == 'seedtest-colibre':
    plot_sfr(sfrloc_id97, 'lightsalmon', 'ID97_no-repos_1e10')
    plot_sfr(sfrloc_id109, 'darkred', 'ID109_ups1em3_1e10')    
    plot_sfr(sfrloc_id118, 'skyblue', 'ID118_no-repos_1e11')
    plot_sfr(sfrloc_id119, 'darkblue', 'ID119_ups1em3_1e11')

if mode == 'bugtest-xl':
    plot_sfr(sfrloc_id22, 'gray', 'ID22_noAGN', boxsize=25, linestyle='-', linewidth=2.0)
    plot_sfr(sfrloc_id134, 'skyblue', 'ID134_25Mpc_XL_fixed', boxsize=25, linestyle='-', linewidth=2.0)
    plot_sfr(sfrloc_id57, 'skyblue', 'ID57_25Mpc_XL_buggy', boxsize=25, linestyle=':', linewidth=2.0)
    plot_sfr(sfrloc_id132, 'royalblue', 'ID132_12Mpc_XL_fixed', boxsize=12.5, linestyle='-', linewidth=0.5)
    plot_sfr(sfrloc_id136, 'darkslateblue', 'ID136_12Mpc_XL_fixed', boxsize=12.5, linestyle='-', linewidth=0.5)
    plot_sfr(sfrloc_id125, 'royalblue', 'ID125_12Mpc_XL_buggy', boxsize=12.5, linestyle=':', linewidth=0.5)
    plot_sfr(sfrloc_id135, 'plum', 'ID135_25Mpc_ups1em3_fixed', boxsize=25, linestyle='-', linewidth=2.0)    
    plot_sfr(sfrloc_id105, 'plum', 'ID105_25Mpc_ups1em3_buggy', boxsize=25.0, linestyle=':', linewidth=2.0)
    plot_sfr(sfrloc_id133, 'crimson', 'ID133_12Mpc_ups1em3_fixed', boxsize=12.5, linestyle='-', linewidth=0.5)
    plot_sfr(sfrloc_id137, 'magenta', 'ID137_12Mpc_ups1em3_fixed', boxsize=12.5, linestyle='-', linewidth=0.5)

if mode == 'bugtest-colibre':
    plot_sfr(sfrloc_id108, 'gray', 'ID108_noAGN', boxsize=25, linestyle='-', linewidth=0.5)
    plot_sfr(sfrloc_id97, 'seagreen', 'ID97_norepos_buggy', boxsize=25.0, linestyle=':', linewidth=0.5)
    plot_sfr(sfrloc_id109, 'plum', 'ID109_ups1em3_buggy', boxsize=25.0, linestyle=':', linewidth=0.5)    
    plot_sfr(sfrloc_id138, 'seagreen', 'ID138_norepos_fixed', boxsize=25.0, linestyle='-', linewidth=0.5)
    plot_sfr(sfrloc_id139, 'plum', 'ID139_ups1em3_fixed', boxsize=25.0, linestyle='-', linewidth=0.5)

if mode == 'angmomtest-xl':
    plot_sfr(sfr_file(22), 'gray', 'ID22_noAGN', boxsize=25, linestyle='-', linewidth=2.0)
    plot_sfr(sfr_file(134), 'skyblue', 'ID134_25Mpc_XL-wAngMom', boxsize=25, linestyle='-', linewidth=2.0)
    plot_sfr(sfr_file(132), 'royalblue', 'ID132_12Mpc_XL-wAngMom', boxsize=12.5, linestyle='-', linewidth=0.5)
    plot_sfr(sfr_file(136), 'darkslateblue', 'ID136_12Mpc_XL-wAngMom', boxsize=12.5, linestyle='-', linewidth=0.5)
    plot_sfr(sfr_file(144), 'blue', 'ID144_12Mpc_XL-wAngMom', boxsize=12.5, linestyle='-', linewidth=0.5)
    plot_sfr(sfr_file(140), 'royalblue', 'ID140_12Mpc_XL-noAngMom', boxsize=12.5, linestyle=':', linewidth=0.5)
    
    plot_sfr(sfr_file(135), 'plum', 'ID135_25Mpc_ups1em3-wAngMom', boxsize=25, linestyle='-', linewidth=2.0)
    plot_sfr(sfr_file(133), 'crimson', 'ID133_12Mpc_ups1em3-wAngMom', boxsize=12.5, linestyle='-', linewidth=0.5)
    plot_sfr(sfr_file(137), 'magenta', 'ID137_12Mpc_ups1em3-wAngMom', boxsize=12.5, linestyle='-', linewidth=0.5)

    plot_sfr(sfr_file(142), 'skyblue', 'ID142_25Mpc_XL-noAngMom', boxsize=25, linestyle=':', linewidth=2.0)
    plot_sfr(sfr_file(143), 'plum', 'ID143_25Mpc_ups1em3-noAngMom', boxsize=25, linestyle=':', linewidth=2.0)
    plot_sfr(sfr_file(141), 'crimson', 'ID140_12Mpc_ups1em3-noAngMom', boxsize=12.5, linestyle=':', linewidth=0.5)

    plot_sfr(sfr_file(105), 'orange', 'ID105_25Mpc_ups1em3-wAngMom-buggy', boxsize=25, linestyle='--', linewidth=2.0, alpha=0.5)

if mode == 'buglrtest-xl':
    plot_sfr(sfr_file(22), 'gray', 'ID22_noAGN', boxsize=25, linestyle='-', linewidth=2.0)
    plot_sfr(sfr_file(134), 'cornflowerblue', 'ID134_25Mpc_XL_fixed', boxsize=25, linestyle='-', linewidth=2.0)
    plot_sfr(sfr_file(57), 'skyblue', 'ID57_25Mpc_XL_buggy', boxsize=25, linestyle=':', linewidth=2.0)
    #plot_sfr(sfrloc_id132, 'royalblue', 'ID132_12Mpc_XL_fixed', boxsize=12.5, linestyle='-', linewidth=0.5)
    #plot_sfr(sfrloc_id136, 'darkslateblue', 'ID136_12Mpc_XL_fixed', boxsize=12.5, linestyle='-', linewidth=0.5)
    #plot_sfr(sfrloc_id125, 'royalblue', 'ID125_12Mpc_XL_buggy', boxsize=12.5, linestyle=':', linewidth=0.5)
    #plot_sfr(sfr_file(135), 'plum', 'ID135_25Mpc_ups1em3_fixed', boxsize=25, linestyle='-', linewidth=2.0)    
    #plot_sfr(sfr_file(105), 'lightpink', 'ID105_25Mpc_ups1em3_buggy', boxsize=25.0, linestyle=':', linewidth=2.0)
    #plot_sfr(sfrloc_id133, 'crimson', 'ID133_12Mpc_ups1em3_fixed', boxsize=12.5, linestyle='-', linewidth=0.5)
    #plot_sfr(sfrloc_id137, 'magenta', 'ID137_12Mpc_ups1em3_fixed', boxsize=12.5, linestyle='-', linewidth=0.5)
    plot_sfr(sfr_file(149), 'lightgreen', 'ID149_25MpcLR_XL_fixed', boxsize=25, linestyle='-', linewidth=2.0)
    plot_sfr(sfr_file(150), 'palegreen', 'ID150_25MpcLR_XL_buggy', boxsize=25, linestyle=':', linewidth=2.0)

if mode == 'seedtest-xl':
    plot_sfr(sfr_file(22), 'gray', 'ID22_noAGN', boxsize=25, linestyle='-', linewidth=2.0)
    plot_sfr(sfr_file(134), 'cornflowerblue', 'ID134_25Mpc_XL_seed1p5e5', boxsize=25, linestyle='-', linewidth=2.0)
    plot_sfr(sfr_file(147), 'skyblue', 'ID147_25Mpc_XL_seed1p5e4', boxsize=25, linestyle=':', linewidth=2.0)
    plot_sfr(sfr_file(135), 'plum', 'ID135_25Mpc_ups1em3_seed1p5e5', boxsize=25, linestyle='-', linewidth=2.0)
    plot_sfr(sfr_file(148), 'lightpink', 'ID148_25Mpc_ups1em3_seed1p5e4', boxsize=25, linestyle=':', linewidth=2.0)

if mode == 'newtest-xl':
    plot_sfr(sfr_file(145), 'grey', 'ID145')
    plot_sfr(sfr_file(172), 'red', 'ID172')
    plot_sfr(sfr_file(173), 'orange', 'ID173')
    plot_sfr(sfr_file(174), 'goldenrod', 'ID174')
    plot_sfr(sfr_file(175), 'limegreen', 'ID175')
    plot_sfr(sfr_file(176), 'skyblue', 'ID176')
    plot_sfr(sfr_file(177), 'royalblue', 'ID177')
    plot_sfr(sfr_file(178), 'purple', 'ID178')
    
if mode == 'sn-fs-test':
    plot_sfr(sfr_file(205), 'limegreen', 'ID205')
    plot_sfr(sfr_file(206), 'seagreen', 'ID206')    
    plot_sfr(sfr_file(226), 'black', 'ID226')
    plot_sfr(sfr_file(227), 'royalblue', 'ID227')
    plot_sfr(sfr_file(228), 'skyblue', 'ID228')
    plot_sfr(sfr_file(229), 'grey', 'ID229')
    #plot_sfr(sfr_file(238), 'aquamarine', 'ID238')
    plot_sfr(sfr_file(268), 'red', 'ID268')
    plot_sfr(sfr_file(269), 'salmon', 'ID269')
    plot_sfr(sfr_file(270), 'fuchsia', 'ID270')
    plot_sfr(sfr_file(271), 'indianred', 'ID271')
    plot_sfr(sfr_file(276), 'yellow', 'ID276')
    plot_sfr(sfr_file(277), 'goldenrod', 'ID277')
    plot_sfr(sfr_file(280), 'purple', 'ID280')

if mode == 'sn-fs-new-test':
    plot_sfr(sfr_file(204), 'grey', 'ID204\_std')

    plot_sfr(sfr_file(276), 'seagreen', 'ID276\_fE0p5-5p0\_fs0p5-1')
    plot_sfr(sfr_file(281), 'aquamarine', 'ID281\_fE0p5-5p0\_fs2-5', linestyle='--')
    plot_sfr(sfr_file(282), 'seagreen', 'ID282\_fE0p5-5p0\_fc-3', linestyle=':')
    plot_sfr(sfr_file(279), 'limegreen', 'ID279\_fE0p5-5p0\_fs0p5-1\_direct-rho')
    plot_sfr(sfr_file(254), 'mediumturquoise', 'ID254\_fE0p5-5p0\_fs0p5-1\_Tmax7p5')

    plot_sfr(sfr_file(277), 'goldenrod', 'ID277\_fE0p3-3p0\_fs0p5-1')
    plot_sfr(sfr_file(273), 'yellow', 'ID273\_fE0p3-3p0\_fs2-5', linestyle='--')
    plot_sfr(sfr_file(274), 'yellow', 'ID274\_fE0p3-3p0\_fc-3', linestyle=':')

    plot_sfr(sfr_file(278), 'red', 'ID278\_fEconst\_fs0p5-1')    
    plot_sfr(sfr_file(268), 'red', 'ID268\_fEconst\_fs1-3', linestyle='--')
    plot_sfr(sfr_file(269), 'salmon', 'ID269\_fEconst\_fs2-5', linestyle='--')
    plot_sfr(sfr_file(270), 'coral', 'ID270\_fEconst\_fs3-6', linestyle='--')

    plot_sfr(sfr_file(271), 'purple', 'ID271\_fEnon-0p3-1p0\_fs2-5', linestyle='--')

    plot_sfr(sfr_file(285), 'brown', 'ID285\_alpha-0p1', linestyle='-')

if mode == 'repos-test':
    plot_sfr(sfr_file(204), 'brown', 'ID204\_StrongRepos\_no-mergers')
    plot_sfr(sfr_file(232), 'black', 'JB-StrongRepos')
    plot_sfr(sfr_file(294), 'grey', 'JB-Ref')
    
    #plot_sfr(sfr_file(255), 'red', 'ID255\_200')
    #plot_sfr(sfr_file(256), 'goldenrod', 'ID256\_40')
    #plot_sfr(sfr_file(257), 'orange', 'ID257\_8')
    #plot_sfr(sfr_file(258), 'seagreen', 'ID258\_1p6')
    #plot_sfr(sfr_file(259), 'orange', 'ID259\_8\_M1', linestyle=":")
    #plot_sfr(sfr_file(260), 'seagreen', 'ID260\_40\_M1', linestyle=":")

    plot_sfr(sfr_file(261), 'aquamarine', 'ID261\_0p25-cs')
    plot_sfr(sfr_file(262), 'skyblue', 'ID262\_1p0-cs')
    plot_sfr(sfr_file(263), 'royalblue', 'ID263\_4p0-cs')
    plot_sfr(sfr_file(264), 'purple', 'ID264\_16p0-cs')
    

if mode == 'repos-test-rgb':
    plot_sfr(sfr_file(204), 'brown', 'ID204\_StrongRepos\_no-mergers', linewidth=1, linestyle=':')
    plot_sfr(sfr_file(232), 'black', 'JB-StrongRepos', linewidth=1, linestyle=':')
    plot_sfr(sfr_file(294), 'grey', 'JB-Ref', linewidth=1, linestyle=':')
    
    plot_sfr(sfr_file(255), 'maroon', 'ID255\_200')
    plot_sfr(sfr_file(256), 'indianred', 'ID256\_40')
    plot_sfr(sfr_file(257), 'lightcoral', 'ID257\_8')
    plot_sfr(sfr_file(258), 'orange', 'ID258\_1p6')
    plot_sfr(sfr_file(259), 'lightcoral', 'ID259\_8\_M1', linestyle="--")
    plot_sfr(sfr_file(260), 'indianred', 'ID260\_40\_M1', linestyle="--")

    plot_sfr(sfr_file(261), 'aquamarine', 'ID261\_0p25-cs')
    #plot_sfr(sfr_file(262), 'skyblue', 'ID262\_1p0-cs')
    #plot_sfr(sfr_file(263), 'royalblue', 'ID263\_4p0-cs')
    plot_sfr(sfr_file(264), 'purple', 'ID264\_16p0-cs')


    
# Add lines indicating (full) snapshots
if 1 == 0:
    for iout, zout in enumerate(output_zred):
        aexp_out = 1/(1+zout)
        if aexp_out < 0.090909: continue
        plt.plot((aexp_out, aexp_out), (5e-4, 2e-1), color = 'grey', linestyle=':',
                linewidth=0.5)
        plt.text(aexp_out, 5.5e-4, f'{iout}', color='grey', fontsize=4,
                bbox={'facecolor':'white', 'alpha':0.5, 'edgecolor':'none'},
                va='bottom', ha='center')
    
ax = plt.gca()
ax.set_xlim((1.0, 0.0625))
ax.set_ylim((5e-4, 2e-1))
ax.set_xscale('log')
ax.set_yscale('log')

zticks = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0])
ztickname = ['0.0', '0.2', '0.5', '1', '2', '3', '5', '7', '10', '15']
ztickloc = 1/(1+zticks)

plt.xticks(ztickloc, ztickname)
ax.tick_params(axis='x', which='minor', bottom=False)

ax.set_xlabel(r'Redshift $z$')
ax.set_ylabel(r'$\rho_\mathrm{SFR}\,[\mathrm{M}_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3}]$')

plt.legend(fontsize=6, bbox_to_anchor=(0, 0.05, 1.0, 0.95))

plt.subplots_adjust(left = 0.15, bottom = 0.15, right = 0.95, top=0.95)
plt.show
plt.savefig(plotloc, dpi = None, transparent = False)




if __name__ == "__main__":
    main()
    print("Done!")
