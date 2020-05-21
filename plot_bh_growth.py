import hydrangea as hy
import numpy as np
from pdb import set_trace
from astropy.cosmology import FlatLambdaCDM
import os
import argparse
import glob
import hydrangea.hdf5 as hd

import matplotlib
matplotlib.use('pdf')

import matplotlib.font_manager as fm
fontpath = '/cosma/home/dp004/dc-bahe1/.matplotlib/Palatino.ttc'
prop = fm.FontProperties(fname=fontpath)
fname = os.path.split(fontpath)[1]
matplotlib.rcParams['font.family'] = prop.get_name()

#matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['font.serif'][0] = 'palatino'
matplotlib.rcParams['mathtext.default'] = 'regular'


import matplotlib.pyplot as plt

print("Parsing input arguments...")
parser = argparse.ArgumentParser(description="Parse input parameters.")
parser.add_argument('sim', type=int, help='Simulation index to analyse')
parser.add_argument('id', type=int, help='Black hole ID to analyse')
parser.add_argument('--num_snap', type=int, help='Maximum number of snapshots',
                    default=2700)
parser.add_argument('--mmax', type=float, help='Maximum log mass to display',
                    default=7.0)
parser.add_argument('--reload', action='store_true', help='Reload data from files?')
parser.add_argument('--width', type=float, help='Plot width in inch',
                    default=9.0)
parser.add_argument('--linewidth', type=float, help='Plot linewidth',
                    default=0.2)
parser.add_argument('--snapname', default='eagle')

args = parser.parse_args()

matplotlib.rcParams['lines.linewidth'] = args.linewidth

dirs = glob.glob(f'/cosma7/data/dp004/dc-bahe1/EXL/ID{args.sim}*/')
if len(dirs) != 1:
    print(f"Could not unambiguously find directory for simulation {args.sim}!")
    set_trace()
wdir = dirs[0] + '/'
print(f"Analysing simulation {wdir}...")

H0 = 67.79
h = H0/100
Oc0 = 0.1187/h**2
Ob0 = 0.02214/h**2
Om0 = Oc0 + Ob0
cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0)

plot_bvel = False
plot_bcvel = True

id_subj = args.id
id_cen = None
nsnap = args.num_snap

msg = np.zeros((nsnap, 2)) - 1
mdyn = np.zeros((nsnap, 2)) - 1
macc = np.zeros((nsnap, 2)) - 1
vbh = np.zeros((nsnap, 2)) - 1
vgas = np.zeros((nsnap, 2)) - 1
vcgas = np.zeros((nsnap, 2)) - 1
cgas = np.zeros((nsnap, 2)) - 1
cgasc = np.zeros((nsnap, 2)) - 1
sgas = np.zeros((nsnap, 2)) - 1
rhogas = np.zeros((nsnap, 2)) - 1
times = np.zeros(nsnap) - 1
num_steps = np.zeros((nsnap, 2), dtype=int) - 1
num_repos = np.zeros((nsnap, 2), dtype=int) - 1
num_repos_att = np.zeros((nsnap, 2), dtype=int) - 1
num_swallows = np.zeros((nsnap, 2), dtype=int) - 1
num_mergers = np.zeros((nsnap, 2), dtype=int) - 1
fvisc = np.zeros((nsnap, 2)) - 1

hdfile = wdir + f'/bh_track_data_id{args.id}.hdf5'

if args.reload:
    for isnap in range(nsnap):

        print("")
        print(f"Snapshot {isnap}")
        print("")

        datafile = wdir + f'{args.snapname}_{isnap:04d}.hdf5'
        if not os.path.isfile(datafile):
            break
    
        time = hy.hdf5.read_attribute(datafile, 'Header', 'Time')[0]
        utime = hy.hdf5.read_attribute(datafile, 'Units', 'Unit time in cgs (U_t)')[0]
        utime /= hy.units.GIGAYEAR
        time *= utime

        times[isnap] = time
    
        aexp_factor = hy.hdf5.read_attribute(datafile, 'Header', 'Scale-factor')[0]

        ulen = hy.hdf5.read_attribute(datafile, 'Units', 'Unit length in cgs (U_L)')[0]
        ulen *= hy.units.CM
        ulen /= hy.units.MEGAPARSEC
    
        if isnap == 0:
            code_branch = hy.hdf5.read_attribute(datafile, 'Code', 'Git Branch')
            code_date = hy.hdf5.read_attribute(datafile, 'Code', 'Git Date')
            code_rev  = hy.hdf5.read_attribute(datafile, 'Code', 'Git Revision')
            sim_name = hy.hdf5.read_attribute(datafile, 'Header', 'RunName')
            
        bids = hy.hdf5.read_data(datafile, 'PartType5/ParticleIDs')
        if bids is None:
            continue

        bpos = hy.hdf5.read_data(datafile, 'PartType5/Coordinates') * ulen * 1e3
        bvel = hy.hdf5.read_data(datafile, 'PartType5/Velocities')
        bsgmass = hy.hdf5.read_data(datafile, 'PartType5/SubgridMasses') * 1e10
        bptmass = hy.hdf5.read_data(datafile, 'PartType5/DynamicalMasses') * 1e10
        bmacc = hy.hdf5.read_data(datafile, 'PartType5/AccretionRates') * 1.023045e-02
        bvgas = hy.hdf5.read_data(datafile, 'PartType5/GasRelativeVelocities')
        bvcgas = hy.hdf5.read_data(datafile, 'PartType5/GasCircularVelocities')
        bcgas = hy.hdf5.read_data(datafile, 'PartType5/GasSoundSpeeds') * aexp_factor
        bcgasc = hy.hdf5.read_data(datafile, 'PartType5/GasSoundSpeeds') / aexp_factor
        bsgas = hy.hdf5.read_data(datafile, 'PartType5/GasVelocityDispersions')
        brhogas = hy.hdf5.read_data(datafile, 'PartType5/GasDensities')
        bsteps = hy.hdf5.read_data(datafile, 'PartType5/NumberOfTimeSteps')
        bswallow = hy.hdf5.read_data(datafile, 'PartType5/NumberOfSwallows')
        bmerger = hy.hdf5.read_data(datafile, 'PartType5/NumberOfMergers')
        brepos = hy.hdf5.read_data(datafile, 'PartType5/NumberOfRepositionings')
        breposatt = hy.hdf5.read_data(datafile, 'PartType5/NumberOfRepositionAttempts')  
        bfvisc = hy.hdf5.read_data(datafile, 'PartType5/ViscosityFactors')  
        if bfvisc is None:
            has_fvisc = False
        else:
            has_fvisc = True

            
        ind_subj = np.nonzero(bids == id_subj)[0]
        if len(ind_subj) > 0:
            ind_subj = ind_subj[0]
        else:
            print("Subject BH not found...")
            ind_subj = None

        ind_cen = np.nonzero(bids == id_cen)[0]
        if len(ind_cen) > 0:
            ind_cen = ind_cen[0]
        else:
            print("Central BH not found...")
            ind_cen = None

        if ind_subj is not None:
            print(f"BH log M_sg [subj] = {np.log10(bsgmass[ind_subj])}")
            print(f"BH accr [subj] = {np.log10(bmacc[ind_subj])}")

            msg[isnap, 0] = np.log10(bsgmass[ind_subj])
            mdyn[isnap, 0] = np.log10(bptmass[ind_subj])
            macc[isnap, 0] = np.log10(bmacc[ind_subj])
            vbh[isnap, 0] = np.log10(np.linalg.norm(bvel[ind_subj, :]))
            vgas[isnap, 0] = np.log10(np.linalg.norm(bvgas[ind_subj, :]))
            vcgas[isnap, 0] = np.log10(np.linalg.norm(bvcgas[ind_subj, :]))
            cgas[isnap, 0] = np.log10(bcgas[ind_subj])
            cgasc[isnap, 0] = np.log10(bcgasc[ind_subj])
            if bsgas is not None:
                sgas[isnap, 0] = np.log10(bsgas[ind_subj])
                has_veldisp = True
            else:
                has_veldisp = False
            rhogas[isnap, 0] = np.log10(brhogas[ind_subj])
            num_steps[isnap, 0] = bsteps[ind_subj]
            num_repos[isnap, 0] = brepos[ind_subj]
            num_swallows[isnap, 0] = bswallow[ind_subj]
            num_repos_att[isnap, 0] = breposatt[ind_subj]
            num_mergers[isnap, 0] = bmerger[ind_subj]
            if has_fvisc:
                fvisc[isnap, 0] = bfvisc[ind_subj]
            
        if ind_cen is not None:
            print(f"BH log M_sg [cen] = {np.log10(bsgmass[ind_cen])}")
            print(f"BH accr [cen] = {np.log10(bmacc[ind_cen])}")
            if bsgas is not None:
                print(f"BH vdisp [cen] = {np.log10(bsgas[ind_cen])}")

            msg[isnap, 1] = np.log10(bsgmass[ind_cen])
            mdyn[isnap, 1] = np.log10(bptmass[ind_cen])
            macc[isnap, 1] = np.log10(bmacc[ind_cen])
            vbh[isnap, 1] = np.log10(np.linalg.norm(bvel[ind_cen, :]))
            vgas[isnap, 1] = np.log10(np.linalg.norm(bvgas[ind_cen, :]))
            vcgas[isnap, 1] = np.log10(np.linalg.norm(bvcgas[ind_cen, :]))
            cgas[isnap, 1] = np.log10(bcgas[ind_cen])
            cgasc[isnap, 1] = np.log10(bcgasc[ind_cen])
            if bsgas is not None:
                sgas[isnap, 1] = np.log10(bsgas[ind_cen])
            rhogas[isnap, 1] = np.log10(brhogas[ind_cen])


    hd.write_data(hdfile, 'SubgridMass', msg)
    hd.write_data(hdfile, 'DynamicalMass', mdyn)
    hd.write_data(hdfile, 'AccretionRate', macc)
    hd.write_data(hdfile, 'BHVelocity', vbh)
    hd.write_data(hdfile, 'GasVelocity', vgas)
    hd.write_data(hdfile, 'GasCircularVelocity', vcgas)
    hd.write_data(hdfile, 'GasSoundSpeed', cgas)
    hd.write_data(hdfile, 'GasSoundSpeedCorr', cgasc)
    hd.write_data(hdfile, 'GasVelocityDispersion', sgas)
    hd.write_data(hdfile, 'GasDensity', rhogas)
    hd.write_data(hdfile, 'Times', times)
    hd.write_data(hdfile, 'NumSteps', num_steps)
    hd.write_data(hdfile, 'NumRepos', num_repos)
    hd.write_data(hdfile, 'NumReposAttempt', num_repos_att)
    hd.write_data(hdfile, 'NumSwallows', num_swallows)
    hd.write_data(hdfile, 'NumMergers', num_mergers)
    hd.write_data(hdfile, 'ViscosityFactor', fvisc)
    
    hd.write_attribute(hdfile, 'Header', 'CodeBranch', code_branch)
    hd.write_attribute(hdfile, 'Header', 'CodeDate', code_date)
    hd.write_attribute(hdfile, 'Header', 'CodeRev', code_rev)
    hd.write_attribute(hdfile, 'Header', 'HasVelDisp', has_veldisp)
    hd.write_attribute(hdfile, 'Header', 'HasViscosity', has_fvisc)
    hd.write_attribute(hdfile, 'Header', 'SimName', sim_name)
    
else:

    msg = hd.read_data(hdfile, 'SubgridMass')
    mdyn = hd.read_data(hdfile, 'DynamicalMass')
    macc = hd.read_data(hdfile, 'AccretionRate')
    vbh = hd.read_data(hdfile, 'BHVelocity')
    vgas = hd.read_data(hdfile, 'GasVelocity')
    vcgas = hd.read_data(hdfile, 'GasCircularVelocity')
    cgas = hd.read_data(hdfile, 'GasSoundSpeed')
    cgasc = hd.read_data(hdfile, 'GasSoundSpeedCorr')
    sgas = hd.read_data(hdfile, 'GasVelocityDispersion')
    rhogas = hd.read_data(hdfile, 'GasDensity')
    times = hd.read_data(hdfile, 'Times')
    num_steps = hd.read_data(hdfile, 'NumSteps')
    num_repos = hd.read_data(hdfile, 'NumRepos')
    num_repos_att = hd.read_data(hdfile, 'NumReposAttempt')
    num_swallows = hd.read_data(hdfile, 'NumSwallows')
    num_mergers = hd.read_data(hdfile, 'NumMergers')
    fvisc = hd.read_data(hdfile, 'ViscosityFactor')
    
    code_branch = hd.read_attribute(hdfile, 'Header', 'CodeBranch')
    code_date = hd.read_attribute(hdfile, 'Header', 'CodeDate')
    code_rev = hd.read_attribute(hdfile, 'Header', 'CodeRev')
    has_veldisp = hd.read_attribute(hdfile, 'Header', 'HasVelDisp')
    has_fvisc = hd.read_attribute(hdfile, 'Header', 'HasViscosity')
    sim_name = hd.read_attribute(hdfile, 'Header', 'SimName')
    
fig = plt.figure(figsize=(args.width, 15))

ind_good_cen = np.nonzero((times >= 0) & (msg[:, 1] > 0))[0]
ind_good_sat = np.nonzero((times >= 0) & (msg[:, 0] > 0))[0]
ind_good = np.nonzero((times >= 0))[0]

# Find times of swallows
ind_sw_cen = np.nonzero((num_swallows[1:, 1] > num_swallows[:-1, 1]) &
                        (num_swallows[1:, 1] > 0))[0]
if len(ind_sw_cen) > 0:
    times_swallow_cen = times[ind_sw_cen]
else:
    times_swallow_cen = None

ind_repos_cen = np.nonzero(num_repos[:, 1] > 0)[0]
if len(ind_repos_cen) > 0:
    time_repos_cen = times[ind_repos_cen[0]]
else:
    time_repos_cen = None

ind_sw_sat = np.nonzero((num_swallows[1:, 0] > num_swallows[:-1, 0]) &
                        (num_swallows[1:, 0] > 0))[0]
ind_mg_sat = np.nonzero((num_mergers[1:, 0] > num_mergers[:-1, 0]) &
                        (num_mergers[1:, 0] > 0))[0]

print(f"Total of {len(ind_sw_sat)} gas swallows") 
if len(ind_sw_sat) > 0:
    times_swallow_sat = times[ind_sw_sat]
else:
    times_swallow_sat = None

print(f"Total of {len(ind_mg_sat)} BH mergers") 
if len(ind_mg_sat) > 0:
    times_merger_sat = times[ind_mg_sat]
else:
    times_merger_sat = None
    
ind_repos_sat = np.nonzero(num_repos[:, 0] > 0)[0]
if len(ind_repos_sat) > 0:
    time_repos_sat = times[ind_repos_sat[0]]
else:
    time_repos_sat = None

def draw_time_lines():
    """Plot lines of swallows and reposition times"""

    if time_repos_cen is not None:
        plt.plot([time_repos_cen, time_repos_cen], yr, color = 'red', alpha = 0.2, zorder=-100)
    if time_repos_sat is not None:
        plt.plot([time_repos_sat, time_repos_sat], yr, color = 'black', alpha = 0.2, zorder=-100)

    if times_merger_sat is not None:
        for itime in times_merger_sat:
            plt.plot([itime, itime], yr, color='royalblue', alpha=0.2, zorder=-100, linestyle=':')
        
    if times_swallow_cen is not None:
        for itime in times_swallow_cen:
            plt.plot([itime, itime], yr, color='red', alpha=0.2, zorder=-100, linestyle='--')
    if times_swallow_sat is not None:
        for itime in times_swallow_sat:
            plt.plot([itime, itime], yr, color='black', alpha=0.2, zorder=-100, linestyle='--')

trange = [np.min(times[ind_good]), np.max(times[ind_good])]
print("trange: ", trange)

ax1 = plt.subplot(6, 1, 1)
yr = [4.8, args.mmax]

plt.plot(times[ind_good_sat], msg[ind_good_sat, 0], color = 'black')
plt.plot(times[ind_good_cen], msg[ind_good_cen, 1], color = 'red')

plt.plot(times[ind_good_sat], mdyn[ind_good_sat, 0], color = 'black', linestyle=':')
plt.plot(times[ind_good_cen], mdyn[ind_good_cen, 1], color = 'red', linestyle=':')

draw_time_lines()

ax1.set_xlim(trange)
ax1.set_ylim(yr)
ax1.set_ylabel(r'BH masses (log $m$ [M$_\odot$])')
ax1.axes.get_xaxis().set_visible(False)

plt.plot(trange[0]+(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[1]-(yr[1]-yr[0])*np.array([0.06, 0.06]),
         linestyle = ':', color='black')
plt.plot(trange[0]+(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[1]-(yr[1]-yr[0])*np.array([0.13, 0.13]),
         linestyle = '-', color='black')

plt.text(trange[0]+(trange[1]-trange[0])*0.12, yr[1]-(yr[1]-yr[0])*0.03,
         f'Dynamical mass', va='top', ha='left', alpha=0.5)
plt.text(trange[0]+(trange[1]-trange[0])*0.12, yr[1]-(yr[1]-yr[0])*0.1,
         f'Subgrid Mass', va='top', ha='left', alpha=1.0)


plt.text(trange[1]-(trange[1]-trange[0])*0.02, yr[0]+(yr[1]-yr[0])*0.08,
         f"{code_branch} ({code_rev})", fontsize=8, va='bottom', ha='right')
plt.text(trange[1]-(trange[1]-trange[0])*0.02, yr[0]+(yr[1]-yr[0])*0.02,
         f"[{code_date}]", fontsize=8, va='bottom', ha='right')
#plt.text(trange[0]+(trange[1]-trange[0])*0.02, 6.5, code_rev, fontsize=8)

ax1b = ax1.twiny()
zreds = np.array([10.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.7, 0.5, 0.2, 0.1, 0.0])
zredtimes = np.array([cosmo.age(_z).value for _z in zreds])
ind_good = np.nonzero((zredtimes >= trange[0]) &
                      (zredtimes <= trange[1]))[0]

ax1b.set_xticks(zredtimes[ind_good])
zreds_good = zreds[ind_good]
ax1b.set_xticklabels([f'{_zt:.1f}' for _zt in zreds_good])
ax1b.set_xlabel('Redshift')
ax1b.set_xlim(trange)
ax1b.set_ylim(yr)

ax2 = plt.subplot(6, 1, 2)
yr = [-8.9, 0]
plt.plot(times[ind_good_sat], macc[ind_good_sat, 0], color = 'black')
plt.plot(times[ind_good_cen], macc[ind_good_cen, 1], color = 'red')
ax2.set_xlim(trange)
ax2.set_ylim(yr)
ax2.set_ylabel('Accretion rates' '\n' r'(log $\dot{m}$ [M$_\odot$/yr])')
ax2.axes.get_xaxis().set_visible(False)

plt.text(trange[0]+(trange[1]-trange[0])*0.03, yr[1]-(yr[1]-yr[0])*0.03,
         f'ID = {id_subj}', va='top', ha='left')

draw_time_lines()

if has_fvisc:
    print("FVISC")
    ax2b = ax2.twinx()
    plt.plot(times[ind_good_sat], np.log10(fvisc[ind_good_sat, 0]), color = 'purple', linestyle=':')
    ax2b.set_xlim(trange)
    ax2b.set_ylim([-4, 0.05])
    ax2b.set_ylabel('log Viscosity factor')
    ax2b.spines['right'].set_color('purple')
    ax2b.yaxis.label.set_color('purple')
    ax2b.tick_params(axis='y', colors='purple')

ax3 = plt.subplot(6, 1, 3)
plt.plot(times[ind_good_sat], vgas[ind_good_sat, 0], color='black')
plt.plot(times[ind_good_cen], vgas[ind_good_cen, 1], color='red')

if plot_bvel:
    plt.plot(times[ind_good_sat], vbh[ind_good_sat, 0], color='black', linestyle = '--')
    plt.plot(times[ind_good_cen], vbh[ind_good_cen, 1], color='red', linestyle = '--')
if plot_bcvel:
    plt.plot(times[ind_good_sat], vcgas[ind_good_sat, 0], color='cornflowerblue', linestyle = '--')
    plt.plot(times[ind_good_cen], vcgas[ind_good_cen, 1], color='red', linestyle = '--')
elif has_veldisp:
    plt.plot(times[ind_good_sat], sgas[ind_good_sat, 0], color='black', linestyle = '--')
    plt.plot(times[ind_good_cen], sgas[ind_good_cen, 1], color='red', linestyle = '--')


plt.plot(times[ind_good_sat], cgasc[ind_good_sat, 0], color='goldenrod', linestyle=':')
#plt.plot(times[ind_good_sat], cgasc[ind_good_sat, 0], color='seagreen', linestyle=':')
plt.plot(times[ind_good_cen], cgasc[ind_good_cen, 1], color='red', linestyle=':')

yr = [-0.9, 3.0]
ax3.set_xlim(trange)
ax3.set_ylim(yr)
#ax3.set_xlabel('Time [Myr]')
ax3.set_ylabel(r'Gas speeds (log $v$ [km/s])')

plt.plot(trange[1]-(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[0]+(yr[1]-yr[0])*np.array([0.06, 0.06]),
         linestyle = ':', color='goldenrod')
if has_veldisp or plot_bvel or plot_bcvel:
    plt.plot(trange[1]-(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[0]+(yr[1]-yr[0])*np.array([0.13, 0.13]),
         linestyle = '--', color='cornflowerblue')
plt.plot(trange[1]-(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[0]+(yr[1]-yr[0])*np.array([0.2, 0.2]),
         linestyle = '-', color='black')
plt.text(trange[1]-(trange[1]-trange[0])*0.12, yr[0]+(yr[1]-yr[0])*0.03,
         f'Sound speed', va='bottom', ha='right', alpha=1.0)
if plot_bvel:
    plt.text(trange[1]-(trange[1]-trange[0])*0.12, yr[0]+(yr[1]-yr[0])*0.1,
         f'Black hole velocity', va='bottom', ha='right', alpha=0.75)
if plot_bcvel:
    plt.text(trange[1]-(trange[1]-trange[0])*0.12, yr[0]+(yr[1]-yr[0])*0.1,
         f'Circular speed', va='bottom', ha='right', alpha=0.75)
elif has_veldisp:
    plt.text(trange[1]-(trange[1]-trange[0])*0.12, yr[0]+(yr[1]-yr[0])*0.1,
         f'Velocity dispersion', va='bottom', ha='right', alpha=0.75)

plt.text(trange[1]-(trange[1]-trange[0])*0.12, yr[0]+(yr[1]-yr[0])*0.17,
         f'Bulk speed', va='bottom', ha='right', alpha=1.0)

draw_time_lines()

plt.text(trange[0]+(trange[1]-trange[0])*0.03, yr[1] - (yr[1]-yr[0])*0.03,
         sim_name, va='top', ha='left', alpha=1.0)

ax3.axes.get_xaxis().set_visible(False)

ax4 = plt.subplot(6, 1, 4)
plt.plot(times[ind_good_sat], rhogas[ind_good_sat, 0], color='black')
plt.plot(times[ind_good_cen], rhogas[ind_good_cen, 1], color='red')

yr = [4.0, 9.0]
ax4.set_xlim(trange)
ax4.set_ylim(yr)
ax4.set_xlabel('Time [Myr]')
ax4.set_ylabel(r'Gas density (log)')

draw_time_lines()

ax4.axes.get_xaxis().set_visible(False)

ax5 = plt.subplot(6, 1, 5)
plt.plot(times[ind_good_sat], num_repos[ind_good_sat, 0], color='black')
plt.plot(times[ind_good_sat], num_repos_att[ind_good_sat, 0], color='black', linestyle=':')

yr = [0, max(np.max(num_repos), np.max(num_repos_att))]
ax5.set_xlim(trange)
ax5.set_ylim(yr)
ax5.set_xlabel('Time [Myr]')
ax5.set_ylabel(r'Reposition numbers')
print(f"Total of {num_repos[ind_good_sat[-1], 0]} repositionings") 

plt.plot(trange[1]-(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[0]+(yr[1]-yr[0])*np.array([0.06, 0.06]),
         linestyle = '-', color='black')
plt.plot(trange[1]-(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[0]+(yr[1]-yr[0])*np.array([0.13, 0.13]),
         linestyle = ':', color='black')

plt.text(trange[1]-(trange[1]-trange[0])*0.12, yr[0]+(yr[1]-yr[0])*0.03,
         f'Actual repositionings', va='bottom', ha='right', alpha=1)
plt.text(trange[1]-(trange[1]-trange[0])*0.12, yr[0]+(yr[1]-yr[0])*0.1,
         f'Attempted repositionings', va='bottom', ha='right', alpha=0.5)

draw_time_lines()
ax5.axes.get_xaxis().set_visible(False)

ax6 = plt.subplot(6, 1, 6)

start_ind_sat = np.clip(np.arange(nsnap)-5, ind_good_sat[0], ind_good_sat[-1])
end_ind_sat = np.clip(np.arange(nsnap)+5, ind_good_sat[0], ind_good_sat[-1])
delta_steps_sat = num_steps[end_ind_sat, 0] - num_steps[start_ind_sat, 0]
delta_repos_sat = num_repos[end_ind_sat, 0] - num_repos[start_ind_sat, 0]
delta_repos_att_sat = num_repos_att[end_ind_sat, 0] - num_repos_att[start_ind_sat, 0]
frac_repos_sat = delta_repos_sat/delta_steps_sat
frac_repos_att_sat = delta_repos_att_sat/delta_steps_sat

plt.plot(times[ind_good_sat], frac_repos_sat[ind_good_sat], color='black')
plt.plot(times[ind_good_sat], frac_repos_att_sat[ind_good_sat], color='black', linestyle=':')

yr = [0, min(max(np.max(frac_repos_sat[ind_good_sat]),
                 np.max(frac_repos_att_sat[ind_good_sat]))*1.01, 1)]
ax6.set_xlim(trange)
ax6.set_ylim(yr)
ax6.set_xlabel('Time [Gyr]')
ax6.set_ylabel(r'Reposition fractions')

plt.plot(trange[0]+(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[1]-(yr[1]-yr[0])*np.array([0.06, 0.06]),
         linestyle = '-', color='black')
plt.plot(trange[0]+(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[1]-(yr[1]-yr[0])*np.array([0.13, 0.13]),
         linestyle = ':', color='black')

plt.text(trange[0]+(trange[1]-trange[0])*0.12, yr[1]-(yr[1]-yr[0])*0.03,
         f'Actual repositionings', va='top', ha='left', alpha=1)
plt.text(trange[0]+(trange[1]-trange[0])*0.12, yr[1]-(yr[1]-yr[0])*0.1,
         f'Attempted repositionings', va='top', ha='left', alpha=0.5)


draw_time_lines()

plt.subplots_adjust(left=0.2/(args.width/9), right=max(0.93/(args.width/9), 0.93), bottom=0.05, top=0.95, hspace=0, wspace=0)

plt.show
plt.savefig(f'evolution_ID{args.sim}_BH{args.id}.pdf', dpi=200)








