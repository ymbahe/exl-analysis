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
parser.add_argument('index', type=int, help='BH index to analyse')
parser.add_argument('--id', type=int, help='Black hole ID to analyse')
parser.add_argument('--mmax', type=float, help='Maximum log mass to display',
                    default=7.0)
parser.add_argument('--width', type=float, help='Plot width in inch',
                    default=9.0)
parser.add_argument('--linewidth', type=float, help='Plot linewidth',
                    default=0.2)
parser.add_argument('--catname', default='black_hole_data.hdf5')

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

hdfile = wdir + args.catname

aexp = 1/(1+hd.read_data(hdfile, 'Redshifts'))

msg = np.log10(hd.read_data(hdfile, 'SubgridMasses') * 1e10)
mdyn = np.log10(hd.read_data(hdfile, 'DynamicalMasses') * 1e10)
macc = np.log10(hd.read_data(hdfile, 'AccretionRates') * 1.023045e-02)
vbh = np.log10(hd.read_data(hdfile, 'Velocities'))
vgas = hd.read_data(hdfile, 'GasRelativeVelocities')
vgas = np.log10(np.linalg.norm(vgas, axis=1))
vcgas = hd.read_data(hdfile, 'GasCircularVelocities')
vcgas = np.log10(np.linalg.norm(vcgas, axis=1))
cgas = np.log10(hd.read_data(hdfile, 'GasSoundSpeeds') / aexp)
sgas = hd.read_data(hdfile, 'GasVelocityDispersions')
if sgas is not None:
    sgas = np.log10(sgas)
rhogas = np.log10(hd.read_data(hdfile, 'GasDensities'))
num_steps = hd.read_data(hdfile, 'NumberOfTimeSteps')
num_repos = hd.read_data(hdfile, 'NumberOfRepositionings')
num_repos_att = hd.read_data(hdfile, 'NumberOfRepositionAttempts')
num_swallows = hd.read_data(hdfile, 'NumberOfSwallows')
num_mergers = hd.read_data(hdfile, 'NumberOfMergers')
fvisc = np.log10(hd.read_data(hdfile, 'ViscosityFactors'))

particle_ids = hd.read_data(hdfile, 'ParticleIDs')
times = hd.read_data(hdfile, 'Times')

nsnap = len(times)

snapfile = wdir + 'eagle_0018.hdf5'
code_branch = hd.read_attribute(snapfile, 'Code', 'Git Branch')
code_date = hd.read_attribute(snapfile, 'Code', 'Git Date')
code_rev = hd.read_attribute(snapfile, 'Code', 'Git Revision')
sim_name = hd.read_attribute(snapfile, 'Header', 'RunName')

if args.index < 0:
    args.index = np.nonzero(particle_ids == id_subj)[0]
    if len(args.index) != 1:
        print("Could not find subject BH...")
        set_trace()
    args.index = args.index[0]
        
else:
    args.id = particle_ids[args.index]

fig = plt.figure(figsize=(args.width, 15))
    
ind_good = np.nonzero((times >= 0) & (msg[args.index, :] != np.nan) & (msg[args.index, :] > 0))[0]
ind_good_all = np.nonzero((times >= 0))[0]


# Find times of swallows, mergers, and first repositioning
ind_sw = np.nonzero((num_swallows[args.index, 1:] > num_swallows[args.index, :-1]) &
                    (num_swallows[args.index, 1:] > 0))[0]
print(f"Total of {len(ind_sw)} gas swallows") 
if len(ind_sw) > 0:
    times_swallow = times[ind_sw]
else:
    times_swallow = None

ind_repos = np.nonzero(num_repos[args.index, :] > 0)[0]
if len(ind_repos) > 0:
    time_repos = times[ind_repos[0]]
else:
    time_repos = None

ind_mg = np.nonzero((num_mergers[args.index, 1:] > num_mergers[args.index, :-1]) &
                    (num_mergers[args.index, 1:] > 0))[0]

print(f"Total of {len(ind_mg)} BH mergers") 
if len(ind_mg) > 0:
    times_merger = times[ind_mg]
else:
    times_merger = None
    

def draw_time_lines():
    """Plot lines of swallows and reposition times"""

    if time_repos is not None:
        plt.plot([time_repos, time_repos], yr, color = 'black', alpha = 0.2, zorder=-100)

    if times_merger is not None:
        for itime in times_merger:
            plt.plot([itime, itime], yr, color='royalblue', alpha=0.2, zorder=-100, linestyle=':')

    if times_swallow is not None:
        for itime in times_swallow:
            plt.plot([itime, itime], yr, color='black', alpha=0.2, zorder=-100, linestyle='--')

# ================================================================================
# ==============  Ok, now draw the evolution lines ===============================
# ================================================================================
            
trange = [np.min(times[ind_good]), np.max(times[ind_good])]
print("trange: ", trange)

# -------------------- Masses -------------------

ax1 = plt.subplot(6, 1, 1)
yr = [4.8, args.mmax]

plt.plot(times[ind_good], msg[args.index, ind_good], color = 'black')
plt.plot(times[ind_good], mdyn[args.index, ind_good], color = 'black', linestyle=':')

draw_time_lines()

ax1.set_xlim(trange)
ax1.set_ylim(yr)
ax1.set_ylabel(r'BH masses (log $m$ [M$_\odot$])')
ax1.axes.get_xaxis().set_visible(False)

# Legend
plt.plot(trange[0]+(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[1]-(yr[1]-yr[0])*np.array([0.06, 0.06]),
         linestyle = ':', color='black')
plt.plot(trange[0]+(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[1]-(yr[1]-yr[0])*np.array([0.13, 0.13]),
         linestyle = '-', color='black')

plt.text(trange[0]+(trange[1]-trange[0])*0.12, yr[1]-(yr[1]-yr[0])*0.03,
         f'Dynamical mass', va='top', ha='left', alpha=0.5)
plt.text(trange[0]+(trange[1]-trange[0])*0.12, yr[1]-(yr[1]-yr[0])*0.1,
         f'Subgrid Mass', va='top', ha='left', alpha=1.0)

# Code info
plt.text(trange[1]-(trange[1]-trange[0])*0.02, yr[0]+(yr[1]-yr[0])*0.08,
         f"{code_branch} ({code_rev})", fontsize=8, va='bottom', ha='right')
plt.text(trange[1]-(trange[1]-trange[0])*0.02, yr[0]+(yr[1]-yr[0])*0.02,
         f"[{code_date}]", fontsize=8, va='bottom', ha='right')
#plt.text(trange[0]+(trange[1]-trange[0])*0.02, 6.5, code_rev, fontsize=8)

# Plot redshifts on top
ax1b = ax1.twiny()
zreds = np.array([10.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.7, 0.5, 0.2, 0.1, 0.0])
zredtimes = np.array([cosmo.age(_z).value for _z in zreds])
ind_good_z = np.nonzero((zredtimes >= trange[0]) &
                        (zredtimes <= trange[1]))[0]

ax1b.set_xticks(zredtimes[ind_good_z])
zreds_good = zreds[ind_good_z]
ax1b.set_xticklabels([f'{_zt:.1f}' for _zt in zreds_good])
ax1b.set_xlabel('Redshift')
ax1b.set_xlim(trange)
ax1b.set_ylim(yr)

# ----------------- Accretion rates --------------------------------

ax2 = plt.subplot(6, 1, 2)
yr = [-8.9, 0]

plt.plot(times[ind_good], macc[args.index, ind_good], color = 'black')
draw_time_lines()

ax2.set_xlim(trange)
ax2.set_ylim(yr)
ax2.set_ylabel('Accretion rates' '\n' r'(log $\dot{m}$ [M$_\odot$/yr])')
ax2.axes.get_xaxis().set_visible(False)

plt.text(trange[0]+(trange[1]-trange[0])*0.03, yr[1]-(yr[1]-yr[0])*0.03,
         f'BH {args.index} [ID = {args.id}]', va='top', ha='left')

if fvisc is not None:
    print("FVISC")
    ax2b = ax2.twinx()
    plt.plot(times[ind_good], fvisc[args.index, ind_good], color = 'purple', linestyle=':')
    ax2b.set_xlim(trange)
    ax2b.set_ylim([-4, 0.05])
    ax2b.set_ylabel('log Viscosity factor')
    ax2b.spines['right'].set_color('purple')
    ax2b.yaxis.label.set_color('purple')
    ax2b.tick_params(axis='y', colors='purple')

# ----------------- Speeds ---------------------------
    
ax3 = plt.subplot(6, 1, 3)
yr = [-0.9, 3.0]

plt.plot(times[ind_good], vgas[args.index, ind_good], color='black')
plt.plot(times[ind_good], cgas[args.index, ind_good], color='goldenrod', linestyle=':')

if plot_bvel:
    plt.plot(times[ind_good], vbh[args.index, ind_good], color='black', linestyle = '--')
if plot_bcvel:
    plt.plot(times[ind_good], vcgas[args.index, ind_good], color='cornflowerblue', linestyle = '--')
elif sgas is not None:
    plt.plot(times[ind_good], sgas[args.index, ind_good], color='cornflowerblue', linestyle = '--')

draw_time_lines()
    
ax3.set_xlim(trange)
ax3.set_ylim(yr)
ax3.set_ylabel(r'Gas speeds (log $v$ [km/s])')
ax3.axes.get_xaxis().set_visible(False)

# Legend
plt.plot(trange[1]-(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[0]+(yr[1]-yr[0])*np.array([0.06, 0.06]),
         linestyle = ':', color='goldenrod')
if sgas is not None  or plot_bvel or plot_bcvel:
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
elif sgas is not None:
    plt.text(trange[1]-(trange[1]-trange[0])*0.12, yr[0]+(yr[1]-yr[0])*0.1,
         f'Velocity dispersion', va='bottom', ha='right', alpha=0.75)

plt.text(trange[1]-(trange[1]-trange[0])*0.12, yr[0]+(yr[1]-yr[0])*0.17,
         f'Bulk speed', va='bottom', ha='right', alpha=1.0)

# Simulation name
plt.text(trange[0]+(trange[1]-trange[0])*0.03, yr[1] - (yr[1]-yr[0])*0.03,
         sim_name, va='top', ha='left', alpha=1.0)

# ----------------------------- Density ----------------------------------------

ax4 = plt.subplot(6, 1, 4)
yr = [4.0, 9.0]

plt.plot(times[ind_good], rhogas[args.index, ind_good], color='black')

draw_time_lines()

ax4.set_xlim(trange)
ax4.set_ylim(yr)
ax4.set_xlabel('Time [Myr]')
ax4.set_ylabel(r'Gas density (log)')
ax4.axes.get_xaxis().set_visible(False)

# ---------------------------- Repositioning -----------------------------------

ax5 = plt.subplot(6, 1, 5)
yr = [0, np.max(num_repos_att[args.index, ind_good])*1.05]

plt.plot(times[ind_good], num_repos[args.index, ind_good], color='black')
plt.plot(times[ind_good], num_repos_att[args.index, ind_good], color='black', linestyle=':')
print(f"Total of {num_repos[args.index, ind_good[-1]]} repositionings") 

draw_time_lines()

ax5.set_xlim(trange)
ax5.set_ylim(yr)
ax5.set_xlabel('Time [Myr]')
ax5.set_ylabel(r'Reposition numbers')
ax5.axes.get_xaxis().set_visible(False)

# Legend
plt.plot(trange[1]-(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[0]+(yr[1]-yr[0])*np.array([0.06, 0.06]),
         linestyle = '-', color='black')
plt.plot(trange[1]-(trange[1]-trange[0])*np.array([0.03, 0.1]), yr[0]+(yr[1]-yr[0])*np.array([0.13, 0.13]),
         linestyle = ':', color='black')

plt.text(trange[1]-(trange[1]-trange[0])*0.12, yr[0]+(yr[1]-yr[0])*0.03,
         f'Actual repositionings', va='bottom', ha='right', alpha=1)
plt.text(trange[1]-(trange[1]-trange[0])*0.12, yr[0]+(yr[1]-yr[0])*0.1,
         f'Attempted repositionings', va='bottom', ha='right', alpha=0.5)

# ------------------------- Repositioning fraction ------------------------

ax6 = plt.subplot(6, 1, 6)

# Here, we can't set the axis range yet, need to compute the plotting quantity first

# For each point in time, find the index range from (relative) -5 to +5
# (clipped to the very first and last output)
start_ind = np.clip(np.arange(nsnap)-5, ind_good[0], ind_good[-1])
end_ind = np.clip(np.arange(nsnap)+5, ind_good[0], ind_good[-1])

# Work out the fraction of repositions and reposition attempts in
# these ranges of outputs
delta_steps = num_steps[args.index, end_ind] - num_steps[args.index, start_ind]
delta_repos = num_repos[args.index, end_ind] - num_repos[args.index, start_ind]
delta_repos_att = num_repos_att[args.index, end_ind] - num_repos_att[args.index, start_ind]
frac_repos = delta_repos / delta_steps
frac_repos_att = delta_repos_att / delta_steps

# Plot the fractions as a function of time
plt.plot(times[ind_good], frac_repos[ind_good], color='black')
plt.plot(times[ind_good], frac_repos_att[ind_good], color='black', linestyle=':')

yr = [0, min(max(np.max(frac_repos[ind_good]),
                 np.max(frac_repos_att[ind_good]))*1.01, 1)]

draw_time_lines()

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


plt.subplots_adjust(left=0.2/(args.width/9),
                    right=max(0.93/(args.width/9), 0.93),
                    bottom=0.05, top=0.95, hspace=0, wspace=0)

plt.show
plt.savefig(f'evolution_ID{args.sim}_BH{args.index}.pdf', dpi=200)









