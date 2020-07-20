"""Flexible script to make images of EAGLE galaxies."""

print("Importing libraries...")
import argparse
import numpy as np
import os
import time

import sys
sys.path.insert(0, '/home/bahe/python/sim-utils')

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from pdb import set_trace

import hydrangea as hy
import hydrangea.hdf5 as hd
import image_routines as ir
import extratools as et
import swiftsimio as sw

import cmocean
import unyt

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.style.use('dark_background')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'][0] = 'palatino'
matplotlib.rc('text', usetex=True)


# ---------- Basic settings: simulation, galaxy, size, type -------------------

vr_halo = -1       # Center on specified VR halo (-1: disabled)
zsize = None       # z size in Mpc (+/- vs. mid-plane); same as x/y if None
save_maps = True   # Save HDF5 images? Can be overridden by args
show_bhs = True    # Show BHs as points? Can be overridden by args

# ---------- Projection settings ----------------------------------------------

projectionPlane = 'xy'  # Image projection plane (may be None)
camDir = [0, 0, -1]     # Camera direction vector (may be left at [0,0,0])
camAngle = [0, 0, 0]    # Camera angles (may left at [0,0,0])
rho = 0
edge_on = False

# ------------- Image settings ------------------------------------------------

scale_hsml_with_age = False
gas_tmax = None      # Only consider gas below a given T (can be None for all)
fixedSmoothingLength = 0   # Leave at 0 to compute adaptive smoothing lengths
desNGB = 58           # Number of neighbours for smoothing calculation    
tempRange = [4.0, 8.0] # Scaling range for (log) temperature
kernel_gamma = 1.936492
xBase = None


# ------------------------------------------------------------------------
# -------- Should not have to adjust anything below this line ------------
# ------------------------------------------------------------------------

print("Parsing input arguments...")
parser = argparse.ArgumentParser(description="Parse input parameters.")
parser.add_argument('rootdir', help='Simulation to image')
parser.add_argument('snapshot', type=int, nargs='+',
    help='Snapshot (range) to image')
parser.add_argument('--snap_name', default='snapshot',
                    help='Snapshot name prefix (default: "snapshot").')
parser.add_argument('--bh_data_file', default='black_hole_data.hdf5',
                    help='Name of the BH data file within the simulation '
                         'directory (default: "black_hole_data.hdf5").')
parser.add_argument('--ptype', type=int, help='Particle type to image',
    default=0)
parser.add_argument('--imsize', type=float,
    help='Half-sidelength of image (cMpc)', default=12.5)
parser.add_argument('--numpix', type=int,
    help='Number of pixels along one side', default=1000)
parser.add_argument('--campos', type=float, nargs='+',
    help='Centre position (cMpc)')
parser.add_argument('--campos_phys', type=float, nargs='+',
                    help='Centre position (pMpc)')
parser.add_argument('--cambh', type=int, help='BH index to centre on.')
parser.add_argument('--cambhid', type=int, help='ID of BH to center on')
parser.add_argument('--cambhbid', type=int, help='Black-ID of BH to center on')
parser.add_argument('--varpos', type=float, nargs='+',
    help='Moving frame coordinates (cMpc), [x0, y0, z0, dx, dy, dz]')
parser.add_argument('--outdir', help='Subdir to store images in',
                    default='full')
parser.add_argument('--scale', type=float, nargs='+', help='Scale range')
parser.add_argument('--absscale', action='store_true', default=False)
parser.add_argument('--propersize', action='store_true', default=False)
parser.add_argument('--imtype',
                    help='Type of image to make. Default: "temp"',
                    default='temp')
parser.add_argument('--nobh', action='store_true')
parser.add_argument('--bhind', action='store_true',
                    help='Show BH indices in image?')
parser.add_argument('--bh_mmax', type=float, help='Max scaling mass of BHs',
    default=5.5)
parser.add_argument('--bh_ftrange', type=float, nargs='+',
                    help='Display only BHs in given formation time frame.')
parser.add_argument('--bh_mrange', type=float, nargs='+',
                    help='Display only BHs in given log subgrid mass range.')
parser.add_argument('--bh_file', help='File containing list of BH indices to plot.')
parser.add_argument('--bh_quant', default='mass', help='Quantity to colour BHs by')
parser.add_argument('--noplot', action='store_true',
    help='Do not generate actual image.')
parser.add_argument('--nosave', action='store_true',
    help='Do not save image data as HDF5')
parser.add_argument('--inch', type=float, help='Image size in inch', default=4)
parser.add_argument('--draw_hsml', action='store_true',
                    help='Draw on Hsml of target BH')
parser.add_argument('--coda', help='String to include at end of filename',
                    default='')
parser.add_argument('--quantrange', type=float, nargs='+',
                    default=tempRange,
                    help='Scaling range for secondary quantity (e.g. log T)')
parser.add_argument('--no_double_image', action='store_true')
parser.add_argument('--replot_existing', action='store_true',
                    help='Re-plot an already existing image?')
args = parser.parse_args()

print("Checking argument consistency...")
if len(args.snapshot) > 3:
    print("Cannot submit more than three parts to snapshot range!")
    set_trace()

# Make sure simulation directory ends with forward slash
if not args.rootdir.endswith('/'):
    args.rootdir = args.rootdir + '/'

if args.scale is None:

    if args.imtype == 'gri':
        args.scale = [28.0, 17.0]
    elif not args.absscale:
        args.scale = [0.1, 99.99]
    elif args.ptype == 0:
        if args.imtype == 'sfr':
            args.scale = [0, 7]
        else:
            args.scale = [-3, 1]
    elif args.ptype == 1:
        args.scale = [-1, 3.5]
    elif args.ptype == 4:
        args.scale = [-1, 3.5]
    else:
        print("Inconsistent scale!")
        set_trace()

if args.bh_ftrange is None:
    args.bh_ftrange = [0, 1]

if args.bh_mrange is None:
    args.bh_mrange = [0.0, 100.0]
    
if args.imtype in ['temp', 'diffusion_parameters', 'sfr'] and args.ptype != 0:
    print(f"{args.imtype} is only available for gas.")
    set_trace()

if args.nosave:
    save_maps = False
    
if zsize is None:
    args.zsize = args.imsize
else:
    args.zsize = zsize

if args.cambhbid is not None:
    black_file = args.rootdir + args.bh_data_file
    args.cambhid = hd.read_data(black_file, 'ParticleIDs',
                                read_index=args.cambhbid)
    
if not save_maps and args.noplot:
    print("If we don't want any output, we can stop right here.")
    set_trace()

    
args.realimsize = args.imsize
args.realzsize = args.zsize

def image_snap(isnap):
    """Main function to image one specified snapshot."""
    
    print(f"Beginning imaging snapshot {isnap}...")
    stime = time.time()

    plotloc = (args.rootdir +
               f'{args.outdir}/image_pt{args.ptype}_{args.imtype}_'
               f'{args.coda}_')
    if args.cambhbid is not None:
        plotloc = plotloc + f'BH-{args.cambhbid}_'
    if not os.path.isdir(os.path.dirname(plotloc)):
        os.makedirs(os.path.dirname(plotloc))
    if not args.replot_existing and os.path.isfile(
            f'{plotloc}{isnap:04d}.png'):
        print(f"Image {plotloc}{isnap:04d}.png already exists, skipping.")
        return
        
    snapdir = args.rootdir + f'{args.snap_name}_{isnap:04d}.hdf5'
    
    mask = sw.mask(snapdir)
    
    # Read metadata
    print("Read metadata...")
    boxsize = max(mask.metadata.boxsize.value)
    
    ut = hd.read_attribute(snapdir, 'Units', 'Unit time in cgs (U_t)')[0]
    um = hd.read_attribute(snapdir, 'Units', 'Unit mass in cgs (U_M)')[0]
    time_int = hd.read_attribute(snapdir, 'Header', 'Time')[0]
    aexp_factor = hd.read_attribute(snapdir, 'Header', 'Scale-factor')[0]
    zred = hd.read_attribute(snapdir, 'Header', 'Redshift')[0]
    num_part = hd.read_attribute(snapdir, 'Header', 'NumPart_Total')

    time_gyr = time_int*ut/(3600*24*365.24*1e9)
    mdot_factor = (um/1.989e33) / (ut/(3600*24*365.24))

    # -----------------------
    # Snapshot-specific setup
    # -----------------------
  
    # Camera position
    camPos = None
    if vr_halo >= 0:
        print("Reading camera position from VR catalogue...")
        vr_file = args.rootdir + f'vr_{isnap:04d}.hdf5'
        camPos = hd.read_data(vr_file, 'MinimumPotential/Coordinates')

    elif args.varpos is not None:
        print("Find camera position...")
        if len(args.varpos) != 6:
            print("Need 6 arguments for moving box")
            set_trace()
        camPos = np.array([args.varpos[0]+args.varpos[3]*time_gyr,
                           args.varpos[1]+args.varpos[4]*time_gyr,
                           args.varpos[2]+args.varpos[5]*time_gyr])
        print(camPos)
        camPos *= aexp_factor

    elif args.campos is not None:
        camPos = np.array(args.campos) * aexp_factor
    elif args.campos_phys is not None:
        camPos = np.array(args.campos)
        
    elif args.cambhid is not None:
        all_bh_ids = hd.read_data(snapdir, 'PartType5/ParticleIDs')
        args.cambh = np.nonzero(all_bh_ids == args.cambhid)[0]
        if len(args.cambh) == 0:
            print(f"BH ID {args.cambhid} does not exist, skipping.")
            return
        
        if len(args.cambh) != 1:
            print(f"Could not unambiguously find BH ID '{args.cambhid}'!")
            set_trace()
        args.cambh = args.cambh[0]
        
    if args.cambh is not None and camPos is None:
        camPos = hd.read_data(snapdir, 'PartType5/Coordinates',
                              read_index=args.cambh) * aexp_factor
        args.hsml = hd.read_data(snapdir, 'PartType5/SmoothingLengths',
                                 read_index=args.cambh) * aexp_factor * kernel_gamma
        
    elif camPos is None:
        print("Setting camera position to box centre...")
        camPos = np.array([0.5, 0.5, 0.5]) * boxsize * aexp_factor

    # Image size conversion, if necessary
    if not args.propersize:
        args.imsize = args.realimsize * aexp_factor
        args.zsize = args.realzsize * aexp_factor
    else:
        args.imsize = args.realimsize
        args.zsize = args.realzsize
    
    max_sel = 1.2 * np.sqrt(3) * max(args.imsize, args.zsize)
    extent = np.array([-1, 1, -1, 1]) * args.imsize

    # Set up loading region
    if max_sel < boxsize * aexp_factor / 2:        

        load_region = np.array(
            [[camPos[0]-args.imsize*1.2, camPos[0]+args.imsize*1.2],
             [camPos[1]-args.imsize*1.2, camPos[1]+args.imsize*1.2],
             [camPos[2]-args.zsize*1.2, camPos[2]+args.zsize*1.2]])
        load_region = sw.cosmo_array(load_region / aexp_factor, "Mpc")
        mask.constrain_spatial(load_region)
        data = sw.load(snapdir, mask=mask)
    else:
        data = sw.load(snapdir)

    pt_names = ['gas', 'dark_matter', None, None, 'stars', 'black_holes']
    datapt = getattr(data, pt_names[args.ptype])    

    pos = datapt.coordinates.value * aexp_factor
    
    # Next bit does periodic wrapping
    def flip_dim(idim):
        full_box_phys = boxsize * aexp_factor
        half_box_phys = boxsize * aexp_factor / 2
        if camPos[idim] < min(max_sel, half_box_phys):
            ind_high = np.nonzero(pos[:, idim] > half_box_phys)[0]
            pos[ind_high, idim] -= full_box_phys
        elif camPos[idim] > max(full_box_phys - max_sel, half_box_phys):
            ind_low = np.nonzero(pos[:, idim] < half_box_phys)[0]
            pos[ind_low, idim] += full_box_phys

    for idim in range(3):
        print(f"Periodic wrapping in dimension {idim}...")
        flip_dim(idim)

    rad = np.linalg.norm(pos-camPos[None, :], axis=1)
    ind_sel = np.nonzero(rad < max_sel)[0]
    pos = pos[ind_sel, :]

    # Read BH properties, if they exist
    if num_part[5] > 0 and not args.nobh:
        bh_hsml = (hd.read_data(snapdir, 'PartType5/SmoothingLengths')
                   * aexp_factor)
        bh_pos = hd.read_data(snapdir, 'PartType5/Coordinates') * aexp_factor
        bh_mass = hd.read_data(snapdir, 'PartType5/SubgridMasses') * 1e10
        bh_maccr = (hd.read_data(snapdir, 'PartType5/AccretionRates')
                    * mdot_factor)
        bh_id = hd.read_data(snapdir, 'PartType5/ParticleIDs')
        bh_nseed = hd.read_data(snapdir, 'PartType5/CumulativeNumberOfSeeds')
        bh_ft = hd.read_data(snapdir, 'PartType5/FormationScaleFactors')
        print(f"Max BH mass: {np.log10(np.max(bh_mass))}")

    else:
        bh_mass = None  # Dummy value

    # Read the appropriate 'mass' quantity
    if args.ptype == 0 and args.imtype == 'sfr':
        mass = datapt.star_formation_rates[ind_sel]
        mass.convert_to_units(unyt.Msun / unyt.yr)
        mass = np.clip(mass.value, 0, None) # Don't care about last SFR aExp
    else:
        mass = datapt.masses[ind_sel]
        mass.convert_to_units(unyt.Msun)
        mass = mass.value
        
    if args.ptype == 0:
        hsml = (datapt.smoothing_lengths.value[ind_sel] * aexp_factor
                * kernel_gamma)
    elif fixedSmoothingLength > 0:
        hsml = np.zeros(mass.shape[0], dtype=np.float32) + fixedSmoothingLength
    else:
        hsml = None
    
    if args.imtype == 'temp':
        quant = datapt.temperatures.value[ind_sel]
    elif args.imtype == 'diffusion_parameters':
        quant = datapt.diffusion_parameters.value[ind_sel]
    else:
        quant = mass

    # Read quantities for gri computation if necessary
    if args.ptype == 4 and args.imtype == 'gri':
        m_init = datapt.initial_masses.value[ind_sel] * 1e10  # in M_sun
        z_star = datapt.metal_mass_fractions.value[ind_sel]
        sft = datapt.birth_scale_factors.value[ind_sel]

        age_star = (time_gyr - hy.aexp_to_time(sft, time_type='age')) * 1e9
        age_star = np.clip(age_star, 0, None)  # Avoid rounding issues

        lum_g = et.imaging.stellar_luminosity(m_init, z_star, age_star, 'g')
        lum_r = et.imaging.stellar_luminosity(m_init, z_star, age_star, 'r')
        lum_i = et.imaging.stellar_luminosity(m_init, z_star, age_star, 'i')

    # ---------------------
    # Generate actual image
    # ---------------------

    xBase = np.zeros(3, dtype=np.float32)
    yBase = np.copy(xBase)
    zBase = np.copy(xBase)

    if args.imtype == 'gri':       
        image_weight_all_g, image_quant, hsml = ir.make_sph_image_new_3d(
            pos, lum_g, lum_g, hsml, DesNgb=desNGB, imsize=args.numpix, zpix=1, 
            boxsize=args.imsize, CamPos=camPos, CamDir=camDir,
            ProjectionPlane=projectionPlane, verbose=True,
            CamAngle=[0,0,rho], rollMode=0, edge_on=edge_on, 
            treeAllocFac=10, xBase=xBase, yBase=yBase, zBase=zBase,
            return_hsml=True)
        image_weight_all_r, image_quant = ir.make_sph_image_new_3d(
            pos, lum_r, lum_r, hsml, DesNgb=desNGB, imsize=args.numpix, zpix=1, 
            boxsize=args.imsize, CamPos=camPos, CamDir=camDir,
            ProjectionPlane=projectionPlane, verbose=True,
            CamAngle=[0,0,rho], rollMode=0, edge_on=edge_on, 
            treeAllocFac=10, xBase=xBase, yBase=yBase, zBase=zBase,
            return_hsml=False)
        image_weight_all_i, image_quant = ir.make_sph_image_new_3d(
            pos, lum_i, lum_i, hsml, DesNgb=desNGB, imsize=args.numpix, zpix=1, 
            boxsize=args.imsize, CamPos=camPos, CamDir=camDir,
            ProjectionPlane=projectionPlane, verbose=True,
            CamAngle=[0,0,rho], rollMode=0, edge_on=edge_on, 
            treeAllocFac=10, xBase=xBase, yBase=yBase, zBase=zBase,
            return_hsml=False)

        map_maas_g = -5/2*np.log10(
            image_weight_all_g[:, :, 1]+1e-15)+5*np.log10(180*3600/np.pi) + 25
        map_maas_r = -5/2*np.log10(
            image_weight_all_r[:, :, 1]+1e-15)+5*np.log10(180*3600/np.pi) + 25
        map_maas_i = -5/2*np.log10(
            image_weight_all_i[:, :, 1]+1e-15)+5*np.log10(180*3600/np.pi) + 25

    else:
        image_weight_all, image_quant = ir.make_sph_image_new_3d(
            pos, mass, quant, hsml, DesNgb=desNGB, imsize=args.numpix, zpix=1, 
            boxsize=args.imsize, CamPos=camPos, CamDir=camDir,
            ProjectionPlane=projectionPlane, verbose=True,
            CamAngle=[0,0,rho], rollMode=0, edge_on=edge_on, 
            treeAllocFac=10, xBase=xBase, yBase=yBase, zBase=zBase,
            zrange=[-args.zsize, args.zsize])

        # Extract surface density in M_sun [/yr] / kpc^2
        sigma = np.log10(image_weight_all[:, :, 1] + 1e-15) - 6
        if args.ptype == 0 and args.imtype in ['temp']:
            tmap = np.log10(image_quant[:, :, 1])
        elif args.ptype == 0 and args.imtype in ['diffusion_parameters']:
            tmap = image_quant[:, :, 1]
            
    # -----------------
    # Save image data
    # -----------------

    if save_maps:
        maploc = plotloc + f'{isnap:04d}.hdf5'

        if args.imtype == 'gri' and args.ptype == 4:
            hd.write_data(maploc, 'g_maas', map_maas_g, new=True)
            hd.write_data(maploc, 'r_maas', map_maas_r)
            hd.write_data(maploc, 'i_maas', map_maas_i)
        else:
            hd.write_data(maploc, 'Sigma', sigma, new=True)
            if args.ptype == 0 and args.imtype == 'temp':
                hd.write_data(maploc, 'Temperature', tmap)
            elif args.ptype == 0 and args.imtype == 'diffusion_parameters':
                hd.write_data(maploc, 'DiffusionParameters', tmap)
                
        hd.write_data(maploc, 'Extent', extent) 
        hd.write_attribute(maploc, 'Header', 'CamPos', camPos)
        hd.write_attribute(maploc, 'Header', 'ImSize', args.imsize)
        hd.write_attribute(maploc, 'Header', 'NumPix', args.numpix)
        hd.write_attribute(maploc, 'Header', 'Redshift', 1/aexp_factor - 1)
        hd.write_attribute(maploc, 'Header', 'AExp', aexp_factor)
        hd.write_attribute(maploc, 'Header', 'Time', time_gyr)
        
        if bh_mass is not None:
            hd.write_data(maploc, 'BH_pos', bh_pos-camPos[None, :],
                comment='Relative position of BHs')
            hd.write_data(maploc, 'BH_mass', bh_mass,
                comment='Subgrid mass of BHs')
            hd.write_data(maploc, 'BH_maccr', bh_maccr,
                comment='Instantaneous BH accretion rate in M_sun/yr')
            hd.write_data(maploc, 'BH_id', bh_id,
                comment='Particle IDs of BHs')
            hd.write_data(maploc, 'BH_nseed', bh_nseed,
                comment='Number of seeds in each BH')
            hd.write_data(maploc, 'BH_aexp', bh_ft,
                comment='Formation scale factor of each BH')
            
    # -------------
    # Plot image...
    # -------------

    if not args.noplot:

        print("Obtained image, plotting...")
        fig = plt.figure(figsize = (args.inch, args.inch))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        plt.sca(ax)

        # Option I: we have really few particles. Plot them individually:
        if pos.shape[0] < 32:
            plt.scatter(pos[:, 0]-camPos[0], pos[:, 1]-camPos[1], color='white')

        else:
            # Main plotting regime

            # Case A: gri image -- very different from rest
            if args.ptype == 4 and args.imtype == 'gri':

                vmin = -args.scale[0] + np.array([-0.5, -0.25, 0.0])
                vmax = -args.scale[1] + np.array([-0.5, -0.25, 0.0])
                
                clmap_rgb = np.zeros((args.numpix, args.numpix, 3))
                clmap_rgb[:, :, 2] = np.clip(
                    ((-map_maas_g)-vmin[0])/((vmax[0]-vmin[0])), 0, 1)
                clmap_rgb[:, :, 1] = np.clip(
                    ((-map_maas_r)-vmin[1])/((vmax[1]-vmin[1])), 0, 1)
                clmap_rgb[:, :, 0] = np.clip(
                    ((-map_maas_i)-vmin[2])/((vmax[2]-vmin[2])), 0, 1)
                
                im = plt.imshow(clmap_rgb, extent=extent, aspect='equal',
                    interpolation='nearest', origin='lower', alpha=1.0)
            
            else:

                # Establish image scaling
                if not args.absscale:
                    ind_use = np.nonzero(sigma > 1e-15)
                    vrange = np.percentile(sigma[ind_use], args.scale)
                else:
                    vrange = args.scale            
                print(f'Sigma range: {vrange[0]:.4f} -- {vrange[1]:.4f}')
                
                # Case B: temperature/diffusion parameter image
                if (args.ptype == 0
                    and args.imtype in ['temp', 'diffusion_parameters']
                    and not args.no_double_image):
                    if args.imtype == 'temp':
                        cmap = None
                    elif args.imtype == 'diffusion_parameters':
                        cmap = cmocean.cm.haline
                    clmap_rgb = ir.make_double_image(sigma, tmap, percSigma=vrange,
                        absSigma=True, rangeQuant=args.quantrange,cmap=cmap)

                    im = plt.imshow(clmap_rgb, extent=extent,
                                    aspect='equal', interpolation='nearest',
                                    origin='lower', alpha=1.0)

                else:
                    # Standard sigma images
                    if args.ptype == 0:
                        if args.imtype == 'hi':
                            cmap = plt.cm.bone
                        elif args.imtype == 'sfr':
                            cmap = plt.cm.magma
                        elif args.imtype == 'diffusion_parameters':
                            cmap = cmocean.cm.haline
                        else:
                            cmap = plt.cm.inferno

                    elif args.ptype == 1:
                        cmap = plt.cm.Greys_r
                    elif args.ptype == 4:
                        cmap = plt.cm.bone

                    if args.no_double_image:
                        plotquant = tmap
                        vmin, vmax = args.quantrange[0], args.quantrange[1]
                    else:
                        plotquant = sigma
                        vmin, vmax = vrange[0], vrange[1]
                        
                    im = plt.imshow(plotquant, cmap=cmap,
                        extent=extent, vmin=vmin, vmax=vmax,
                        origin='lower', interpolation='nearest', aspect='equal')

        # Plot BHs if desired:
        if show_bhs and bh_mass is not None:

            if args.bh_file is not None:
                bh_inds = np.loadtxt(args.bh_file, dtype=int)
            else:
                bh_inds = np.arange(bh_pos.shape[0])
            
            ind_show = np.nonzero(
                (np.abs(bh_pos[bh_inds, 0]-camPos[0]) < args.imsize) &
                (np.abs(bh_pos[bh_inds, 1]-camPos[1]) < args.imsize) &
                (np.abs(bh_pos[bh_inds, 2]-camPos[2]) < args.zsize) &
                (bh_ft[bh_inds] >= args.bh_ftrange[0]) &
                (bh_ft[bh_inds] <= args.bh_ftrange[1]) &
                (bh_mass[bh_inds] >= 10.0**args.bh_mrange[0]) &
                (bh_mass[bh_inds] <= 10.0**args.bh_mrange[1]))[0]
            ind_show = bh_inds[ind_show]
            
            if args.bh_quant == 'mass':
                sorter = np.argsort(bh_mass[ind_show])
                sc = plt.scatter(bh_pos[ind_show[sorter], 0]-camPos[0],
                             bh_pos[ind_show[sorter], 1]-camPos[1], marker='o',
                             c=np.log10(bh_mass[ind_show[sorter]]),
                             edgecolor='grey', vmin=5.0, vmax=args.bh_mmax,
                             s=5.0, linewidth=0.2)
                bticks = np.linspace(5.0, args.bh_mmax, num=6, endpoint=True)
                blabel = r'log$_{10}$ ($m_\mathrm{BH}$ [M$_\odot$])'
                
            elif args.bh_quant == 'formation':
                sorter = np.argsort(bh_ft[ind_show])
                sc = plt.scatter(bh_pos[ind_show[sorter], 0]-camPos[0],
                             bh_pos[ind_show[sorter], 1]-camPos[1], marker='o',
                             c=bh_ft[ind_show[sorter]],
                             edgecolor='grey', vmin=0, vmax=1.0,
                             s=5.0, linewidth=0.2)
                bticks = np.linspace(0.0, 1.0, num=6, endpoint=True)
                blabel = 'Formation scale factor'
                
            if args.bhind:
                for ibh in ind_show[sorter]:
                    c = plt.cm.viridis(
                        (np.log10(bh_mass[ibh])-5.0)/(args.bh_mmax-5.0))
                    plt.text(bh_pos[ibh, 0] - camPos[0] + args.imsize/200,
                             bh_pos[ibh, 1] - camPos[1] + args.imsize/200,
                             f'{ibh}', color=c, fontsize=4,
                             va='bottom', ha='left')

            if args.draw_hsml:
                phi = np.arange(0, 2.01*np.pi, 0.01)
                plt.plot(args.hsml * np.cos(phi),
                         args.hsml * np.sin(phi),
                         color='white', linestyle=':', linewidth=0.5)
                
                    
            # Add colour bar for BH masses                        
            if args.imtype != 'sfr':
                ax2 = fig.add_axes([0.6, 0.07, 0.35, 0.02])
                ax2.set_xticks([])
                ax2.set_yticks([])
                cbar = plt.colorbar(sc, cax=ax2, orientation='horizontal',
                                    ticks=bticks)
                cbar.ax.tick_params(labelsize=8)
                fig.text(0.775, 0.1, blabel,
                         rotation=0.0, va='bottom', ha='center', color='white',
                         fontsize=8)

        # Done with main image, some embellishments...
        plt.sca(ax)
        plt.text(-0.045/0.05*args.imsize, 0.045/0.05*args.imsize, 'z = {:.3f}' 
                 .format(1/aexp_factor - 1), va='center', ha='left', 
                 color='white')
        plt.text(-0.045/0.05*args.imsize, 0.041/0.05*args.imsize,
                 't = {:.3f} Gyr' .format(time_gyr), va='center', ha='left',
                 color='white', fontsize=8)

        plot_bar()

        # Plot colorbar for SFR if appropriate
        if args.ptype == 0 and args.imtype == 'sfr':
            ax2 = fig.add_axes([0.6, 0.07, 0.35, 0.02])
            ax2.set_xticks([])
            ax2.set_yticks([])

            scc = plt.scatter([-1e10], [-1e10], c=[0], cmap=plt.cm.magma,
                              vmin=vrange[0], vmax=vrange[1])
            cbar = plt.colorbar(scc, cax=ax2, orientation='horizontal',
                                ticks=np.linspace(np.floor(vrange[0]),
                                                  np.ceil(vrange[1]), 5,
                                                  endpoint=True))
            cbar.ax.tick_params(labelsize=8)
            fig.text(0.775, 0.1,
                     r'log$_{10}$ ($\Sigma_\mathrm{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$])',
                     rotation=0.0, va='bottom', ha='center', color='white',
                     fontsize=8)

        ax.set_xlabel(r'$\Delta x$ [pMpc]')
        ax.set_ylabel(r'$\Delta y$ [pMpc]')
            
        ax.set_xlim((-args.imsize, args.imsize))
        ax.set_ylim((-args.imsize, args.imsize))

        plt.savefig(plotloc + str(isnap).zfill(4) + '.png',
            dpi=args.numpix/args.inch)
        plt.close()

    print(f"Finished snapshot {isnap} in {(time.time() - stime):.2f} sec.")
    print(f"Image saved in {plotloc}{isnap:04d}.png")


def plot_bar():

    lengths = [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01,
               0.005, 0.002, 0.001]

    for ilen in lengths:
        frac_len = ilen / args.imsize
        if frac_len > 0.8:
            continue

        if ilen >= 1:
            label = '{:.0f} pMpc' .format(ilen)
        else:
            label = '{:.0f} pkpc' .format(ilen*1000)

        lower_pt = -args.imsize*0.045/0.05
        upper_pt = lower_pt + ilen
        cen_pt = lower_pt + ilen/2

        plt.plot((lower_pt, upper_pt),
            (-0.045/0.05*args.imsize, -0.045/0.05*args.imsize),
            color='white', linewidth=2)
        plt.text(cen_pt, -0.0435/0.05*args.imsize, label,
                 va='bottom', ha='center', color='white', fontsize=9)
        break


if __name__ == "__main__":

    if len(args.snapshot) == 1:
        image_snap(args.snapshot[0])
    elif len(args.snapshot) == 2:
        print(f"Will image snapshots {args.snapshot[0]} to "
              f"{args.snapshot[1]}...")
        for isnap in range(args.snapshot[0], args.snapshot[1]+1):
            image_snap(isnap)
    else:
        for insap in np.arange(args.snapshot[0], args.snapshot[1]+1, args.snapshot[2]):
            image_snap(isnap)

    print("Done with images!")

