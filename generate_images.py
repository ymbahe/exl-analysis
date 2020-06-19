"""Set up the creation of a website displaying galaxy plots and images."""

import numpy as np
import hydrangea.hdf5 as hd
import os
import local
import xltools as xl
from astropy.cosmology import FlatLambdaCDM
import argparse

from pdb import set_trace

# ----- Define general settings -----

# Required fields from the unified BH catalogue
bh_props_list = ['ParticleIDs', 'Haloes', 'Halo_MStar', 'Halo_M200c',
                 'Halo_SFR', 'HaloTypes', 'Redshifts', 'SubgridMasses']

# List of image half-sizes, in pMpc
ap_list = [1.5, 0.3, 0.03, 0.003]

# List of image types to create (ptype, imtype, options)
image_list = [(0, 'temp', '--draw_hsml'),
              (0, 'sfr', '--draw_hsml --scale -4.0 -0.5 --absscale'),
              (4, 'gri', '--nobh --absscale')]

vr_plots = [('M200c', 'MStar', 'sSFR'),
            ('M200c', 'MBH', 'MStar'),
            ('M200c', 'Size', 'MStar')]

def main():

    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    parser.add_argument('sim',
                        help='Simulation index or name to analyse')
    parser.add_argument('--base_dir',
                        help=f'Simulation base directory, default: '
                             f'{local.BASE_DIR}',
                        default=local.BASE_DIR)
    parser.add_argument('--bh_mmax', type=float,
                        help='Maximum BH mass, for the scaling in the images '
                        'in log (M/M_Sun), default: 8.5.', default=8.5)
    parser.add_argument('--numpix', type=int,
                        help='Size of images in pixels, default: 1000',
                        default=1000)
    parser.add_argument('--plot_prefix', default='gallery/vr-plots',
                        help='Prefix for plot files, default: '
                             'gallery/vr-plots')
    parser.add_argument('--snapshots', nargs='+', type=int,
                        help='Snapshots for which to set up websites.')
    parser.add_argument('--snap_name', default='snapshot',
                        help='Snapshot prefix, default: "snapshot".')
    parser.add_argument('--mstar_min', type=float,
                        help='Minimum stellar mass of the host galaxy for '
                             'a black hole to be included, in M_Sun '
                             '(default: 3e10)', default=3e10)
    parser.add_argument('--m200_min', type=float,
                        help='Minimum M200c of the host galaxy for '
                             'a black hole to be included, in M_Sun '
                             '(default: 0, i.e. select on stellar mass only)',
                             default=0.0)
    parser.add_argument('--bh_data_file', default='black_hole_data.hdf5',
                        help='Name of the file containing the BH data, '
                             'default: "black_hole_data.hdf5"')
    parser.add_argument('--vr_prefix', default='vr',
                        help='Prefix of (combined) VR catalogue, default: vr.')
    parser.add_argument('--snap_frontpage', type=int,
                        help='Snapshot for images on frontpage (default: 36)',
                        default=36)
    parser.add_argument('--size_frontpage', type=float,
                        help='Size of image for front page, in pMpc '
                             '(default: 0.03)', default=0.03)

    args = parser.parse_args()

    if len(args.snapshots) == 0:
        print("No snapshots selected, aborting.")
    
    # Adjust selected front page size to closest match in list
    frontpage_sizeind = np.argmin(np.abs(np.array(ap_list)
                                         - args.size_frontpage))
    args.size_frontpage = ap_list[frontpage_sizeind]

    # Construct the full working directory of the simulation
    args.wdir = xl.get_sim_dir(args.base_dir, args.sim)

    # Look up the snapshot redshifts
    get_snapshot_redshifts(args)
    
    args.plotdata_file = f'{args.wdir}{args.plot_prefix}.hdf5'
    if os.path.isfile(args.plotdata_file):
        bh_list = hd.read_data(args.plotdata_file, 'BlackHoleBIDs')
        select_list = None

    else:    
        # Find BHs we are intereste in, load data
        select_list = [
            ["Halo_M200c", '>=', args.m200_min],
            ["Halo_MStar", '>=', args.mstar_min],
            ["Flag_MostMassiveInHalo", '==', 1],
            ["HaloTypes", '==', 10]
        ]
        bh_list = None
            
    bh_data, bh_sel = xl.lookup_bh_data(args.wdir + args.bh_data_file,
                                        bh_props_list, select_list)
    if bh_list is None:
        bh_list = bh_sel

    if len(bh_list) == 0:
        print("No black holes selected, aborting.")
        return

    
    # Generate the script to auto-generate all the required images
    generate_image_script(args, bh_list)

    # Generate the script to auto-generate all the tracks
    generate_track_script(args, bh_list)

    generate_website(args, bh_data, bh_list)


def get_snapshot_redshifts(args):
    """Look up the redshifts of all snapshots."""

    args.snaps_zred = np.zeros(max(args.snapshots)+1) - 1
    for isnap in args.snapshots:
        snap_file = f'{args.wdir}{args.snap_name}_{isnap:04d}.hdf5'
        args.snaps_zred[isnap] = hd.read_attribute(snap_file, 'Header',
                                                   'Redshift')[0]


def generate_image_script(args, bh_list):
    """Generate a script for auto-generation of images."""

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    
    with open(f'{args.wdir}/image_script.sh', 'w') as writer:
        writer.write('#!/bin/tcsh\n\n')

        for isnap in args.snapshots:
            for ibh in bh_list:
                for iap in ap_list:
                    for iim in image_list:
                        coda = get_coda(iap)
                        if iap < 0.01 and iim[1] == 'gri':
                            specific_options = '--scale 28.0 13.0'
                        elif iap > 1.0 and 'nobh' not in iim[2]:
                            specific_options = '--nobh'
                        else:
                            specific_options = ''

                        writer.write(f'python {curr_dir}/eagle_image.py '
                                    f'{args.wdir} {isnap} '
                                    f'--ptype {iim[0]} --imsize {iap} '
                                    f'--cambhbid {ibh} --outdir gallery '
                                    f'--bh_mmax {args.bh_mmax} '
                                    f'--imtype {iim[1]} --coda {get_coda(iap)} '
                                    f'--numpix {args.numpix} --nosave '
                                    f'{specific_options} '
                                    f'{iim[2]}\n')


def generate_track_script(args, bh_list):
    """Generate a script for auto-generation of growth tracks."""

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    
    with open(f'{args.wdir}/track_script.sh', 'w') as writer:
        writer.write('#!/bin/tcsh\n\n')

        for ibh in bh_list:
            writer.write(f'python {curr_dir}/bh_growth_comparison.py '
                         f'--sim {args.sim} --bh_bid {ibh} '
                         f'--basedir {args.base_dir} '
                         '\n')

                    
def generate_website(args, bh_data, bh_list):
    """Generate a simple website displaying all images."""
    
    if not os.path.isdir(f'{args.wdir}/gallery'):
        os.makedirs(f'{args.wdir}/gallery')
    with open(f'{args.wdir}/gallery/index.html', 'w') as writer:

        # Write HTML opening lines
        writer.write('<!DOCTYPE html>\n<html>\n<head>\n'
                     '  <link rel="stylesheet" href="style.css">\n'
                     '</head><body>')

        # Actual BH-specific sites are written one per snapshot.
        for isnap in args.snapshots:

            if isnap == args.snap_frontpage:
                zind_bh_frontpage = np.argmin(
                    np.abs(bh_data['Redshifts'] - args.snaps_zred[isnap]))
                print(f"BH output index for frontpage is {zind_bh_frontpage}.")

            
            # Need to explicitly connect the BHs to the VR catalogue
            # at the current snapshot, not necessarily in BH catalogue itself.
            # Returns None if VR catalogue does not exist
            vr_data = xl.connect_to_galaxies(
                bh_data['ParticleIDs'][bh_list],
                f'{args.wdir}{args.vr_prefix}_{isnap:04d}')
               
            
            # Now process each BH in turn...
            for iibh, ibh in enumerate(bh_list):

                if isnap == args.snap_frontpage:

                    # Write the entry image for the front page:
                    m_bh = (bh_data['SubgridMasses'][ibh, zind_bh_frontpage]
                            * 1e10)
                    if 'Halo_M200c' in bh_data:
                        m200_bh = bh_data['Halo_M200c'][ibh]
                    else:
                        m200_bh = None
                    print(f"iibh = {iibh}, m_bh = {m_bh} M_Sun")
                    write_gallery_image(
                        writer, ibh, isnap=isnap, size=args.size_frontpage,
                        m_subgrid=m_bh, m200=m200_bh)

                # Exctact the halo data we need. Note that vr_data only
                # contains info for our selected BHs, need to index with iibh
                if vr_data is not None:
                    vr_bhdata = {}
                    vr_bhdata['log_MStar']  = np.log10(vr_data["MStar"][iibh])
                    vr_bhdata['log_M200c'] = np.log10(vr_data["M200c"][iibh])
                    vr_bhdata['SFR'] = vr_data["SFR"][iibh]
                    vr_bhdata['log_sSFR']  = (np.log10(vr_bhdata['SFR'])
                                              - vr_bhdata['log_MStar'])
                else:
                    vr_bhdata = None
                    
                # Now write actual BH site
                with open(f'{args.wdir}/gallery/'
                          f'index_bh-{ibh}_snap-{isnap}.html', 'w'
                          ) as writer_bh:

                    write_bh_header(writer_bh, ibh, isnap, bh_data, vr_bhdata,
                                    args)
                    write_bh_plots(writer_bh, ibh, isnap, bh_list,
                                   args.plotdata_file)
                    write_bh_images(writer_bh, ibh, isnap)
                    
                    # Add html closing lines to BH site
                    writer_bh.write('</body>\n</html>\n')
                         
        # Add html closing lines to front page
        writer.write('</body>\n</html>\n')


def write_gallery_image(writer, ibh, isnap, size, m_subgrid, m200):
    """Write the front page gallery image for a specific BH."""

    curr_im = (f'image_pt0_temp_{get_coda(size)}_'
               f'BH-{ibh}_{isnap:04d}.png')
    m_exp = int(np.log10(m_subgrid))
    m_pre = m_subgrid / 10.0**m_exp

    if m200 is not None:
        m200_exp = int(np.log10(m200))
        m200_pre = m200 / 10.0**m200_exp
            
    writer.write(f'<div class="gallery">\n'
                 f'<div class="desc">Black hole {ibh} '
                 f'(m<sub>BH</sub> = {m_pre:.1f}&times;10<sup>{m_exp}</sup> '
                 f'M<sub>&#9737</sub>')
    if m200 is None:
        writer.write(f')</div>')
    else:
        writer.write(f', M<sub>200c</sub> = {m200_pre:.1f}&times;10<sup>{m200_exp}</sup> M<sub>&#9737</sub>)</div>')

    writer.write(f'<a href="index_bh-{ibh}_snap-{isnap}.html">\n'
                 f'  <img src="{curr_im}" alt="BH {ibh}" '
                 f'width=300>\n</a>\n'
                 f'</div>\n')

    
def write_bh_header(writer, ibh, isnap, bh_data, vr_bhdata, args):
    """Write the header for the BH site."""

    # Write HTML opening lines
    writer.write('<!DOCTYPE html>\n<html>\n<head>\n'
                 '  <link rel="stylesheet" href="style_bh.css">\n'
                 '</head><body>')

    writer.write('<a href="index.html">Back to front page</a>')

    writer.write(f'<h2>BH-ID {ibh} '
                 f'[ID={bh_data["ParticleIDs"][ibh]}]</h2>\n')

    writer.write(f'<h3> ')
    for ixsnap in args.snapshots:
        writer.write(f'<a href="index_bh-{ibh}_snap-{ixsnap}.html">'
                     f'Snap {ixsnap} (z = {args.snaps_zred[ixsnap]:.1f}) '
                     f'</a> &nbsp; &nbsp; &nbsp;')
    writer.write(f'</h3>')
    
    if vr_bhdata is not None:
        writer.write(f'<h3>log (M<sub>Star</sub> / M<sub>Sun</sub>) = '
                     f'{vr_bhdata["log_MStar"]:.3f} &nbsp &nbsp &nbsp '
                     f'log(M<sub>200c</sub> / M<sub>Sun</sub>) = '
                     f'{vr_bhdata["log_M200c"]:.3f} &nbsp &nbsp &nbsp '
                     f'SFR = {vr_bhdata["SFR"]:.3f} M<sub>Sun</sub> '
                     f'yr<sup>-1</sup> &nbsp &nbsp &nbsp '
                     f'log (sSFR / yr<sup>-1</sup>) = '
                     f'{vr_bhdata["log_sSFR"]:.3f} </h3>')


def write_bh_plots(writer, ibh, isnap, bh_list, plotdata_file):
    """Write all the plots for a BH."""

    # Black hole growth tracks
    bh_growth_im = f'bh_growth_tracks_BH-BID-{ibh}.png'
    writer.write(f'<a href="{bh_growth_im}">\n'
                 f'  <img src="{bh_growth_im}" alt="BH growth track" '
                 f'width=500>\n</a>\n')

    # Individual detailed growth chart for this BH
    bh_chart_im = f'bh_evolution_BH-BID-{ibh}.png'
    writer.write(f'<a href="{bh_chart_im}">\n'
                 f'  <img src="{bh_chart_im}" alt="BH evolution" '
                 f'height=450>\n</a>\n')

    # Stellar birth densities
    stellar_densities_im = f'stellar_birth_densities_BH-BID-{ibh}.png'
    writer.write(f'<a href="{stellar_densities_im}">\n'
                 f'  <img src="{stellar_densities_im}" '
                 f'alt="Stellar birth densities" '
                 f'width=500>\n</a>\n')

    writer.write(f'<br>')
    
    for iiplot, iplot in enumerate(vr_plots):
        write_vr_plot(writer, ibh, isnap, bh_list, plotdata_file,
                      iiplot, iplot)

    writer.write(f'<br>')
    

def write_vr_plot(writer, ibh, isnap, bh_list, plotdata_file, iiplot, iplot):
    """Write one single VR plot."""

    plotloc = (f'vr-plots_{iplot[0]}-{iplot[1]}-{iplot[2]}_'
               f'BH-{ibh}_snap-{isnap}.png')
        
    writer.write(f'<img src="{plotloc}" alt="{iplot[0]}-{iplot[1]}-'
                 f'{iplot[2]}" ')
    if os.path.isfile(plotdata_file):
        writer.write(f'usemap="#map-{iplot[0]}-{iplot[1]}" ')
    writer.write(f'width=450>\n')

    # Interactive map is only possible if we have the data file
    if not os.path.isfile(plotdata_file):
        return

    writer.write('\n')
    writer.write(f'<map name="map-{iplot[0]}-{iplot[1]}">')
    writer.write('\n')
    
    imx = hd.read_data(plotdata_file, f'S{isnap}/{iplot[0]}-{iplot[1]}/xpt')
    imy = hd.read_data(plotdata_file, f'S{isnap}/{iplot[0]}-{iplot[1]}/ypt')    

    # Add a link for each individual BH to the map...
    for ixbh, xbh in enumerate(bh_list):

        x_this = int(imx[ixbh] * 450)
        y_this = int(imy[ixbh] * 450)
        rad = 5
                
        writer.write(f'<area shape="circle" coords="{x_this}, {y_this}, '
                     f'{rad}" alt="BH {xbh}" href="index_bh-{xbh}_'
                     f'snap-{isnap}.html">\n')

    writer.write(f'</map>\n')


def write_bh_images(writer, ibh, isnap):
    """Write all the images for a BH."""

    for iiap, iap in enumerate(ap_list):

        writer.write(f'<h4>{iap*1e3:.0f} kpc</h4>\n')
    
        for iiim, iim in enumerate(image_list):
            curr_im = (f'image_pt{iim[0]}_{iim[1]}_{get_coda(iap)}_'
                       f'BH-{ibh}_{isnap:04d}.png')

            writer.write(f'<a href="{curr_im}">\n'
                         f'  <img src="{curr_im}" alt="Image" '
                         f'width=500">\n</a>\n')

        writer.write('<br>\n\n')
    

def get_coda(iap):

    coda = f'{iap:.3f}'
    coda = coda.replace('.', 'p')
    return coda



if __name__ == '__main__':
    main()
