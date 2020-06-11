"""Set up the creation of a website displaying galaxy plots and images."""

import numpy as np
import hydrangea.hdf5 as hd
import os
import local
import xltools as xl

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

# List of snapshots for which to create images and websites
snap_list = [24, 36]

# Need to improve the following bit...
snap_zred = np.zeros(37)
snap_zred[24] = 1.08
snap_zred[36] = 0.00

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
                        'in log (M/M_Sun)', default=8.5)
    parser.add_argument('--numpix', type=int,
                        help='Size of images in pixels, default: 1000',
                        default: 1000)
    parser.add_argument('--mstar_min', type=float,
                        help='Minimum stellar mass of the host galaxy for '
                             'a black hole to be included, in M_Sun '
                             '(default: 3e10)', default=3e10)
    parser.add_argument('--m200_min', type=float,
                        help='Minimum M200c of the host galaxy for '
                             'a black hole to be included, in M_Sun '
                             '(default: 0, i.e. select on stellar mass only)',
                             default=0.0)
    parser.add_argument('--bh_data_file',
                        help='Name of the file containing the BH data, '
                             'default: "black_hole_data.hdf5"')
    parser.add_argument('--snap_frontpage', type=int,
                        help='Snapshot for images on frontpage (default: 36)',
                        default=36)
    parser.add_argument('--size_frontpage', type=float,
                        help='Size of image for front page, in pMpc '
                             '(default: 0.03)', default=0.03)

    args = parser.parse_args()

    # Adjust selected front page size to closest match in list
    frontpage_sizeind = np.argmin(np.abs(np.array(ap_list)
                                         - args.size_frontpage))
    args.size_frontpage = ap_list[frontpage_sizeind]

    # Construct the full working directory of the simulation
    args.wdir = xl.get_sim_dir(args.base_dir, args.sim)
    
    # Find BHs we are intereste in, load data
    bh_data, bh_list = lookup_bh_data(args)

    # Generate the script to auto-generate all the required images
    generate_image_script(args, bh_list)

    # Generate the script to auto-generate all the tracks
    generate_track_script(args, bh_list)

    generate_website(args, bh_data)

    
def lookup_bh_data(args):
    """Load info from BH file into arrays and find target BHs."""

    bh_file = f'{args.wdir}{args.bh_data_file}'
    bh_data = {}

    # Load the required data from the evolution tables
    for idata in bh_props_list:
        data = hd.read_data(bh_file, idata)
        bh_data[idata] = data

    # Also, find list of BHs we are interested in (based on z=0 props)
    bh_list = np.nonzero((bh_maxflag == 1) &
                         (bh_data['Halo_MStar'] >= args.mstar_min) &
                         (bh_data['Halo_M200c'] >= args.m200_min) &
                         (bh_data['HaloTypes'] == 10))[0]

    # Sort BHs by index
    bh_list = np.sort(bh_list)
    print(f"There are {len(bh_list)} BHs in selection list.")

    return bh_data, bh_list


def generate_image_script(args, bh_list):
    """Generate a script for auto-generation of images."""

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    
    with open(f'{args.wdir}/image_script.sh', 'w') as writer:
        writer.write('#!/bin/tcsh\n\n')

        for isnap in snap_list:
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
        for isnap in snap_list:

            # Need to explicitly connect the BHs to the VR catalogue
            # at the current snapshot, not necessarily in BH catalogue itself.
            # Returns None if VR catalogue does not exist
            vr_data = xl.connect_to_galaxies(bh_data['ParticleIDs'][bh_list],
                                             args.wdir, isnap)
            
            # Now process each BH in turn...
            for iibh, ibh in enumerate(bh_list):

                if isnap == args.snap_frontpage:
                    # Write the entry image for the front page:
                    zind_bh_frontpage = np.argmin(
                        np.abs(bh_data['Redshifts'] - snap_zred[isnap]))
                    m_bh = bh_data['SubgridMasses'][ibh, zind_bh_frontpage]

                    write_gallery_image(
                        writer, ibh, isnap=isnap, size=args.size_frontpage,
                        m_subgrid=m_bh)

                # Exctact the halo data we need. Note that vr_data only
                # contains info for our selected BHs, need to index with iibh
                if vr_data is not None:
                    vr_bhdata = {}
                    vr_bhdata['log_MStar']  = np.log10(vr_data["MStar"][iibh])
                    vr_bhdata['log_M200'] = np.log10(vr_data["M200"][iibh])
                    vr_bhdata['SFR'] = vr_data["SFR"][iibh]
                    vr_bhdata['log_sSFR']  = (np.log10(vr_bhdata['SFR'])
                                              - vr_bhdata['log_MStar'])
                else:
                    vr_bhdata = None
                    
                # Now write actual BH site
                with open(f'{args.wdir}/gallery/'
                          f'index_bh-{ibh}_snap-{isnap}.html', 'w'
                          ) as writer_bh:

                    write_bh_header(writer_bh, ibh, isnap, bh_data, vr_bhdata)
                    write_bh_plots(writer_bh, ibh, isnap, vr_data)
                    write_bh_images(writer_bh, ibh, isnap)
                    
                    # Add html closing lines to BH site
                    writer_bh.write('</body>\n</html>\n')
                         
        # Add html closing lines to front page
        writer.write('</body>\n</html>\n')


def write_gallery_image(writer, ibh, isnap, size, m_subgrid):
    """Write the front page gallery image for a specific BH."""

    curr_im = (f'image_pt0_temp_{get_coda(size)}_'
               f'BH-{ibh}_{isnap:04d}.png')
    m_exp = int(np.log10(m_subgrid))
    m_pre = m_subgrid / 10.0**m_exp
    
    writer.write(f'<div class="gallery">\n'
                 f'<div class="desc">Black hole {ibh} '
                 f'(m<sub>BH</sub> = {m_pre:.1f}&times;10<sup>{m_exp}</sup> '
                 f'M<sub>&#9737</sub>)</div>'
                 f'<a href="index_bh-{ibh}_snap-{isnap}.html">\n'
                 f'  <img src="{curr_im}" alt="BH {ibh}" '
                 f'width=300>\n</a>\n'
                 f'</div>\n')

    
def write_bh_header(writer, ibh, isnap, bh_data, vr_bhdata):
    """Write the header for the BH site."""

    # Write HTML opening lines
    writer.write('<!DOCTYPE html>\n<html>\n<head>\n'
                 '  <link rel="stylesheet" href="style_bh.css">\n'
                 '</head><body>')

    writer.write('<a href="index.html">Back to front page</a>')

    writer.write(f'<h2>BH-ID {ibh} '
                 f'[ID={bh_data["ParticleIDs"][ibh]}]</h2>\n')

    writer.write(f'<h3> ')
    for ixsnap in snap_list:
        writer.write(f'<a href="index_bh-{ibh}_snap-{ixsnap}.html">'
                     f'Snap {ixsnap} (z = {snap_zred[ixsnap]:.1f}) '
                     f'&nbsp; &nbsp; &nbsp;'
                     f'</a>')
    writer.write(f'</h3>')
    
    if vr_bhdata is not None:
        writer.write(f'<h3>log (M<sub>Star</sub> / M<sub>Sun</sub>) = '
                     f'{vr_bhdata["logmstar"]:.3f} &nbsp &nbsp &nbsp '
                     f'log(M<sub>200c</sub> / M<sub>Sun</sub>) = '
                     f'{vr_bhdata["logm200"]:.3f} &nbsp &nbsp &nbsp '
                     f'SFR = {vr_bhdata["sfr"]:.3f} M<sub>Sun</sub> '
                     f'yr<sup>-1</sup> &nbsp &nbsp &nbsp '
                     f'log (sSFR / yr<sup>-1</sup>) = '
                     f'{vr_bhdata["logssfr"]:.3f} </h3>')


def write_bh_plots(writer, ibh, isnap, bh_list, vr_data):
    """Write all the plots for a BH."""

    bh_growth_im = f'bh_growth_tracks_BH-BID-{ibh}.png'
    writer.write(f'<a href="{bh_growth_im}">\n'
                 f'  <img src="{bh_growth_im}" alt="BH growth track" '
                 f'width=500>\n</a>\n')

    # Can only do this if we have successfully linked to VR
    if vr_data is None:
        return

    gal_props_im = f'mstar-m200-ssfr_bh-bid-{ibh}_snap-{isnap}.png'
    writer.write(f'<img src="{gal_props_im}" alt="Galaxy properties" '
                 f'usemap="#workmap" width={500*5.5/5.0}>\n')

    writer.write(f'<map name="workmap">\n')

    # Add a link for each individual BH to the map...
    for ixbh, xbh in enumerate(bh_list):

        if vr_data["M200"][ixbh] <= 0 or vr_data["MStar"][ixbh] <= 0:
            continue

        imx = (np.log10(vr_data["M200"][ixbh])-11.7) / (13.3-11.7)
        imx = int((imx*0.67 + 0.15) * (500*5.5/5.0))

        imy = (np.log10(vr_data["MStar"][ixbh])-10.4) / (11.3-10.4)
        imy = (imy*0.8 + 0.15) * (500*4.5/5.0)
        imy = int((500*4.5/5.0) - imy)

        rad = 5
                
        writer.write(f'<area shape="circle" coords="{imx}, {imy}, {rad}" '
                     f'alt="BH {xbh}" href="index_bh-{xbh}_'
                     f'snap-{isnap}.html">\n')

    writer.write(f'</map>\n<br>\n')


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
