"""Generate a whole bunch of images."""

import numpy as np
from pdb import set_trace
import hydrangea.hdf5 as hd
import os
import local

from reduce_bh_data import connect_to_galaxies

# Define general settings

simdir = local.BASE_DIR + 'ID179_JE25/'
sim = 179
max_bh_mass = 8.5

black_file = simdir + 'black_hole_data.hdf5'
bh_maxflag = hd.read_data(black_file, 'Flag_MostMassiveInHalo')
bh_mstar = hd.read_data(black_file, 'Halo_MStar')
bh_halotype = hd.read_data(black_file, 'HaloTypes')
bh_msg = hd.read_data(black_file, 'SubgridMasses')[:, -1]*1e10
bh_ids = hd.read_data(black_file, 'ParticleIDs')

bh_list = np.nonzero((bh_maxflag == 1) &
                     (bh_mstar >= 3e10) &
                     (bh_halotype == 10))[0]

print(f"There are {len(bh_list)} BHs in selection list.")

#bh_list = [3452, 1729, 3619, 2987, 1803, 299, 3713]
ap_list = [1.5, 0.3, 0.03, 0.003]
snap_list = [24, 36]
snap_zred = np.zeros(37)
snap_zred[24] = 1.08
snap_zred[36] = 0.00

image_list = [(0, 'temp', '--draw_hsml'),
              (0, 'sfr', '--draw_hsml --scale -4.0 -0.5 --absscale'),
              (4, 'gri', '--nobh --absscale')]

def main():

    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    parser.add_argument('sim',
                        help='Simulation index or name to analyse')
    parser.add_argument('--base_dir', help='Simulation base directory',
                        default=local.BASE_DIR)
    parser.add_argument('--bh_mmax',
                        help='Maximum BH mass, for the scaling in the images',
                        default='1')
    parser.add_argument('--output_names',
                        help='Prefix of existing outputs (default: "output")',
                        default='output')
    parser.add_argument('--link_names', help='Prefix of links (default: "eagle")',
                        default='eagle')

    args = parser.parse_args()

    if args.sims[0].lower() == 'all':
        args.sims = local.get_all_sims(args.base_dir)
        have_full_sim_dir = True
    else:
        have_full_sim_dir = False

    for isim in args.sims:
        process_sim(args, isim, have_full_sim_dir)

    
    bh_data = lookup_bh_data()
    generate_image_script()
    generate_track_script()
    generate_website(bh_data)

    
def lookup_bh_data():
    """Load info from BH file into arrays."""

    bh_file = simdir + 'black_hole_data.hdf5'
    bh_data = {}
    for idata in ['ParticleIDs', 'Haloes', 'Halo_MStar', 'Halo_M200c',
                  'Halo_SFR']:
        data = hd.read_data(bh_file, idata)
        bh_data[idata] = data

    return bh_data


def generate_image_script():
    """Generate a script for auto-generation of images."""

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    
    with open(f'{simdir}/image_script.sh', 'w') as writer:
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
                                    f'{simdir} {isnap} '
                                    f'--ptype {iim[0]} --imsize {iap} '
                                    f'--cambhbid {ibh} --outdir gallery '
                                    f'--bh_mmax {max_bh_mass} '
                                    f'--imtype {iim[1]} --coda {get_coda(iap)} '
                                    f'--numpix 1000 --nosave '
                                    f'{specific_options} '
                                    f'{iim[2]}\n')

def generate_track_script():
    """Generate a script for auto-generation of growth tracks."""

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    
    with open(f'{simdir}/track_script.sh', 'w') as writer:
        writer.write('#!/bin/tcsh\n\n')

        for ibh in bh_list:
            writer.write(f'python {curr_dir}/bh_growth_comparison.py '
                         f'--sim {sim} --bh_bid {ibh} '
                         '\n')
                    
def generate_website(bh_data):
    """Generate a simple website displaying all images."""
    
    if not os.path.isdir(f'{simdir}/gallery'):
        os.makedirs(f'{simdir}/gallery')
    with open(f'{simdir}/gallery/index.html', 'w') as writer:

        # Write HTML opening lines
        writer.write('<!DOCTYPE html>\n<html>\n<head>\n'
                     '  <link rel="stylesheet" href="style.css">\n'
                     '</head><body>')

        # Actual BH-specific sites are written one per snapshot.
        for isnap in snap_list:

            args = type('test', (object,), {})()
            args.wdir = simdir
            args.vrsnap = isnap
            vr_data = connect_to_galaxies(bh_ids[np.sort(bh_list)], args)
            
            # Now process each BH in turn...
            for iibh, ibh in enumerate(np.sort(bh_list)):

                if isnap == 36:
                    # Write the entry image for the front page:
                    write_gallery_image(writer, ibh, isnap=isnap)
                    
                # Currently, we only do VR linking at z = 0
                if isnap == isnap:
                    vr_bhdata = {}
                    vr_bhdata['logmstar']  = np.log10(vr_data["MStar"][iibh])
                    vr_bhdata['logm200'] = np.log10(vr_data["M200"][iibh])
                    vr_bhdata['sfr'] = vr_data["SFR"][iibh]
                    vr_bhdata['logssfr']  = (np.log10(vr_bhdata['sfr'])
                                           - vr_bhdata['logmstar'])
                else:
                    vr_bhdata = None
                    
                # Now write actual BH site
                with open(f'{simdir}/gallery/index_bh-{ibh}_snap-{isnap}.html',
                          'w') as writer_bh:

                    write_bh_header(writer_bh, ibh, isnap, bh_data, vr_bhdata)
                    write_bh_plots(writer_bh, ibh, isnap, vr_data)
                    write_bh_images(writer_bh, ibh, isnap)
                    
                    # Add html closing lines to BH site
                    writer_bh.write('</body>\n</html>\n')
                         
        # Add html closing lines to front page
        writer.write('</body>\n</html>\n')


def write_gallery_image(writer, ibh, isnap):
    """Write the gallery image for a specific BH."""

    curr_im = (f'image_pt0_temp_{get_coda(0.03)}_'
               f'BH-{ibh}_{isnap:04d}.png')
    msg = bh_msg[ibh]
    m_exp = int(np.log10(msg))
    m_pre = msg / 10.0**m_exp
    
    writer.write(f'<div class="gallery">\n'
                 f'<div class="desc">Black hole {ibh} '
                 f'(m<sub>BH</sub> = {m_pre:.1f}&times;10<sup>{m_exp}</sup> '
                 f'M<sub>&#9737</sub>)</div>'
                 f'<a href="index_bh-{ibh}_snap-{isnap}.html">\n'
                 f'  <img src="{curr_im}" alt="BH {ibh}" '
                 f'width=300>\n</a>\n'
                 f'</div>\n')

    
def write_bh_header(writer, ibh, isnap, bh_data, vr_data=None):
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
    
    if vr_data is not None:
        writer.write(f'<h3>log (M<sub>Star</sub> / M<sub>Sun</sub>) = '
                     f'{vr_data["logmstar"]:.3f} &nbsp &nbsp &nbsp '
                     f'log(M<sub>200c</sub> / M<sub>Sun</sub>) = '
                     f'{vr_data["logm200"]:.3f} &nbsp &nbsp &nbsp '
                     f'SFR = {vr_data["sfr"]:.3f} M<sub>Sun</sub> '
                     f'yr<sup>-1</sup> &nbsp &nbsp &nbsp '
                     f'log (sSFR / yr<sup>-1</sup>) = '
                     f'{vr_data["logssfr"]:.3f} </h3>')


def write_bh_plots(writer, ibh, isnap, vr_data):
    """Write all the plots for a BH."""

    bh_growth_im = f'bh_growth_tracks_BH-BID-{ibh}.png'
    writer.write(f'<a href="{bh_growth_im}">\n'
                 f'  <img src="{bh_growth_im}" alt="BH growth track" '
                 f'width=500>\n</a>\n')

    if None is None:
        gal_props_im = f'mstar-m200-ssfr_bh-bid-{ibh}_snap-{isnap}.png'
        writer.write(f'<img src="{gal_props_im}" alt="Galaxy properties" '
                     f'usemap="#workmap" width={500*5.5/5.0}>\n')

        writer.write(f'<map name="workmap">\n')

        # Add a link for each individual BH to the map...
        for ixbh, xbh in enumerate(np.sort(bh_list)):

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
