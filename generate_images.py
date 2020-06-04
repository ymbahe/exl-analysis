"""Generate a whole bunch of images."""

import numpy as np
from pdb import set_trace
import hydrangea.hdf5 as hd
import os
import local

# Define general settings

simdir = local.BASE_DIR + 'ID145_E25_xlrepos/'
sim = 145
snapshot = 36
max_bh_mass = 8.5

black_file = simdir + 'black_hole_data.hdf5'
bh_maxflag = hd.read_data(black_file, 'Flag_MostMassiveInHalo')
bh_mstar = hd.read_data(black_file, 'Halo_MStar')
bh_halotype = hd.read_data(black_file, 'HaloTypes')
bh_msg = hd.read_data(black_file, 'SubgridMasses')[:, -1]*1e10

bh_list = np.nonzero((bh_maxflag == 1) &
                     (bh_mstar >= 3e10) &
                     (bh_halotype == 10))[0]

print(f"There are {len(bh_list)} BHs in selection list.")

#bh_list = [3452, 1729, 3619, 2987, 1803, 299, 3713]
ap_list = [1.5, 0.3, 0.03, 0.003]
image_list = [(0, 'temp', '--draw_hsml'),
              (0, 'sfr', '--draw_hsml --scale -4.0 -0.5 --absscale'),
              (4, 'gri', '--nobh --absscale')]

def main():

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
                                 f'{simdir} {snapshot} '
                                 f'--ptype {iim[0]} --imsize {iap} '
                                 f'--cambhbid {ibh} --outdir gallery '
                                 f'--bh_mmax {max_bh_mass} '
                                 f'--imtype {iim[1]} --coda {get_coda(iap)} '
                                 f'--numpix 2000 '
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
        
        for iibh, ibh in enumerate(np.sort(bh_list)):

            logmstar = np.log10(bh_data["Halo_MStar"][ibh])
            logm200 = np.log10(bh_data["Halo_M200c"][ibh])
            sfr = bh_data["Halo_SFR"][ibh]
            logssfr = np.log10(sfr) - logmstar

            msg = bh_msg[ibh]
            m_exp = int(np.log10(msg))
            m_pre = msg / 10.0**m_exp
            
            with open(f'{simdir}/gallery/index_bh-{ibh}.html', 'w') as writer_bh:

                # Write HTML opening lines
                writer_bh.write('<!DOCTYPE html>\n<html>\n<head>\n'
                                '  <link rel="stylesheet" href="style_bh.css">\n'
                                '</head><body>')

                #writer_bh.write('<!DOCTYPE html>\n<html>\n<style>\na {\n    '
                #                'text-decoration: none;\n}\n</style>\n\n<body>\n\n')

                writer_bh.write('<a href="index.html">Back to front page</a>')
                
                writer_bh.write(f'<h2>BH-ID {ibh} '
                                f'[ID={bh_data["ParticleIDs"][ibh]}]</h2>\n')
                writer_bh.write(f'<h3>log (M<sub>Star</sub> / M<sub>Sun</sub>) = '
                                f'{logmstar:.3f} &nbsp &nbsp &nbsp '
                                f'log(M<sub>200c</sub> / M<sub>Sun</sub>) = '
                                f'{logm200:.3f} &nbsp &nbsp &nbsp '
                                f'SFR = {sfr:.3f} M<sub>Sun</sub> yr<sup>-1</sup> '
                                f'&nbsp &nbsp &nbsp '
                                f'log (sSFR / yr<sup>-1</sup>) = {logssfr:.3f} '
                                f'</h3>')
                #writer_bh.write(f'<h3>log M_200 = {logm200:.3f} </h3>')
                #writer_bh.write(f'<h3>SFR  = {sfr:.3f} M_sun/yr </h3>')
                #writer_bh.write(f'<h3>log sSFR  = {logssfr:.3f} yr^-1 </h3>')

                bh_growth_im = f'bh_growth_tracks_BH-BID-{ibh}.png'
                writer_bh.write(f'<a href="{bh_growth_im}">\n'
                                f'  <img src="{bh_growth_im}" alt="BH growth track" '
                                f'width=500>\n</a>\n')

                gal_props_im = f'mstar-m200-ssfr_bh-bid-{ibh}.png'
                writer_bh.write(f'<img src="{gal_props_im}" alt="Galaxy properties" '
                                f'usemap="#workmap" width={500*5.5/5.0}>\n')
                writer_bh.write(f'<map name="workmap">\n')
                
                for ixbh in bh_list:

                    imx = (np.log10(bh_data["Halo_M200c"][ixbh]) - 11.7) / (13.3 - 11.7)
                    imx *= 0.67
                    imx += 0.15
                    imx *= (500*5.5/5.0)
                    imx = int(imx)

                    imy = (np.log10(bh_data["Halo_MStar"][ixbh]) - 10.4) / (11.3 - 10.4)
                    #imy = (11.0 - 10.4) / (11.3 - 10.4)
                    imy *= 0.8
                    imy += 0.15
                    imy *= (500*(4.5/5.0))
                    imy = (500*(4.5/5.0)) - imy
                    imy = int(imy)
                    
                    rad = 5
                    
                    writer_bh.write(f'<area shape="circle" coords="{imx}, {imy}, {rad}" '
                                    f'alt="BH {ixbh}" href="index_bh-{ixbh}.html">\n')

                writer_bh.write(f'</map>\n<br>\n')
                        
                for iiap, iap in enumerate(ap_list):
                    
                    writer_bh.write(f'<h4>{iap*1e3:.0f} kpc</h4>\n')

                    for iiim, iim in enumerate(image_list):
                        curr_im = (f'image_pt{iim[0]}_{iim[1]}_{get_coda(iap)}_'
                                   f'BH-{ibh}_{snapshot:04d}.png')

                        if iiap == 2 and iiim == 0:
                            writer.write(f'<div class="gallery">\n'
                                         f'<div class="desc">Black hole {ibh} '
                                         f'(m<sub>BH</sub> = {m_pre:.1f}&times;10<sup>{m_exp}</sup> M<sub>&#9737</sub>)</div>'
                                         f'<a href="index_bh-{ibh}.html">\n'
                                         f'  <img src="{curr_im}" alt="BH {ibh}" '
                                         f'width=300>\n</a>\n'
                                         f'</div>\n')
                            
                            #if iibh % 50 == 0:
                            #    writer.write(f'<br>\n')
                            
                        writer_bh.write(f'<a href="{curr_im}">\n'
                                        f'  <img src="{curr_im}" alt="Image" '
                                        f'width=500">\n</a>\n')

                    writer_bh.write('<br>\n\n')

                # Add html closing lines
                writer_bh.write('</body>\n</html>\n')

                        
        # Add html closing lines
        writer.write('</body>\n</html>\n')


def get_coda(iap):

    coda = f'{iap:.3f}'
    coda = coda.replace('.', 'p')
    return coda



if __name__ == '__main__':
    main()
