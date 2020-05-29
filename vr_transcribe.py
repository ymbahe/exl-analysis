"""Transcribe the VR catalogue to meaningful structure."""

import numpy as np

from pdb import set_trace
import argparse
import glob
from hydrangea.hdf5 import read_data, write_data, write_attribute
import h5py as h5

print("Parsing input arguments...")
parser = argparse.ArgumentParser(description="Parse input parameters.")
parser.add_argument('sim', type=int, help='Simulation index to transcribe')
parser.add_argument('snapshot', type=int, help='Snapshot to transcribe')
args = parser.parse_args()

dirs = glob.glob(f'/cosma7/data/dp004/dc-bahe1/EXL/ID{args.sim}*/')
if len(dirs) != 1:
    print(f"Could not unambiguously find directory for simulation {args.sim}!")
    set_trace()
vrfile = dirs[0] + f'vr/halos_{args.snapshot:04d}.properties'
vrfile_CPU = dirs[0] + f'vr/halos_{args.snapshot:04d}.catalog_particles.unbound'
vrfile_CP = dirs[0] + f'vr/halos_{args.snapshot:04d}.catalog_particles'
vrfile_prof = dirs[0] + f'vr/halos_{args.snapshot:04d}.profiles'

print(f"Analysing output file {vrfile}...")

outfile = dirs[0] + f'vr_{args.snapshot:04d}.hdf5'
outfile_particles = dirs[0] + f'vr_{args.snapshot:04d}_particles.hdf5'

# Define translation table for 'simple' data sets...
# List of tuples, each with structure: [VR_name] || [out_name] || [comment] || conversion factor

descr_sfr = 'Star formation rates in M_sun / yr'
descr_mass = 'Mass in M_sun'
descr_zmet = 'Metal mass fractions'
descr_npart = 'Number of particles'
descr_halfmass = 'Half mass radius in ???'
descr_veldisp = '1D velocity dispersion in km/s'
descr_efrac = ''
descr_ekin = 'Energy'
descr_epot = 'Energy'
descr_id = 'Halo ID'
descr_idp = 'Particle ID'
descr_kapparot = 'Rotational support parameter'
descr_lambdab = 'LambdaB spin parameter'
descr_q = 'Major/intermediate axis ratio'
descr_s = 'Minor/intermediate axis ratio'
descr_rad = 'Radius in ???'
descr_stype = 'Structure type code'
descr_temp = 'Temperature in K'
descr_c = 'NFW concentration parameter'
descr_substr = 'Number of substructures'
descr_sage = 'Average stellar age in ???'
descr_angmom = 'Angular momentum in ???'
descr_pos = 'Coordinates in ???'
descr_vel = 'Velocities in km/s'
descr_eig = 'Eigenvectors in ???'

conv_sfr = 1.023045e-02

type_in = {
    'total': '',
    'bh': '_bh',
    'gas': '_gas',
    'gas_nsf': '_gas_nsf',
    'gas_sf': '_gas_sf',
    'star': '_star',
}

type_out = {
    'total': '',
    'bh': 'BlackHoles/',
    'gas': 'Gas/',
    'gas_nsf': 'NonStarFormingGas/',
    'gas_sf': 'StarFormingGas/',
    'star': 'Stars/'
}

aperture_list = [100, 50, 30, 10, 5]
dimsymbols = ['x', 'y', 'z']
dimsymbols_cap = ['X', 'Y', 'Z']

list_apertures = [
    ('SFR_gas', 'SFR', descr_sfr, conv_sfr, None, True),
    ('SubgridMasses_aperture_total_solar_mass_bh', 'Masses/BH_SubgridMasses', descr_sfr, conv_sfr, None, False),
    ('Zmet', 'Metallicities/Gas', descr_zmet, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star'], True),
    ('mass', 'Masses', descr_mass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star'], True),
    ('mass_bh', 'Masses/BH_Dynamical', descr_mass, 1.0, None, False),
    ('npart', 'ParticleNumbers', descr_npart, 1, None, False),
    ('rhalfmass', 'HalfMassRadii', descr_halfmass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star'], True),
    ('veldisp', 'VelocityDispersions', descr_veldisp, 1.0, ['total', 'gas', 'gas_nsf', 'star'], False),
    ('veldips', 'VelocityDispersions', descr_veldisp, 1.0, ['gas_sf'], False)
]

list_simple = [
    ('Efrac', 'BoundFractions', descr_efrac, 1.0, ['total', 'gas', 'star']),
    ('Ekin', 'KineticEnergies', descr_ekin, 1.0, None),
    ('Epot', 'PotentialEnergies', descr_epot, 1.0, None),
    ('ID', 'HaloIDs', descr_id, 1, None),
    ('ID_mbp', 'ParticleIDs/MostBound', descr_idp, 1, None),
    ('ID_minpot', 'ParticleIDs/MinimumPotential', descr_idp, 1, None),
    ('Krot', 'KappaRotParameters', descr_kapparot, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('M', 'Masses', descr_mass, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),
    ('M_bh', 'Masses/BH_Dynamical', descr_mass, 1.0, None),
    ('M_gas_500c', 'Masses/Special/Gas_R500c', descr_mass, 1.0, None),
    ('M_gas_Rvmax', 'Masses/Special/Gas_RVmax', descr_mass, 1.0, None),
    ('M_star_500c', 'Masses/Special/Stars_R500c', descr_mass, 1.0, None),
    ('M_star_Rvmax', 'Masses/Special/Stars_RVmax', descr_mass, 1.0, None),
    ('Mass_200crit', 'Masses/R200crit_All', descr_mass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_200crit_excl', 'Masses/R200crit_Halo', descr_mass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_200mean', 'Masses/R200mean_All', descr_mass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_200mean_excl', 'Masses/R200mean_Halo', descr_mass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_BN98', 'Masses/BN98_All', descr_mass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_BN98_excl', 'Masses/BN98_Halo', descr_mass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_FOF', 'Masses/FOF', descr_mass, 1.0, None),
    ('Mass_tot', 'Masses/Total', descr_mass, 1.0, None),
    ('Mvir', 'Masses/Virial', descr_mass, 1.0, None),
    ('RVmax_lambda_B', 'LambdaBSpins_RVmax', descr_lambdab, 1.0, None),
    ('RVmax_q', 'MajorAxisRatios_RVmax', descr_q, 1.0, None),
    ('RVmax_s', 'MinorAxisRatios_RVmax', descr_s, 1.0, None),
    ('RVmax_sigV', 'VelocityDispersions_RVmax', descr_veldisp, 1.0, None),
    ('R_200crit', 'Radii/R200crit_All', descr_rad, 1.0, None),
    ('R_200crit_excl', 'Radii/R200crit_Halo', descr_rad, 1.0, None),
    ('R_200mean', 'Radii/R200mean_All', descr_rad, 1.0, None),
    ('R_200mean_excl', 'Radii/R200mean_Halo', descr_rad, 1.0, None),
    ('R_BN98', 'Radii/BN98_All', descr_rad, 1.0, None),
    ('R_BN98_excl', 'Radii/BN98_Halo', descr_rad, 1.0, None),
    ('R_HalfMass', 'Radii/HalfMass', descr_halfmass, 1.0, ['total', 'gas', 'gas_sf', 'gas_nsf', 'star']),
    ('R_size', 'HaloExtents', descr_rad, 1.0, None),
    ('Rmax', 'Radii/RVmax', descr_rad, 1.0, None),
    ('Rvir', 'Radii/Virial', descr_mass, 1.0, None),
    ('SFR_gas', 'StarFormationRates', descr_sfr, 1.0, None),
    ('Structuretype', 'StructureTypes', descr_stype, 1, None),
    ('SubgridMasses_average_solar_mass_bh', 'Masses/BH_Average_SubgridMasses', descr_mass, 1.0, None),
    ('SubgridMasses_max_solar_mass_bh', 'Masses/BH_Max_SubgridMasses', descr_mass, 1.0, None),
    ('SubgridMasses_min_solar_mass_bh', 'Masses/BH_Min_SubgridMasses', descr_mass, 1.0, None),
    ('T', 'Temperatures', descr_temp, 1.0, ['gas', 'gas_sf', 'gas_nsf']),
    ('Vmax', 'Vmax', descr_vel, 1.0, None),
    ('Zmet', 'Metallicities', descr_zmet, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),
    ('cNFW', 'Concentrations', descr_c, 1.0, None),
    ('hostHaloID', 'HostHaloIDs', descr_id, 1.0, None),
    ('lambda_B', 'LambdaBSpins', descr_lambdab, 1.0, None),
    ('n', 'Particles/Numbers', descr_npart, 1, ['bh', 'gas', 'star']),
    ('npart', 'Particles/Numbers/Total', descr_npart, 1, None),
    ('numSubStruct', 'SubstructureCounts', descr_substr, 1, None),
    ('q', 'MajorAxisRatios', descr_q, 1.0, ['total', 'gas', 'star']),
    ('s', 'MinorAxisRatios', descr_s, 1.0, ['total', 'gas', 'star']),
    ('sigV', 'VelocityDispersions', descr_veldisp, 1.0, ['total', 'gas_nsf', 'gas_sf']),
    ('tage_star', 'StellarAges', descr_sage, 1.0, None)
]

list_3d_array = [
    ('L?', 'AngularMomenta/Total', descr_angmom, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),

    ('L?_200c_excl', 'AngularMomenta/R200crit_Halo', descr_angmom, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),
    ('L?_200c', 'AngularMomenta/R200crit_All', descr_angmom, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),
    ('L?_200crit', 'AngularMomenta/R200crit_All/Total', descr_angmom, 1.0, None),
    ('L?_200crit_excl', 'AngularMomenta/R200crit_Halo/Total', descr_angmom, 1.0, None),

    ('L?_200m_excl', 'AngularMomenta/R200mean_Halo', descr_angmom, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),
    ('L?_200m', 'AngularMomenta/R200mean_All', descr_angmom, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),
    ('L?_200mean', 'AngularMomenta/R200mean_All/Total', descr_angmom, 1.0, None),
    ('L?_200mean_excl', 'AngularMomenta/R200mean_Halo/Total', descr_angmom, 1.0, None),

    ('L?_BN98', 'AngularMomenta/BN98_All', descr_angmom, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('L?_BN98_excl', 'AngularMomenta/BN98_Halo', descr_angmom, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),

    ('RVmax_L?', 'AngularMomenta/RVmax', descr_angmom, 1.0, None),

    ('V?c', 'Velocities/CentreOfMass', descr_vel, 1.0, ['total', 'gas', 'star']),
    ('V?cmbp', 'Velocities/MostBoundParticle', descr_vel, 1.0, None),
    ('V?cminpot', 'Velocities/MinimumPotential', descr_vel, 1.0, None),
    ('?c', 'Coordinates/CentreOfMass', descr_pos, 1.0, ['total', 'gas', 'star']),
    ('?cmbp', 'Coordinates/MostBoundParticle', descr_vel, 1.0, None),
    ('?cminpot', 'Coordinates/MinimumPotential', descr_vel, 1.0, None)
]

list_3d3d_array = [
    ('RVmax_eig_?*', 'Eigenvectors_RVmax', descr_eig, 1.0, None),
    ('RVmax_veldisp_?*', 'VelocityDispersionTensors_RVmax', descr_veldisp, 1.0, None),
    ('eig_?*', 'Eigenvectors', descr_eig, 1.0, ['total', 'gas', 'star']),
    ('veldisp_?*', 'VelocityDispersionTensors', descr_veldisp, 1.0, ['total', 'gas', 'star'])        
]

list_other_files = [
    ('hierarchy', 'Parent_halo_ID', 'ParentHaloIDs', 'Pointer to the parent halo (-1 for field haloes).'),
    #('catalog_groups', 'Offset', 'Particles/Offsets', '???'),
    #('catalog_groups', 'Offset_unbound', 'Particles/Unbound/Offsets', '???'),
    #('catalog_parttypes', 'Particle_types', 'Particles/PartTypes', '', False),
    #('catalog_particles', 'Particle_IDs', 'Particles/IDs', '', False),
    #('catalog_parttypes.unbound', 'Particle_types', 'Particles/Unbound/PartTypes', '', False),
    #('catalog_particles.unbound', 'Particle_IDs', 'Particles/Unbound/IDs', '', False),
    #('catalog_SOlist', 'Offset', 'Particles/Groups/Offset', ''),
    #('catalog_SOlist', 'Particle_IDs', 'Particles/Groups/IDs', '', False),
    #('catalog_SOlist', 'Particle_types', 'Particles/Groups/PartTypes', '', False),
    #('catalog_SOlist', 'SO_size', 'Particles/Groups/Numbers', '')
]


list_particles = [
    ('catalog_groups', 'Offset', 'Haloes/Offsets', '???'),
    ('catalog_groups', 'Offset_unbound', 'Unbound/Offsets', '???'),
    ('catalog_parttypes', 'Particle_types', 'Haloes/PartTypes', ''),
    ('catalog_particles', 'Particle_IDs', 'Haloes/IDs', ''),
    ('catalog_parttypes.unbound', 'Particle_types', 'Unbound/PartTypes', ''),
    ('catalog_particles.unbound', 'Particle_IDs', 'Unbound/IDs', ''),
    ('catalog_SOlist', 'Offset', 'Groups/Offset', ''),
    ('catalog_SOlist', 'Particle_IDs', 'Groups/IDs', ''),
    ('catalog_SOlist', 'Particle_types', 'Groups/PartTypes', ''),
    ('catalog_SOlist', 'SO_size', 'Groups/Numbers', ''),
    ('properties', 'npart', 'Haloes/Numbers', '')
]

       
list_profiles = [
    ('Mass_inclusive_profile', 'Profiles/Masses_All', descr_mass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_profile', 'Profiles/Masses_Halo', descr_mass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Npart_inclusive_profile', 'Profiles/Numbers_All', descr_npart, 1, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Npart_profile', 'Profiles/Numbers_Halo', descr_npart, 1, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star'])
]

# Transcribe metadata

f_in = h5.File(vrfile, 'r')
f_out = h5.File(outfile, 'w')

f_in.copy('Configuration', f_out)
f_in.copy('SimulationInfo', f_out)
f_in.copy('UnitInfo', f_out)

f_in.close()
f_out.close()


num_haloes_total = read_data(vrfile, 'Total_num_of_groups')[0]
num_haloes = read_data(vrfile, 'Num_of_groups')[0]         # Note confusing nomenclature
num_groups = read_data(vrfile_prof, 'Num_of_halos')[0]     # ....
num_groups_total = read_data(vrfile_prof, 'Total_num_of_halos')[0]     # ....

file_id = read_data(vrfile, 'File_id')[0]
file_num = read_data(vrfile, 'Num_of_files')[0]
num_bound = read_data(vrfile_CP, 'Num_of_particles_in_groups')[0]
num_bound_total = read_data(vrfile_CP, 'Total_num_of_particles_in_all_groups')[0]
num_unbound = read_data(vrfile_CPU, 'Num_of_particles_in_groups')[0]
num_unbound_total = read_data(vrfile_CPU, 'Total_num_of_particles_in_all_groups')[0]

flag_inclusive_profiles = read_data(vrfile_prof, 'Inclusive_profiles_flag')[0]
num_bin_edges = read_data(vrfile_prof, 'Num_of_bin_edges')[0]
radial_bin_edges = read_data(vrfile_prof, 'Radial_bin_edges')
type_radial_bins = read_data(vrfile_prof, 'Radial_norm')[0]

write_attribute(outfile, 'Header', 'NumberOfHaloes', num_haloes)
write_attribute(outfile, 'Header', 'NumberOfHaloes_Total', num_haloes_total)
write_attribute(outfile, 'Header', 'NumberOfGroups', num_groups)
write_attribute(outfile, 'Header', 'NumberOfGroups_Total', num_groups_total)
write_attribute(outfile, 'Header', 'NumberOfFiles', file_num)
write_attribute(outfile, 'Header', 'FileID', file_id)
write_attribute(outfile, 'Header', 'NumberOfBoundParticles', num_bound)
write_attribute(outfile, 'Header', 'NumberOfBoundParticles_Total', num_bound_total)
write_attribute(outfile, 'Header', 'NumberOfUnboundParticles', num_unbound)
write_attribute(outfile, 'Header', 'NumberOfUnboundParticles_Total', num_unbound_total)

write_attribute(outfile, 'Profiles', 'Flag_Inclusive', flag_inclusive_profiles)
write_attribute(outfile, 'Profiles', 'NumberOfBinEdges', num_bin_edges)
write_attribute(outfile, 'Profiles', 'TypeOfBinEdges', type_radial_bins)
write_attribute(outfile, 'Profiles', 'RadialBinEdges', radial_bin_edges)

f_out = h5.File(outfile, 'r')
f_part = h5.File(outfile_particles, 'w')

f_out.copy('Header', f_part)

f_out.close()
f_part.close()


# Actually transcribe data...

for ikey in list_simple:

    if len(ikey) < 5: set_trace()

    if ikey[4] is None:
        types = [None]
    else:
        types = ikey[4]

    for itype in types:
        if itype is None:
            typefix_in = ''
            typefix_out = ''
        else:
            typefix_in = type_in[itype]
            typefix_out = type_out[itype]

    vrname = ikey[0] + typefix_in
    outname = typefix_out + ikey[1]

    data = read_data(vrfile, vrname, require=True)
    write_data(outfile, outname, data*ikey[3], comment=ikey[2])


for ikey in list_profiles:

    if len(ikey) < 5: set_trace()
    if ikey[4] is None:
        data = read_data(vrfile_prof, ikey[0], require=True)
        write_data(outfile, ikey[1], data*ikey[3], comment=ikey[2])

    else:
        for itype in ikey[4]:
            data = read_data(vrfile_prof, ikey[0] + type_in[itype], require=True)
            write_data(outfile, ikey[1] + '/' + type_out[itype], data*ikey[3], comment=ikey[2])


for ikey in list_apertures:
    for iap in aperture_list:
        for iproj in range(4):

            # Don't do projected quantities if switched off
            if iproj > 0 and ikey[5] is False:
                break
            
            if iproj == 0:
                prefix = 'Aperture_'
                outfix = ''
            else:
                prefix = f'Projected_aperture_{iproj}_'
                outfix = f'/Projection{iproj}/'

            if ikey[4] is None:
                types = [None]
            else:
                types = ikey[4]

            for itype in types:

                if itype is None:
                    typefix_in = ''
                    typefix_out = ''
                else:
                    typefix_in = type_in[itype]
                    typefix_out = type_out[itype]

                vrname = prefix + ikey[0] + typefix_in + f'_{iap}_kpc'
                outname = 'ApertureMeasurements/' + outfix + ikey[1] + typefix_out + f'/{iap}kpc'

                data = read_data(vrfile, vrname, require=True)
                write_data(outfile, outname, data*ikey[3], comment=ikey[2])


# More fun: deal with 3D array quantities
for ikey in list_3d_array:

    if ikey[4] is None:
        types = [None]
    else:
        types = ikey[4]

    for itype in types:

        if itype is None:
            typefix_in = ''
            typefix_out = ''
        else:
            typefix_in = type_in[itype]
            typefix_out = type_out[itype]

        vrname = ikey[0] + typefix_in
        outname = ikey[1] + typefix_out

        outdata = np.zeros((num_haloes, 3), dtype=np.float32) - 1

        for idim in range(3):

            if ikey[0] in ['V?c', 'V?cmbp', 'V?cminpot', '?c', '?cmbp', '?cminpot']:
                vrname_dim = vrname.replace('?', dimsymbols_cap[idim])
            else:
                vrname_dim = vrname.replace('?', dimsymbols[idim])
        
            outdata[:, idim] = read_data(vrfile, vrname_dim, require=True)

        write_data(outfile, outname, outdata*ikey[3], comment=ikey[2])


# Most fun: deal with 3x2D matrix quantities
for ikey in list_3d3d_array:

    if ikey[4] is None:
        types = [None]
    else:
        types = ikey[4]

    for itype in types:

        if itype is None:
            typefix_in = ''
            typefix_out = ''
        else:
            typefix_in = type_in[itype]
            typefix_out = type_out[itype]

        vrname = ikey[0] + typefix_in
        outname = ikey[1] + typefix_out

        outdata = np.zeros((num_haloes, 3, 3), dtype=np.float32) - 1

        for idim1 in range(3):
            for idim2 in range(3):

                vrname_dim = vrname.replace('?', dimsymbols[idim1]).replace('*', dimsymbols[idim2])        
                outdata[:, idim1, idim2] = read_data(vrfile, vrname_dim, require=True)

        write_data(outfile, outname, outdata*ikey[3], comment=ikey[2])


# Finally, transcribe data from other files
for ikey in list_other_files:

    if len(ikey) != 4: set_trace()

    infile = dirs[0] + f'vr/halos_{args.snapshot:04d}.{ikey[0]}'

    data = read_data(infile, ikey[1], require=True)
    write_data(outfile, ikey[2], data, comment=ikey[3])


for ikey in list_particles:

    if len(ikey) != 4: set_trace()

    infile = dirs[0] + f'vr/halos_{args.snapshot:04d}.{ikey[0]}'

    data = read_data(infile, ikey[1], require=True)
    write_data(outfile_particles, ikey[2], data, comment=ikey[3])
