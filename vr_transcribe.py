"""Transcribe the VR catalogue to meaningful structure."""

import numpy as np

import time
from pdb import set_trace
import argparse
import glob
from hydrangea.hdf5 import read_data, write_data, write_attribute, read_attribute
import h5py as h5
import os
import local
import xltools as xl

print("Parsing input arguments...")
parser = argparse.ArgumentParser(description="Parse input parameters.")
parser.add_argument('sims', nargs='+',
                    help='Simulation index/names to transcribe '
                         '("all" for all runs in base_dir).')
parser.add_argument('--base_dir', default=local.BASE_DIR,
                    help='Simulation base directory, default: '
                         f'{local.BASE_DIR}')

parser.add_argument('--snaps', type=int, nargs='+',
                    help='Snapshot indices to transcribe.')
parser.add_argument('--vr_snaps', type=int, nargs='+',
                    help='Snapshot indices of input VR catalogues, '
                         'if different from the (output) numbering '
                         'specified in --snaps.')
parser.add_argument('--vr_name', default='vr/halos',
                    help='Name prefix of (input) VR files '
                         '(default: "vr/halos").')
parser.add_argument('--out_file', default='vr',
                    help='Output file prefix (default: "vr").')

parser.add_argument('--verbose', action='store_true',
                    help='Display additional progress messages.')

args = parser.parse_args()

# Define the general translation structure

descr_sfr = 'Star formation rates [M_sun/yr]'
descr_mass = 'Masses [M_sun]'
descr_zmet = 'Metal mass fractions [Solar?]'
descr_npart = 'Number of particles'
descr_halfmass = 'Half mass radii [Mpc]'
descr_veldisp = '1D velocity dispersions [km/s]'
descr_efrac = 'Bound fractions'
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

type_self = {
    'total': '',
    'bh': 'black holes',
    'gas': 'gas',
    'gas_nsf': 'not star forming gas',
    'gas_sf': 'star forming',
    'star': 'star'
}
    
aperture_list = [100, 50, 30, 10, 5]
dimsymbols = ['x', 'y', 'z']
dimsymbols_cap = ['X', 'Y', 'Z']

list_apertures = [
    ('SFR_gas', 'SFR', descr_sfr, conv_sfr, None, True),
    ('SubgridMasses_aperture_total_solar_mass_bh', 'BlackHoles/SubgridMasses', descr_sfr, conv_sfr, None, False),
    ('Zmet', 'Metallicities', descr_zmet, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star'], True),
    ('mass', 'Masses', descr_mass, 1e10, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star'], True),
    ('mass_bh', 'BlackHoles/DynamicalMasses', descr_mass, 1e10, None, False),
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
    ('ID_mbp', 'MostBoundParticles/ParticleIDs', descr_idp, 1, None),
    ('ID_minpot', 'MinimumPotential/ParticleIDs', descr_idp, 1, None),
    ('Krot', 'KappaRotParameters', descr_kapparot, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('M', 'Masses', descr_mass, 1e10, ['gas', 'gas_nsf', 'gas_sf', 'star']),
    ('M_bh', 'BlackHoles/DynamicalMasses', descr_mass, 1e10, None),
    ('M_gas_500c', 'Gas/M500c', descr_mass, 1e10, None),
    ('M_gas_Rvmax', 'ApertureMeasurements/RVmax/GasMasses', descr_mass, 1e10, None),
    ('M_star_500c', 'Stars/M500c', descr_mass, 1e10, None),
    ('M_star_Rvmax', 'ApertureMeasurements/RVmax/StellarMasses', descr_mass, 1e10, None),
    ('Mass_200crit', 'M200crit', descr_mass, 1e10, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_200crit_excl', 'HaloSO/M200crit', descr_mass, 1e10, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_200mean', 'M200mean', descr_mass, 1e10, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_200mean_excl', 'HaloSO/M200mean', descr_mass, 1e10, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_BN98', 'Mvir_BN98', descr_mass, 1e10, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_BN98_excl', 'HaloSO/Mvir_BN98', descr_mass, 1e10, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_FOF', 'FOF_Masses', descr_mass, 1e10, None),
    ('Mass_tot', 'Masses', descr_mass, 1e10, None),
    ('Mvir', 'VirialMasses', descr_mass, 1e10, None),
    ('RVmax_lambda_B', 'ApertureMeasurements/RVmax/LambdaBSpins', descr_lambdab, 1.0, None),
    ('RVmax_q', 'ApertureMeasurements/RVmax/MajorAxisRatios', descr_q, 1.0, None),
    ('RVmax_s', 'ApertureMeasurements/RVmax/MinorAxisRatios', descr_s, 1.0, None),
    ('RVmax_sigV', 'ApertureMeasurements/RVmax/VelocityDispersions', descr_veldisp, 1.0, None),
    ('R_200crit', 'R200crit', descr_rad, 1.0, None),
    ('R_200crit_excl', 'HaloSO/R200crit', descr_rad, 1.0, None),
    ('R_200mean', 'R200mean', descr_rad, 1.0, None),
    ('R_200mean_excl', 'HaloSO/R200mean', descr_rad, 1.0, None),
    ('R_BN98', 'Rvir_BN98', descr_rad, 1.0, None),
    ('R_BN98_excl', 'HaloSO/Rvir_BN98', descr_rad, 1.0, None),
    ('R_HalfMass', 'HalfMassRadii', descr_halfmass, 1.0, ['total', 'gas', 'gas_sf', 'gas_nsf', 'star']),
    ('R_size', 'HaloExtents', descr_rad, 1.0, None),
    ('Rmax', 'RVmax', descr_rad, 1.0, None),
    ('Rvir', 'VirialRadii', descr_rad, 1.0, None),
    ('SFR_gas', 'StarFormationRates', descr_sfr, conv_sfr, None),
    ('Structuretype', 'StructureTypes', descr_stype, 1, None),
    ('SubgridMasses_average_solar_mass_bh', 'BlackHoles/AverageSubgridMasses', descr_mass, 1e10, None),
    ('SubgridMasses_max_solar_mass_bh', 'BlackHoles/MaxSubgridMasses', descr_mass, 1e10, None),
    ('SubgridMasses_min_solar_mass_bh', 'BlackHoles/MinSubgridMasses', descr_mass, 1e10, None),
    ('T', 'Temperatures', descr_temp, 1.0, ['gas', 'gas_sf', 'gas_nsf']),
    ('Vmax', 'MaximumCircularVelocities', descr_vel, 1.0, None),
    ('Zmet', 'Metallicities', descr_zmet, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),
    ('cNFW', 'Concentrations', descr_c, 1.0, None),
    ('hostHaloID', 'CentralHaloIDs', 'Pointer to the central halo in the same group; -1 if this halo is'
                                     'itself a central ("field") halo.', 1, None),
    ('lambda_B', 'LambdaBSpins', descr_lambdab, 1.0, None),
    ('n', 'ParticleNumbers', 'Number of # particles in the halo.', 1, ['bh', 'gas', 'star']),
    ('npart', 'ParticleNumbers', descr_npart, 1, None),
    ('numSubStruct', 'SubstructureCounts', descr_substr, 1, None),
    ('q', 'MajorAxisRatios', descr_q, 1.0, ['total', 'gas', 'star']),
    ('s', 'MinorAxisRatios', descr_s, 1.0, ['total', 'gas', 'star']),
    ('sigV', 'VelocityDispersions', descr_veldisp, 1.0, ['total', 'gas_nsf', 'gas_sf']),
    ('tage_star', 'Stars/AverageAges', descr_sage, 1.0, None)
]

list_3d_array = [
    ('L?', 'AngularMomenta/Total', descr_angmom, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),

    ('L?_200c_excl', 'HaloSO/AngularMomenta/R200crit', descr_angmom, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),
    ('L?_200c', 'AngularMomenta/R200crit', descr_angmom, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),
    ('L?_200crit', 'AngularMomenta/R200crit', descr_angmom, 1.0, None),
    ('L?_200crit_excl', 'HaloSO/AngularMomenta/R200crit', descr_angmom, 1.0, None),

    ('L?_200m_excl', 'HaloSO/AngularMomenta/R200mean', descr_angmom, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),
    ('L?_200m', 'AngularMomenta/R200mean', descr_angmom, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star']),
    ('L?_200mean', 'AngularMomenta/R200mean', descr_angmom, 1.0, None),
    ('L?_200mean_excl', 'HaloSO/AngularMomenta/R200mean', descr_angmom, 1.0, None),

    ('L?_BN98', 'AngularMomenta/Rvir_BN98', descr_angmom, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('L?_BN98_excl', 'HaloSO/AngularMomenta/Rvir_BN98', descr_angmom, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),

    ('RVmax_L?', 'ApertureMeasurements/RVmax/AngularMomenta', descr_angmom, 1.0, None),

    ('V?c', 'CentreOfMassVelocities', descr_vel, 1.0, ['total', 'gas', 'star']),
    ('V?cmbp', 'MostBoundParticles/Velocities', descr_vel, 1.0, None),
    ('V?cminpot', 'MinimumPotential/Velocities', descr_vel, 1.0, None),
    ('?c', 'CentresOfMass', descr_pos, 1.0, ['total', 'gas', 'star']),
    ('?cmbp', 'MostBoundParticles/Coordinates', descr_pos, 1.0, None),
    ('?cminpot', 'MinimumPotential/Coordinates', descr_pos, 1.0, None)
]

list_3x3_matrix = [
    ('RVmax_eig_?*', 'ApertureMeasurements/RVmax/Eigenvectors', descr_eig, 1.0, None),
    ('RVmax_veldisp_?*', 'ApertureMeasurements/RVmax/VelocityDispersionTensors', descr_veldisp, 1.0, None),
    ('eig_?*', 'Eigenvectors', descr_eig, 1.0, ['total', 'gas', 'star']),
    ('veldisp_?*', 'VelocityDispersionTensors', descr_veldisp, 1.0, ['total', 'gas', 'star'])        
]

# Data to be transferred from hierarchy VR file
list_other_files = [
    ('Parent_halo_ID', 'ParentHaloIDs', 
        'Pointer to the parent halo (-1 for field haloes).',
        None, None, None, 'hierarchy'),
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
    ('Offset', 'Haloes/Offsets',
        'Index of the first particle ID in "IDs" that belongs to each halo. '
        'There is one more entry here than the number of haloes: the last one '
        'is equal to the total number of IDs. This makes it possible to '
        'access the IDs of all haloes with [Offsets[i]] : [Offsets[i+1]].', 
        None, None, None, 'catalog_groups'),
    ('Offset_unbound', 'Unbound/Offsets',
        'Index of the first particle ID in "IDs" that belongs to each halo. '
        'There is one more entry here than the number of haloes: the last one '
        'is equal to the total number of IDs. This makes it possible to '
        'access the IDs of all haloes with [Offsets[i]] : [Offsets[i+1]].',
        None, None, None, 'catalog_groups'),
    ('Offset', 'Groups/Offset', 
        '',
        None, None, None, 'catalog_SOlist'),

    ('Particle_types', 'Haloes/PartTypes',
        'Types of all particles associated with haloes. Particles belonging to '
        'halo i are listed in indices [Offsets[i]] : [Offsets[i+1]].',
        None, None, None, 'catalog_parttypes'),
    ('Particle_IDs', 'Haloes/IDs',
        'Particle IDs of all particles bound to haloes. The IDs for halo i '
        'are stored at Offsets[i] : Offsets[i+1].',
        None, None, None, 'catalog_particles'),

    ('Particle_types', 'Unbound/PartTypes', '', None, None, None, 'catalog_parttypes.unbound'),
    ('Particle_IDs', 'Unbound/IDs', '', None, None, None, 'catalog_particles.unbound'),
    ('Particle_IDs', 'Groups/IDs', '', None, None, None, 'catalog_SOlist'),
    ('Particle_types', 'Groups/PartTypes', '', None, None, None, 'catalog_SOlist'),
    ('SO_size', 'Groups/Numbers', '', None, None, None, 'catalog_SOlist')
]

       
list_profiles = [
    ('Mass_inclusive_profile', 'Groups/Masses', descr_mass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Mass_profile', 'Masses', descr_mass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Npart_inclusive_profile', 'Groups/Numbers', descr_npart, 1, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
    ('Npart_profile', 'Numbers_Halo', descr_npart, 1, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star'])
]


def main():
    """Main loop over simulations and snapshots."""

    if args.sims[0].lower() == 'all':
        args.sims = xl.get_all_sims(args.base_dir)
        have_full_sim_dir = True
    else:
        have_full_sim_dir = False
    
    for isim in args.sims:

        if have_full_sim_dir:
            wdir = isim
        else:
            wdir = xl.get_sim_dir(args.base_dir, isim)
        
        print("")
        print("====================================================================")
        print(f"=== Processing {wdir} ===")
        print("====================================================================")        
        print("")
        
        for iisnap, isnap in enumerate(args.snaps):

            # Account for possibly different VR numbering than (desired) output
            if args.vr_snaps is None:
                ivsnap = isnap
            else:
                ivsnap = args.vr_snaps[iisnap]
                
            process_snap(wdir, args.out_file, isnap, ivsnap)


def process_snap(wdir, out_file_base, isnap, ivsnap):
    """Process one simulation snapshot."""

    global num_haloes
    global num_groups
    
    stime = time.time()
    print("")
    print(f"Transcribing simulation {wdir}, snapshot {isnap}...")
    if ivsnap != isnap:
        print(f"   (fetching VR snapshot {ivsnap} ...)")
    print("")
    
    # Form the various VR file names
    vrfile_base = wdir + f'{args.vr_name}_{ivsnap:04d}.'
    vrfile = vrfile_base + 'properties'
    vrfile_CPU = vrfile_base + 'catalog_particles.unbound'
    vrfile_CP = vrfile_base + 'catalog_particles'
    vrfile_prof = vrfile_base + 'profiles'
    vrfile_hierarchy = vrfile_base + 'hierarchy'

    # Construct the output files (for main catalogue and for particle info)
    outfile = wdir + f'{out_file_base}_{isnap:04d}.hdf5'
    outfile_particles = wdir + f'{out_file_base}_{isnap:04d}_particles.hdf5'

    print("Transcribing metadata...")
    num_haloes, num_groups = transcribe_metadata(
        vrfile_base, outfile, outfile_particles)

    print("Transcribing apertures...")
    transcribe_data(list_apertures, vrfile, outfile, kind='apertures')

    print("Transcribing scalar quantities...")
    transcribe_data(list_simple, vrfile, outfile)

    print("Transcribing 3D arrays...")
    transcribe_data(list_3d_array, vrfile, outfile, form='3darray')

    print("Transcribing 3x3 matrices...")
    transcribe_data(list_3x3_matrix, vrfile, outfile, form='3x3matrix')

    print("Transcribing scalar quantities from auxiliary files...")
    transcribe_data(list_other_files, vrfile_base, outfile,
                    mixed_source=True)

    print("Transcribing particle links...")
    transcribe_data(list_particles, vrfile_base, outfile_particles,
                    mixed_source=True)
    add_coda_to_offsets(outfile_particles)

    print("Transcribing profiles...")
    transcribe_data(list_profiles, vrfile_prof, outfile, kind='profiles')

    print("\n"
          f"Finished transcribing simulation {wdir},\nsnapshot {isnap} "
          f"in {(time.time() - stime):.3f} sec."
          "\n")


def transcribe_metadata(vrfile_base, outfile, outfile_particles):
    """Transcribe the metadata"""
    
    # Set up the required VR file paths
    vrfile = vrfile_base + 'properties'
    vrfile_CPU = vrfile_base + 'catalog_particles.unbound'
    vrfile_CP = vrfile_base + 'catalog_particles'
    vrfile_prof = vrfile_base + 'profiles'
    vrfile_CS = vrfile_base + 'catalog_SOlist'

    # Copy the Header groups.
    # In future, this should be done properly, converting strings to numbers...
    f_in = h5.File(vrfile, 'r')
    f_out = h5.File(outfile, 'w')
    f_in.copy('Configuration', f_out)
    f_in.copy('SimulationInfo', f_out)
    f_in.copy('UnitInfo', f_out)
    f_in.close()
    f_out.close()

    # Read metadata stored in datasets
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
    num_part_so = read_data(vrfile_CS, 'Num_of_particles_in_SO_regions')[0]
    num_part_so_total = read_data(vrfile_CS, 'Total_num_of_particles_in_SO_regions')[0]
    flag_inclusive_profiles = read_data(vrfile_prof, 'Inclusive_profiles_flag')[0]
    num_bin_edges = read_data(vrfile_prof, 'Num_of_bin_edges')[0]
    radial_bin_edges = read_data(vrfile_prof, 'Radial_bin_edges')
    type_radial_bins = read_data(vrfile_prof, 'Radial_norm')[0]

    # Write general metadata to 'Header'
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
    write_attribute(outfile, 'Header', 'NumberOfSOParticles', num_part_so)
    write_attribute(outfile, 'Header', 'NumberOfSOParticles_Total', num_part_so_total)

    # Write profile-specific metadata directly to that group
    write_attribute(outfile, 'Profiles', 'Flag_Inclusive', flag_inclusive_profiles)
    write_attribute(outfile, 'Profiles', 'NumberOfBinEdges', num_bin_edges)
    write_attribute(outfile, 'Profiles', 'TypeOfBinEdges', type_radial_bins)
    write_attribute(outfile, 'Profiles', 'RadialBinEdges', radial_bin_edges)

    # Copy Header to particles file
    f_out = h5.File(outfile, 'r')
    f_part = h5.File(outfile_particles, 'w')
    f_out.copy('Header', f_part)
    f_out.close()
    f_part.close()

    return num_haloes, num_groups


def transcribe_data(data_list, vrfile_in, outfile, kind='simple',
                    form='scalar', mixed_source=False):
    """Transcribe data sets.

    Parameters
    ----------
    data_list : tuple
        A list of the transcription keys to process. Each key is a tuple
        of (VR_name, Out_name, Comment, Conversion_Factor, Type_list).
    vrfile_in : str
        The VR file to transcribe data from.
    outfile : str
        The output file to store transcribed data in.
    kind : str
        The kind of transcription we are doing. Options are
            - 'main' --> main data transcription
            - 'profiles' --> transcribe profile data
            - 'apertures' --> transcribe aperture measurements
    form : str
        Form of data elements. Options are
            - 'scalar' --> simple scalar data quantity
            - '3darray' --> transcribe 3d array quantities
            - '3x3matrix' --> transcribe 3x3 matrix quantities
    mixed_source : bool, optional
        If True, index 6 specifies the source VR file and 'vrfile' is 
        assumed to be the common base instead.
    """
    for ikey in data_list:
        if len(ikey) < 5: set_trace()
        #if len(ikey) > 5 and kind != 'apertures': set_trace()
            
        # Deal with possibility of 'None' in Type_list (no type iteration)
        if ikey[4] is None:
            types = [None]
        else:
            types = ikey[4]

        # Deal with possibility of mixed-source input
        if mixed_source:
            if len(ikey) < 7:
                print("Need to specify source file in index 6 for "
                      "mixed source transcription!")
            vrfile = vrfile_in + ikey[6]
        else:
            vrfile = vrfile_in
            
        if not os.path.isfile(vrfile):
            print("Could not find input VR file...")
            set_trace()
            
        # Some quantities use capital X/Y/Z in VR...
        if ikey[0] in ['V?c', 'V?cmbp', 'V?cminpot', 
                       '?c', '?cmbp', '?cminpot']:
            dimsyms = dimsymbols_cap
        else:
            dimsyms = dimsymbols

        # Iterate over aperture types (only relevant for apertures)
        if kind == 'apertures':
            n_proj = 4
            ap_list = aperture_list
        else:
            n_proj = 1
            ap_list = [None]

        for iap in ap_list:
            for iproj in range(n_proj):

                # Some special treatment for apertures
                if kind == 'apertures':
                    if iproj > 0 and ikey[5] is False:
                        break

                    if iproj == 0:
                        ap_prefix = 'Aperture_'
                        ap_outfix = '/'
                    else:
                        ap_prefix = f'Projected_aperture_{iproj}_'
                        ap_outfix = f'/Projection{iproj}/'
                else:
                    ap_prefix = ''
                    ap_outfix = ''
                        
                # Iterate over required types
                for itype in types:

                    # Construct type specifiers in in- and output
                    if itype is None:
                        typefix_in = ''
                        typefix_out = ''
                    else:
                        typefix_in = type_in[itype]
                        typefix_out = type_out[itype]

                    # Adjust comment
                    if itype is None:
                        comment = ikey[2].replace('#', '')
                    else:
                        comment = ikey[2].replace('#', type_self[itype])
                
                    # Construct the full data set names in in- and output
                    vrname = ikey[0] + typefix_in
                    outname = typefix_out + ikey[1]

                    # Deal with names for special cases
                    if kind == 'profiles':
                        outname = f'Profiles/{outname}'
                    elif kind == 'apertures':
                        vrname = f'{ap_prefix}{vrname}_{iap}_kpc'
                        outname = (f'ApertureMeasurements{ap_outfix}{iap}kpc/'
                                   f'{outname}')

                    # Transcribe data
                    if args.verbose:
                        print(f"{vrname} --> {outname}")

                    if form == '3darray':
                        outdata = np.zeros((num_haloes, 3), dtype=np.float32)-1

                        # Load individual dimensions' data sets into output
                        for idim in range(3):
                            vrname_dim = vrname.replace('?', dimsyms[idim])        
                            outdata[:, idim] = read_data(vrfile, vrname_dim,
                                                         require=True)
                    
                    elif form == '3x3matrix':
                        outdata = np.zeros((num_haloes, 3, 3), 
                                           dtype=np.float32) - 1                        
                        for idim1 in range(3):
                            for idim2 in range(3):
                                vrname_dim = (
                                    vrname.replace('?', dimsyms[idim1]).replace('*', dimsyms[idim2]))
                                outdata[:, idim1, idim2] = read_data(
                                    vrfile, vrname_dim, require=True)
                    else:
                        # Standard case (scalar quantity)
                        outdata = read_data(vrfile, vrname, require=True)

                    if ikey[3] is not None:
                        outdata *= ikey[3]
  

def add_coda_to_offsets(vr_part_file):
    """Add the coda to particle offsets."""

    num_name = {'Haloes': 'NumberOfBoundParticles_Total',
                'Unbound': 'NumberOfUnboundParticles_Total',
                'Groups': 'NumberOfSOParticles_Total'}

    for grp in ['Haloes', 'Unbound', 'Groups']:
        offsets = read_data(vr_part_file, f'{grp}/Offsets')
        num_ids = read_attribute(vr_part_file, 'Header', num_name[grp]

        offsets = np.concatenate((offsets, [num_ids]))
        write_data(vr_part_file, f'{grp}/Offsets', offsets)


if __name__ == '__main__':
    main()
            
"""                    
for ikey in list_profiles:
    print("Profiles...")
    
    if len(ikey) < 5: set_trace()

    if ikey[4] is None:
        data = read_data(vrfile_prof, ikey[0], require=True)
        write_data(outfile, f'Profiles/{ikey[1]}', data*ikey[3], comment=ikey[2])

    else:
        for itype in ikey[4]:
            data = read_data(vrfile_prof, ikey[0] + type_in[itype], require=True)
            write_data(outfile, f'Profiles/{type_out[itype]}/{ikey[1]}', data*ikey[3], comment=ikey[2])


for ikey in list_apertures:
    for iap in aperture_list:
        for iproj in range(4):

            # Don't do projected quantities if switched off
            if iproj > 0 and ikey[5] is False:
                break
            
            if iproj == 0:
                prefix = 'Aperture_'
                outfix = '/'
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
                outname = f'ApertureMeasurements' + outfix + f'{iap}kpc/' + typefix_out + ikey[1]
                print(outname)

                if kind == '3darray':
                    pass
                elif kind == '3x3matrix':
                    pass
                else:
                    outdata = read_data(vrfile, vrname, require=True)
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
        outname = typefix_out + ikey[1]
        print(outname)
        
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
        outname = typefix_out + ikey[1]
        print(outname)
        
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
"""
