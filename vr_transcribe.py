"""Transcribe the VR catalogue to meaningful structure."""

import numpy as np

from pdb import set_trace
import argparse
import glob
from hydrangea.hdf5 import read_data, write_data, write_attribute

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
print(f"Analysing output file {vrfile}...")

outfile = dirs[0] + f'vr_{args.snapshot:04d}.hdf5'

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
	'total': '/Total',
	'bh': '/BH',
	'gas': '/Gas',
	'gas_nsf': '/NonStarFormingGas',
	'gas_sf': '/StarFormingGas',
	'star': '/Stars'
}

aperture_list = [100, 50, 30, 10, 5]

list_apertures = [
	('SFR_gas', 'SFR', descr_sfr, conv_sfr, None, True),
	('SubgridMasses_aperture_total_solar_mass_bh', 'Masses/BH_SubgridMasses', descr_sfr, conv_sfr, None, False),
	('Zmet', 'Metallicities/Gas', descr_zmet, 1.0, ['gas', 'gas_nsf', 'gas_sf', 'star'], True),
	('mass', 'Masses', descr_mass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star'], True),
	('mass_bh', 'Masses/BH_Dynamical', descr_mass, 1.0, None, False),
	('npart', 'ParticleNumbers', descr_npart, 1, None, True),
	('rhalfmass', 'HalfMassRadii', descr_halfmass, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star'], True),
	('veldisp', 'VelocityDispersions', descr_veldisp, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star'], True)
]

list_simple = [
	('Efrac', 'BoundFractions', descr_efrac, 1.0, ['total', 'gas', 'star']),
	('Ekin' 'KineticEnergies', descr_ekin, 1.0, None),
	('Epot', 'PotentialEnergies', descr_epot, 1.0, None),
	('ID', 'HaloIDs', descr_id, 1, None),
	('ID_mbp', 'ParticleIDs/MostBound', descr_idp, 1, None),
	('ID_minpot', 'ParticleIDs/MinimumPotential', descr_idp, 1, None),
	('Krot', 'RotationalKappas', descr_kapparot, 1.0, ['total', 'gas', 'gas_nsf', 'gas_sf', 'star']),
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
	('R_size', 'HaloExtent', descr_rad, 1.0, None),
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
	('cNFW', 'Concentration', descr_c, 1.0, None),
	('hostHaloID', 'HostHaloIDs', descr_id, 1.0, None),
	('lambda_B', 'LambdaBSpins', descr_lambdab, 1.0, None),
	('n', 'ParticleNumbers', descr_npart, 1, ['bh', 'gas', 'star']),
	('npart', 'ParticleNumbers/Total', descr_npart, 1, None),
	('numSubStruct', 'SubstructureCounts', descr_substr, 1, None),
	('q', 'MajorAxisRatios', descr_q, 1.0, ['total', 'gas', 'star']),
	('s', 'MinorAxisRatios', descr_s, 1.0, ['total', 'gas', 'star']),
	('sigV', 'VelocityDispersions', descr_veldisp, 1.0, ['total', 'gas_nsf', 'gas_sf']),
	('tage_star', 'StellarAge', descr_sage, 1.0, None)
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
	('V?cminpot', 'Velocities/MinimumPotentialParticle', descr_vel, 1.0, None),
	('?c', 'Coordinates/CentreOfMass', descr_pos, 1.0, ['total', 'gas', 'star']),
	('?cmbp', 'Coordinates/MostBoundParticle', descr_vel, 1.0, None),
	('?cminpot', 'Coordinates/MinimumPotentialParticle', descr_vel, 1.0, None)
]

list_3d3d_array = [
	('RVmax_eig_??', 'Eigenvectors_RVmax', descr_eig, 1.0, None),
	('RVmax_veldisp_??', 'VelocityDispersionTensor_RVmax', descr_veldisp, 1.0, None),
	('eig_??', 'Eigenvectors', descr_eig, 1.0, ['total', 'gas', 'star']),
	('veldisp_??', 'VelocityDispersionTensor', descr_veldisp, 1.0, ['total', 'gas', 'star'])		
]

# Actually transcribe data...

for ikey in list_simple:

	if ikey[4] is None:
		data = read_data(vrfile, ikey[0], require=True)
		write_data(outfile, ikey[1], data*ikey[3], comment=ikey[2])

	else:
		for itype in ikey[4]:
            set_trace()
            data = read_data(vrfile, ikey[0] + type_in[itype], require=True)
            write_data(outfile, ikey[1] + '/' + type_out[ikey], data*ikey[3], comment=ikey[2])

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
				outname = 'ApertureMeasurements' + outfix + ikey[1] + typefix_out

				data = read_data(vrfile, vrname, require=True)
				write_data(outfile, outname, data*ikey[3], comment=ikey[2])




# To do otherwise: File_id, Num_of_files, Num_of_groups, SimulationInfo, Total_num_of_groups, UnitInfo
