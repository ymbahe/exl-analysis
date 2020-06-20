"""Link a VR catalogue to its DM-only counterpart."""

import numpy as np
import hydrangea.hdf5 as hd
import hydrangea.crossref as hx
from pdb import set_trace
import argparse
import time
import os
import local
import xltools as xl

def main():
	"""Main program."""

	argvals = None
    args = get_args(argvals)
    args.dmo_dir = f'{args.base_dir}{args.dmo_sim}'

    if args.sims[0].lower() == 'all':
        args.sims = xl.get_all_sims(args.base_dir)
        args.have_full_sim_dir = True
    else:
        args.have_full_sim_dir = False

    print(f"Processing {len(args.sims)} simulations")

    for isim in args.sims:
        process_sim(isim, args)


def get_args(argv=None):
    """Parse input arguments."""

    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")

    parser.add_argument('sims', nargs='+',
                        help='Simulation index/name to analyse ("all" to '
                             'process all simulations in base_dir.')
    parser.add_argument('--base_dir', default=local.BASE_DIR,
                        help='Base directory of the simulation, default: '
                             f'{local.BASE_DIR}')
    parser.add_argument('--dmo_sim',
    	                help='Directory of the DM-only simulation to match '
    	                     'to. This must be identical for all sims '
    	                     'processed here.')
    parser.add_argument('--snapshots', type=int, nargs='+',
    	                help='Snapshots to match.')
    parser.add_argument('--vr_name', default='vr',
    	                help='Name prefix of VR catalogue (default: "vr").')

    return parser.parse_args(argv)


def process_sim(isim, args):
	"""Process one simulation."""
	if args.have_full_sim_dir:
        args.wdir = isim
    else:
        args.wdir = xl.get_sim_dir(args.base_dir, isim)
    args.dmo_dir = xl.get_sim_dir(args.base_dir, args.dmo_sim)
    print(f"Matching simulation {args.wdir} to DMO sim {args.dmo_dir}...")

    for isnap in args.snapshots:
    	process_snapshot(isim, isnap, args)


def process_snapshot(isim, isnap, args):
	"""Process one snapshot."""
	print(f"Processing snapshot {isnap}...")

    snap_this = Snapshot(f'{args.wdir}{args.vr_name}_{isnap:04d}')
	snap_dmo = Snapshot(f'{args.dmo_dir}{args.vr_name}_{isnap:04d}')

	# Find the best match in the DMO snapshot for each halo in this sim
	match_in_dmo = match_haloes(snap_this, snap_dmo)

	# ... and do the same in reverse
	match_in_this = match_haloes(snap_dmo, snap_this)

	# Reject matches that are not bijective
	ind_non_bijective = np.nonzero((match_in_dmo < 0) |
		                           (match_in_this[match_in_dmo] !=
		                            np.arange(snap_this.n_haloes)))[0]
	match_in_dmo[ind_non_bijective] = -1

	# Write out results
	vr_file_this = f'{args.wdir}{args.vr_name}_{isnap:04d}.hdf5'
	vr_file_dmo = f'{args.dmo_dir}{args.vr_name}_{isnap:04d}.hdf5'

	hd.write_data(vr_file_this, 'MatchInDMO/Haloes', match_in_dmo)

	for iset in ['M200crit', 'Masses', 'MaximumCircularVelocities']:
		data_dmo = hd.read_data(vr_file_dmo, iset)
		hd.write_data(vr_file_this, f'MatchInDMO/{iset}')


def match_haloes(snap_a, snap_b, gate_ab=None):
	"""Core matching function, returns halo matches in another sim.

	It analyses the haloes in simulation A, and finds the matching haloes
	in simulation B (if any).

	Parameters
	----------
	snap_a : snapshot class
	    Class representing snapshot A.
	snap_b : snapshot class
	    Class representing snapshot B.
	gate_ab : gate instance
	    Gate linking snaps A and B. If None (default), it is created
	    internally.

	Returns
	-------
	match_in_b : ndarray (int)
	    Matching indices in B for each halo in A.
	gate_ab : gate instance
	    (Updated) gate linking A and B.
	"""

	# Step 1: Set up the Gate if not done already.
	if gate_ab is None:
		gate_ab = hx.Gate(snap_a.ids, snap_b.ids)

	for ihalo_a in range(snap_a.n_haloes):
		
		# Get IDs for current halo in A
		inds_a = snap_a.get_halo_ids(ihalo_a, ptype='DM')

		# Get haloes of these IDs in B
		inds_in_b = gate_ab.in_int(inds_a)
		haloes_in_b, in_halo_in_b = snap_b.ind_to_halo(inds_in_b)

		# Check if most common halo accounts for >= 1/2 of IDs
		if len(ind_halo_in_b) >= (0.5 * len(inds_a)):
			match_hist = np.bincount(haloes_in_b[in_halo_in_b])
			best_match = np.argmax(match_hist)

			if match_hist[best_match] >= len(inds_a) / 2:
				match_in_b[ihalo_a] = best_match

	return match_in_b, gate_ab


class Snapshot:
	"""Class to represent (the relevant info of) a snapshot.

	Parameters
	----------
	vr_file : str
	    The name of the VR catalogue of the snapshot to be represented.
	"""

	def __init__(self, vr_file):
		vr_file = f'{vr_file}.hdf5'
	    vr_part_file = f'{vr_file}_particles.hdf5'

		self.ids = hd.read_data(vr_part_file, 'Haloes/IDs')
		self.offsets = hd.read_data(vr_part_file, 'Haloes/Offsets')
		self.lengths = hd.read_data(vr_part_file, 'Haloes/Numbers')
		self.n_haloes = len(self.lengths)

	def get_halo_ids(self, halo, ptype=None):
		"""Retrieve the IDs for one particular halo.

		Parameters
		----------
		halo : int
		    The index of the halo for which to retrieve IDs.
		ptype : str or None, optional
		    If not None (default), retrieve only IDs of the selected type
		    (options: 'DM', 'baryons').
		"""
		halo_ids = self.ids[self.offsets[halo] :
		                    self.offsets[halo] + self.lengths[halo]]

		if ptype is None:
			return halo_ids
		if ptype.lower() == 'dm':
			ind_dm = np.nonzero(halo_ids % 2 == 0)[0]
			return halo_ids[ind_dm]
		if ptype.lower().startswith('baryon'):
			ind_baryon = np.nonzero(halo_ids % 2 == 1)[0]
			return halo_ids[ind_baryon]

		# If we get here, something has gone wrong
		print("Ehm... You should not have gotten here.")
		set_trace()

	def ind_to_halo(self, inds):
		"""Find the haloes to which the input indices belong.

		The function can cope with input indices being lower (higher) than
		the lowest (highest) offset in the internal catalogue; these will
		return a halo value of -1. The same value is returned for indices
		sitting between halo blocks, if the internal ID list is
		non-contiguous.
		"""
		halo_candidate = np.searchsorted(self.offsets, inds, side='right') - 1
		ind_actual = np.nonzero((halo_candidate >= 0) &
			                    (inds < self.offsets[halo_candidate]
			                            + self.lengths[halo_candidate]))[0]

		haloes = np.zeros(len(inds), dtype=int) - 1
		haloes[ind_actual] = halo_candidate[ind_actual]

		return haloes, ind_actual


if __name__ == "__main__":
	main()

