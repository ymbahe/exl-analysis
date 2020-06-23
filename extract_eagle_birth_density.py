"""Extract stellar birth density distribution from EAGLE sims."""

import numpy as np
import argparse
import hydrangea as hy

from pdb import set_trace

def main():
    
    print("Parsing input arguments...")
    parser = argparse.ArgumentParser(description="Parse input parameters.")
    parser.add_argument('sim', help='Simulation path to analyse.')
    parser.add_argument('--snapshots', type=int, help='Snapshots to analyse',
                        nargs='+')
    parser.add_argument('--sim_name', help='Name under which to store '
                                            'the results.')

    args = parser.parse_args()
    sim = hy.Simulation(run_dir=args.sim)

    if args.snapshots is None:
        args.snapshots = np.arange(3, 29)
    
    out_file = './comparison_data/EAGLE_birth_densities.hdf5'

    aexps = hy.hdf5.read_data(out_file, f'{args.sim_name}/ScaleFactors')
    if aexps is None:
        aexps = np.zeros(29) - 1 
    
    for isnap in args.snapshots:
        snap_file = sim.get_snap_file(isnap)
        stars = hy.SplitFile(snap_file, part_type=4)

        if stars.num_elem == 0:
            print("No stars, skipping snapshot.")
            continue
        else:
            print(f"Expecting {stars.num_elem} stars.")
        
        n_crit = 10.0 * (1.81)**(-1/2)  # in n_H [cm^-3]
        n_H_by_crit = stars.BirthDensity * 0.752 / n_crit
        n_H_by_crit = np.log10(n_H_by_crit)
        
        sorter = np.argsort(n_H_by_crit)
        xquant = n_H_by_crit[sorter]
        yquant = (np.cumsum(stars.InitialMass[sorter].astype(float))
                  / np.sum(stars.InitialMass))

        # Sample evenly in x at 1000 points
        ind_sample = np.linspace(0, len(xquant)-1, num=1000).astype(int)
        ind_sample = np.unique(ind_sample)

        hy.hdf5.write_data(out_file, f'{args.sim_name}/S{isnap}/nH_by_nCrit',
                           xquant[ind_sample])
        hy.hdf5.write_data(out_file, f'{args.sim_name}/S{isnap}/'
                                      'CumulativeMassFraction',
                           yquant[ind_sample])
        hy.hdf5.write_attribute(out_file, f'{args.sim_name}/S{isnap}',
            'ScaleFactor', stars.aexp)
        hy.hdf5.write_attribute(out_file, f'{args.sim_name}/S{isnap}',
            'Redshift', stars.redshift)
        hy.hdf5.write_attribute(out_file, f'{args.sim_name}/S{isnap}',
            'Time', stars.time)

        aexps[isnap] = stars.aexp

    hy.hdf5.write_data(out_file, f'{args.sim_name}/ScaleFactors', aexps)
        
if __name__ == "__main__":
    main()
