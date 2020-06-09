import glob
import os

BASE_DIR = '/cosma7/data/dp004/dc-bahe1/EXL/'

def get_sim_dir(base_dir, isim):
    """Construct the directory of a simulation.

    Parameters
    ----------
    base_dir : str
        The base directory of the simulation family
    isim : str or int
        The simulation index (if beginning with 'ID[xxx]') or name.
        It is assumed that the simulation lives within [base_dir].

    Returns
    -------
    sim_dir : str
        The full directory of the specified simulation.
    """

    if isinstance(isim, int):

        # Case A: simulations labelled ID[xxx]

        dirs = glob.glob(f'{base_dir}/ID{isim}*/')

        # Make sure nothing stupid has happened
        if len(dirs) != 1:
            print(f"Could not unambiguously find directory for simulation {args.sim}!")
            set_trace()

        wdir = dirs[0]

    else:

        # Case B: simulations labelled with string name
        wdir = f'{base_dir}/{isim}'
        
    if not wdir.endswith('/'):
        wdir = wdir + '/'

    return wdir


def get_all_sims(base_dir):
    """Find all simulations in a directory."""

    candidates = glob.glob(f'{base_dir}/*')
    simulations = []
    
    for icand in candidates:
        if os.path.isdir(icand):
            if not icand.endswith('/'):
                icand = icand + '/'
            simulations.append(icand)

    return simulations

