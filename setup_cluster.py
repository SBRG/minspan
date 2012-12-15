import os
from IPython.parallel import Client
from time import sleep

minspan_dir = os.path.abspath(os.path.dirname(__file__))
profile_minspan_dir = os.path.join(minspan_dir, "profile_minspan")
profile_minspan_mpi_dir = os.path.join(minspan_dir, "profile_minspan_mpi")

def start_engines(mpi=False, verbose=False):
    profile = profile_minspan_mpi_dir if mpi else profile_minspan_dir
    silence = "" if verbose else "2>/dev/null"
    os.system('ipcluster start --profile-dir="%s" --daemonize %s' %
              (profile, silence))
    return get_client_and_view(verbose=verbose)

def get_client_and_view(verbose=True):
    c = None
    v = None
    while c is None:
        try:
            c = Client(profile_dir=profile_minspan_dir)
        except:
            sleep(5)
    while len(c.ids) == 0:
        if verbose:
            print "waiting for connections"
        sleep(5)
    while v is None:
        try:
            v = c.direct_view()
        except Exception, e:
            print e
            sleep(5)
    return c, v


def stop_engines(mpi=False, verbose=False):
    profile = profile_minspan_mpi_dir if mpi else profile_minspan_dir
    silence = "" if verbose else "2>/dev/null"
    os.system('ipcluster stop --profile-dir="%s" %s' % (profile, silence))

if __name__ == "__main__":
    stop_engines()
    c, v = start_engines()
    print "%d engines connected" % (len(c.ids))
    from minspan import *
    from cobra.io import load_matlab_model
    from time import time
    model = load_matlab_model("testing_models.mat", "inj661")
    S = model.to_array_based_model().S
    start = time()
    solved_fluxes = minspan(model, cores=1, verbose=True, mapper=v.map_sync, starting_fluxes="auto")
    print "solved in %.2f seconds" % (time() - start)
    print "nnz", nnz(solved_fluxes)
    print "rank", matrix_rank(solved_fluxes)
    print "max(S * v) =", abs(S * solved_fluxes).max()
    
    #from IPython import embed; embed()
    stop_engines()
