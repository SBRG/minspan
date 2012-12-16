c = get_config()
c.IPClusterStart.controller_launcher_class = 'MPI'
c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'
c.MPIEngineSetLauncher.mpi_args = ["--pernode"]
