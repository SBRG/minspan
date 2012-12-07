minspan
=======

This project calculates the minimal spanning vectors of the null space.

It is written in python, and requires [cobrapy](https://github.com/opencobra/cobrapy) with a suitable solver, such as
* [gurobi](http://www.gurobi.com/)
* [cplex](http://www-01.ibm.com/software/integration/optimization/cplex-optimizer/)

To calculate minspan, load a model and run the minspan function.

```python
from cobra.io import load_matlab_model
from minspan import minspan, nnz
model = load_matlab_model("testing_models.mat", "ecoli_core")
solved_fluxes = minspan(model, cores=1, verbose=True)
print "nnz", nnz(solved_fluxes)
```

Documentation for the arguments to the minspan function are containted in the docstring.