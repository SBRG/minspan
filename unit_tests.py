from minspan import *

from unittest import TestCase, TestLoader, TextTestRunner, skipIf
import sys

import cobra
from cobra.io import load_matlab_model

from multiprocessing import Pool

testing_models_filepath = "testing_models.mat"  # todo absolute path

def test_ecoli_core_model(testcase, solved_fluxes):
    testcase.assertEqual(matrix_rank(solved_fluxes), 23)
    testcase.assertEqual(nnz(solved_fluxes), 479)
    testcase.assertAlmostEqual(abs(testcase.S * solved_fluxes).max(), 0)

class TestMinspanEcoliCore(TestCase):
    def setUp(self):
        self.model = load_matlab_model(testing_models_filepath, "ecoli_core")
        self.S = self.model.to_array_based_model().S

    @skipIf("gurobi" not in cobra.solvers.solver_dict, "gurobi required")
    def test_EcoliCore_gurobi(self):
        solved_fluxes = minspan(self.model, cores=4, timelimit=30,
            verbose=False, solver_name="gurobi")
        test_ecoli_core_model(self, solved_fluxes)

    @skipIf("cplex" not in cobra.solvers.solver_dict, "cplex required")
    def test_EcoliCore_cplex(self):
        solved_fluxes = minspan(self.model, cores=4, timelimit=30,
            verbose=False, solver_name="cplex")
        test_ecoli_core_model(self, solved_fluxes)

    @skipIf("gurobi" not in cobra.solvers.solver_dict, "gurobi required")
    def test_EcoliCore_gurobi_multiprocessing(self):
        pool = Pool()
        solved_fluxes = minspan(self.model, cores=1, timelimit=30,
            verbose=False, solver_name="gurobi", mapper=pool.map)
        test_ecoli_core_model(self, solved_fluxes)

    @skipIf("cplex" not in cobra.solvers.solver_dict, "cplex required")
    def test_EcoliCore_cplex_multiprocessing(self):
        pool = Pool()
        solved_fluxes = minspan(self.model, cores=2, timelimit=45,
            verbose=False, solver_name="cplex", mapper=pool.map)
        test_ecoli_core_model(self, solved_fluxes)

class TestMathFunctions(TestCase):
    def test_nnz(self):
        S = load_matlab_model(testing_models_filepath, "ecoli_core").to_array_based_model().S
        self.assertEqual(nnz(S), 309) # lil sparse matrix
        self.assertEqual(nnz(S.tocsc()), 309) # csc sparse matrix
        self.assertEqual(nnz(S.todok()), 309) # dok sparse matrix
        self.assertEqual(nnz(S.todense()), 309) # matrix
        self.assertEqual(nnz(array(S.todense())), 309) # ndarray

        self.assertEqual(nnz(array([0])), 0)
        self.assertEqual(nnz(array([0, 1])), 1)
        self.assertEqual(nnz(array([[0, 0], [0, 0]])), 0)
        self.assertEqual(nnz(array([[0, 1], [0, 0]])), 1)
        self.assertEqual(nnz(array([[1, 0], [0, 1]])), 2)

    def test_scaling(self):
        S = matrix("0 0; 0 0")
        lb = array([0, 0])
        ub = array([100, 100])
        scaled = scale_vector(array([0.9, 2.25000001]), S, lb, ub)
        self.assertTrue((scaled == array((2., 5.))).all())
        test_matrix = array([[0.9, 1.1], [2.499999999, 3.111]])
        scale_matrix(test_matrix, S, lb, ub)
        error = (array([[9, 1], [25, 2.82818182]]) - test_matrix).max()
        self.assertAlmostEqual(error, 0)

    def test_null(self):
        S = load_matlab_model(testing_models_filepath, "ecoli_core").to_array_based_model().S.todense()
        N = null(S)
        self.assertEqual(N.shape, (84, 23))
        self.assertAlmostEqual(abs(S * N).max(), 0)

if __name__ == "__main__":
    import sys
    loader = TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    TextTestRunner(verbosity=2).run(suite)
