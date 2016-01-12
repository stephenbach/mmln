from unittest import TestCase

import mmln
import mmln.infer


class TestInference(TestCase):

    def test_infer(self):
        for inf in self._get_inference_methods():
            v1 = mmln.infer.Variable()
            v2 = mmln.infer.Variable()
            v3 = mmln.infer.Variable()
            v4 = mmln.infer.Variable()
            v5 = mmln.infer.Variable()

            inf.set_potential(1, 1, v1, -0.5, two_sided=True, squared=True)
            inf.set_potential(1, 1, v2, -0.5, two_sided=True, squared=True)
            inf.set_potential(1, 1, v3, -0.5, two_sided=True, squared=True)
            inf.set_potential(1, 1, v4, -0.5, two_sided=True, squared=True)
            inf.set_potential(1, 1, v5, -0.5, two_sided=True, squared=True)

            inf.set_potential(5, -1, v1, 1, squared=True)
            inf.set_potential(5, -1, v2, 1, squared=True)

            inf.set_potential(2, (1, -1), (v1, v3), 0, squared=True)
            inf.set_potential(2, (1, -1), (v3, v1), 0, squared=True)
            inf.set_potential(2, (1, -1), (v2, v3), 0, squared=True)
            inf.set_potential(2, (1, -1), (v3, v2), 0, squared=True)
            inf.set_potential(2, (1, -1), (v3, v4), 0, squared=True)
            inf.set_potential(2, (1, -1), (v4, v3), 0, squared=True)
            inf.set_potential(2, (1, -1), (v3, v5), 0, squared=True)
            inf.set_potential(2, (1, -1), (v5, v3), 0, squared=True)

            inf.set_potential(5, 1, v4, 0, squared=True)
            inf.set_potential(2, 1, v5, 0, squared=True)

            inf.infer()

            self.assertAlmostEqual(v1.value, .821, 3)
            self.assertAlmostEqual(v2.value, .821, 3)
            self.assertAlmostEqual(v3.value, .534, 3)
            self.assertAlmostEqual(v4.value, .196, 3)
            self.assertAlmostEqual(v5.value, .313, 3)

    def _get_inference_methods(self):
        return {self._get_hlmrf()}

    def _get_hlmrf(self):
        return mmln.infer.HLMRF(max_iter=1000)


class TestHLMRF(TestCase):

    def test_admm_one_var_bowl_potential(self):
        var = mmln.infer.Variable()

        var.value = 0.9
        pot = mmln.infer._ADMMOneVarBowlPotential(mmln.infer.HLMRF(), 9, 1.0, var, 0.2)
        pot.lagrange = 3.0
        pot.optimize_local_copies()
        # self.assertAlmostEqual(pot.local_copy, 0.05)
        # todo

    def test_admm_one_var_hinge_potential(self):
        var = mmln.infer.Variable()

        var.value = 0.1
        pot = mmln.infer._ADMMOneVarHingePotential(mmln.infer.HLMRF(), 2.0, 1.0, var, 0.0)
        pot.lagrange = -0.15
        pot.optimize_local_copies()
        self.assertAlmostEqual(pot.local_copy, 0.05)

    def test_admm_two_var_hinge_potential(self):
        var1 = mmln.infer.Variable()
        var2 = mmln.infer.Variable()

        var1.value = 0.7
        var2.value = 0.5

        pot = mmln.infer._ADMMTwoVarHingePotential(mmln.infer.HLMRF(), 1.0, 1.0, var1, -1.0, var2, 0.0)
        pot.lagrange1 = 0.0
        pot.lagrange2 = 0.0
        pot.optimize_local_copies()
        self.assertAlmostEqual(pot.local_copy1, 0.62)
        self.assertAlmostEqual(pot.local_copy2, 0.58)
