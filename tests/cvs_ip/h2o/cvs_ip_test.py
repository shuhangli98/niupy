import niupy
import unittest

mo_spaces = {
    "FROZEN_DOCC": [1, 0, 0, 0],
    "RESTRICTED_DOCC": [1, 0, 0, 1],
    "ACTIVE": [2, 0, 1, 1],
}

eom_dsrg = niupy.EOM_DSRG(
    opt_einsum=True,
    mo_spaces=mo_spaces,
    nroots=10,
    basis_per_root=20,
    collapse_per_root=2,
    max_cycle=200,
    tol_s=1e-10,
    tol_semi=1e-10,
    method_type="cvs_ip",
)


class KnownValues(unittest.TestCase):

    def test_niupy_full(self):

        eom_dsrg.kernel_full()

        e = eom_dsrg.evals
        p = eom_dsrg.spec_info

        self.assertAlmostEqual(e[0], 545.9415248558, 4)
        self.assertAlmostEqual(e[1], 571.7665663112, 4)
        self.assertAlmostEqual(e[2], 572.0315519843, 4)
        self.assertAlmostEqual(e[3], 572.2423427205, 4)
        self.assertAlmostEqual(e[8], 574.5615314963, 4)

        self.assertAlmostEqual(p[0], 1.61761416, 4)
        self.assertAlmostEqual(p[1], 0.00000000, 4)
        self.assertAlmostEqual(p[2], 0.00000000, 4)
        self.assertAlmostEqual(p[3], 0.00000000, 4)
        self.assertAlmostEqual(p[8], 0.01134412, 4)

    def test_niupy(self):

        eom_dsrg.kernel()

        e = eom_dsrg.evals
        p = eom_dsrg.spec_info

        self.assertAlmostEqual(e[0], 545.9415248558, 4)
        self.assertAlmostEqual(e[1], 571.7665663109, 4)
        self.assertAlmostEqual(e[2], 572.0315519842, 4)
        self.assertAlmostEqual(e[3], 572.2423427206, 4)
        self.assertAlmostEqual(e[8], 574.5615314963, 4)

        self.assertAlmostEqual(p[0], 1.61761416, 4)
        self.assertAlmostEqual(p[1], 0.00000000, 4)
        self.assertAlmostEqual(p[2], 0.00000000, 4)
        self.assertAlmostEqual(p[3], 0.00000000, 4)
        self.assertAlmostEqual(p[8], 0.01134412, 4)


if __name__ == "__main__":
    print("CVS-IP calculations")
    unittest.main()
