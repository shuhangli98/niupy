import niupy
import unittest

eom_dsrg = niupy.EOM_DSRG(
    opt_einsum=True,
    nroots=10,
    basis_per_root=20,
    collapse_per_root=2,
    max_cycle=200,
    tol_s=1e-10,
    tol_semi=1e-10,
    method_type="ip",
)


class KnownValues(unittest.TestCase):

    def test_niupy_full(self):

        eom_dsrg.kernel_full()

        e = eom_dsrg.evals
        p = eom_dsrg.spec_info

        self.assertAlmostEqual(e[0], 11.0712299825, 4)
        self.assertAlmostEqual(e[1], 12.9096937411, 4)
        self.assertAlmostEqual(e[2], 17.2339756217, 4)
        self.assertAlmostEqual(e[3], 17.3971330715, 4)

        self.assertAlmostEqual(p[0], 1.97180714, 4)
        self.assertAlmostEqual(p[1], 1.95473561, 4)
        self.assertAlmostEqual(p[2], 0.00000000, 4)
        self.assertAlmostEqual(p[3], 0.00213609, 4)

    def test_niupy(self):

        eom_dsrg.kernel()

        e = eom_dsrg.evals
        p = eom_dsrg.spec_info

        self.assertAlmostEqual(e[0], 11.0712299826, 4)
        self.assertAlmostEqual(e[1], 12.9096937411, 4)
        self.assertAlmostEqual(e[2], 17.2339756217, 4)
        self.assertAlmostEqual(e[3], 17.3971330810, 4)

        self.assertAlmostEqual(p[0], 1.97180714, 4)
        self.assertAlmostEqual(p[1], 1.95473561, 4)
        self.assertAlmostEqual(p[2], 0.00000000, 4)
        self.assertAlmostEqual(p[3], 0.00213609, 4)


if __name__ == "__main__":
    print("IP calculations")
    unittest.main()
