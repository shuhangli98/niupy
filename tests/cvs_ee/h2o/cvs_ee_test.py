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
    nroots=4,
    basis_per_root=50,
    collapse_per_root=2,
    max_cycle=200,
    tol_s=1e-10,
    tol_semi=1e-5,
    method_type="cvs_ee",
)


class KnownValues(unittest.TestCase):

    def test_niupy_full(self):

        eom_dsrg.kernel_full()

        e = eom_dsrg.evals
        p = eom_dsrg.spec_info

        self.assertAlmostEqual(e[0], 540.2887714820, 4)
        self.assertAlmostEqual(e[1], 540.9660608351, 4)
        self.assertAlmostEqual(e[2], 542.4188308903, 4)
        self.assertAlmostEqual(e[3], 543.3395502334, 4)

        self.assertAlmostEqual(p[0], 0.00000000, 4)
        self.assertAlmostEqual(p[1], 0.01996767, 4)
        self.assertAlmostEqual(p[2], 0.00000000, 4)
        self.assertAlmostEqual(p[3], 0.04296089, 4)

    def test_niupy(self):

        eom_dsrg.kernel()

        e = eom_dsrg.evals
        p = eom_dsrg.spec_info

        self.assertAlmostEqual(e[0], 540.2887714846, 4)
        self.assertAlmostEqual(e[1], 540.9660608382, 4)
        self.assertAlmostEqual(e[2], 542.4188308949, 4)
        self.assertAlmostEqual(e[3], 543.3395502382, 4)

        self.assertAlmostEqual(p[0], 0.00000000, 4)
        self.assertAlmostEqual(p[1], 0.01991209, 4)
        self.assertAlmostEqual(p[2], 0.00000000, 4)
        self.assertAlmostEqual(p[3], 0.04335803, 4)


if __name__ == "__main__":
    print("CVS-EE calculations")
    unittest.main()
