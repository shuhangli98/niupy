import niupy
import numpy as np

mo_spaces = {
    "FROZEN_DOCC": [1, 0, 0, 0],
    "RESTRICTED_DOCC": [1, 0, 0, 1],
    "ACTIVE": [2, 0, 1, 1],
}

eom_dsrg = niupy.EOM_DSRG(
    mo_spaces=mo_spaces,
    nroots=5,
    max_cycle=200,
    max_space=200,
    tol_s=1e-10,
    tol_semi=1e-5,
    method_type="cvs_ee",
    diagonal_type="compute",
)

eom_dsrg.kernel()

# converged 49 162  |r|= 8.17e-06  e= [1.42108547e-14 1.98552461e+01 1.98801361e+01 1.99335244e+01
#  1.99673602e+01]  max|de|= -3.34e-11
# All EOM-DSRG roots converged.
# =====================================================================================
#                                   EOM-DSRG summary
# -------------------------------------------------------------------------------------
# Root  Energy (eV)          f                    Symmetry   Spin
# -------------------------------------------------------------------------------------
# 1     0.0000000000         0.00000000           A1         Singlet
# 2     540.2887714864       0.00000000           A1         Triplet
# 3     540.9660608402       0.01991207           A1         Singlet
# 4     542.4188308967       0.00000000           B2         Triplet
# 5     543.3395502401       0.04335803           B2         Singlet
# =====================================================================================
