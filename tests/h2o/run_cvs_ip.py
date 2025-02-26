import niupy
import numpy as np

mo_spaces = {
    "FROZEN_DOCC": [1, 0, 0, 0],
    "RESTRICTED_DOCC": [1, 0, 0, 1],
    "ACTIVE": [2, 0, 1, 1],
}

eom_dsrg = niupy.EOM_DSRG(
    mo_spaces=mo_spaces,
    nroots=10,
    max_cycle=200,
    max_space=200,
    tol_s=1e-10,
    tol_semi=1e-10,
    method_type="cvs_ip",
    diagonal_type="compute",
)

eom_dsrg.kernel_full()

# =====================================================================================
#                                   EOM-DSRG summary
# -------------------------------------------------------------------------------------
# Root  Energy (eV)          P                    Symmetry   Spin
# -------------------------------------------------------------------------------------
# 1     545.9415248560       0.80880708           A1         Doublet
# 2     571.7665663116       0.00000000           B1         Quartet
# 3     572.0315519879       0.00000000           A2         Doublet
# 4     572.2423427208       0.00000000           B2         Quartet
# 5     573.1372280107       0.00000000           B1         Quartet
# 6     574.1792207528       0.00000000           A2         Doublet
# 7     574.2708362695       0.00000000           A1         Quartet
# 8     574.4478900269       0.00000000           B1         Quartet
# 9     574.5615314965       0.00567206           A1         Doublet
# 10    574.9812037551       0.00000000           B2         Quartet
# =====================================================================================
