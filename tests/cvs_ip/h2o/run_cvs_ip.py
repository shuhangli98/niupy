import niupy
import numpy as np

mo_spaces = {
    "FROZEN_DOCC": [1, 0, 0, 0],
    "RESTRICTED_DOCC": [1, 0, 0, 1],
    "ACTIVE": [2, 0, 1, 1],
}

eom_dsrg = niupy.EOM_DSRG(
    opt_einsum=True,
    mo_spaces=mo_spaces,
    nroots=10,
    max_cycle=200,
    max_space=200,
    tol_s=1e-10,
    tol_semi=1e-10,
    method_type="cvs_ip",
    diagonal_type="compute",
)

eom_dsrg.kernel()

# =====================================================================================
#                                   EOM-DSRG summary
# -------------------------------------------------------------------------------------
# Root  Energy (eV)          P                    Symmetry   Spin
# -------------------------------------------------------------------------------------
# 1     545.9415248559       1.61761419           A1         Doublet
# 2     571.7665663112       0.00000000           B1         Quartet
# 3     572.0315519839       0.00000000           A2         Quartet
# 4     572.2423427208       0.00000000           B2         Quartet
# 5     573.1372280105       0.00000000           B1         Quartet
# 6     574.1792207497       0.00000000           A2         Doublet
# 7     574.2708362696       0.00000000           A1         Doublet
# 8     574.4478900263       0.00000000           B1         Quartet
# 9     574.5615314966       0.01134412           A1         Doublet
# 10    574.9812037553       0.00000000           B2         Quartet
# =====================================================================================
