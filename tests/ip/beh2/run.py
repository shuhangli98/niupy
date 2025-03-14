import psi4
import forte, forte.utils
import niupy
import numpy as np

eom_dsrg = niupy.EOM_DSRG(
    opt_einsum=False,
    nroots=5,
    max_cycle=200,
    max_space=200,
    tol_s=1e-10,
    tol_semi=1e-5,
    method_type="ip",
    diagonal_type="compute",
    verbose=5,
)
eom_dsrg.kernel()

# =====================================================================================
#                                   EOM-DSRG summary
# -------------------------------------------------------------------------------------
# Root  Energy (eV)          P                    Symmetry   Spin
# -------------------------------------------------------------------------------------
# 1     11.0712299825        1.97176273           B2         Doublet
# 2     12.9096937411        1.95473542           A1         Doublet
# 3     17.2339756217        0.00000000           B2         Doublet
# 4     17.3971330715        0.00213352           A1         Doublet
# 5     17.8104691260        0.00000000           A2         Quartet
# =====================================================================================
