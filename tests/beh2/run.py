import psi4
import forte, forte.utils
import niupy
import numpy as np

eom_dsrg = niupy.EOM_DSRG(
    opt_einsum=False,
    nroots=5,
    max_cycle=200,
    max_space=200,
    tol_s=1e-5,
    tol_semi=1e-5,
    method_type="ip",
    diagonal_type="compute",
    verbose = 5,
)
eom_dsrg.kernel_full()
