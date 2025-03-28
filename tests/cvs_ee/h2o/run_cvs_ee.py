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
    nroots=4,
    max_cycle=200,
    max_space=200,
    tol_s=1e-10,
    tol_semi=1e-5,
    method_type="cvs_ee",
    diagonal_type="compute",
    first_row=False,
)

eom_dsrg.kernel()

# First_row False
# kernel_full
# [19.85524613 19.88013606 19.93352437 19.96736018]
# Root 0: [(np.float64(4.409755545679286), (np.int64(0), np.int64(1), np.int64(2), np.int64(3)), 'iAaV'), (np.float64(-4.409755489970687), (np.int64(1), np.int64(0), np.int64(3), np.int64(2)), 'aIvA'), (np.float64(-3.871468633112351), (np.int64(1), np.int64(0), np.int64(2), np.int64(3)), 'aIaV')]
# Root 1: [(np.float64(4.8900972106367275), (np.int64(0), np.int64(1), np.int64(2), np.int64(3)), 'iAaV'), (np.float64(4.890097150616002), (np.int64(1), np.int64(0), np.int64(3), np.int64(2)), 'aIvA'), (np.float64(4.377352916004487), (np.int64(1), np.int64(0), np.int64(2), np.int64(3)), 'aIaV')]
# Root 2: [(np.float64(-23.402089723548475), (np.int64(3), np.int64(0), np.int64(3), np.int64(2)), 'aIvA'), (np.float64(23.40208966929598), (np.int64(0), np.int64(3), np.int64(2), np.int64(3)), 'iAaV'), (np.float64(-20.86313857954252), (np.int64(3), np.int64(0), np.int64(2), np.int64(3)), 'aIaV')]
# Root 3: [(np.float64(22.335084750075154), (np.int64(3), np.int64(0), np.int64(3), np.int64(2)), 'aIvA'), (np.float64(22.335084701413244), (np.int64(0), np.int64(3), np.int64(2), np.int64(3)), 'iAaV'), (np.float64(19.6181941145706), (np.int64(0), np.int64(3), np.int64(3), np.int64(2)), 'iAvA')]
# =====================================================================================
#                                   EOM-DSRG summary
# -------------------------------------------------------------------------------------
# Root  Energy (eV)          f                    Symmetry   Spin
# -------------------------------------------------------------------------------------
# 1     540.2887714820       0.00000000           A1         Triplet
# 2     540.9660608351       0.01996767           A1         Singlet
# 3     542.4188308903       0.00000000           B2         Triplet
# 4     543.3395502334       0.04296089           B2         Singlet
# =====================================================================================
# kernel
# length of precond: 1614
# Time(s) for Davidson Setup:  1.914297103881836
# davidson 0 4  |r|=  1.2  e= [20.51820154 20.62848812 20.67845914 20.68319994]  max|de|= 20.7  lindep=    1
# davidson 1 8  |r|= 0.721  e= [20.00871691 20.06714944 20.07252843 20.10146441]  max|de|= -0.606  lindep= 0.979
# davidson 2 12  |r|= 0.206  e= [19.87215554 19.89966032 19.95639158 19.99033309]  max|de|= -0.167  lindep= 0.948
# davidson 3 16  |r|= 0.0918  e= [19.85702132 19.88208598 19.93645779 19.97029271]  max|de|= -0.02  lindep= 0.987
# davidson 4 20  |r|= 0.0228  e= [19.85541386 19.88035617 19.93374988 19.96761641]  max|de|= -0.00271  lindep= 0.983
# davidson 5 24  |r|= 0.00979  e= [19.85527456 19.88017251 19.93355891 19.96739055]  max|de|= -0.000226  lindep= 0.981
# davidson 6 28  |r|= 0.00363  e= [19.85525057 19.88014145 19.9335301  19.96736439]  max|de|= -3.11e-05  lindep= 0.989
# davidson 7 32  |r|= 0.00138  e= [19.85524681 19.88013676 19.93352507 19.96736083]  max|de|= -5.03e-06  lindep= 0.995
# davidson 8 36  |r|= 0.000527  e= [19.85524624 19.88013614 19.93352445 19.96736027]  max|de|= -6.21e-07  lindep= 0.982
# davidson 9 40  |r|= 0.000195  e= [19.85524615 19.88013607 19.93352438 19.96736019]  max|de|= -9.26e-08  lindep= 0.989
# davidson 10 44  |r|= 7.53e-05  e= [19.85524613 19.88013606 19.93352437 19.96736018]  max|de|= -1.43e-08  lindep= 0.987
# davidson 11 48  |r|= 3.05e-05  e= [19.85524613 19.88013606 19.93352437 19.96736018]  max|de|= -1.95e-09  lindep= 0.99
# root 1 converged  |r|= 7.93e-06  e= 19.880136055681895  max|de|= -1.93e-10
# root 2 converged  |r|= 5.66e-06  e= 19.933524370771693  max|de|= -1.15e-10
# root 3 converged  |r|= 5.95e-06  e= 19.96736018256316  max|de|= -9.63e-11
# davidson 12 52  |r|= 1.16e-05  e= [19.85524613 19.88013606 19.93352437 19.96736018]  max|de|= -3.45e-10  lindep= 0.99
# root 0 converged  |r|= 3.61e-06  e= 19.855246131035596  max|de|= -4.42e-11
# converged 13 53  |r|= 7.91e-06  e= [19.85524613 19.88013606 19.93352437 19.96736018]  max|de|= -4.42e-11
# All EOM-DSRG roots converged.
# Root 0: [(np.float64(-4.409871123420219), (np.int64(1), np.int64(0), np.int64(3), np.int64(2)), 'aIvA'), (np.float64(4.409804741709669), (np.int64(0), np.int64(1), np.int64(2), np.int64(3)), 'iAaV'), (np.float64(3.8714563445824166), (np.int64(0), np.int64(1), np.int64(3), np.int64(2)), 'iAvA')]
# Root 1: [(np.float64(-4.890004263638869), (np.int64(1), np.int64(0), np.int64(3), np.int64(2)), 'aIvA'), (np.float64(-4.889905572990854), (np.int64(0), np.int64(1), np.int64(2), np.int64(3)), 'iAaV'), (np.float64(-4.377180359936779), (np.int64(1), np.int64(0), np.int64(2), np.int64(3)), 'aIaV')]
# Root 2: [(np.float64(-23.40201530811817), (np.int64(0), np.int64(3), np.int64(2), np.int64(3)), 'iAaV'), (np.float64(23.40192830610863), (np.int64(3), np.int64(0), np.int64(3), np.int64(2)), 'aIvA'), (np.float64(20.86295940808126), (np.int64(3), np.int64(0), np.int64(2), np.int64(3)), 'aIaV')]
# Root 3: [(np.float64(-22.335020285625163), (np.int64(0), np.int64(3), np.int64(2), np.int64(3)), 'iAaV'), (np.float64(-22.335013802709327), (np.int64(3), np.int64(0), np.int64(3), np.int64(2)), 'aIvA'), (np.float64(-19.6180238487215), (np.int64(3), np.int64(0), np.int64(2), np.int64(3)), 'aIaV')]
# =====================================================================================
#                                   EOM-DSRG summary
# -------------------------------------------------------------------------------------
# Root  Energy (eV)          f                    Symmetry   Spin
# -------------------------------------------------------------------------------------
# 1     540.2887714806       0.00000000           A1         Triplet
# 2     540.9660608338       0.01991216           A1         Singlet
# 3     542.4188308967       0.00000000           B2         Triplet
# 4     543.3395502403       0.04335809           B2         Singlet
# =====================================================================================
