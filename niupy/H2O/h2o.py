import niupy
import numpy as np

# Unrelaxed
eom_dsrg = niupy.EOM_DSRG(
    opt_einsum=False,
    nroots=3,
    verbose=5,
    max_cycle=100,
    diag_shift=0.0,
    target_sym=0,
    tol_s=1e-6,
    tol_semi=1e-6,
    method_type="cvs_ee",
    S_12_type="compute",
    diagonal_type="block",
)
conv, e, u, spin, osc_strength = eom_dsrg.kernel()
for idx, i_e in enumerate(e):
    if idx == 0:
        print(f"Root {idx}: {i_e - e[0]} Hartree, spin: {spin[idx]}")
    else:
        print(
            f"Root {idx}: {i_e - e[0]} Hartree, spin: {spin[idx]}, osc_strength: {osc_strength[idx-1]}"
        )

# 1e-4
# Root 0: 0.0 Hartree, spin: Singlet
# Root 1: 19.85461234583798 Hartree, spin: Triplet, osc_strength: 7.249436968879093e-15
# Root 2: 19.879439913696377 Hartree, spin: Singlet, osc_strength: 0.0202853669790235

# 1e-2
# Root 0: 0.0 Hartree, spin: Singlet
# Root 1: 19.86514670365444 Hartree, spin: Triplet, osc_strength: 4.7014998284772014e-18
# Root 2: 19.897151962315075 Hartree, spin: Singlet, osc_strength: 0.01970109089722534
