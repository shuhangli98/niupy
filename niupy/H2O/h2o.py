import niupy
import numpy as np

# Unrelaxed
eom_dsrg = niupy.EOM_DSRG(opt_einsum=False, nroots=3, verbose=5, max_cycle=100, diag_shift=0.0,
                          target_sym=0, method_type='cvs_ee', S_12_type='compute', diagonal_type='block')
conv, e, u, spin, osc_strength = eom_dsrg.kernel()
for idx, i_e in enumerate(e):
    if idx == 0:
        print(f"Root {idx}: {i_e - e[0]} Hartree, spin: {spin[idx]}")
    else:
        print(f"Root {idx}: {i_e - e[0]} Hartree, spin: {spin[idx]}, osc_strength: {osc_strength[idx-1]}")
