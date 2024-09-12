import forte
import psi4
import math
import forte.utils

mol = psi4.geometry(f"""
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
""")

psi4.set_options({
    'basis': '6-31g',
    'scf_type': 'direct',
    'e_convergence': 12,
    'reference': 'rhf',
})

e_scf, wfn_scf = psi4.energy('scf', return_wfn=True)

forte_options = {
    'basis': '6-31g',
    'job_type': 'mcscf_two_step',
    'active_space_solver': 'fci',
    'frozen_docc': [0, 0, 0, 0],
    'restricted_docc': [2, 0, 0, 1],
    'active': [2, 0, 1, 1],
    'root_sym': 0,
    'maxiter': 100,
    'e_convergence': 1e-8,
    'r_convergence': 1e-8,
    'casscf_e_convergence': 1e-8,
    'casscf_g_convergence': 1e-6,
}

E_casscf, wfn_cas = psi4.energy(
    'forte', forte_options=forte_options, return_wfn=True, ref_wfn=wfn_scf)

print(f'CASSCF Energy = {E_casscf}')

forte_options = {
    'basis': '6-31g',
    'frozen_docc': [0, 0, 0, 0],
    'restricted_docc': [2, 0, 0, 1],
    'active': [2, 0, 1, 1],
    'root_sym': 0,
    'job_type': 'newdriver',
    'FULL_HBAR': True,
    'active_space_solver': 'fci',
    'correlation_solver': 'MRDSRG',
    'CORR_LEVEL': 'LDSRG2',
    'DSRG_S': 0.5,
    'RELAX_REF': 'None',
}

E_dsrg = psi4.energy('forte', forte_options=forte_options, ref_wfn=wfn_cas)

# mo_spaces = {'frozen_docc': [1, 0, 0, 0], 'RESTRICTED_DOCC': [1, 0, 0, 1], 'ACTIVE': [2, 0, 1, 1]}
# res = forte.utils.prepare_forte_objects(wfn_cas, mo_spaces)
# mo_space_info = res['mo_space_info']
# core_sym = mo_space_info.symmetry('FROZEN_DOCC')
# occ_sym = mo_space_info.symmetry('RESTRICTED_DOCC')
# act_sym = mo_space_info.symmetry('ACTIVE')
# vir_sym = mo_space_info.symmetry('VIRTUAL')
# print(f"Core Symmetry: {core_sym}")
# print(f"Occ Symmetry: {occ_sym}")
# print(f"Act Symmetry: {act_sym}")
# print(f"Vir Symmetry: {vir_sym}")
