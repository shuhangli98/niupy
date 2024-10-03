import niupy
import psi4
import forte
import forte.utils
import numpy as np

x = 3.000
mol = psi4.geometry(f"""
Be 0.0   0.0             0.0
H  {x}   {2.54-0.46*x}   0.0
H  {x}  -{2.54-0.46*x}   0.0
symmetry c2v
units bohr
""")

psi4.set_options({
    'basis': 'sto-6g',
    'reference': 'rhf',
})

forte_options = {
    'basis': 'sto-6g',
    'job_type': 'mcscf_two_step',
    'active_space_solver': 'genci',
    'restricted_docc': [2, 0, 0, 0],
    'active': [1, 0, 0, 1],
    'root_sym': 0,
    'maxiter': 100,
    'e_convergence': 1e-8,
    'r_convergence': 1e-8,
    'casscf_e_convergence': 1e-8,
    'casscf_g_convergence': 1e-6,
}

E_casscf, wfn_cas = psi4.energy(
    'forte', forte_options=forte_options, return_wfn=True)

print(f'CASSCF Energy = {E_casscf}')

forte_options = {
    'basis': 'sto-6g',
    'restricted_docc': [2, 0, 0, 0],
    'active': [1, 0, 0, 1],
    'root_sym': 0,
    'job_type': 'newdriver',
    'FULL_HBAR': True,
    'active_space_solver': 'genci',
    'correlation_solver': 'MRDSRG',
    'CORR_LEVEL': 'LDSRG2',
    'DSRG_S': 0.5,
    'RELAX_REF': 'NONE',
    'FOURPDC': 'MK',
}

E_dsrg = psi4.energy('forte', forte_options=forte_options, ref_wfn=wfn_cas)
