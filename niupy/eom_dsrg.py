from niupy.eom_tools import *
import psi4
import forte
import forte.utils
import numpy as np
import niupy.eom_dsrg_compute as eom_dsrg_compute

class EOM_DSRG:
    def __init__(self, Hbar, gamma1, eta1, lambda2, lambda3, diag_shift=0.0, tol_e=1e-8, max_space=100, max_cycle=100, tol_davidson=1e-5, tol_s=1e-4, tol_s_act=1e-4, target_sym=0, target_spin=0, nroots=6, verbose=0, wfn=None, mo_spaces=None, method_type='ee', diagonal_type='exact'):
        # Get MO information
        if wfn is not None and mo_spaces is not None:
            res = forte.utils.prepare_forte_objects(wfn, mo_spaces)
            mo_space_info = res['mo_space_info']
            occ_sym = mo_space_info.symmetry('RESTRICTED_DOCC')
            act_sym = mo_space_info.symmetry('ACTIVE')
            vir_sym = mo_space_info.symmetry('VIRTUAL')
            occ_sym = np.array(occ_sym)
            act_sym = np.array(act_sym)
            vir_sym = np.array(vir_sym)
        else:
            print("Running BeH2/STO-6G since no wfn and mo_spaces are provided.")
            occ_sym = np.array([0, 0])
            act_sym = np.array([0, 3])
            vir_sym = np.array([0, 2, 3])

        print("\n")
        print("  occ_sym: ", occ_sym)
        print("  act_sym: ", act_sym)
        print("  vir_sym: ", vir_sym)

        self.nocc = len(occ_sym)
        self.nact = len(act_sym)
        self.nvir = len(vir_sym)

        self.method_type = method_type
        self.diagonal_type = diagonal_type  # 'exact' or 'block'
        self.verbose = verbose

        self.nroots = nroots              # Number of EOM-DSRG roots requested
        self.max_space = max_space        # Maximum size of the Davidson trial space
        self.max_cycle = max_cycle        # Maximum number of iterations in the Davidson procedure
        self.tol_e = tol_e                # Tolerance for the energy in the Davidson procedure
        self.tol_davidson = tol_davidson  # Tolerance for the residual in the Davidson procedure
        self.tol_s = tol_s                # Tolerance for the orthogonalization of excitation spaces
        self.tol_s_act = tol_s_act        # Tolerance for the orthogonalization of the active space
        self.diag_shift = diag_shift      # Shift for the diagonal of the effective Hamiltonian

        # Get the target symmetry and spin
        self.target_sym = target_sym
        self.target_spin = target_spin

        # Get Hbar and RDMs
        self.Hbar = Hbar
        self.gamma1 = gamma1
        self.eta1 = eta1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # templates
        self.template_c, self.full_template_c = eom_dsrg_compute.get_templates(self)

        # Generate symmetry (dictionary and vector) for excitation operators.
        self.sym = sym_dir(self.full_template_c, occ_sym, act_sym, vir_sym)
        self.sym_vec = dict_to_vec(self.sym, 1).flatten()

    def kernel(self):
        conv, e, u, spin = eom_dsrg_compute.kernel(self)
        return conv, e, u, spin


if __name__ == "__main__":
    test = 1
    if test == 1:
        import os
        import sys
        script_dir = os.path.dirname(__file__)
        rel_path = "BeH2"

        gamma1 = np.load(f'{rel_path}/save_gamma1.npz')
        eta1 = np.load(f'{rel_path}/save_eta1.npz')
        lambda2 = np.load(f'{rel_path}/save_lambda2.npz')
        lambda3 = np.load(f'{rel_path}/save_lambda3.npz')
        Hbar = np.load(f'{rel_path}/save_Hbar.npz')

        eom_dsrg = EOM_DSRG(Hbar, gamma1, eta1, lambda2, lambda3, nroots=3,
                            verbose=5, max_cycle=100, target_sym=0, method_type='ee', diagonal_type='block')
        conv, e, u, spin = eom_dsrg.kernel()
        for idx, i_e in enumerate(e):
            print(f"Root {idx}: {i_e - e[0]} Hartree, spin: {spin[idx]}")
