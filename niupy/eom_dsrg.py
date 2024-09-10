from niupy.eom_tools import *
import forte
import forte.utils
import numpy as np
import niupy.eom_dsrg_compute as eom_dsrg_compute


class EOM_DSRG:
    def __init__(
        self,
        Hbar, gamma1, eta1, lambda2, lambda3, dp1, diag_shift=0.0,
        tol_e=1e-8, max_space=100, max_cycle=100,
        tol_davidson=1e-5, tol_s=1e-4, tol_s_act=1e-4,
        target_sym=0, target_spin=0, nroots=6,
        verbose=0, wfn=None, mo_spaces=None,
        method_type='ee', diagonal_type='exact'
    ):

        # Initialize MO symmetry information
        self._initialize_mo_symmetry(wfn, mo_spaces, method_type)

        # Print symmetry information
        self._print_symmetry_info()

        # Set system sizes
        self.ncore = len(self.core_sym)
        self.nocc = len(self.occ_sym)
        self.nact = len(self.act_sym)
        self.nvir = len(self.vir_sym)

        self.method_type = method_type
        self.diagonal_type = diagonal_type  # 'exact' or 'block'
        self.verbose = verbose
        self.nroots = nroots                # Number of EOM-DSRG roots requested
        self.max_space = max_space          # Maximum size of the Davidson trial space
        self.max_cycle = max_cycle          # Maximum number of iterations in the Davidson procedure
        self.tol_e = tol_e                  # Tolerance for the energy in the Davidson procedure
        self.tol_davidson = tol_davidson    # Tolerance for the residual in the Davidson procedure
        self.tol_s = tol_s                  # Tolerance for the orthogonalization of excitation spaces
        self.tol_s_act = tol_s_act          # Tolerance for the orthogonalization of the active space
        self.diag_shift = diag_shift        # Shift for the diagonal of the effective Hamiltonian
        self.target_sym = target_sym
        self.target_spin = target_spin

        # Set Hamiltonian and RDMs
        self.Hbar = Hbar
        self.gamma1 = gamma1
        self.eta1 = eta1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.dp1 = dp1

        # Initialize templates and sigma vectors
        self.template_c, self.full_template_c = eom_dsrg_compute.get_templates(self)
        self._initialize_sigma_vectors()

        # Generate symmetry information
        self.sym = sym_dir(self.full_template_c, self.core_sym, self.occ_sym, self.act_sym, self.vir_sym)
        self.sym_vec = dict_to_vec(self.sym, 1).flatten()

    def _initialize_mo_symmetry(self, wfn, mo_spaces, method_type):
        if wfn is not None and mo_spaces is not None:
            res = forte.utils.prepare_forte_objects(wfn, mo_spaces)
            mo_space_info = res['mo_space_info']
            self.core_sym = np.array(mo_space_info.symmetry('FROZEN_DOCC'))
            self.occ_sym = np.array(mo_space_info.symmetry('RESTRICTED_DOCC'))
            self.act_sym = np.array(mo_space_info.symmetry('ACTIVE'))
            self.vir_sym = np.array(mo_space_info.symmetry('VIRTUAL'))
        else:
            self._set_default_symmetry(method_type)

    def _set_default_symmetry(self, method_type):
        if method_type == 'ee':
            print("Running BeH2/STO-6G since no wfn and mo_spaces are provided.")
            self.core_sym = np.array([])
            self.occ_sym = np.array([0, 0])
            self.act_sym = np.array([0, 3])
            self.vir_sym = np.array([0, 2, 3])
        elif method_type == 'cvs-ee':
            print("Running H2O/6-31g since no wfn and mo_spaces are provided.")
            # 6-31g
            self.core_sym = np.array([0])
            self.occ_sym = np.array([0, 0, 2, 3])
            self.act_sym = np.array([0, 2, 3, 3])
            self.vir_sym = np.array([0, 0, 0, 3])
            # aug-cc-pvdz
            # self.core_sym = np.array([0])
            # self.occ_sym = np.array([0, 3])
            # self.act_sym = np.array([0, 0, 2, 3])
            # self.vir_sym = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            #                         1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

    def _print_symmetry_info(self):
        print("\n")
        print("  Symmetry information:")
        print(f"  core_sym: {self.core_sym}")
        print(f"  occ_sym: {self.occ_sym}")
        print(f"  act_sym: {self.act_sym}")
        print(f"  vir_sym: {self.vir_sym}")

    def _initialize_sigma_vectors(self):
        (
            self.build_first_row,
            self.build_sigma_vector_Hbar,
            self.build_sigma_vector_s,
            self.get_S_12,
            self.compute_preconditioner_exact,
            self.compute_preconditioner_block
        ) = eom_dsrg_compute.get_sigma_build(self)

    def kernel(self):
        conv, e, u, spin, osc_strength = eom_dsrg_compute.kernel(self)
        return conv, e, u, spin, osc_strength


if __name__ == "__main__":
    import os
    test = 2
    script_dir = os.path.dirname(__file__)

    def load_data(rel_path):
        abs_file_path = os.path.join(script_dir, rel_path)
        gamma1 = np.load(f'{abs_file_path}/save_gamma1.npz')
        eta1 = np.load(f'{abs_file_path}/save_eta1.npz')
        lambda2 = np.load(f'{abs_file_path}/save_lambda2.npz')
        lambda3 = np.load(f'{abs_file_path}/save_lambda3.npz')
        Hbar = np.load(f'{abs_file_path}/save_Hbar.npz')
        dp1 = np.load(f'{abs_file_path}/save_dp1.npy', allow_pickle=True)
        return Hbar, gamma1, eta1, lambda2, lambda3, dp1

    if test == 1:
        Hbar, gamma1, eta1, lambda2, lambda3, dp1 = load_data("BeH2")
        eom_dsrg = EOM_DSRG(Hbar, gamma1, eta1, lambda2, lambda3, dp1, nroots=3,
                            verbose=5, max_cycle=100, target_sym=0, method_type='ee', diagonal_type='block')
        conv, e, u, spin, _ = eom_dsrg.kernel()
        for idx, i_e in enumerate(e):
            print(f"Root {idx}: {i_e - e[0]} Hartree, spin: {spin[idx]}")
    elif test == 2:
        Hbar, gamma1, eta1, lambda2, lambda3, dp1 = load_data("H2O")
        Hbar = slice_H_core(Hbar, 1)
        eom_dsrg = EOM_DSRG(Hbar, gamma1, eta1, lambda2, lambda3, dp1, nroots=3,
                            verbose=5, max_cycle=100, target_sym=0, method_type='cvs-ee', diagonal_type='identity')
        conv, e, u, spin, osc_strength = eom_dsrg.kernel()
        for idx, i_e in enumerate(e):
            if idx == 0:
                print(f"Root {idx}: {i_e - e[0]} Hartree, spin: {spin[idx]}")
            else:
                print(f"Root {idx}: {i_e - e[0]} Hartree, spin: {spin[idx]}, osc_strength: {osc_strength[idx-1]}")

    # elif test == 3:
    #     # Disabled for now
    #     Hbar, gamma1, eta1, lambda2, lambda3 = load_data("H2O_Prism")
    #     Hbar = slice_H_core(Hbar, 1)
    #     eom_dsrg = EOM_DSRG(Hbar, gamma1, eta1, lambda2, lambda3, nroots=3,
    #                         verbose=5, max_cycle=100, target_sym=0, method_type='cvs-ee', diagonal_type='block')
    #     conv, e, u, spin = eom_dsrg.kernel()
    #     # for idx, i_e in enumerate(e):
    #     #     print(f"Root {idx}: {i_e - e[0]} Hartree")  # , spin: {spin[idx]}
