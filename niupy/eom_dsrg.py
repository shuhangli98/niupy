from niupy.eom_tools import *
import forte
import forte.utils
import numpy as np
import niupy.eom_dsrg_compute as eom_dsrg_compute
import os


class EOM_DSRG:
    def __init__(
        self, rel_path, diag_shift=0.0,
        tol_e=1e-8, max_space=100, max_cycle=100,
        tol_davidson=1e-5, tol_s=1e-4,
        target_sym=0, target_spin=0, nroots=6,
        verbose=0, wfn=None, mo_spaces=None,
        method_type='ee', diagonal_type='exact'
    ):
        script_dir = os.getcwd()
        self.abs_file_path = os.path.join(script_dir, rel_path)

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
        self.diag_shift = diag_shift        # Shift for the diagonal of the effective Hamiltonian
        self.target_sym = target_sym
        self.target_spin = target_spin

        # Set Hamiltonian and RDMs
        self.gamma1 = np.load(f'{self.abs_file_path}/save_gamma1.npz')
        self.eta1 = np.load(f'{self.abs_file_path}/save_eta1.npz')
        self.lambda2 = np.load(f'{self.abs_file_path}/save_lambda2.npz')
        self.lambda3 = np.load(f'{self.abs_file_path}/save_lambda3.npz')
        # self.Hbar = np.load(f'{self.abs_file_path}/save_Hbar.npz')
        self.Mbar0 = np.load(f'{self.abs_file_path}/Mbar0.npy')
        Mbar1_x = np.load(f'{self.abs_file_path}/Mbar1_0.npz')
        Mbar1_y = np.load(f'{self.abs_file_path}/Mbar1_1.npz')
        Mbar1_z = np.load(f'{self.abs_file_path}/Mbar1_2.npz')
        Mbar2_x = np.load(f'{self.abs_file_path}/Mbar2_0.npz')
        Mbar2_y = np.load(f'{self.abs_file_path}/Mbar2_1.npz')
        Mbar2_z = np.load(f'{self.abs_file_path}/Mbar2_2.npz')
        Mbar_x = {**Mbar1_x, **Mbar2_x}
        Mbar_y = {**Mbar1_y, **Mbar2_y}
        Mbar_z = {**Mbar1_z, **Mbar2_z}
        self.Mbar = [Mbar_x, Mbar_y, Mbar_z]

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
            self.occ_sym = np.array([0, 3])
            self.act_sym = np.array([0, 0, 2, 3])
            self.vir_sym = np.array([0, 0, 0, 2, 3, 3])

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
            self.build_transition_dipole,
            self.get_S_12,
            self.compute_preconditioner_exact,
            self.compute_preconditioner_block
        ) = eom_dsrg_compute.get_sigma_build(self)

    def kernel(self):
        conv, e, u, spin, osc_strength = eom_dsrg_compute.kernel(self)
        return conv, e, u, spin, osc_strength


if __name__ == "__main__":
    test = 1
    if test == 1:
        # Hbar, gamma1, eta1, lambda2, lambda3, Mbar, Mbar0 = load_data("H2O")
        rel_path = "niupy/H2O"
        eom_dsrg = EOM_DSRG(rel_path, nroots=3, verbose=5, max_cycle=100,
                            target_sym=0, method_type='cvs-ee', diagonal_type='block')
        conv, e, u, spin, osc_strength = eom_dsrg.kernel()
        for idx, i_e in enumerate(e):
            if idx == 0:
                print(f"Root {idx}: {i_e - e[0]} Hartree, spin: {spin[idx]}")
            else:
                print(f"Root {idx}: {i_e - e[0]} Hartree, spin: {spin[idx]}, osc_strength: {osc_strength[idx-1]}")

# Root 0: 0.0 Hartree, spin: Singlet
# Root 1: 19.85801514477547 Hartree, spin: Triplet, osc_strength: 8.882149755332066e-15
# Root 2: 19.884122785946218 Hartree, spin: Singlet, osc_strength: 0.02027260029312239
