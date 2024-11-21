from niupy.eom_tools import *
import forte
import forte.utils
import numpy as np
import os
import subprocess
from niupy.code_generator import cvs_ee, ee, ip


class EOM_DSRG:
    def __init__(
        self,
        diag_shift=0.0,
        opt_einsum=True,
        einsum_type="greedy",
        tol_e=1e-8,
        max_space=100,
        max_cycle=100,
        tol_davidson=1e-5,
        tol_s=1e-10,
        tol_semi=1e-6,
        ref_sym=0,
        target_sym=0,
        target_spin=0,
        nroots=6,
        verbose=0,
        wfn=None,
        mo_spaces=None,
        S_12_type="compute",
        method_type="cvs_ee",
        diagonal_type="exact",
        diag_val=1.0,
        davidson_type="traditional",
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

        self.abs_file_path = os.getcwd()

        # package_dir = os.path.dirname(os.path.abspath(__file__))
        # print(f"Package directory: {package_dir}")
        # code_generator_dir = os.path.join(package_dir, "code_generator")

        if method_type == "cvs_ee":
            cvs_ee.generator(
                self.abs_file_path, self.ncore, self.nocc, self.nact, self.nvir
            )
        elif method_type == "ee":
            ee.generator(
                self.abs_file_path, self.ncore, self.nocc, self.nact, self.nvir
            )
        elif method_type == "ip":
            ip.generator()
        else:
            raise ValueError(f"Method type {method_type} not supported.")

        subprocess.run(
            [
                "sed",
                "-i",
                "-e",
                "s/optimize='optimal'/optimize=einsum_type/g; s/np\\.einsum/einsum/g",
                os.path.join(self.abs_file_path, f"{method_type}_eom_dsrg.py"),
            ]
        )

        self.einsum_type = einsum_type
        if opt_einsum:
            print("Using opt_einsum...")
            from opt_einsum import contract

            self.einsum = contract
        else:
            self.einsum = np.einsum
        self.method_type = method_type
        self.diagonal_type = diagonal_type  # 'exact', 'block' or 'load'
        self.verbose = verbose
        self.nroots = nroots
        self.max_space = max_space
        self.max_cycle = max_cycle
        self.tol_e = tol_e
        self.tol_davidson = tol_davidson
        self.tol_s = tol_s
        self.tol_semi = tol_semi
        self.diag_shift = diag_shift
        self.target_sym = target_sym
        self.ref_sym = ref_sym
        if self.ref_sym != 0:
            raise NotImplementedError(
                "Reference symmetry other than 0 is not implemented."
            )
        self.target_spin = target_spin
        self.diag_val = diag_val  # Diagonal value for identity preconditioner
        self.S_12_type = S_12_type  # 'compute' or 'load'
        self.davidson_type = davidson_type  # 'traditional' or 'generalized'

        if self.diagonal_type == "load":
            self.S_12_type = "load"

        # Set Hamiltonian and RDMs
        self.gamma1 = np.load(f"{self.abs_file_path}/save_gamma1.npz")
        self.eta1 = np.load(f"{self.abs_file_path}/save_eta1.npz")
        self.lambda2 = np.load(f"{self.abs_file_path}/save_lambda2.npz")
        self.lambda3 = np.load(f"{self.abs_file_path}/save_lambda3.npz")
        self.lambda4 = np.load(f"{self.abs_file_path}/save_lambda4.npz")

        # self.Hbar = np.load(f'{self.abs_file_path}/save_Hbar.npz')
        self.Mbar0 = np.load(f"{self.abs_file_path}/Mbar0.npy")
        Mbar1_x = np.load(f"{self.abs_file_path}/Mbar1_0.npz")
        Mbar1_y = np.load(f"{self.abs_file_path}/Mbar1_1.npz")
        Mbar1_z = np.load(f"{self.abs_file_path}/Mbar1_2.npz")
        Mbar2_x = np.load(f"{self.abs_file_path}/Mbar2_0.npz")
        Mbar2_y = np.load(f"{self.abs_file_path}/Mbar2_1.npz")
        Mbar2_z = np.load(f"{self.abs_file_path}/Mbar2_2.npz")
        Mbar_x = {**Mbar1_x, **Mbar2_x}
        Mbar_y = {**Mbar1_y, **Mbar2_y}
        Mbar_z = {**Mbar1_z, **Mbar2_z}
        self.Mbar = [Mbar_x, Mbar_y, Mbar_z]

        # Initialize templates and sigma vectors
        import niupy.eom_dsrg_compute as eom_dsrg_compute

        self.eom_dsrg_compute = eom_dsrg_compute
        self.template_c, self.full_template_c = self.eom_dsrg_compute.get_templates(
            self
        )
        self._initialize_sigma_vectors()

        # Generate symmetry information
        self.sym = sym_dir(
            self.full_template_c,
            self.core_sym,
            self.occ_sym,
            self.act_sym,
            self.vir_sym,
        )
        self.sym_vec = dict_to_vec(self.sym, 1).flatten()

    def _initialize_mo_symmetry(self, wfn, mo_spaces, method_type):
        if wfn is not None and mo_spaces is not None:
            res = forte.utils.prepare_forte_objects(wfn, mo_spaces)
            mo_space_info = res["mo_space_info"]
            self.core_sym = np.array(mo_space_info.symmetry("FROZEN_DOCC"))
            self.occ_sym = np.array(mo_space_info.symmetry("RESTRICTED_DOCC"))
            self.act_sym = np.array(mo_space_info.symmetry("ACTIVE"))
            self.vir_sym = np.array(mo_space_info.symmetry("VIRTUAL"))
        else:
            self._set_default_symmetry(method_type)

    def _set_default_symmetry(self, method_type):
        if method_type == "ee":
            print("Running BeH2/STO-6G since no wfn and mo_spaces are provided.")
            self.core_sym = np.array([])
            self.occ_sym = np.array([0, 0])
            self.act_sym = np.array([0, 3])
            self.vir_sym = np.array([0, 2, 3])
        elif method_type == "cvs_ee":
            print("Running H2O/6-31g since no wfn and mo_spaces are provided.")
            # 6-31g
            self.core_sym = np.array([0])
            self.occ_sym = np.array([0, 3])
            self.act_sym = np.array([0, 0, 2, 3])
            self.vir_sym = np.array([0, 0, 0, 2, 3, 3])
            # Test no occ
            # self.core_sym = np.array([0, 0, 3])
            # self.occ_sym = np.array([])
            # self.act_sym = np.array([0, 0, 2, 3])
            # self.vir_sym = np.array([0, 0, 0, 2, 3, 3])

    def _print_symmetry_info(self):
        print("\n")
        print("  Symmetry information:")
        print(f"  core_sym: {self.core_sym}")
        print(f"  occ_sym: {self.occ_sym}")
        print(f"  act_sym: {self.act_sym}")
        print(f"  vir_sym: {self.vir_sym}")
        print("\n")

    def _initialize_sigma_vectors(self):
        (
            self.build_first_row,
            self.build_sigma_vector_Hbar,
            self.build_sigma_vector_s,
            self.build_transition_dipole,
            self.get_S_12,
            self.compute_preconditioner_exact,
            self.compute_preconditioner_block,
            self.compute_preconditioner_only_H,
        ) = self.eom_dsrg_compute.get_sigma_build(self)

    def kernel(self):
        conv, e, u, spin, osc_strength = self.eom_dsrg_compute.kernel(self)
        # if os.path.exists(f"{self.method_type}_eom_dsrg.py"):
        #     os.remove(f"{self.method_type}_eom_dsrg.py")
        # if os.path.exists(f"{self.method_type}_eom_dsrg.py-e"):
        #     os.remove(f"{self.method_type}_eom_dsrg.py-e")
        return conv, e, u, spin, osc_strength


# if __name__ == "__main__":
#     test = 1
#     if test == 1:
