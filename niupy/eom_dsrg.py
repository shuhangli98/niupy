from niupy.eom_tools import *
import forte
import forte.utils
import psi4
import numpy as np
import os
import subprocess
from niupy.code_generator import cvs_ee, ee, ip, cvs_ip


class EOM_DSRG:
    def __init__(self, method_type, wfn=None, **kwargs):
        self.method_type = method_type
        # Set defaults
        self.tol_e = kwargs.get("tol_e", 1e-8)
        self.max_space = kwargs.get("max_space", 100)
        self.max_cycle = kwargs.get("max_cycle", 100)
        self.tol_davidson = kwargs.get("tol_davidson", 1e-5)
        self.tol_s = kwargs.get("tol_s", 1e-10)
        self.tol_semi = kwargs.get("tol_semi", 1e-6)
        self.nroots = kwargs.get("nroots", 6)
        self.verbose = kwargs.get("verbose", 5)
        self.diagonal_type = kwargs.get("diagonal_type", "compute")
        self.ref_sym = kwargs.get("ref_sym", 0)
        if self.ref_sym != 0:
            raise NotImplementedError(
                "Reference symmetry other than 0 is not implemented."
            )

        opt_einsum = kwargs.get("opt_einsum", True)
        mo_spaces = kwargs.get("mo_spaces", None)

        # Initialize MO symmetry information
        self._initialize_mo_symmetry(wfn, mo_spaces, method_type)

        # Print symmetry information
        self._print_symmetry_info()

        # Set system sizes
        self.ncore = len(self.core_sym)
        self.nocc = len(self.occ_sym)
        self.nact = len(self.act_sym)
        self.nvir = len(self.vir_sym)
        if self.verbose:
            print(f"ncore: {self.ncore}")
            print(f"nocc: {self.nocc}")
            print(f"nact: {self.nact}")
            print(f"nvir: {self.nvir}")

        self.abs_file_path = os.getcwd()

        # package_dir = os.path.dirname(os.path.abspath(__file__))
        # print(f"Package directory: {package_dir}")
        # code_generator_dir = os.path.join(package_dir, "code_generator")

        if method_type == "cvs_ee":
            cvs_ee.generator(
                self.abs_file_path, self.ncore, self.nocc, self.nact, self.nvir
            )
        elif method_type == "ip":
            ip.generator(self.abs_file_path)
        elif method_type == "cvs_ip":
            cvs_ip.generator(
                self.abs_file_path, self.ncore, self.nocc, self.nact, self.nvir
            )
        else:
            raise ValueError(f"Method type {method_type} is not supported.")

        if opt_einsum:
            print("Using opt_einsum...")
            from opt_einsum import contract

            self.einsum = contract
        else:
            self.einsum = np.einsum

        self.S12 = lambda: None

        # Initialize templates and sigma vectors
        import niupy.eom_dsrg_compute as eom_dsrg_compute

        self.eom_dsrg_compute = eom_dsrg_compute
        self.template_c, self.full_template_c = self.eom_dsrg_compute.get_templates(
            self
        )
        self._initialize_sigma_vectors()
        self._get_integrals()

        # Generate symmetry information
        self.sym = sym_dir(
            self.full_template_c,
            self.core_sym,
            self.occ_sym,
            self.act_sym,
            self.vir_sym,
        )
        self.sym_vec = dict_to_vec(self.sym, 1).flatten()

    def _get_integrals(self):
        self.gamma1 = np.load(f"{self.abs_file_path}/save_gamma1.npz")
        self.eta1 = np.load(f"{self.abs_file_path}/save_eta1.npz")
        self.lambda2 = np.load(f"{self.abs_file_path}/save_lambda2.npz")
        self.lambda3 = np.load(f"{self.abs_file_path}/save_lambda3.npz")
        self.lambda4 = np.load(f"{self.abs_file_path}/save_lambda4.npz")

        # self.Hbar is loaded in setup_davidson
        if self.build_transition_dipole is not NotImplemented:
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
        else:
            self.Mbar = [None, None, None]

    def _initialize_mo_symmetry(self, wfn, mo_spaces, method_type):
        if wfn is not None:
            if mo_spaces is None:
                psi4_options = psi4.core.get_options()
                spaces = ["frozen_docc", "restricted_docc", "active", "restricted_uocc", "frozen_uocc"]
                mo_spaces = {}
                for s in spaces:
                    if len(psi4_options.get_int_vector(s)) > 0:
                        mo_spaces[s] = psi4_options.get_int_vector(s)
            nmopi = wfn.nmopi()
            point_group = wfn.molecule().point_group().symbol()
            mo_space_info = forte.make_mo_space_info_from_map(nmopi, point_group, mo_spaces)
            self.core_sym = np.array(mo_space_info.symmetry("FROZEN_DOCC"))
            self.occ_sym = np.array(mo_space_info.symmetry("RESTRICTED_DOCC"))
            self.act_sym = np.array(mo_space_info.symmetry("ACTIVE"))
            self.vir_sym = np.array(mo_space_info.symmetry("VIRTUAL"))
        else:
            self._set_default_symmetry(method_type)

    def _set_default_symmetry(self, method_type):
        if method_type == "cvs_ee" or method_type == "cvs_ip":
            print("Running H2O/6-31g since no wfn and mo_spaces are provided.")
            # 6-31g
            self.core_sym = np.array([0])
            self.occ_sym = np.array([0, 3])
            self.act_sym = np.array([0, 0, 2, 3])
            self.vir_sym = np.array([0, 0, 0, 2, 3, 3])
        elif method_type == "ip":
            print("Running BeH2/STO-6G since no wfn and mo_spaces are provided.")
            self.core_sym = np.array([])
            self.occ_sym = np.array([0, 0])
            self.act_sym = np.array([0, 3])
            self.vir_sym = np.array([0, 2, 3])

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
            self.get_S12,
            self.apply_S12,
            self.compute_preconditioner,
        ) = self.eom_dsrg_compute.get_sigma_build(self)

    def kernel(self):
        conv, e, u, spin, symmetry, spec_info = self.eom_dsrg_compute.kernel(self)
        if os.path.exists(f"{self.method_type}_eom_dsrg.py"):
            os.remove(f"{self.method_type}_eom_dsrg.py")
        if os.path.exists(f"{self.method_type}_eom_dsrg.py-e"):
            os.remove(f"{self.method_type}_eom_dsrg.py-e")
        return conv, e, u, spin, symmetry, spec_info
