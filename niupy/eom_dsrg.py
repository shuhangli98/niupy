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

        opt_einsum = kwargs.get("opt_einsum", False)
        mo_spaces = kwargs.get("mo_spaces", None)

        self.guess = kwargs.get("guess", "ones")

        self.sequential_ortho = kwargs.get("sequential_ortho", True)
        self.blocked_ortho = kwargs.get("blocked_ortho", True)

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

        if method_type == "cvs_ee":
            cvs_ee.generator(
                self.abs_file_path, self.ncore, self.nocc, self.nact, self.nvir
            )
        elif method_type == "ip":
            self.ops, self.single_space, self.composite_space = ip.generator_full(
                self.abs_file_path, blocked_ortho=self.blocked_ortho
            )
            self.nmos = {
                "i": self.ncore,
                "c": self.nocc,
                "a": self.nact,
                "v": self.nvir,
                "I": self.ncore,
                "C": self.nocc,
                "A": self.nact,
                "V": self.nvir,
            }
            self.nops, self.slices = get_slices(self.ops, self.nmos)
            self.delta = {
                "cc": np.eye(self.nmos["c"]),
                "vv": np.eye(self.nmos["v"]),
                "CC": np.eye(self.nmos["C"]),
                "VV": np.eye(self.nmos["V"]),
            }
            ip.generator(
                self.abs_file_path,
                sequential_ortho=self.sequential_ortho,
                blocked_ortho=self.blocked_ortho,
            )
        elif method_type == "cvs_ip":
            self.ops, self.single_space, self.composite_space = cvs_ip.generator_full(
                self.abs_file_path,
                self.ncore,
                self.nocc,
                self.nact,
                self.nvir,
                blocked_ortho=self.blocked_ortho,
            )
            self.nmos = {
                "i": self.ncore,
                "c": self.nocc,
                "a": self.nact,
                "v": self.nvir,
                "I": self.ncore,
                "C": self.nocc,
                "A": self.nact,
                "V": self.nvir,
            }
            self.nops, self.slices = get_slices(self.ops, self.nmos)
            self.delta = {
                "ii": np.eye(self.nmos["i"]),
                "cc": np.eye(self.nmos["c"]),
                "vv": np.eye(self.nmos["v"]),
                "II": np.eye(self.nmos["I"]),
                "CC": np.eye(self.nmos["C"]),
                "VV": np.eye(self.nmos["V"]),
            }
            cvs_ip.generator(
                self.abs_file_path, self.ncore, self.nocc, self.nact, self.nvir
            )
        else:
            raise ValueError(f"Method type {method_type} is not supported.")

        # Disable opt_einsum for now
        # if opt_einsum:
        #     print("Using opt_einsum...")
        #     from opt_einsum import contract

        #     self.einsum = contract

        #     subprocess.run(
        #         [
        #             "sed",
        #             "-i",
        #             "-e",
        #             "s/optimize=True/optimize='greedy'/g; s/np\\.einsum/einsum/g",
        #             os.path.join(self.abs_file_path, f"{method_type}_eom_dsrg.py"),
        #         ]
        #     )

        # else:
        #     self.einsum = np.einsum

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

    def _initialize_mo_symmetry(self, method_type, mo_spaces=None):
        try:
            mo_space_save = np.load("save_mo_space.npz")
        except FileNotFoundError:
            self._set_default_symmetry(method_type)
        nmopi = mo_space_save["nmopi"]
        self.point_group = mo_space_save["point_group"]
        if mo_spaces is None:
            mo_spaces = {}
            mo_spaces["FROZEN_DOCC"] = mo_space_save["FROZEN_DOCC"]
            mo_spaces["RESTRICTED_DOCC"] = mo_space_save["RESTRICTED_DOCC"]
            mo_spaces["ACTIVE"] = mo_space_save["ACTIVE"]
            mo_spaces["VIRTUAL"] = mo_space_save["VIRTUAL"]
        mo_space_info = forte.make_mo_space_info_from_map(
            psi4.core.Dimension(list(nmopi)), self.point_group, mo_spaces
        )
        
        self.core_sym = np.array(mo_space_info.symmetry("FROZEN_DOCC"))
        self.occ_sym = np.array(mo_space_info.symmetry("RESTRICTED_DOCC"))
        self.act_sym = np.array(mo_space_info.symmetry("ACTIVE"))
        self.vir_sym = np.array(mo_space_info.symmetry("VIRTUAL"))

    def _set_default_symmetry(self, method_type):
        if method_type == "cvs_ee" or "cvs_ip" in method_type:
            print("Running H2O/6-31g since no wfn and mo_spaces are provided.")
            # 6-31g
            self.point_group = "c2v"
            self.core_sym = np.array([0])
            self.occ_sym = np.array([0, 3])
            self.act_sym = np.array([0, 0, 2, 3])
            self.vir_sym = np.array([0, 0, 0, 2, 3, 3])
        elif method_type == "ip":
            print("Running BeH2/STO-6G since no wfn and mo_spaces are provided.")
            self.point_group = "c2v"
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
            # self.build_sigma_vector_Hbar_singles,
            # self.build_sigma_vector_s_singles
        ) = self.eom_dsrg_compute.get_sigma_build(self)

    def _pretty_print_info(self, e, spin, symmetry, spec_info):
        def _irrep(x):
            if isinstance(x, list):
                return ",".join([irrep_table[self.point_group.lower()][i] for i in x])
            else:
                return irrep_table[self.point_group.lower()][x]

        nroot = len(e)
        if "ee" in self.method_type:
            spec = "f"
        elif "ip" in self.method_type:
            spec = "P"
        if self.point_group.lower() not in irrep_table:
            _irrep = lambda x: x
        print("=" * 85)
        print(f"{'EOM-DSRG summary':^85}")
        print("-" * 85)
        print(
            f"{'Root':<5} {'Energy (eV)':<20} {spec:<20} {'Symmetry':<10} {'Spin':<20}"
        )
        print("-" * 85)
        for i in range(nroot):
            print(
                f"{i+1:<5} {e[i]*eh_to_ev:<20.10f} {spec_info[i]:<20.8f} {_irrep(symmetry[i]):<10} {spin[i]:<20}"
            )
        print("=" * 85)

    def kernel(self):
        conv, e, u, nop = self.eom_dsrg_compute.kernel(self)

        if not all(conv):
            unconv = [i for i, c in enumerate(conv) if not c]
            print("Some roots did not converge.")
            print(f"Unconverged roots: {unconv}")
        else:
            print("All EOM-DSRG roots converged.")

        eigvec, eigvec_dict = self.eom_dsrg_compute.get_original_basis_evecs(
            self, u, nop
        )
        spin, symmetry, spec_info = self.eom_dsrg_compute.post_process(
            self, e, eigvec, eigvec_dict
        )
        self._pretty_print_info(e, spin, symmetry, spec_info)

        # if os.path.exists(f"{self.method_type}_eom_dsrg.py"):
        #     os.remove(f"{self.method_type}_eom_dsrg.py")

        return conv, e, u, eigvec, eigvec_dict, spin, symmetry, spec_info

    def kernel_full(self, dump_vectors=False):
        _available_methods = ["ip", "cvs_ip"]
        assert (
            self.method_type in _available_methods
        ), f"Full EOM-DSRG is only supported for {_available_methods}."
        evals, evecs = self.eom_dsrg_compute.kernel_full(
            self, sequential=self.sequential_ortho
        )
        eigvec_dict = full_vec_to_dict(
            self.full_template_c, self.slices, evecs[:, : self.nroots], self.nmos
        )
        if dump_vectors:
            pickle.dump(eigvec_dict, open(f"niupy_save.pkl", "wb"))
        eigvec = dict_to_vec(eigvec_dict, self.nroots)
        e = evals[: self.nroots]
        spin, symmetry, spec_info = self.eom_dsrg_compute.post_process(
            self, e, eigvec, eigvec_dict
        )
        self._pretty_print_info(e, spin, symmetry, spec_info)

        # if os.path.exists(f"{self.method_type}_eom_dsrg_full.py"):
        #     os.remove(f"{self.method_type}_eom_dsrg_full.py")

        return e, eigvec, eigvec_dict, spin, symmetry, spec_info
