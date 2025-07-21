from niupy.eom_tools import *
import niupy.lib.logger as logger
import forte
import psi4
import numpy as np
import os, sys
from niupy.code_generator import cvs_ee, ip, cvs_ip
import inspect
import importlib


class EOM_DSRG:
    def __init__(self, method_type, **kwargs):
        # Logger initialization
        self.verbose = kwargs.get("verbose", 4)
        log = logger.Logger(sys.stdout, self.verbose)
        log.niupy_header()
        log.info("Initializing NiuPy...")
        self.log = log

        self.method_type = method_type

        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        self.file_dir = os.path.dirname(os.path.abspath(caller_file))
        log.info(f"Running file dir: {self.file_dir}")

        # Set defaults
        self.e_tol = kwargs.get("e_tol", 1e-8)
        self.r_tol = kwargs.get("r_tol", 1e-5)
        self.basis_per_root = kwargs.get("basis_per_root", 4)
        self.collapse_per_root = kwargs.get("collapse_per_root", 2)
        self.max_cycle = kwargs.get("max_cycle", 100)
        self.tol_s = kwargs.get("tol_s", 1e-10)
        self.tol_semi = kwargs.get("tol_semi", 1e-6)
        self.nroots = kwargs.get("nroots", 6)
        self.diagonal_type = kwargs.get("diagonal_type", "compute")
        self.ref_sym = kwargs.get("ref_sym", 0)
        opt_einsum = kwargs.get("opt_einsum", True)

        if opt_einsum:
            log.info("Using opt_einsum...")
            from opt_einsum import contract

            self.einsum = contract
            self.einsum_type = "'greedy'"
        else:
            self.einsum = np.einsum
            self.einsum_type = "True"

        mo_spaces = kwargs.get("mo_spaces", None)
        self.guess = kwargs.get("guess", "ones")
        self.sequential_ortho = kwargs.get("sequential_ortho", True)
        self.blocked_ortho = kwargs.get("blocked_ortho", True)
        self.first_row = kwargs.get("first_row", False)

        # Initialize MO symmetry information
        self._initialize_mo_symmetry(mo_spaces)
        # Set system sizes
        self.ncore = len(self.core_sym)
        self.nocc = len(self.occ_sym)
        self.nact = len(self.act_sym)
        self.nvir = len(self.vir_sym)

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

        self.delta = {
            "ii": np.eye(self.nmos["i"]),
            "cc": np.eye(self.nmos["c"]),
            "vv": np.eye(self.nmos["v"]),
            "II": np.eye(self.nmos["I"]),
            "CC": np.eye(self.nmos["C"]),
            "VV": np.eye(self.nmos["V"]),
        }

        self._generate_code()

        self.S12 = lambda: None

        # Initialize templates and sigma vectors
        import niupy.eom_dsrg_compute as eom_dsrg_compute

        importlib.reload(eom_dsrg_compute)

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

    def _generate_code(self):
        match self.method_type:
            case "cvs_ee":
                self.ops, self.single_space, self.composite_space = (
                    cvs_ee.generator_full(
                        self.log,
                        self.file_dir,
                        self.ncore,
                        self.nocc,
                        self.nact,
                        self.nvir,
                        blocked_ortho=self.blocked_ortho,
                    )
                )
                self.nops, self.slices = get_slices(self.ops, self.nmos)
                self.small_ops, self.small_space = cvs_ee.generator(
                    self.log,
                    self.file_dir,
                    self.ncore,
                    self.nocc,
                    self.nact,
                    self.nvir,
                    self.einsum_type,
                    sequential_ortho=self.sequential_ortho,
                    blocked_ortho=self.blocked_ortho,
                    first_row=self.first_row,
                )
                self.small_nops, self.small_slices = get_slices(
                    self.small_ops, self.nmos
                )

                # Memory estimation
                mem = self.small_nops * self.small_nops * np.dtype(np.float64).itemsize
                mem /= 1e9  # Convert to GB
                self.log.info(f"\nEstimated memory usage for Hsmall: {mem:.2f} GB")

                if self.guess == "singles":
                    self.ops_sub, self.single_space_sub, self.composite_space_sub = (
                        cvs_ee.generator_subspace(
                            self.log,
                            self.file_dir,
                            self.ncore,
                            self.nocc,
                            self.nact,
                            self.nvir,
                            blocked_ortho=self.blocked_ortho,
                        )
                    )
                    self.nops_sub, self.slices_sub = get_slices(self.ops_sub, self.nmos)
            case "ip":
                self.ops, self.single_space, self.composite_space = ip.generator_full(
                    self.log, self.file_dir, blocked_ortho=self.blocked_ortho
                )
                self.nops, self.slices = get_slices(self.ops, self.nmos)
                ip.generator(
                    self.log,
                    self.file_dir,
                    self.einsum_type,
                    sequential_ortho=self.sequential_ortho,
                    blocked_ortho=self.blocked_ortho,
                )
            case "cvs_ip":
                self.ops, self.single_space, self.composite_space = (
                    cvs_ip.generator_full(
                        self.log,
                        self.file_dir,
                        self.ncore,
                        self.nocc,
                        self.nact,
                        self.nvir,
                        blocked_ortho=self.blocked_ortho,
                    )
                )
                self.nops, self.slices = get_slices(self.ops, self.nmos)
                cvs_ip.generator(
                    self.log,
                    self.file_dir,
                    self.ncore,
                    self.nocc,
                    self.nact,
                    self.nvir,
                    self.einsum_type,
                    sequential_ortho=self.sequential_ortho,
                    blocked_ortho=self.blocked_ortho,
                )
            case _:
                msg = "Unknown method %s" % self.method_type
                self.log.error(msg)
                raise Exception(msg)

    def _get_integrals(self):
        self.gamma1 = np.load(f"{self.file_dir}/save_gamma1.npz")
        self.eta1 = np.load(f"{self.file_dir}/save_eta1.npz")
        self.lambda2 = np.load(f"{self.file_dir}/save_lambda2.npz")
        self.lambda3 = np.load(f"{self.file_dir}/save_lambda3.npz")
        self.lambda4 = np.load(f"{self.file_dir}/save_lambda4.npz")

    def _initialize_mo_symmetry(self, mo_spaces=None):
        try:
            mo_space_save = np.load(os.path.join(self.file_dir, "save_mo_space.npz"))
        except FileNotFoundError:
            raise FileNotFoundError("No save_mo_space.npz file found.") from None
        nmopi = mo_space_save["nmopi"]
        self.point_group = str(mo_space_save["point_group"])
        if mo_spaces is None:
            mo_spaces = {}
            mo_spaces["frozen_docc"] = mo_space_save["frozen_docc"]
            mo_spaces["restricted_docc"] = mo_space_save["restricted_docc"]
            mo_spaces["active"] = mo_space_save["active"]
            mo_spaces["virtual"] = mo_space_save["virtual"]
        else:
            mo_spaces["virtual"] = mo_space_save["virtual"]
        mo_space_info = forte.make_mo_space_info_from_map(
            psi4.core.Dimension(list(nmopi)), self.point_group, mo_spaces
        )

        self.core_sym = np.array(mo_space_info.symmetry("FROZEN_DOCC"))
        self.occ_sym = np.array(mo_space_info.symmetry("RESTRICTED_DOCC"))
        self.act_sym = np.array(mo_space_info.symmetry("ACTIVE"))
        self.vir_sym = np.array(mo_space_info.symmetry("VIRTUAL"))

    def _initialize_sigma_vectors(self):
        (
            self.build_first_row,
            self.build_sigma_vector_Hbar,
            self.build_transition_dipole,
            self.get_S12,
            self.apply_S12,
            self.compute_preconditioner,
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
            self.log.info("Some roots did not converge.")
            self.log.info(f"Unconverged roots: {unconv}")
        else:
            self.log.info("All roots converged.")

        eigvec, eigvec_dict = self.eom_dsrg_compute.get_original_basis_evecs(
            self, u, nop
        )
        spin, symmetry, spec_info = self.eom_dsrg_compute.post_process(
            self, e, eigvec, eigvec_dict
        )
        self._pretty_print_info(e, spin, symmetry, spec_info)

        self.evals = e * eh_to_ev
        self.evecs = eigvec
        self.spin = spin
        self.symmetry = symmetry
        self.spec_info = spec_info

        # if os.path.exists(f"{self.method_type}_eom_dsrg.py"):
        #     os.remove(f"{self.method_type}_eom_dsrg.py")

        # return conv, e, u, eigvec, eigvec_dict, spin, symmetry, spec_info

    def kernel_full(self, dump_vectors=False, skip_spec=False):
        _available_methods = ["ip", "cvs_ip", "cvs_ee"]
        assert (
            self.method_type in _available_methods
        ), f"Full EOM-DSRG is only supported for {_available_methods}."
        evals, evecs = self.eom_dsrg_compute.kernel_full(
            self, sequential=self.sequential_ortho
        )
        eigvec_dict = full_vec_to_dict(
            self.template_c, self.slices, evecs[:, : self.nroots], self.nmos
        )
        if dump_vectors:
            pickle.dump(eigvec_dict, open(f"niupy_save.pkl", "wb"))
        eigvec = dict_to_vec(eigvec_dict, self.nroots)
        e = evals[: self.nroots]
        spin, symmetry, spec_info = self.eom_dsrg_compute.post_process(
            self, e, eigvec, eigvec_dict, skip_spec=skip_spec
        )
        self._pretty_print_info(e, spin, symmetry, spec_info)

        self.evals = e * eh_to_ev
        self.evecs = eigvec
        self.spin = spin
        self.symmetry = symmetry
        self.spec_info = spec_info

        # if os.path.exists(f"{self.method_type}_eom_dsrg_full.py"):
        #     os.remove(f"{self.method_type}_eom_dsrg_full.py")

        # return e, eigvec, eigvec_dict, spin, symmetry, spec_info
