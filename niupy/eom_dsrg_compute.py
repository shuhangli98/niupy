import os
import functools
import pickle
from niupy.eom_tools import (
    eigh_gen_composite,
    tensor_label_to_full_tensor_label,
    dict_to_vec,
    vec_to_dict,
    full_vec_to_dict_short,
    antisymmetrize,
    slice_H_core,
)

if os.path.exists("cvs_ee_eom_dsrg.py"):
    print("Importing cvs_ee_eom_dsrg")
    import cvs_ee_eom_dsrg
if os.path.exists("cvs_ee_eom_dsrg_full.py"):
    print("Importing cvs_ee_eom_dsrg_full")
    import cvs_ee_eom_dsrg_full
if os.path.exists("cvs_ip_eom_dsrg.py"):
    print("Importing cvs_ip_eom_dsrg")
    import cvs_ip_eom_dsrg
if os.path.exists("cvs_ip_eom_dsrg_full.py"):
    print("Importing cvs_ip_eom_dsrg_full")
    import cvs_ip_eom_dsrg_full
if os.path.exists("ee_eom_dsrg.py"):
    print("Importing ee_eom_dsrg")
    import ee_eom_dsrg
if os.path.exists("ip_eom_dsrg.py"):
    print("Importing ip_eom_dsrg")
    import ip_eom_dsrg
if os.path.exists("ip_eom_dsrg_full.py"):
    print("Importing ip_eom_dsrg_full")
    import ip_eom_dsrg_full
import numpy as np
import copy
from pyscf import lib
import time

davidson = lib.linalg_helper.davidson1


def kernel_full(eom_dsrg, sequential=True):
    eom_dsrg.Hbar = np.load(f"{eom_dsrg.abs_file_path}/save_Hbar.npz")

    if "cvs" in eom_dsrg.method_type:
        eom_dsrg.Hbar = slice_H_core(eom_dsrg.Hbar, eom_dsrg.core_sym, eom_dsrg.occ_sym)

    if eom_dsrg.method_type == "cvs_ip":
        driver = cvs_ip_eom_dsrg_full.driver
    elif eom_dsrg.method_type == "cvs_ee":
        driver = cvs_ee_eom_dsrg_full.driver
    elif eom_dsrg.method_type == "ip":
        driver = ip_eom_dsrg_full.driver

    heff, ovlp = driver(
        eom_dsrg.Hbar,
        eom_dsrg.delta,
        eom_dsrg.gamma1,
        eom_dsrg.eta1,
        eom_dsrg.lambda2,
        eom_dsrg.lambda3,
        eom_dsrg.lambda4,
        eom_dsrg.nops,
        eom_dsrg.slices,
        eom_dsrg.nmos,
    )
    singles = [tensor_label_to_full_tensor_label(_) for _ in eom_dsrg.single_space]
    composite = [
        [tensor_label_to_full_tensor_label(_) for _ in __]
        for __ in eom_dsrg.composite_space
    ]
    print(singles)
    print(composite)
    eigval, eigvec = eigh_gen_composite(
        heff,
        ovlp,
        singles,
        composite,
        eom_dsrg.slices,
        eom_dsrg.tol_s,
        eom_dsrg.tol_semi,
        sequential=sequential,
    )
    return eigval, eigvec


def kernel(eom_dsrg):
    """
    Main function that sets up the Davidson algorithm, solves it, and determines the spin multiplicity.

    Parameters:
        eom_dsrg: An object with configuration parameters and functions required for calculations.

    Returns:
        conv: Convergence status of the Davidson algorithm.
        e: Eigenvalues from the Davidson algorithm.
        eigvec: Eigenvectors after processing.
        spin: Spin multiplicities ('Singlet', 'Triplet', or 'Incorrect spin').
        spec_info: Computed oscillator strengths for cvs_ee method, spectroscopic factors for ip method.
    """
    start = time.time()
    print("Setting up Davidson algorithm...", flush=True)

    # Setup Davidson algorithm
    apply_M, precond, x0, nop = setup_davidson(eom_dsrg)
    print("Time(s) for Davidson Setup: ", time.time() - start, flush=True)
    # Davidson algorithm
    conv, e, u = davidson(
        lambda xs: [apply_M(x) for x in xs],
        x0,
        precond,
        nroots=eom_dsrg.nroots,
        verbose=eom_dsrg.verbose,
        max_space=eom_dsrg.max_space,
        max_cycle=eom_dsrg.max_cycle,
        tol=eom_dsrg.tol_e,
        tol_residual=eom_dsrg.tol_davidson,
    )

    return conv, e, u, nop


def post_process(eom_dsrg, e, eigvec, eigvec_dict):
    # Get spin multiplicity and process eigenvectors
    excitation_analysis = find_top_values(eigvec_dict, 3)
    for key, values in excitation_analysis.items():
        print(f"Root {key}: {values}")

    spin = []
    symmetry = []

    for i_idx in range(eom_dsrg.nroots):
        current_vec = eigvec[:, i_idx]
        current_vec_dict = vec_to_dict(
            eom_dsrg.full_template_c, current_vec.reshape(-1, 1)
        )
        spin.append(assign_spin_multiplicity(eom_dsrg, current_vec_dict))
        symmetry.append(assign_spatial_symmetry(eom_dsrg, current_vec))

    del eom_dsrg.Hbar

    # # Get spin multiplicity and process eigenvectors
    # spin, eigvec = get_spin_multiplicity(eom_dsrg, u, nop, S_12)

    if eom_dsrg.build_transition_dipole is not NotImplemented:
        # Optimize slicing by vectorizing
        eom_dsrg.Mbar = [
            slice_H_core(M, eom_dsrg.core_sym, eom_dsrg.occ_sym) for M in eom_dsrg.Mbar
        ]
        # Compute oscillator strengths
        spec_info = compute_oscillator_strength(eom_dsrg, e, eigvec)
    elif "ip" in eom_dsrg.method_type:
        spec_info = compute_spectroscopic_factors(eom_dsrg, eigvec)
    else:
        print(
            f"Warning: Spectroscopic info not implemented for kernel_full of {eom_dsrg.method_type}!"
        )
        spec_info = np.zeros(eom_dsrg.nroots)

    return spin, symmetry, spec_info


def compute_spectroscopic_factors(eom_dsrg, eigvec):
    assert "ip" in eom_dsrg.method_type
    if eom_dsrg.method_type == "ip":
        p_compute = ip_eom_dsrg.build_sigma_vector_p
    elif eom_dsrg.method_type == "cvs_ip":
        p_compute = cvs_ip_eom_dsrg.build_sigma_vector_p
    p = np.zeros(eigvec.shape[1])
    for i in range(eigvec.shape[1]):
        current_dict = vec_to_dict(
            eom_dsrg.full_template_c, eigvec[:, i].reshape(-1, 1)
        )
        pi = p_compute(
            eom_dsrg.einsum,
            current_dict,
            None,
            eom_dsrg.gamma1,
            eom_dsrg.eta1,
            eom_dsrg.lambda2,
            eom_dsrg.lambda3,
            eom_dsrg.lambda4,
            first_row=False,
        )
        pi = antisymmetrize(dict_to_vec(pi, 1), ea=False)
        p[i] = (pi**2).sum() * 2.0  # Ms = +/- 1/2
    return p


def compute_oscillator_strength(eom_dsrg, eigval, eigvec):
    """
    Compute oscillator strengths for each eigenvector.
    """
    return [0.0] + [
        2.0 / 3.0 * (eigval[i] - eigval[0]) * compute_dipole(eom_dsrg, eigvec[:, i])
        for i in range(1, eigvec.shape[1])
    ]


def compute_dipole(eom_dsrg, current_vec):
    """
    Compute the squared dipole moment for a given eigenvector.
    """
    current_vec_dict = vec_to_dict(eom_dsrg.full_template_c, current_vec.reshape(-1, 1))

    # Reshape dictionaries only when necessary
    for key, val in current_vec_dict.items():
        if key != "first":
            current_vec_dict[key] = val.reshape(val.shape[1:])

    dipole_sum_squared = 0.0

    for i, Mbar_i in enumerate(eom_dsrg.Mbar):
        HT = eom_dsrg.build_transition_dipole(
            eom_dsrg.einsum,
            current_vec_dict,
            Mbar_i,
            eom_dsrg.gamma1,
            eom_dsrg.eta1,
            eom_dsrg.lambda2,
            eom_dsrg.lambda3,
            eom_dsrg.lambda4,
        )
        if eom_dsrg.first_row:
            HT_first = HT + current_vec_dict["first"][0, 0] * eom_dsrg.Mbar0[i]
        else:
            HT_first = HT
        dipole_sum_squared += HT_first**2

    return dipole_sum_squared


def calculate_norms(current_vec_dict):
    """
    Calculate subtraction and addition norms for each length-2 key pair in the vector dictionary.

    Parameters:
        current_vec_dict (dict): Dictionary containing vector components.

    Returns:
        subtraction_norms (list): Norms of the differences between matched lower and upper case keys.
        addition_norms (list): Norms of the sums between matched lower and upper case keys.
    """
    subtraction_norms = []
    addition_norms = []

    for key in current_vec_dict.keys():
        if len(key) == 2 and key.islower():
            upper_key = key.upper()
            if upper_key in current_vec_dict:
                sub_norm = np.linalg.norm(
                    current_vec_dict[key] - current_vec_dict[upper_key]
                )
                add_norm = np.linalg.norm(
                    current_vec_dict[key] + current_vec_dict[upper_key]
                )
                subtraction_norms.append(sub_norm)
                addition_norms.append(add_norm)

    return subtraction_norms, addition_norms


def get_original_basis_evecs(eom_dsrg, u, nop, ea=False):
    eigvec = np.array(
        [eom_dsrg.apply_S12(eom_dsrg, nop, vec, transpose=False).flatten() for vec in u]
    ).T
    eigvec_dict = antisymmetrize(vec_to_dict(eom_dsrg.full_template_c, eigvec), ea=ea)
    eigvec = dict_to_vec(eigvec_dict, len(u))

    return eigvec, eigvec_dict


def assign_spin_multiplicity(eom_dsrg, current_vec_dict):
    if "ee" in eom_dsrg.method_type:
        subtraction_norms, addition_norms = calculate_norms(current_vec_dict)

        # Check spin classification based on calculated norms
        minus = all(
            norm < 1e-2 for norm in subtraction_norms
        )  # The parent state is assumed to be singlet.
        plus = all(norm < 1e-2 for norm in addition_norms) and not all(
            norm < 1e-2 for norm in subtraction_norms
        )
        if plus:
            return "Triplet"
        elif minus:
            return "Singlet"
        else:
            return "Incorrect spin"
    elif "ip" in eom_dsrg.method_type:
        hole_norm = 0.0
        for k, v in current_vec_dict.items():
            if len(k) == 1:
                hole_norm += np.linalg.norm(v)
        if hole_norm > 1e-8:
            return "Doublet"

        pairs = (
            [("iAa", "IAA"), ("iAv", "IAV")]
            if "cvs_ip" in eom_dsrg.method_type
            else [("cCv", "CCV"), ("aAa", "AAA")]
        )
        ab_sum = 0.0
        bb_sum = 0.0
        for p in pairs:
            ab = current_vec_dict[p[0]][0, ...]
            bb = current_vec_dict[p[1]][0, ...]
            ab_sum += ab.sum()
            for i in range(bb.shape[0]):
                for j in range(i + 1, bb.shape[1]):
                    for a in range(bb.shape[2]):
                        bb_sum += bb[i, j, a]
        if abs(bb_sum) - abs(ab_sum) < 1e-8:
            return "Quartet"
        else:
            return "Doublet"


def assign_spatial_symmetry(eom_dsrg, current_vec):
    large_indices = np.where(abs(current_vec) > 1e-2)[0]
    first_value = eom_dsrg.sym_vec[large_indices[0]]
    if all(eom_dsrg.sym_vec[index] == first_value for index in large_indices):
        return first_value
    else:
        irreps = list(set(eom_dsrg.sym_vec[index] for index in large_indices))
        if len(irreps) > 2:
            return "Incorrect symmetry"
        else:
            if irreps[0] ^ irreps[1] == 1:
                # this means that it is a mixture of En(g/u)x and En(g/u)y irreps in Dinfh or Cinfv
                return irreps
            else:
                return "Incorrect symmetry"


def find_top_values(data, num):
    random_key = next(iter(data))
    num_slices = data[random_key].shape[0]
    results = {i: [] for i in range(num_slices)}

    for i in range(num_slices):
        # Initialize a list to store the top values for the current slice index
        slice_top_values = []

        # Iterate over each array in the dictionary
        for key, array in data.items():
            # Extract the slice corresponding to the current index i
            current_slice = array[i]
            # Flatten the slice and get the indices
            flat_slice = current_slice.flatten()
            indices = np.arange(flat_slice.size)

            # Create tuples of (absolute value, original value, index, key)
            values_with_info = [
                (abs(val), val, np.unravel_index(idx, current_slice.shape), key)
                for idx, val in zip(indices, flat_slice)
            ]
            # Extend the list with current slice values
            slice_top_values.extend(values_with_info)

        # Sort by absolute value in descending order and take the top three
        top_three = sorted(slice_top_values, key=lambda x: -x[0])[:num]

        # Store the top three as (original value, original index in the slice, key)
        results[i] = [(val[1], val[2], val[3]) for val in top_three]

    return results


def setup_davidson(eom_dsrg):
    """
    Set up parameters and functions required for the Davidson algorithm.

    Parameters:
        eom_dsrg: Object with method and algorithm-specific parameters.

    Returns:
        apply_M: Function that applies the effective Hamiltonian.
        precond: Preconditioner vector.
        guess: Initial guess vectors.
        nop: Dimension size.
    """
    eom_dsrg.Hbar = np.load(f"{eom_dsrg.abs_file_path}/save_Hbar.npz")
    if "cvs" in eom_dsrg.method_type:
        eom_dsrg.Hbar = slice_H_core(eom_dsrg.Hbar, eom_dsrg.core_sym, eom_dsrg.occ_sym)

    start = time.time()
    print("Starting S12...", flush=True)
    eom_dsrg.get_S12(eom_dsrg)
    print("Time(s) for S12: ", time.time() - start, flush=True)

    if eom_dsrg.build_first_row is NotImplemented:
        eom_dsrg.first_row = None
    else:
        eom_dsrg.first_row = eom_dsrg.build_first_row(
            eom_dsrg.einsum,
            eom_dsrg.full_template_c,
            eom_dsrg.Hbar,
            eom_dsrg.gamma1,
            eom_dsrg.eta1,
            eom_dsrg.lambda2,
            eom_dsrg.lambda3,
            eom_dsrg.lambda4,
        )
    eom_dsrg.build_H = eom_dsrg.build_sigma_vector_Hbar

    # Compute Preconditioner
    start = time.time()
    print("Starting Preconditioner...", flush=True)
    if eom_dsrg.diagonal_type == "compute":
        precond = eom_dsrg.compute_preconditioner(eom_dsrg)
        np.save(f"{eom_dsrg.abs_file_path}/precond", precond)
    elif eom_dsrg.diagonal_type == "load":
        print("Loading Preconditioner from file")
        precond = np.load(f"{eom_dsrg.abs_file_path}/precond.npy")
    print("Time(s) for Preconditioner: ", time.time() - start, flush=True)

    northo = len(precond)
    nop = dict_to_vec(eom_dsrg.full_template_c, 1).shape[0]
    apply_M = lambda x: define_effective_hamiltonian(x, eom_dsrg, nop, northo)

    if eom_dsrg.guess == "singles":
        print("Computing singles...", flush=True)
        assert eom_dsrg.method_type == "cvs_ee"
        guess_evals, guess_evecs = eom_dsrg.eom_dsrg_compute.kernel_full(
            eom_dsrg, sequential=eom_dsrg.sequential_ortho
        )
        dict_template_short = {}
        for kt in eom_dsrg.slices.keys():
            half_kt = len(kt) // 2
            new_key = kt[half_kt:] + kt[:half_kt]
            dict_template_short[new_key] = eom_dsrg.full_template_c[new_key].copy()
        guess_evecs_dict = full_vec_to_dict_short(
            eom_dsrg.full_template_c,
            dict_template_short,
            eom_dsrg.slices,
            guess_evecs[:, : eom_dsrg.nroots],
            eom_dsrg.nmos,
        )

        pickle.dump(guess_evecs_dict, open(f"niupy_save.pkl", "wb"))
        x0 = read_guess_vectors(eom_dsrg, nop, northo)
    elif eom_dsrg.guess == "ones":
        x0 = compute_guess_vectors(eom_dsrg, precond)

    return apply_M, precond, x0, nop


def define_effective_hamiltonian(x, eom_dsrg, nop, northo, ea=False):
    """
    Define the effective Hamiltonian application function.

    Parameters:
        eom_dsrg: Object with method-specific parameters.
        nop: Dimension size of operators.
        northo: Dimension size of orthogonal components.

    Returns:
        Function that applies the effective Hamiltonian to a vector.
    """

    # nop and northo include the first row/column
    Xt = eom_dsrg.apply_S12(eom_dsrg, nop, x, transpose=False)
    Xt_dict = vec_to_dict(eom_dsrg.full_template_c, Xt)
    Xt_dict = antisymmetrize(Xt_dict, ea=ea)

    HXt_dict = eom_dsrg.build_H(
        eom_dsrg.einsum,
        Xt_dict,
        eom_dsrg.Hbar,
        eom_dsrg.gamma1,
        eom_dsrg.eta1,
        eom_dsrg.lambda2,
        eom_dsrg.lambda3,
        eom_dsrg.lambda4,
        eom_dsrg.first_row,
    )

    HXt_dict = antisymmetrize(HXt_dict, ea=ea)
    HXt = dict_to_vec(HXt_dict, 1).flatten()
    XHXt = eom_dsrg.apply_S12(eom_dsrg, northo, HXt, transpose=True)
    XHXt = XHXt.flatten()
    return XHXt


def read_guess_vectors(eom_dsrg, nops, northo, ea=False):
    assert eom_dsrg.method_type == "cvs_ee"
    print("Reading guess vectors...", flush=True)
    guess = pickle.load(open(f"{eom_dsrg.abs_file_path}/niupy_save.pkl", "rb"))
    x0s = []

    unit_vector = np.zeros(northo)
    unit_vector[0] = 1.0
    x0s.append(unit_vector)

    print("Projecting guess vectors...", flush=True)
    for i in range(eom_dsrg.nroots - 1):
        x0 = np.zeros((nops, 1))
        x0 = vec_to_dict(eom_dsrg.full_template_c, x0)
        for k, v in guess.items():
            x0[k] = (v[i, ...])[np.newaxis, ...]
        HX0_dict = eom_dsrg.build_H(
            eom_dsrg.einsum,
            x0,
            eom_dsrg.Hbar,
            eom_dsrg.gamma1,
            eom_dsrg.eta1,
            eom_dsrg.lambda2,
            eom_dsrg.lambda3,
            eom_dsrg.lambda4,
            eom_dsrg.first_row,
        )

        HX0_dict = antisymmetrize(HX0_dict, ea=ea)
        HX0 = dict_to_vec(HX0_dict, 1).flatten()
        XHXt = eom_dsrg.apply_S12(eom_dsrg, northo, HX0, transpose=True)
        XHXt = XHXt.flatten()
        x0s.append(XHXt)

    return x0s


def compute_guess_vectors(eom_dsrg, precond, ascending=True):
    """
    Compute initial guess vectors for the Davidson algorithm.

    Parameters:
        eom_dsrg: Object containing the number of roots and preconditioner.
        precond (np.ndarray): Preconditioner vector.
        ascending (bool): Whether to sort the preconditioner in ascending order.

    Returns:
        List of initial guess vectors.
    """
    sort_ind = np.argsort(precond) if ascending else np.argsort(precond)[::-1]
    print(f"length of precond: {len(precond)}")

    x0s = np.zeros((precond.shape[0], eom_dsrg.nroots))
    min_shape = min(precond.shape[0], eom_dsrg.nroots)
    x0s[:min_shape, :min_shape] = np.identity(min_shape)

    x0 = np.zeros((precond.shape[0], eom_dsrg.nroots))
    x0[sort_ind] = x0s.copy()

    x0s = []
    for p in range(x0.shape[1]):
        x0s.append(x0[:, p])
    return x0s


def get_templates(eom_dsrg, nlow=1):
    """Generate the initial and full templates based on the method type."""
    # Dictionary mapping method types to the appropriate template functions
    template_funcs = {
        "ee": ee_eom_dsrg.get_template_c if os.path.exists("ee_eom_dsrg.py") else None,
        "cvs_ee": (
            cvs_ee_eom_dsrg.get_template_c
            if os.path.exists("cvs_ee_eom_dsrg.py")
            else None
        ),
        "cvs_ip": (
            cvs_ip_eom_dsrg.get_template_c
            if os.path.exists("cvs_ip_eom_dsrg.py")
            else None
        ),
        "ip": ip_eom_dsrg.get_template_c if os.path.exists("ip_eom_dsrg.py") else None,
        # Additional mappings for other methods can be added here
    }

    # Fetch the correct template function based on method type
    template_func = template_funcs.get(eom_dsrg.method_type)
    if template_func is None:
        raise ValueError(f"Invalid method type: {eom_dsrg.method_type}")

    # Generate the template with the specified parameters
    template = template_func(
        nlow, eom_dsrg.ncore, eom_dsrg.nocc, eom_dsrg.nact, eom_dsrg.nvir
    )

    # Create a deep copy of the template and adjust its structure
    full_template = copy.deepcopy(template)
    if eom_dsrg.method_type == "cvs_ee" and eom_dsrg.first_row:
        full_template["first"] = np.zeros(nlow).reshape(nlow, 1)
        full_template = {"first": full_template.pop("first"), **full_template}

    return template, full_template


def get_sigma_build(eom_dsrg):
    """Get the appropriate sigma build functions based on the method type."""
    if "full" in eom_dsrg.method_type:
        return (NotImplemented,) * 7
    # Dictionary mapping method types to the appropriate sigma build functions
    sigma_funcs = {
        "ee": ee_eom_dsrg if os.path.exists("ee_eom_dsrg.py") else None,
        "cvs_ee": cvs_ee_eom_dsrg if os.path.exists("cvs_ee_eom_dsrg.py") else None,
        "cvs_ip": cvs_ip_eom_dsrg if os.path.exists("cvs_ip_eom_dsrg.py") else None,
        "ip": ip_eom_dsrg if os.path.exists("ip_eom_dsrg.py") else None,
        # Additional mappings for other methods can be added here
    }

    # Fetch the correct module based on the method type
    sigma_module = sigma_funcs.get(eom_dsrg.method_type)
    if sigma_module is None:
        raise ValueError(f"Invalid method type: {eom_dsrg.method_type}")

    # Extract the specific functions from the module
    build_first_row = sigma_module.build_first_row
    build_sigma_vector_Hbar = sigma_module.build_sigma_vector_Hbar
    build_transition_dipole = sigma_module.build_transition_dipole
    get_S12 = sigma_module.get_S12
    apply_S12 = sigma_module.apply_S12
    compute_preconditioner = sigma_module.compute_preconditioner

    return (
        build_first_row,
        build_sigma_vector_Hbar,
        build_transition_dipole,
        get_S12,
        apply_S12,
        compute_preconditioner,
    )
