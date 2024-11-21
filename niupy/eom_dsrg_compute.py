import os

if os.path.exists("cvs_ee_eom_dsrg.py"):
    print("Importing cvs_ee_eom_dsrg")
    import cvs_ee_eom_dsrg
if os.path.exists("ee_eom_dsrg.py"):
    print("Importing ee_eom_dsrg")
    import ee_eom_dsrg
if os.path.exists("ip_eom_dsrg.py"):
    print("Importing ip_eom_dsrg")
    import ip_eom_dsrg
import numpy as np
import copy
from niupy.eom_tools import (
    dict_to_vec,
    vec_to_dict,
    antisymmetrize,
    slice_H_core,
)
from pyscf import lib
import time

davidson = lib.linalg_helper.davidson1
dgeev1 = lib.linalg_helper.dgeev1


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
        osc_strength: Computed oscillator strengths.
    """
    start = time.time()
    print("Setting up Davidson algorithm...", flush=True)

    # Setup Davidson algorithm
    if eom_dsrg.davidson_type == "traditional":
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

        # Get spin multiplicity and process eigenvectors
        spin, eigvec, symmetry = get_information(eom_dsrg, u, nop)

    else:
        raise ValueError(f"Invalid Davidson type: {eom_dsrg.davidson_type}")

    del eom_dsrg.Hbar

    # # Get spin multiplicity and process eigenvectors
    # spin, eigvec = get_spin_multiplicity(eom_dsrg, u, nop, S_12)

    if eom_dsrg.build_transition_dipole is NotImplemented:
        osc_strength = None
    else:
        # Optimize slicing by vectorizing
        eom_dsrg.Mbar = [
            slice_H_core(M, eom_dsrg.core_sym, eom_dsrg.occ_sym) for M in eom_dsrg.Mbar
        ]

        # Compute oscillator strengths
        osc_strength = compute_oscillator_strength(eom_dsrg, e, eigvec)

    return conv, e, eigvec, spin, symmetry, osc_strength


def compute_oscillator_strength(eom_dsrg, eigval, eigvec):
    """
    Compute oscillator strengths for each eigenvector.
    """
    return [
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
            eom_dsrg.einsum_type,
            current_vec_dict,
            Mbar_i,
            eom_dsrg.gamma1,
            eom_dsrg.eta1,
            eom_dsrg.lambda2,
            eom_dsrg.lambda3,
            eom_dsrg.lambda4,
        )
        HT_first = HT + current_vec_dict["first"][0, 0] * eom_dsrg.Mbar0[i]
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


def get_information(eom_dsrg, u, nop):
    """
    Process vectors to classify their spin multiplicities.

    Parameters:
        eom_dsrg: Object with a full_template_c attribute, used for vector processing.
        u: List of eigenvectors.
        nop: Operation parameter.

    Returns:
        spin (list): List of spin classifications ("Singlet", "Triplet", or "Incorrect spin").
        eigvec (np.ndarray): Processed eigenvectors.
    """

    eigvec = np.array(
        [eom_dsrg.apply_S12(eom_dsrg, nop, vec, transpose=False).flatten() for vec in u]
    ).T
    eigvec_dict = antisymmetrize(vec_to_dict(eom_dsrg.full_template_c, eigvec))

    excitation_analysis = find_top_values(eigvec_dict, 3)
    for key, values in excitation_analysis.items():
        print(f"Root {key}: {values}")

    eigvec = dict_to_vec(eigvec_dict, len(u))

    spin = []
    symmetry = []

    for i_idx in range(len(u)):
        current_vec = eigvec[:, i_idx]
        current_vec_dict = vec_to_dict(
            eom_dsrg.full_template_c, current_vec.reshape(-1, 1)
        )

        subtraction_norms, addition_norms = calculate_norms(current_vec_dict)
        print(f"Norms {key}: {subtraction_norms}, {addition_norms}")

        # Check spin classification based on calculated norms
        singlet = all(
            norm < 1e-2 for norm in subtraction_norms
        )  # The parent state is assumed to be singlet.
        triplet = all(norm < 1e-2 for norm in addition_norms) and not all(
            norm < 1e-2 for norm in subtraction_norms
        )

        if triplet:
            spin.append("Triplet")
        elif singlet:
            spin.append("Singlet")
        else:
            spin.append("Incorrect spin")

        large_indices = np.where(abs(current_vec) > 1e-2)[0]
        first_value = eom_dsrg.sym_vec[large_indices[0]]
        if all(eom_dsrg.sym_vec[index] == first_value for index in large_indices):
            symmetry.append(first_value)
        else:
            symmetry.append("Spin contamination.")

    return spin, eigvec, symmetry


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
        x0: Initial guess vectors.
        nop: Dimension size.
    """

    start = time.time()
    print("Starting S12...", flush=True)
    eom_dsrg.get_S12(eom_dsrg)
    print("Time(s) for S12: ", time.time() - start, flush=True)

    eom_dsrg.Hbar = np.load(f"{eom_dsrg.abs_file_path}/save_Hbar.npz")
    if eom_dsrg.method_type == "cvs_ee":
        eom_dsrg.Hbar = slice_H_core(eom_dsrg.Hbar, eom_dsrg.core_sym, eom_dsrg.occ_sym)

    if eom_dsrg.build_first_row is not NotImplemented:
        eom_dsrg.first_row = None
    else:
        eom_dsrg.first_row = eom_dsrg.build_first_row(
            eom_dsrg.einsum,
            eom_dsrg.einsum_type,
            eom_dsrg.full_template_c,
            eom_dsrg.Hbar,
            eom_dsrg.gamma1,
            eom_dsrg.eta1,
            eom_dsrg.lambda2,
            eom_dsrg.lambda3,
            eom_dsrg.lambda4,
        )
    eom_dsrg.build_H = eom_dsrg.build_sigma_vector_Hbar

    # Compute preconditioner
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
    apply_M = define_effective_hamiltonian(eom_dsrg, nop, northo)

    x0 = compute_guess_vectors(eom_dsrg, precond)

    # Symmetry check
    # test = np.zeros((len(x0), len(x0)))
    # for i_x, x in enumerate(x0):
    #     for i_y, y in enumerate(x0):
    #         Mx = apply_M(x)
    #         test[i_x, i_y] = np.dot(y.T, Mx)

    # print(f"Test matrix: {np.allclose(test, test.T)}")

    return apply_M, precond, x0, nop


def define_effective_hamiltonian(eom_dsrg, nop, northo):
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
    def apply_M(x):
        Xt = eom_dsrg.apply_S12(eom_dsrg, nop, x, transpose=False)
        Xt_dict = vec_to_dict(eom_dsrg.full_template_c, Xt)
        Xt_dict = antisymmetrize(Xt_dict)
        HXt_dict = eom_dsrg.build_H(
            eom_dsrg.einsum,
            eom_dsrg.einsum_type,
            Xt_dict,
            eom_dsrg.Hbar,
            eom_dsrg.gamma1,
            eom_dsrg.eta1,
            eom_dsrg.lambda2,
            eom_dsrg.lambda3,
            eom_dsrg.lambda4,
            eom_dsrg.first_row,
        )
        HXt_dict = antisymmetrize(HXt_dict)
        HXt = dict_to_vec(HXt_dict, 1).flatten()
        XHXt = eom_dsrg.apply_S12(eom_dsrg, northo, HXt, transpose=True)
        XHXt = XHXt.flatten()
        return XHXt

    return apply_M


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
    # print(f"precond:{precond[sort_ind]}")
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


def get_templates(eom_dsrg):
    """Generate the initial and full templates based on the method type."""
    # Dictionary mapping method types to the appropriate template functions
    template_funcs = {
        "ee": ee_eom_dsrg.get_template_c if os.path.exists("ee_eom_dsrg.py") else None,
        "cvs_ee": (
            cvs_ee_eom_dsrg.get_template_c
            if os.path.exists("cvs_ee_eom_dsrg.py")
            else None
        ),
        # Additional mappings for other methods can be added here
    }

    # Fetch the correct template function based on method type
    template_func = template_funcs.get(eom_dsrg.method_type)
    if template_func is None:
        raise ValueError(f"Invalid method type: {eom_dsrg.method_type}")

    # Generate the template with the specified parameters
    template = template_func(
        1, eom_dsrg.ncore, eom_dsrg.nocc, eom_dsrg.nact, eom_dsrg.nvir
    )

    # Create a deep copy of the template and adjust its structure
    full_template = copy.deepcopy(template)
    full_template["first"] = np.zeros(1).reshape(1, 1)
    full_template = {"first": full_template.pop("first"), **full_template}

    return template, full_template


def get_sigma_build(eom_dsrg):
    """Get the appropriate sigma build functions based on the method type."""
    # Dictionary mapping method types to the appropriate sigma build functions
    sigma_funcs = {
        "ee": ee_eom_dsrg if os.path.exists("ee_eom_dsrg.py") else None,
        "cvs_ee": cvs_ee_eom_dsrg if os.path.exists("cvs_ee_eom_dsrg.py") else None,
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
    build_sigma_vector_s = sigma_module.build_sigma_vector_s
    build_transition_dipole = sigma_module.build_transition_dipole
    get_S12 = sigma_module.get_S12
    apply_S12 = sigma_module.apply_S12
    compute_preconditioner = sigma_module.compute_preconditioner

    return (
        build_first_row,
        build_sigma_vector_Hbar,
        build_sigma_vector_s,
        build_transition_dipole,
        get_S12,
        apply_S12,
        compute_preconditioner,
    )
