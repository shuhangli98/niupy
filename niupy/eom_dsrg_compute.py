import os
import functools

if os.path.exists("cvs_ee_eom_dsrg.py"):
    print("Importing cvs_ee_eom_dsrg")
    import cvs_ee_eom_dsrg
if os.path.exists("cvs_ip_eom_dsrg.py"):
    print("Importing cvs_ip_eom_dsrg")
    import cvs_ip_eom_dsrg
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
    eigh_gen,
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

    # Get spin multiplicity and process eigenvectors
    spin, eigvec, symmetry = get_information(eom_dsrg, u, nop)

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
    elif eom_dsrg.method_type in ["ip", "cvs_ip"]:
        spec_info = compute_spectroscopic_factors(eom_dsrg, eigvec)

    return conv, e, eigvec, spin, symmetry, spec_info

def compute_spectroscopic_factors(eom_dsrg, eigvec):
    assert eom_dsrg.method_type in ["ip", "cvs_ip"]
    if eom_dsrg.method_type == "ip":
        p_compute = ip_eom_dsrg.build_sigma_vector_p
    elif eom_dsrg.method_type == "cvs_ip":
        p_compute = cvs_ip_eom_dsrg.build_sigma_vector_p
    p = np.zeros(eigvec.shape[1])
    for i in range(eigvec.shape[1]):
        current_dict = vec_to_dict(eom_dsrg.full_template_c, eigvec[:, i].reshape(-1, 1))
        pi = p_compute(eom_dsrg.einsum, current_dict, None, eom_dsrg.gamma1, eom_dsrg.eta1, eom_dsrg.lambda2, eom_dsrg.lambda3, eom_dsrg.lambda4, first_row=False)
        pi = antisymmetrize(dict_to_vec(pi, 1), ea=False)
        p[i] = (pi**2).sum()
    return p

def compute_oscillator_strength(eom_dsrg, eigval, eigvec):
    """
    Compute oscillator strengths for each eigenvector.
    """
    return [0.0] + [2.0 / 3.0 * (eigval[i] - eigval[0]) * compute_dipole(eom_dsrg, eigvec[:, i]) for i in range(1, eigvec.shape[1])]


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


def get_information(eom_dsrg, u, nop, ea=False):
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
    eigvec_dict = antisymmetrize(vec_to_dict(eom_dsrg.full_template_c, eigvec), ea=ea)

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

        spin.append(assign_spin_multiplicity(eom_dsrg, current_vec_dict))
        symmetry.append(assign_spatial_symmetry(eom_dsrg, current_vec))
    
    return spin, eigvec, symmetry

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
        for k,v in current_vec_dict.items():
            if len(k) == 1:
                hole_norm += np.linalg.norm(v)
        if hole_norm > 1e-8:
            return "Doublet"
        cCv = current_vec_dict["cCv"][0,...]
        CCV = current_vec_dict["CCV"][0,...]
        ab_sum = cCv.sum()
        bb_sum = 0.0
        for i in range(CCV.shape[0]):
            for j in range(i+1, CCV.shape[1]):
                for a in range(CCV.shape[2]):
                    bb_sum += CCV[i,j,a]
        if (abs(bb_sum) - abs(ab_sum) < 1e-8):
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
        x0: Initial guess vectors.
        nop: Dimension size.
    """

    start = time.time()
    print("Starting S12...", flush=True)
    eom_dsrg.get_S12(eom_dsrg)
    print("Time(s) for S12: ", time.time() - start, flush=True)

    eom_dsrg.Hbar = np.load(f"{eom_dsrg.abs_file_path}/save_Hbar.npz")
    if "cvs" in eom_dsrg.method_type:
        eom_dsrg.Hbar = slice_H_core(eom_dsrg.Hbar, eom_dsrg.core_sym, eom_dsrg.occ_sym)

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
    eom_dsrg.build_S = eom_dsrg.build_sigma_vector_s

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
        x0 = compute_guess_vectors_from_singles(eom_dsrg, northo)
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

def compute_guess_vectors_from_singles(eom_dsrg, northo, ascending=True, ea=False):
    start = time.time()
    print("Computing initial guess...", flush=True)
    
    nsingles = 0
    temp_dict = {}
    
    for key, value in eom_dsrg.full_template_c.items():
        if (key.count('v') + key.count('V') < 1) or len(key) == 2:
            shape_block = value.shape[1:]
            nsingles += np.prod(shape_block)
            print(f"key: {key}, value: {value.shape}, nsingles: {np.prod(shape_block)}")
    for key, value in eom_dsrg.full_template_c.items():
        if (key.count('v') + key.count('V') < 1) or len(key) == 2:
            shape_block = value.shape[1:]
            temp_dict[key] = np.zeros((nsingles, *shape_block))
            
    temp_dict_vec = dict_to_vec(temp_dict, nsingles)
    np.fill_diagonal(temp_dict_vec, 1.0)
    temp_dict = vec_to_dict(temp_dict, temp_dict_vec)
    temp_dict = antisymmetrize(temp_dict, ea=ea)
    del temp_dict_vec
    
    _, temp_full_c = get_templates(eom_dsrg, nsingles)
    
    for key, value in temp_full_c.items():
        if key in temp_dict.keys():
            temp_full_c[key] = temp_dict[key].copy()
    temp_full_c_vec = dict_to_vec(temp_full_c, nsingles)
    
    H_singles = eom_dsrg.build_sigma_vector_Hbar_singles(eom_dsrg.einsum, temp_full_c, eom_dsrg.Hbar,\
        eom_dsrg.gamma1,eom_dsrg.eta1,eom_dsrg.lambda2,eom_dsrg.lambda3,eom_dsrg.lambda4)
    S_singles = eom_dsrg.build_sigma_vector_s_singles(eom_dsrg.einsum, temp_full_c, eom_dsrg.Hbar,\
        eom_dsrg.gamma1,eom_dsrg.eta1,eom_dsrg.lambda2,eom_dsrg.lambda3,eom_dsrg.lambda4)
    H_singles = antisymmetrize(H_singles, ea=ea)
    S_singles = antisymmetrize(S_singles, ea=ea)
    H_singles_vec = dict_to_vec(H_singles, nsingles)
    S_singles_vec = dict_to_vec(S_singles, nsingles)
    print("Time(s) for H, S construction: ", time.time() - start, flush=True)
    start = time.time()
    print("Start diagonalization...", flush=True)
    eigval, eigvec = eigh_gen(H_singles_vec, S_singles_vec, eta=1e-10)
    print("Time(s) for diagonalization: ", time.time() - start, flush=True)
    start = time.time()
    sort_ind = np.argsort(eigval) if ascending else np.argsort(eigval)[::-1]
    unit_vec = np.zeros(northo)
    unit_vec[0] = 1.0
    x0s = [unit_vec]
    for p in range(eom_dsrg.nroots):
        Sx = temp_full_c_vec @ eigvec[:, sort_ind[p]].reshape(-1,1)
        Sx_dict = vec_to_dict(temp_full_c, Sx)
        HSx_dict = eom_dsrg.build_H(eom_dsrg.einsum,Sx_dict,eom_dsrg.Hbar,eom_dsrg.gamma1,eom_dsrg.eta1,\
            eom_dsrg.lambda2,eom_dsrg.lambda3,eom_dsrg.lambda4,eom_dsrg.first_row)
        HSx_dict = antisymmetrize(HSx_dict, ea=ea)
        HSx = dict_to_vec(HSx_dict, 1).flatten()
        SHSx = eom_dsrg.apply_S12(eom_dsrg, northo, HSx, transpose=True).flatten()
        if np.linalg.norm(SHSx) > 1e-4:
            x0s.append(SHSx/np.linalg.norm(SHSx))
        if len(x0s) == eom_dsrg.nroots:
            break
    print("Time(s) for projecting guess vectors: ", time.time() - start, flush=True)
    print("Initial guess done.", flush=True)
    return x0s

def get_templates(eom_dsrg, nlow = 1):
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
    if eom_dsrg.method_type == "cvs_ee":
        full_template["first"] = np.zeros(nlow).reshape(nlow, 1)
        full_template = {"first": full_template.pop("first"), **full_template}

    return template, full_template


def get_sigma_build(eom_dsrg):
    """Get the appropriate sigma build functions based on the method type."""
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
    build_sigma_vector_s = sigma_module.build_sigma_vector_s
    build_transition_dipole = sigma_module.build_transition_dipole
    get_S12 = sigma_module.get_S12
    apply_S12 = sigma_module.apply_S12
    compute_preconditioner = sigma_module.compute_preconditioner
    build_sigma_vector_Hbar_singles = sigma_module.build_sigma_vector_Hbar_singles
    build_sigma_vector_s_singles = sigma_module.build_sigma_vector_s_singles

    return (
        build_first_row,
        build_sigma_vector_Hbar,
        build_sigma_vector_s,
        build_transition_dipole,
        get_S12,
        apply_S12,
        compute_preconditioner,
        build_sigma_vector_Hbar_singles,
        build_sigma_vector_s_singles,
    )
