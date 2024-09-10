import niupy.ee_eom_dsrg as ee_eom_dsrg
import niupy.cvs_ee_eom_dsrg as cvs_ee_eom_dsrg
import numpy as np
import copy
from niupy.eom_tools import dict_to_vec, vec_to_dict, antisymmetrize
from pyscf import lib
import time

davidson = lib.linalg_helper.davidson1


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
    # Setup Davidson algorithm parameters
    start = time.time()
    print("Setting up Davidson algorithm...")
    apply_M, precond, x0, nop, S_12 = setup_davidson(eom_dsrg)
    print("Time(s) for Davidson Setup: ", time.time() - start)
    conv, e, u = davidson(lambda xs: [apply_M(x) for x in xs], x0, precond, nroots=eom_dsrg.nroots, verbose=eom_dsrg.verbose,
                          max_space=eom_dsrg.max_space, max_cycle=eom_dsrg.max_cycle, tol=eom_dsrg.tol_e, tol_residual=eom_dsrg.tol_davidson)

    spin, eigvec = get_spin_multiplicity(eom_dsrg, u, nop, S_12)
    osc_strength = compute_oscillator_strength(eom_dsrg, e, eigvec)

    return conv, e, eigvec, spin, osc_strength


def compute_oscillator_strength(eom_dsrg, eigval, eigvec):
    """
    Computes the oscillator strengths based on eigenvalues and eigenvectors.

    Parameters:
        eom_dsrg: An object with configuration parameters and functions required for calculations.
        eigval: Array of eigenvalues.
        eigvec: Array of eigenvectors.

    Returns:
        osc_strength: List of oscillator strengths.
    """
    dp1_list = build_dp1_list(eom_dsrg)

    # Compute oscillator strengths for each eigenvector
    osc_strength = [
        2.0 / 3.0 * (eigval[i_v] - eigval[0]) * compute_dipole(eom_dsrg, eigvec[:, i_v], dp1_list)
        for i_v in range(1, eigvec.shape[1])
    ]

    return osc_strength


def build_dp1_list(eom_dsrg):
    """
    Builds a list of dp1 vectors for dipole moment calculations.

    Parameters:
        eom_dsrg: An object with configuration parameters and functions required for calculations.

    Returns:
        dp1_list: List of dp1 vectors for x, y, z directions.
    """
    index_dict = {
        "c": slice(eom_dsrg.ncore, eom_dsrg.ncore+eom_dsrg.nocc),
        "a": slice(eom_dsrg.ncore+eom_dsrg.nocc, eom_dsrg.ncore+eom_dsrg.nocc+eom_dsrg.nact),
        "v": slice(eom_dsrg.ncore+eom_dsrg.nocc+eom_dsrg.nact, eom_dsrg.ncore+eom_dsrg.nocc+eom_dsrg.nact+eom_dsrg.nvir),
        "C": slice(eom_dsrg.ncore, eom_dsrg.ncore+eom_dsrg.nocc),
        "A": slice(eom_dsrg.ncore+eom_dsrg.nocc, eom_dsrg.ncore+eom_dsrg.nocc+eom_dsrg.nact),
        "V": slice(eom_dsrg.ncore+eom_dsrg.nocc+eom_dsrg.nact, eom_dsrg.ncore+eom_dsrg.nocc+eom_dsrg.nact+eom_dsrg.nvir),
        "i": slice(0, eom_dsrg.ncore),
        "I": slice(0, eom_dsrg.ncore)
    }

    dp1_list = [
        dict_to_vec({
            key: eom_dsrg.dp1[i][index_dict[key[0]], index_dict[key[1]]
                                 ].reshape(1, *eom_dsrg.template_c[key].shape[1:])
            if len(key) == 2 else np.zeros((1, *eom_dsrg.template_c[key].shape[1:]))
            for key in eom_dsrg.template_c.keys()
        }, 1).flatten()
        for i in range(3)
    ]

    return dp1_list


def compute_dipole(eom_dsrg, current_vec, dp1_list):
    """
    Computes the squared transition dipole moment for a given eigenvector.

    Parameters:
        eom_dsrg: An object with configuration parameters and functions required for calculations.
        current_vec: Current eigenvector under consideration.
        dp1_list: List of dp1 vectors for x, y, z directions.

    Returns:
        current_dipole_2: Squared transition dipole moment
    """
    current_vec_dict = vec_to_dict(eom_dsrg.full_template_c, current_vec.reshape(-1, 1))
    Sv_dict = eom_dsrg.build_sigma_vector_s(
        current_vec_dict, eom_dsrg.Hbar, eom_dsrg.gamma1,
        eom_dsrg.eta1, eom_dsrg.lambda2, eom_dsrg.lambda3, None
    )
    Sv = dict_to_vec(antisymmetrize(Sv_dict), 1).flatten()

    return sum(np.dot(dp1, Sv[1:])**2 for dp1 in dp1_list)


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
                sub_norm = np.linalg.norm(current_vec_dict[key] - current_vec_dict[upper_key])
                add_norm = np.linalg.norm(current_vec_dict[key] + current_vec_dict[upper_key])
                subtraction_norms.append(sub_norm)
                addition_norms.append(add_norm)

    return subtraction_norms, addition_norms


def get_spin_multiplicity(eom_dsrg, u, nop, S_12):
    """
    Process vectors to classify their spin multiplicities.

    Parameters:
        eom_dsrg: Object with a full_template_c attribute, used for vector processing.
        u: List of eigenvectors.
        nop: Operation parameter.
        S_12: Transformation matrix/function parameter.

    Returns:
        spin (list): List of spin classifications ("Singlet", "Triplet", or "Incorrect spin").
        eigvec (np.ndarray): Processed eigenvectors.
    """
    eigvec = np.array([apply_S_12(S_12, nop, vec, transpose=False).flatten() for vec in u]).T
    eigvec_dict = antisymmetrize(vec_to_dict(eom_dsrg.full_template_c, eigvec))
    eigvec = dict_to_vec(eigvec_dict, len(u))

    spin = []

    for i_idx in range(len(u)):
        current_vec = eigvec[:, i_idx]
        current_vec_dict = vec_to_dict(eom_dsrg.full_template_c, current_vec.reshape(-1, 1))

        subtraction_norms, addition_norms = calculate_norms(current_vec_dict)

        # Check spin classification based on calculated norms
        singlet = all(norm < 1e-4 for norm in subtraction_norms)  # The parent state is assumed to be singlet.
        triplet = all(norm < 1e-4 for norm in addition_norms) and not all(norm < 1e-4 for norm in subtraction_norms)

        if triplet:
            spin.append("Triplet")
        elif singlet:
            spin.append("Singlet")
        else:
            spin.append("Incorrect spin")

    return spin, eigvec


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
        S_12: Transformation matrix/function parameter.
    """
    eom_dsrg.first_row = eom_dsrg.build_first_row(
        eom_dsrg.full_template_c, eom_dsrg.Hbar, eom_dsrg.gamma1,
        eom_dsrg.eta1, eom_dsrg.lambda2, eom_dsrg.lambda3
    )
    eom_dsrg.build_H = eom_dsrg.build_sigma_vector_Hbar
    start = time.time()
    print("Starting S_12...")
    S_12 = eom_dsrg.get_S_12(
        eom_dsrg.template_c, eom_dsrg.gamma1, eom_dsrg.eta1,
        eom_dsrg.lambda2, eom_dsrg.lambda3, eom_dsrg.sym,
        eom_dsrg.target_sym, tol=eom_dsrg.tol_s, tol_act=eom_dsrg.tol_s_act
    )
    print("Time(s) for S_12: ", time.time() - start)
    start = time.time()
    print("Starting Precond...")

    nop = 0
    for i_tensor in S_12:
        nop += i_tensor.shape[1]
    if eom_dsrg.diagonal_type == "identity":
        print("Using Identity Preconditioner")
        precond = np.ones(nop+1)
    elif eom_dsrg.diagonal_type == "exact":
        precond = eom_dsrg.compute_preconditioner_exact(eom_dsrg.template_c, S_12, eom_dsrg.Hbar, eom_dsrg.gamma1,
                                                        eom_dsrg.eta1, eom_dsrg.lambda2, eom_dsrg.lambda3) + eom_dsrg.diag_shift
    else:
        precond = eom_dsrg.compute_preconditioner_block(eom_dsrg.template_c, S_12, eom_dsrg.Hbar, eom_dsrg.gamma1,
                                                        eom_dsrg.eta1, eom_dsrg.lambda2, eom_dsrg.lambda3) + eom_dsrg.diag_shift
    print("Precond: ", len(precond), precond.shape, type(precond))
    print("Time(s) for Precond: ", time.time() - start)
    northo = len(precond)
    nop = dict_to_vec(eom_dsrg.full_template_c, 1).shape[0]
    apply_M = define_effective_hamiltonian(eom_dsrg, S_12, nop, northo)

    x0 = compute_guess_vectors(eom_dsrg, precond)
    return apply_M, precond, x0, nop, S_12


def define_effective_hamiltonian(eom_dsrg, S_12, nop, northo):
    """
    Define the effective Hamiltonian application function.

    Parameters:
        eom_dsrg: Object with method-specific parameters.
        S_12: Transformation matrix/function parameter.
        nop: Dimension size of operators.
        northo: Dimension size of orthogonal components.

    Returns:
        Function that applies the effective Hamiltonian to a vector.
    """
    # nop and northo include the first row/column
    def apply_M(x):
        Xt = apply_S_12(S_12, nop, x, transpose=False)
        Xt_dict = vec_to_dict(eom_dsrg.full_template_c, Xt)
        Xt_dict = antisymmetrize(Xt_dict)
        HXt_dict = eom_dsrg.build_H(Xt_dict, eom_dsrg.Hbar, eom_dsrg.gamma1, eom_dsrg.eta1,
                                    eom_dsrg.lambda2, eom_dsrg.lambda3, eom_dsrg.first_row)
        HXt_dict = antisymmetrize(HXt_dict)
        HXt = dict_to_vec(HXt_dict, 1).flatten()
        XHXt = apply_S_12(S_12, northo, HXt, transpose=True)
        XHXt = XHXt.flatten()
        XHXt += eom_dsrg.diag_shift * x
        return XHXt
    return apply_M


def apply_S_12(S_12, ndim, t, transpose=False):
    """
    Apply the S_12 transformation matrix/function to a vector.

    Parameters:
        S_12: List of transformation matrices/functions.
        ndim: Resulting vector size.
        t: Input vector.
        transpose (bool): Whether to apply the transpose of the transformation.

    Returns:
        Transformed vector.
    """
    # t is a vector. S_half is a list of ndarray. ndim is the resulting vector size.
    Xt = np.zeros((ndim, 1))  # With first column/row.
    i_start_xt = 1
    i_start_t = 1
    Xt[0, 0] = t[0]

    for i_tensor in S_12:
        num_op, num_ortho = i_tensor.shape
        i_end_xt, i_end_t = i_start_xt + (num_op if not transpose else num_ortho), i_start_t + \
            (num_ortho if not transpose else num_op)
        # (nop * northo) @ (northo * 1) if not transpose else (northo * nop) @ (nop * 1)
        Xt[i_start_xt:i_end_xt, :] += (i_tensor @ t[i_start_t:i_end_t].reshape(-1, 1)
                                       if not transpose else i_tensor.T @ t[i_start_t:i_end_t].reshape(-1, 1))
        i_start_xt, i_start_t = i_end_xt, i_end_t

    return Xt


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
        "ee": ee_eom_dsrg.get_template_c,
        "cvs-ee": cvs_ee_eom_dsrg.get_template_c
        # Additional mappings for other methods can be added here
    }

    # Fetch the correct template function based on method type
    template_func = template_funcs.get(eom_dsrg.method_type)
    if template_func is None:
        raise ValueError(f"Invalid method type: {eom_dsrg.method_type}")

    # Generate the template with the specified parameters
    template = template_func(1, eom_dsrg.ncore, eom_dsrg.nocc, eom_dsrg.nact, eom_dsrg.nvir)

    # Create a deep copy of the template and adjust its structure
    full_template = copy.deepcopy(template)
    full_template['first'] = np.zeros(1).reshape(1, 1)
    full_template = {'first': full_template.pop('first'), **full_template}

    return template, full_template


def get_sigma_build(eom_dsrg):
    """Get the appropriate sigma build functions based on the method type."""
    # Dictionary mapping method types to the appropriate sigma build functions
    sigma_funcs = {
        "ee": ee_eom_dsrg,
        "cvs-ee": cvs_ee_eom_dsrg
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
    get_S_12 = sigma_module.get_S_12
    compute_preconditioner_exact = sigma_module.compute_preconditioner_exact
    compute_preconditioner_block = sigma_module.compute_preconditioner_block

    return build_first_row, build_sigma_vector_Hbar, build_sigma_vector_s, get_S_12, compute_preconditioner_exact, compute_preconditioner_block
