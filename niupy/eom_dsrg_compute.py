import niupy.ee_eom_dsrg as ee_eom_dsrg
import numpy as np
import copy
from niupy.eom_tools import dict_to_vec, vec_to_dict, antisymmetrize
from pyscf import lib
import time

davidson = lib.linalg_helper.davidson1


def kernel(eom_dsrg):
    # TODO: Logging system

    # Setup Davidson algorithm parameters
    start = time.time()
    print("Setting up Davidson algorithm...")
    apply_M, precond, x0, nop, S_12 = setup_davidson(eom_dsrg)
    print("Time(s) for Davidson Setup: ", time.time() - start)
    conv, e, u = davidson(lambda xs: [apply_M(x) for x in xs], x0, precond, nroots=eom_dsrg.nroots, verbose=eom_dsrg.verbose,
                          max_space=eom_dsrg.max_space, max_cycle=eom_dsrg.max_cycle, tol=eom_dsrg.tol_e, tol_residual=eom_dsrg.tol_davidson)

    # This part should be in a separate function
    eigvec = []
    for i_vec in range(len(u)):
        vec = apply_S_12(S_12, nop, u[i_vec], transpose=False)
        eigvec.append(vec.flatten())
    eigvec = np.array(eigvec).T

    eigvec_dict = vec_to_dict(eom_dsrg.full_template_c, eigvec)
    eigvec_dict = antisymmetrize(eigvec_dict)
    eigvec = dict_to_vec(eigvec_dict, len(e))
    eigvec = eigvec.T

    symmetry = []
    spin = []

    sym_dict = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2'}  # C2v test, should be changed.

    for i_idx, i_v in enumerate(e):

        current_vec = eigvec[i_idx]

        current_vec_dict = vec_to_dict(eom_dsrg.full_template_c, current_vec.reshape(-1, 1))
        cv_norm = np.linalg.norm(current_vec_dict['cv'] - current_vec_dict['CV'])
        ca_norm = np.linalg.norm(current_vec_dict['ca'] - current_vec_dict['CA'])
        av_norm = np.linalg.norm(current_vec_dict['av'] - current_vec_dict['AV'])

        cv_norm_tri = np.linalg.norm(current_vec_dict['cv'] + current_vec_dict['CV'])
        ca_norm_tri = np.linalg.norm(current_vec_dict['ca'] + current_vec_dict['CA'])
        av_norm_tri = np.linalg.norm(current_vec_dict['av'] + current_vec_dict['AV'])

        if cv_norm_tri < 1e-4 and ca_norm_tri < 1e-4 and av_norm_tri < 1e-4:
            spin.append("Triplet")
        elif cv_norm < 1e-4 and ca_norm < 1e-4 and av_norm < 1e-4:
            spin.append("Singlet")
        else:
            spin.append("Incorrect spin")

        large_indices = np.where(abs(current_vec) > 1e-4)[0]
        first_value = eom_dsrg.sym_vec[large_indices[0]]

        if all(eom_dsrg.sym_vec[index] == first_value for index in large_indices):
            symmetry.append(sym_dict[first_value])
        else:
            symmetry.append("Incorrect symmetry")

    return conv, e, eigvec, spin, symmetry


def setup_davidson(eom_dsrg):
    if eom_dsrg.method_type == "ee":
        eom_dsrg.first_row = ee_eom_dsrg.build_first_row(eom_dsrg.full_template_c, eom_dsrg.Hbar, eom_dsrg.gamma1, eom_dsrg.eta1,
                                                         eom_dsrg.lambda2, eom_dsrg.lambda3)
        eom_dsrg.build_H = ee_eom_dsrg.build_sigma_vector_Hbar
        S_12 = ee_eom_dsrg.get_S_12(eom_dsrg.template_c, eom_dsrg.gamma1, eom_dsrg.eta1,
                                    eom_dsrg.lambda2, eom_dsrg.lambda3, tol=eom_dsrg.tol_s)

        if eom_dsrg.diagonal_type == "exact":
            precond = ee_eom_dsrg.compute_preconditioner_exact(
                eom_dsrg.template_c, S_12, eom_dsrg.Hbar, eom_dsrg.gamma1, eom_dsrg.eta1, eom_dsrg.lambda2, eom_dsrg.lambda3)
        elif eom_dsrg.diagonal_type == "block":
            precond = ee_eom_dsrg.compute_preconditioner_block(
                eom_dsrg.template_c, S_12, eom_dsrg.Hbar, eom_dsrg.gamma1, eom_dsrg.eta1, eom_dsrg.lambda2, eom_dsrg.lambda3)
        else:
            raise ValueError("Invalid diagonal type.")
        precond += eom_dsrg.diag_shift
        northo = len(precond)
        nop = dict_to_vec(eom_dsrg.full_template_c, 1).shape[0]
        apply_M = define_effective_hamiltonian(eom_dsrg, S_12, nop, northo)

    elif eom_dsrg.method_type == "ip":
        pass
    elif eom_dsrg.method_type == "ea":
        pass
    elif eom_dsrg.method_type == "cvs-ip":
        pass
    elif eom_dsrg.method_type == "cvs-ee":
        pass
    else:
        raise ValueError("Invalid method type.")

    x0 = np.zeros(len(precond))
    x0[0] = 1.0
    return apply_M, precond, x0, nop, S_12


def define_effective_hamiltonian(eom_dsrg, S_12, nop, northo):
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
    # t is a vector. S_half is a list of ndarray. ndim is the resulting vector size.
    Xt = np.zeros((ndim, 1))  # With first column/row.
    i_start_xt = 1
    i_start_t = 1
    Xt[0, 0] = t[0]
    if not transpose:
        for i_tensor in S_12:
            num_op = i_tensor.shape[0]
            num_ortho = i_tensor.shape[1]
            i_end_xt = i_start_xt + num_op
            i_end_t = i_start_t + num_ortho
            # (nop * northo) @ (northo * 1)
            Xt[i_start_xt:i_end_xt, :] += i_tensor @ (t[i_start_t:i_end_t].reshape(-1, 1))
            i_start_t = i_end_t
            i_start_xt = i_end_xt
    else:
        for i_tensor in S_12:
            num_op = i_tensor.shape[0]
            num_ortho = i_tensor.shape[1]
            i_end_xt = i_start_xt + num_ortho
            i_end_t = i_start_t + num_op

            # (northo * nop) @ (nop * 1)
            Xt[i_start_xt:i_end_xt, :] += i_tensor.T @ (t[i_start_t:i_end_t].reshape(-1, 1))
            i_start_t = i_end_t
            i_start_xt = i_end_xt

    return Xt


def get_templates(eom_dsrg):
    if eom_dsrg.method_type == "ee":
        template = ee_eom_dsrg.get_template_c(1, eom_dsrg.nocc, eom_dsrg.nact, eom_dsrg.nvir)
        full_template = copy.deepcopy(template)
        full_template['first'] = np.zeros(1).reshape(1, 1)
        full_template = {'first': full_template.pop('first'), **full_template}

    elif eom_dsrg.method_type == "ip":
        pass
    elif eom_dsrg.method_type == "ea":
        pass
    elif eom_dsrg.method_type == "cvs-ip":
        pass
    elif eom_dsrg.method_type == "cvs-ee":
        pass
    else:
        raise ValueError("Invalid method type.")

    return template, full_template
