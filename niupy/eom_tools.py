import wicked as w
import numpy as np
import copy


def compile_sigma_vector(equation, bra_name='bra', ket_name='c'):
    eq, d = w.compile_einsum(equation, return_eq_dict=True)
    for idx, t in enumerate(d['rhs']):
        if t[0] == bra_name:
            bra_idx = idx
        if t[0] == ket_name:
            ket_idx = idx

    d['factor'] = float(d['factor'])
    d['rhs'][ket_idx][2] = 'p' + d['rhs'][ket_idx][2]
    bra = d['rhs'].pop(bra_idx)
    nbody = len(bra[2])//2
    bra[0] = 'sigma'
    bra[2] = 'p' + bra[2][nbody:] + bra[2][:nbody]
    bra[1] = bra[1][nbody:] + bra[1][:nbody]
    d['lhs'] = [bra]
    return w.dict_to_einsum(d)


def compile_first_row_safe(equation, ket_name='c'):
    eq, d = w.compile_einsum(equation, return_eq_dict=True)
    for idx, t in enumerate(d['rhs']):
        if t[0] == ket_name:
            ket_idx = idx

    d['rhs'][ket_idx][2] = 'p' + d['rhs'][ket_idx][2]
    d['lhs'][0][2] = 'p'
    return w.dict_to_einsum(d)


def compile_first_row(equation, ket_name='c'):
    eq, d = w.compile_einsum(equation, return_eq_dict=True)
    for idx, t in enumerate(d['rhs']):
        if t[0] == ket_name:
            ket_idx = idx
    ket = d['rhs'].pop(ket_idx)
    ket[0] = 'sigma'
    d['lhs'] = [ket]
    return w.dict_to_einsum(d)


def generate_sigma_build(mbeq, matrix, first_row=True):
    code = [f"def build_sigma_vector_{matrix}(c, Hbar, gamma1, eta1, lambda2, lambda3, first_row):",
            "    sigma = {}",
            "    for key in c.keys():",
            "        sigma[key] = np.zeros(c[key].shape)"]

    for eq in mbeq['|']:
        no_print = False
        rhs = eq.rhs()
        for t in rhs.tensors():
            if t.label() in ['lambda4', 'lambda5', 'lambda6']:
                no_print = True
                break
        if not no_print:
            code.append(f"    {compile_sigma_vector(eq, bra_name='bra', ket_name='c')}")

    if matrix == 'Hbar' and first_row:
        code.append("    for key in first_row.keys():")
        code.append("        if len(key) == 2:")
        code.append("           tmp = first_row[key] * c['first'][:, :, np.newaxis]")
        code.append("        elif len(key) == 4:")
        code.append("           tmp = first_row[key] * c['first'][:, :, np.newaxis, np.newaxis, np.newaxis]")
        code.append("        sigma[key] += tmp")
        code.append("    c_vec = dict_to_vec(c, c[list(c.keys())[0]].shape[0])")
        code.append("    first_row_vec = dict_to_vec(first_row, 1)")
        code.append("    sigma['first'] += np.einsum('ik, ij->jk', first_row_vec, c_vec[1:, :], optimize=True)")
    elif matrix == 's':
        code.append("    sigma['first'] = c['first'].copy()")

    code.append("    return sigma")
    funct = "\n".join(code)
    return funct


def generate_template_c(block_list, ket_name='c'):
    index_dict = {"c": "nocc", "a": "nact", "v": "nvir",
                  "C": "nocc", "A": "nact", "V": "nvir",
                  "i": "ncore", "I": "ncore"}

    code = [f"def get_template_c(nlow, ncore, nocc, nact, nvir):",
            "    c = {"]

    for i in block_list:
        shape_strings = ['nlow'] + [f"{index_dict[item]}" for item in i]
        shape_formatted = ', '.join(shape_strings)
        code.append(f"         '{i}': np.zeros(({shape_formatted})),")

    code.append("        }")
    code.append("    return c")
    code = '\n'.join(code)
    return code


def generate_first_row_safe(mbeq):
    # This is a safe way to generate (\sum_i <\Phi|Hbar|\Phi_i> c_i).
    code = [f"def build_first_row_safe(c, Hbar, gamma1, eta1, lambda2, lambda3):",
            "    nlow = c[list(c.keys())[0]].shape[0]",
            "    sigma = np.zeros(nlow)"]
    for eq in mbeq['|']:
        no_print = False
        rhs = eq.rhs()
        for t in rhs.tensors():
            if t.label() in ['lambda4', 'lambda5', 'lambda6']:
                no_print = True
                break
        if not no_print:
            code.append(f"    {compile_first_row_safe(eq, ket_name='c')}")

    code.append("    return sigma")
    funct = "\n".join(code)
    return funct


def generate_first_row(mbeq):
    code = [f"def build_first_row(c, Hbar, gamma1, eta1, lambda2, lambda3):",
            "    sigma = {}",
            "    for key in c.keys():",
            "       if key == 'first':",
            "           continue",
            "       sigma[key] = np.zeros((1, *c[key].shape[1:]))"]
    for eq in mbeq['|']:
        no_print = False
        rhs = eq.rhs()
        for t in rhs.tensors():
            if t.label() in ['lambda4', 'lambda5', 'lambda6']:
                no_print = True
                break
        if not no_print:
            code.append(f"    {compile_first_row(eq, ket_name='c')}")

    code.append("    return sigma")
    funct = "\n".join(code)
    return funct


def dict_to_vec(dictionary, n_lowest):
    reshape_vec = [np.reshape(value, (n_lowest, -1)) for value in dictionary.values()]
    vec = np.concatenate(reshape_vec, axis=1)
    return vec.T


def vec_to_dict(dict, vec):
    new_dict = {}
    ncol = vec.shape[1]
    start = 0
    for key, value in dict.items():
        shape = list(value.shape)[1:]
        num_elements = np.prod(shape)
        end = start + num_elements
        array_slice = vec[start:end, :]
        new_value = np.zeros((ncol, *shape))
        for i in range(ncol):
            new_value[i] = array_slice[:, i].reshape(shape)
        new_dict[key] = new_value
        start = end
    return new_dict


def generate_block_contraction(block_str, mbeq, block_type='single', indent='once', bra_name='bra', ket_name='c'):
    code = []
    if indent == 'once':
        space = "    "
    elif indent == 'twice':
        space = "        "
    for eq in mbeq['|']:
        no_print = False
        correct_contraction = False
        rhs = eq.rhs()
        for t in rhs.tensors():
            if t.label() in ['lambda4', 'lambda5', 'lambda6']:
                no_print = True
                break
            elif t.label() == bra_name:
                bra_label = ''.join([str(_)[0] for _ in t.lower()]) + ''.join([str(_)[0] for _ in t.upper()])
            elif t.label() == ket_name:
                ket_label = ''.join([str(_)[0] for _ in t.upper()]) + ''.join([str(_)[0] for _ in t.lower()])

        if block_type == 'single':
            correct_contraction = bra_label == ket_label == block_str
        elif block_type == 'composite':
            correct_contraction = (bra_label in block_str) and (ket_label in block_str)

        if (not no_print) and correct_contraction:
            code.append(f"{space}{compile_sigma_vector(eq, bra_name=bra_name, ket_name=ket_name)}")

    code.append(f"{space}sigma = antisymmetrize(sigma)")

    func = "\n".join(code)
    return func


def generate_S_12(mbeq, single_space, composite_space, tol=1e-4, tol_act=1e-2):
    '''
    single_space: a list of string.
    composite_space: a list of list of string.
    '''
    code = [
        f"def get_S_12(template_c, gamma1, eta1, lambda2, lambda3, sym_dict, target_sym, tol = {tol}, tol_act = {tol_act}):"]
    code.append("    sigma = {}")
    code.append("    c = {}")
    code.append("    S_12 = []")
    code.append("    sym_space = {}\n")

    # The single space
    for key in single_space:
        code.append(f"    # {key} block")
        code.append(f"    shape_block = template_c['{key}'].shape[1:]")
        code.append(f"    shape_size = np.prod(shape_block)")
        code.append(f"    c['{key}'] = np.zeros((shape_size, *shape_block))")
        code.append(f"    sigma['{key}'] = np.zeros((shape_size, *shape_block))")

        code.append(f"    sym_space['{key}'] = sym_dict['{key}']")
        code.append(f"    sym_vec = dict_to_vec(sym_space, 1).flatten()")

        code.append(f"    c_vec = dict_to_vec(c, shape_size)")
        code.append(f"    np.fill_diagonal(c_vec, 1)")
        code.append(f"    c = vec_to_dict(c, c_vec)")
        code.append(f"    c = antisymmetrize(c)\n")

        func = generate_block_contraction(key, mbeq, block_type='single', indent='once')
        code.append(f"{func}\n")

        code.append(f"    vec = dict_to_vec(sigma, shape_size)")

        code.append(f"    x_index, y_index = np.ogrid[:vec.shape[0], :vec.shape[1]]")
        code.append(f"    mask = (sym_vec[x_index] == target_sym) & (sym_vec[y_index] == target_sym)")
        code.append(f"    vec[~mask] = 0")

        code.append(f"    sevals, sevecs = np.linalg.eigh(vec)")
        code.append(f"    trunc_indices = np.where(sevals > tol)[0]")
        code.append(f"    X = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])")
        code.append(f"    S_12.append(X)\n")

        code.append("    sigma.clear()")
        code.append("    sym_space.clear()")
        code.append("    c.clear()\n")

    # The composite space
    for space in composite_space:
        code.append(f"    # {space} composite block")
        code.append(f"    shape_size = 0")
        code.append(f"    for key in {space}:")
        code.append(f"        shape_block = template_c[key].shape[1:]")
        code.append(f"        shape_size += np.prod(shape_block)")
        code.append(f"        sym_space[key] = sym_dict[key]")
        code.append(f"    for key in {space}:")
        code.append(f"        shape_block = template_c[key].shape[1:]")
        code.append(f"        c[key] = np.zeros((shape_size, *shape_block))")
        code.append(f"        sigma[key] = np.zeros((shape_size, *shape_block))")

        code.append(f"    sym_vec = dict_to_vec(sym_space, 1).flatten()")

        code.append(f"    c_vec = dict_to_vec(c, shape_size)")
        code.append(f"    np.fill_diagonal(c_vec, 1)")
        code.append(f"    c = vec_to_dict(c, c_vec)")
        code.append(f"    c = antisymmetrize(c)\n")

        func = generate_block_contraction(space, mbeq, block_type='composite', indent='once')
        code.append(f"{func}\n")

        code.append(f"    vec = dict_to_vec(sigma, shape_size)")

        code.append(f"    x_index, y_index = np.ogrid[:vec.shape[0], :vec.shape[1]]")
        code.append(f"    mask = (sym_vec[x_index] == target_sym) & (sym_vec[y_index] == target_sym)")
        code.append(f"    vec[~mask] = 0")

        code.append(f"    sevals, sevecs = np.linalg.eigh(vec)")
        if space == ['aa', 'AA', 'aaaa', 'AAAA', 'aAaA']:
            code.append(f"    trunc_indices = np.where(sevals > tol_act)[0]")
        else:
            code.append(f"    trunc_indices = np.where(sevals > tol)[0]")
        code.append(f"    X = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])")
        code.append(f"    S_12.append(X)\n")

        code.append("    sigma.clear()")
        code.append("    sym_space.clear()")
        code.append("    c.clear()\n")

    code.append("    return S_12")
    funct = "\n".join(code)
    return funct


def generate_preconditioner(mbeq, single_space, composite_space, diagonal_type='exact'):
    code = [f"def compute_preconditioner_{diagonal_type}(template_c, S_12, Hbar, gamma1, eta1, lambda2, lambda3):"]
    code.append("    sigma = {}")
    code.append("    c = {}")
    code.append("    diagonal = [np.array([0.0])]")
    for i_key, key in enumerate(single_space):
        code.append(f"    # {key} block")
        code.append(f"    shape_block = template_c['{key}'].shape[1:]")
        code.append(f"    tensor = S_12[{i_key}]")
        code.append(f"    northo = tensor.shape[1]")
        code.append(f"    if northo != 0:")
        code.append(f"        c['{key}'] = np.zeros((northo, *shape_block))")
        code.append(f"        sigma['{key}'] = np.zeros((northo, *shape_block))")
        code.append(f"        c = vec_to_dict(c, tensor)")
        code.append(f"        c = antisymmetrize(c)\n")

        func_h = generate_block_contraction(key, mbeq, block_type='single', indent='twice')
        code.append(f"{func_h}\n")

        code.append(f"        vec = dict_to_vec(sigma, northo)")
        code.append(f"        vmv = tensor.T @ vec")
        code.append(f"        diagonal.append(vmv.diagonal())\n")

        code.append("    sigma.clear()")
        code.append("    c.clear()\n")

    code.append(f"    start = len({single_space})\n")

    for space in composite_space:
        code.append(f"    # {space} composite block")
        code.append(f"    tensor = S_12[start]")
        code.append(f"    northo = tensor.shape[1]")
        code.append(f"    if northo != 0:")
        code.append(f"        vmv = np.zeros((northo, northo))")
        if diagonal_type == 'exact':
            code.append(f"        for key in {space}:")
            code.append(f"            shape_block = template_c[key].shape[1:]")
            code.append(f"            c[key] = np.zeros((northo, *shape_block))")
            code.append(f"            sigma[key] = np.zeros((northo, *shape_block))")
            code.append(f"        c = vec_to_dict(c, tensor)")
            code.append(f"        c = antisymmetrize(c)\n")

            func_h = generate_block_contraction(space, mbeq, block_type='composite', indent='twice')
            code.append(f"{func_h}\n")

            code.append(f"        vec = dict_to_vec(sigma, northo)")
            code.append(f"        vmv = tensor.T @ vec")
        elif diagonal_type == 'block':
            code.append(f"        slice_tensor = 0")
            for key_space in space:
                code.append(f"        # {key_space} sub-block")
                code.append(f"        shape_block = template_c['{key_space}'].shape[1:]")
                code.append(f"        shape_size = np.prod(shape_block)")
                code.append(f"        c['{key_space}'] = np.zeros((shape_size, *shape_block))")
                code.append(f"        sigma['{key_space}'] = np.zeros((shape_size, *shape_block))")
                code.append(f"        c_vec = dict_to_vec(c, shape_size)")
                code.append(f"        np.fill_diagonal(c_vec, 1)")
                code.append(f"        c = vec_to_dict(c, c_vec)")
                code.append(f"        c = antisymmetrize(c)\n")

                func_h = generate_block_contraction(key_space, mbeq, block_type='single', indent='twice')
                code.append(f"{func_h}\n")

                code.append(f"        H_temp = dict_to_vec(sigma, shape_size)")
                code.append(f"        S_temp = tensor[slice_tensor:slice_tensor+shape_size, :]")
                code.append(f"        vmv += S_temp.T @ H_temp @ S_temp")
                code.append(f"        slice_tensor += shape_size")
                code.append(f"        sigma.clear()")
                code.append(f"        c.clear()\n")

        code.append(f"        diagonal.append(vmv.diagonal())\n")

        code.append("    start += 1")
        code.append("    sigma.clear()")
        code.append("    c.clear()\n")

    code.append("    full_diag = np.concatenate(diagonal)")
    code.append("    return full_diag")

    funct = "\n".join(code)
    return funct


def antisymmetrize_tensor_2_2(Roovv, nlow, nocc, nvir):
    # antisymmetrize the tensor
    Roovv_anti = np.zeros((nlow, nocc, nocc, nvir, nvir))
    Roovv_anti += np.einsum("pijab->pijab", Roovv)
    Roovv_anti -= np.einsum("pijab->pjiab", Roovv)
    Roovv_anti -= np.einsum("pijab->pijba", Roovv)
    Roovv_anti += np.einsum("pijab->pjiba", Roovv)
    return Roovv_anti


def antisymmetrize_tensor_2_1(Rccav, nlow, nocc, nact, nvir, method='ee'):
    # antisymmetrize the tensor
    if method == 'ee':
        Rccav_anti = np.zeros((nlow, nocc, nocc, nact, nvir))
        Rccav_anti += np.einsum("pijab->pijab", Rccav)
        Rccav_anti -= np.einsum("pijab->pjiab", Rccav)
    # elif method == 'ip':
    #     Rccav_anti = np.zeros((nlow, nocc, nocc, nact))
    #     Rccav_anti += np.einsum("pija->pija", Rccav)
    #     Rccav_anti -= np.einsum("pija->pjia", Rccav)
    return Rccav_anti


def antisymmetrize_tensor_1_2(Rcavv, nlow, nocc, nact, nvir, method='ee'):
    # antisymmetrize the tensor
    Rcavv_anti = np.zeros((nlow, nocc, nact, nvir, nvir))
    Rcavv_anti += np.einsum("pijab->pijab", Rcavv)
    Rcavv_anti -= np.einsum("pijab->pijba", Rcavv)
    return Rcavv_anti


def antisymmetrize(input_dict, method='ee'):
    if (type(input_dict) is dict):
        if method == 'ee':
            for key in input_dict.keys():
                if len(key) == 4:
                    if key[0] == key[1] and key[2] != key[3]:
                        tensor = input_dict[key]
                        input_dict[key] = antisymmetrize_tensor_2_1(
                            tensor, tensor.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[4])
                    elif key[0] != key[1] and key[2] == key[3]:
                        tensor = input_dict[key]
                        input_dict[key] = antisymmetrize_tensor_1_2(
                            tensor, tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3])
                    elif key[0] == key[1] and key[2] == key[3]:
                        tensor = input_dict[key]
                        input_dict[key] = antisymmetrize_tensor_2_2(
                            tensor, tensor.shape[0], tensor.shape[1], tensor.shape[3])
                    else:
                        continue
        # elif method == 'ip':
        #     for key in input_dict.keys():
        #         if len(key) == 3:
        #             if key[0] == key[1]:
        #                 tensor = input_dict[key]
        #                 input_dict[key] = antisymmetrize_tensor_2_1(
        #                     tensor, tensor.shape[0], tensor.shape[1], tensor.shape[3], None, method = 'ip')
        #             else:
        #                 continue
    return input_dict


def is_hermitian(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        print("Matrix is not square")
        return False

    # Check if the matrix is equal to its conjugate transpose
    # print(matrix)
    return np.allclose(matrix, matrix.conj().T)


def eigh_gen(A, S, eta=1e-14):
    if not is_hermitian(S):
        raise ValueError(f"Matrix S is not Hermitian. Max non-Hermicity is {np.max(np.abs(S - S.conj().T))}")
    if not is_hermitian(A):
        raise ValueError(f"Matrix A is not Hermitian. Max non-Hermicity is {np.max(np.abs(A - A.conj().T))}")
    sevals, sevecs = np.linalg.eigh(S)
    trunc_indices = np.where(sevals > eta)[0]
    X = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])
    Ap = X.T @ A @ X
    eigval, eigvec = np.linalg.eigh(Ap)
    eigvec = X @ eigvec
    return eigval, eigvec


def is_antisymmetric_tensor_2_2(tensor):
    return np.allclose(tensor, -tensor.transpose(0, 2, 1, 3, 4)) and np.allclose(tensor, -tensor.transpose(0, 1, 2, 4, 3)) and np.allclose(tensor, tensor.transpose(0, 2, 1, 4, 3))


def is_antisymmetric_tensor_2_1(tensor):
    return np.allclose(tensor, -tensor.transpose(0, 2, 1, 3, 4))


def is_antisymmetric_tensor_1_2(tensor):
    return np.allclose(tensor, -tensor.transpose(0, 1, 2, 4, 3))


def is_antisymmetric(tensor_dict):
    for key, tensor in tensor_dict.items():
        if len(key) == 4:
            if key[0] == key[1] and key[2] != key[3]:
                if not is_antisymmetric_tensor_2_1(tensor):
                    raise ValueError('Subspace is not antisymmetric.')
            elif key[0] != key[1] and key[2] == key[3]:
                if not is_antisymmetric_tensor_1_2(tensor):
                    raise ValueError('Subspace is not antisymmetric.')
            elif key[0] == key[1] and key[2] == key[3]:
                if not is_antisymmetric_tensor_2_2(tensor):
                    raise ValueError('Subspace is not antisymmetric.')
    return True


def sym_dir(c, occ_sym, act_sym, vir_sym):
    out_dir = {}
    dir = {'c': occ_sym, 'C': occ_sym, 'a': act_sym, 'A': act_sym, 'v': vir_sym, 'V': vir_sym}
    for key in c.keys():
        if len(key) == 2:
            out_dir[key] = dir[key[0]][:, None] ^ dir[key[1]][None, :]
        elif len(key) == 4:
            out_dir[key] = dir[key[0]][:, None, None, None] ^ dir[key[1]][None, :, None,
                                                                          None] ^ dir[key[2]][None, None, :, None] ^ dir[key[3]][None, None, None, :]
        else:
            out_dir[key] = np.array([0])  # First
    return out_dir
