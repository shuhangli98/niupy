import wicked as w
import numpy as np
import copy
import itertools


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
    code = [
        f"def build_sigma_vector_{matrix}(c, Hbar, gamma1, eta1, lambda2, lambda3, first_row):",
        "    sigma = {key: np.zeros(c[key].shape) for key in c.keys()}"
    ]

    for eq in mbeq['|']:
        if not any(t.label() in ['lambda4', 'lambda5', 'lambda6'] for t in eq.rhs().tensors()):
            code.append(f"    {compile_sigma_vector(eq)}")

    if matrix == 'Hbar' and first_row:
        code.extend([
            "    for key in first_row.keys():",
            "        tmp = first_row[key] * c['first'][..., np.newaxis]",
            "        sigma[key] += tmp",
            "    c_vec = dict_to_vec(c, c[list(c.keys())[0]].shape[0])",
            "    first_row_vec = dict_to_vec(first_row, 1)",
            "    sigma['first'] += np.einsum('ik, ij->jk', first_row_vec, c_vec[1:, :], optimize=True)"
        ])
    elif matrix == 's':
        code.append("    sigma['first'] = c['first'].copy()")

    code.append("    return sigma")
    return "\n".join(code)


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


def generate_first_row(mbeq):
    code = [f"def build_first_row(c, Hbar, gamma1, eta1, lambda2, lambda3):",
            "    sigma = {key: np.zeros((1, *tensor.shape[1:])) for key, tensor in c.items() if key != 'first'}"]
    for eq in mbeq['|']:
        if not any(t.label() in ['lambda4', 'lambda5', 'lambda6'] for t in eq.rhs().tensors()):
            code.append(f"    {compile_first_row(eq, ket_name='c')}")

    code.append("    return sigma")
    funct = "\n".join(code)
    return funct


def generate_transition_dipole(mbeq):
    code = [f"def build_transition_dipole(c, Hbar, gamma1, eta1, lambda2, lambda3):",
            "    sigma = 0.0"]
    for eq in mbeq['|']:
        if not any(t.label() in ['lambda4', 'lambda5', 'lambda6'] for t in eq.rhs().tensors()):
            code.append(f"    {w.compile_einsum(eq)}")

    code.append("    return sigma")
    funct = "\n".join(code)
    return funct


def dict_to_vec(dictionary, n_lowest):
    reshape_vec = [np.reshape(value, (n_lowest, -1)) for value in dictionary.values()]
    vec = np.concatenate(reshape_vec, axis=1)
    return vec.T


def vec_to_dict(dict_template, vec):
    new_dict = {}
    ncol = vec.shape[1]
    start = 0
    for key, value in dict_template.items():
        shape = value.shape[1:]
        num_elements = np.prod(shape)
        end = start + num_elements
        array_slice = vec[start:end, :].T.reshape((ncol, *shape))
        new_dict[key] = array_slice
        start = end
    return new_dict


def generate_block_contraction(block_str, mbeq, block_type='single', indent='once', bra_name='bra', ket_name='c'):
    indent_spaces = {"once": "    ", "twice": "        "}
    space = indent_spaces.get(indent, "    ")
    code = []

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

        if not no_print:
            if block_type == 'single':
                correct_contraction = bra_label == ket_label == block_str
            elif block_type == 'composite':
                correct_contraction = bra_label in block_str and ket_label in block_str

            if correct_contraction:
                code.append(f"{space}{compile_sigma_vector(eq, bra_name=bra_name, ket_name=ket_name)}")

    code.append(f"{space}sigma = antisymmetrize(sigma)")

    func = "\n".join(code)
    return func


def generate_S_12(mbeq, single_space, composite_space, tol=1e-4, tol_act=1e-2):
    '''
    single_space: a list of strings.
    composite_space: a list of lists of strings.
    '''
    code = [
        f"def get_S_12(template_c, gamma1, eta1, lambda2, lambda3, sym_dict, target_sym, tol={tol}, tol_act={tol_act}):",
        "    sigma = {}",
        "    c = {}",
        "    S_12 = []",
        "    sym_space = {}"
    ]

    def add_single_space_code(key):
        return [
            f"    # {key} block",
            f'    print("Starts {key} block", flush=True)',
            f"    shape_block = template_c['{key}'].shape[1:]",
            f"    shape_size = np.prod(shape_block)",
            f"    c['{key}'] = np.zeros((shape_size, *shape_block))",
            f"    sigma['{key}'] = np.zeros((shape_size, *shape_block))",
            f"    sym_space['{key}'] = sym_dict['{key}']",
            f"    sym_vec = dict_to_vec(sym_space, 1).flatten()",
            f"    c_vec = dict_to_vec(c, shape_size)",
            f"    np.fill_diagonal(c_vec, 1)",
            f"    c = vec_to_dict(c, c_vec)",
            f"    c = antisymmetrize(c)",
            f"    print('Starts contraction', flush=True)",
            generate_block_contraction(key, mbeq, block_type='single', indent='once'),
            f"    vec = dict_to_vec(sigma, shape_size)",
            f"    x_index, y_index = np.ogrid[:vec.shape[0], :vec.shape[1]]",
            f"    mask = (sym_vec[x_index] == target_sym) & (sym_vec[y_index] == target_sym)",
            f"    vec[~mask] = 0",
            f"    print('Starts diagonalization', flush=True)",
            f"    sevals, sevecs = scipy.linalg.eigh(vec)",
            f"    print('Diagonalization done', flush=True)",
            f"    trunc_indices = np.where(sevals > tol)[0]",
            f"    X = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])",
            "    S_12.append(X)",
            "    sigma.clear()",
            "    sym_space.clear()",
            "    c.clear()",
            "    del sym_vec, c_vec, vec, x_index, y_index, mask, sevals, sevecs, trunc_indices, X"
        ]

    def add_composite_space_code(space):
        code_block = [
            f"    # {space} composite block",
            f'    print("Starts {space} composite block", flush=True)',
            f"    shape_size = 0"
        ]
        for key in space:
            code_block.extend([
                f"    shape_block = template_c['{key}'].shape[1:]",
                f"    shape_size += np.prod(shape_block)",
                f"    sym_space['{key}'] = sym_dict['{key}']"
            ])
        code_block.extend([
            f"    for key in {space}:",
            f"        shape_block = template_c[key].shape[1:]",
            f"        c[key] = np.zeros((shape_size, *shape_block))",
            f"        sigma[key] = np.zeros((shape_size, *shape_block))",
            f"    sym_vec = dict_to_vec(sym_space, 1).flatten()",
            f"    c_vec = dict_to_vec(c, shape_size)",
            f"    np.fill_diagonal(c_vec, 1)",
            f"    c = vec_to_dict(c, c_vec)",
            f"    c = antisymmetrize(c)",
            f"    print('Starts contraction', flush=True)",
        ])
        code_block.append(generate_block_contraction(space, mbeq, block_type='composite', indent='once'))
        code_block.extend([
            f"    vec = dict_to_vec(sigma, shape_size)",
            f"    x_index, y_index = np.ogrid[:vec.shape[0], :vec.shape[1]]",
            f"    mask = (sym_vec[x_index] == target_sym) & (sym_vec[y_index] == target_sym)",
            f"    vec[~mask] = 0",
            f"    print('Starts diagonalization', flush=True)",
            f"    sevals, sevecs = scipy.linalg.eigh(vec)",
            f"    print('Diagonalization done', flush=True)",
        ])
        if space == ['aa', 'AA', 'aaaa', 'AAAA', 'aAaA']:
            code_block.append(f"    trunc_indices = np.where(sevals > tol_act)[0]")
        else:
            code_block.append(f"    trunc_indices = np.where(sevals > tol)[0]")
        code_block.extend([
            f"    X = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])",
            "    S_12.append(X)",
            "    sigma.clear()",
            "    sym_space.clear()",
            "    c.clear()",
            "    del sym_vec, c_vec, vec, x_index, y_index, mask, sevals, sevecs, trunc_indices, X "
        ])
        return code_block

    # Add single space code blocks
    for key in single_space:
        code.extend(add_single_space_code(key))
        code.append("")  # Blank line for separation

    # Add composite space code blocks
    for space in composite_space:
        code.extend(add_composite_space_code(space))
        code.append("")  # Blank line for separation

    code.append("    return S_12")
    return "\n".join(code)


def generate_preconditioner(mbeq, single_space, composite_space, diagonal_type='exact'):
    def add_single_space_code(key, i_key):
        return [
            f"    # {key} block",
            f'    print("Starts {key} block precond", flush=True)',
            f"    shape_block = template_c['{key}'].shape[1:]",
            f"    tensor = S_12[{i_key}]",
            f"    northo = tensor.shape[1]",
            f"    if northo != 0:",
            f"        c['{key}'] = np.zeros((northo, *shape_block))",
            f"        sigma['{key}'] = np.zeros((northo, *shape_block))",
            f"        c = vec_to_dict(c, tensor)",
            f"        c = antisymmetrize(c)",
            generate_block_contraction(key, mbeq, block_type='single', indent='twice'),
            f"        vec = dict_to_vec(sigma, northo)",
            f"        vmv = tensor.T @ vec",
            f"        diagonal.append(vmv.diagonal())",
            f"        del vec, vmv",
            "    sigma.clear()",
            "    c.clear()"
        ]

    def add_composite_space_code(space, start):
        code_block = [
            f"    # {space} composite block",
            f'    print("Starts {space} composite block precond", flush=True)',
            f"    tensor = S_12[{start}]",
            f"    northo = tensor.shape[1]",
            f"    if northo != 0:",
            f"        vmv = np.zeros((northo, northo))"
        ]

        if diagonal_type == 'exact':
            code_block.extend([
                f"        for key in {space}:",
                f"            shape_block = template_c[key].shape[1:]",
                f"            c[key] = np.zeros((northo, *shape_block))",
                f"            sigma[key] = np.zeros((northo, *shape_block))",
                f"        c = vec_to_dict(c, tensor)",
                f"        c = antisymmetrize(c)",
                generate_block_contraction(space, mbeq, block_type='composite', indent='twice'),
                f"        vec = dict_to_vec(sigma, northo)",
                f"        vmv = tensor.T @ vec",
                f"        diagonal.append(vmv.diagonal())",
                f"        del vec, vmv",
            ])
        elif diagonal_type == 'block':
            code_block.extend([
                f"        slice_tensor = 0"
            ])
            for key_space in space:
                code_block.extend([
                    f"        # {key_space} sub-block",
                    f'        print("Starts {key_space} sub-block precond", flush=True)',
                    f"        shape_block = template_c['{key_space}'].shape[1:]",
                    f"        shape_size = np.prod(shape_block)",
                    f"        c['{key_space}'] = np.zeros((shape_size, *shape_block))",
                    f"        sigma['{key_space}'] = np.zeros((shape_size, *shape_block))",
                    f"        c_vec = dict_to_vec(c, shape_size)",
                    f"        np.fill_diagonal(c_vec, 1)",
                    f"        c = vec_to_dict(c, c_vec)",
                    f"        c = antisymmetrize(c)",
                    generate_block_contraction(key_space, mbeq, block_type='single', indent='twice'),
                    f"        H_temp = dict_to_vec(sigma, shape_size)",
                    f"        S_temp = tensor[slice_tensor:slice_tensor+shape_size, :]",
                    f"        vmv += S_temp.T @ H_temp @ S_temp",
                    f"        slice_tensor += shape_size",
                    f"        sigma.clear()",
                    f"        c.clear()",
                    f"        del c_vec, H_temp, S_temp",
                ])
        code_block.append(f"        diagonal.append(vmv.diagonal())")
        code_block.extend([
            "    sigma.clear()",
            "    c.clear()"
        ])
        return code_block

    code = [
        f"def compute_preconditioner_{diagonal_type}(template_c, S_12, Hbar, gamma1, eta1, lambda2, lambda3):",
        "    sigma = {}",
        "    c = {}",
        "    diagonal = [np.array([0.0])]",
    ]

    # Add single space code blocks
    for i_key, key in enumerate(single_space):
        code.extend(add_single_space_code(key, i_key))
        code.append("")  # Blank line for separation

    start = len(single_space)

    # Add composite space code blocks
    for space in composite_space:
        code.extend(add_composite_space_code(space, start))
        start += 1  # Update start index
        code.append("")  # Blank line for separation

    code.append("    full_diag = np.concatenate(diagonal)")
    code.append("    return full_diag")

    return "\n".join(code)


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


def sym_dir(c, core_sym, occ_sym, act_sym, vir_sym):
    out_dir = {}
    dir = {'c': occ_sym, 'C': occ_sym, 'a': act_sym, 'A': act_sym,
           'v': vir_sym, 'V': vir_sym, 'i': core_sym, 'I': core_sym}
    for key in c.keys():
        if len(key) == 2:
            out_dir[key] = dir[key[0]][:, None] ^ dir[key[1]][None, :]
        elif len(key) == 4:
            out_dir[key] = dir[key[0]][:, None, None, None] ^ dir[key[1]][None, :, None,
                                                                          None] ^ dir[key[2]][None, None, :, None] ^ dir[key[3]][None, None, None, :]
        else:
            out_dir[key] = np.array([0])  # First
    return out_dir


def slice_H_core(Hbar_old, core_sym, occ_sym):
    # Combine and sort all orbitals by symmetry to get the correct order
    all_orbitals = np.sort(np.concatenate((core_sym, occ_sym)))

    # Initialize core and occupied indices lists
    core_indices = []
    occ_indices = []
    used_indices = set()

    # Create a counter for core and occupied symmetries
    core_sym_count = {sym: np.sum(core_sym == sym) for sym in np.unique(core_sym)}
    occ_sym_count = {sym: np.sum(occ_sym == sym) for sym in np.unique(occ_sym)}

    # Find core indices based on required counts in core_sym
    for sym, count in core_sym_count.items():
        indices = np.where(all_orbitals == sym)[0]
        selected_indices = [idx for idx in indices if idx not in used_indices][:count]
        core_indices.extend(selected_indices)
        used_indices.update(selected_indices)

    # Find occupied indices based on required counts in occ_sym
    for sym, count in occ_sym_count.items():
        indices = np.where(all_orbitals == sym)[0]
        selected_indices = [idx for idx in indices if idx not in used_indices][:count]
        occ_indices.extend(selected_indices)
        used_indices.update(selected_indices)

    # Initialize indices dictionary
    indices_dict = {
        'i': core_indices, 'c': occ_indices, 'a': ..., 'v': ...,
        'I': core_indices, 'C': occ_indices, 'A': ..., 'V': ...
    }

    # Initialize the new dictionary to hold the sliced tensors
    Hbar = {}

    for key, tensor in Hbar_old.items():
        indices_c = [index for index, char in enumerate(key) if char == 'c']
        indices_C = [index for index, char in enumerate(key) if char == 'C']
        count_c = len(indices_c)
        count_C = len(indices_C)

        # Generate all combinations of 'c'/'i' and 'C'/'I' replacements
        combinations_c = list(itertools.product(['c', 'i'], repeat=count_c)) if count_c > 0 else [[]]
        combinations_C = list(itertools.product(['C', 'I'], repeat=count_C)) if count_C > 0 else [[]]

        # Iterate through all combinations of 'c' and 'C' replacements
        for comb_c in combinations_c:
            for comb_C in combinations_C:
                new_key = list(key)
                for i, char in zip(indices_c, comb_c):
                    new_key[i] = char
                for i, char in zip(indices_C, comb_C):
                    new_key[i] = char
                new_key = ''.join(new_key)

                if len(new_key) == 2:
                    first_dim_indices = indices_dict[new_key[0]]
                    second_dim_indices = indices_dict[new_key[1]]
                    Hbar[new_key] = tensor[first_dim_indices, :][:, second_dim_indices]
                elif len(new_key) == 4:
                    first_dim_indices = indices_dict[new_key[0]]
                    second_dim_indices = indices_dict[new_key[1]]
                    third_dim_indices = indices_dict[new_key[2]]
                    fourth_dim_indices = indices_dict[new_key[3]]
                    Hbar[new_key] = tensor[first_dim_indices, :, :, :][:, second_dim_indices,
                                                                       :, :][:, :, third_dim_indices, :][:, :, :, fourth_dim_indices]

    return Hbar
