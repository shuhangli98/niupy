import wicked as w
import numpy as np
import copy
import itertools


def compile_sigma_vector(equation, bra_name="bra", ket_name="c"):
    eq, d = w.compile_einsum(equation, return_eq_dict=True)
    for idx, t in enumerate(d["rhs"]):
        if t[0] == bra_name:
            bra_idx = idx
        if t[0] == ket_name:
            ket_idx = idx

    d["factor"] = float(d["factor"])
    d["rhs"][ket_idx][2] = "p" + d["rhs"][ket_idx][2]
    bra = d["rhs"].pop(bra_idx)
    nbody = len(bra[2]) // 2
    bra[0] = "sigma"
    bra[2] = "p" + bra[2][nbody:] + bra[2][:nbody]
    bra[1] = bra[1][nbody:] + bra[1][:nbody]
    d["lhs"] = [bra]
    return w.dict_to_einsum(d)


def compile_first_row_safe(equation, ket_name="c"):
    eq, d = w.compile_einsum(equation, return_eq_dict=True)
    for idx, t in enumerate(d["rhs"]):
        if t[0] == ket_name:
            ket_idx = idx

    d["rhs"][ket_idx][2] = "p" + d["rhs"][ket_idx][2]
    d["lhs"][0][2] = "p"
    return w.dict_to_einsum(d)


def compile_first_row(equation, ket_name="c"):
    eq, d = w.compile_einsum(equation, return_eq_dict=True)
    for idx, t in enumerate(d["rhs"]):
        if t[0] == ket_name:
            ket_idx = idx
    ket = d["rhs"].pop(ket_idx)
    ket[0] = "sigma"
    d["lhs"] = [ket]
    return w.dict_to_einsum(d)


def generate_sigma_build(mbeq, matrix, first_row=True, algo="normal"):
    code = [
        f"def build_sigma_vector_{matrix}(einsum, einsum_type, c, Hbar, gamma1, eta1, lambda2, lambda3, lambda4, first_row):",
        "    sigma = {key: np.zeros(c[key].shape) for key in c.keys()}",
    ]

    for eq in mbeq["|"]:
        if algo == "normal":
            if not any(
                t.label() in ["lambda4", "lambda5", "lambda6"]
                for t in eq.rhs().tensors()
            ):
                code.append(f"    {compile_sigma_vector(eq)}")
        elif algo == "ee":
            if not any(t.label() in ["lambda5", "lambda6"] for t in eq.rhs().tensors()):
                code.append(f"    {compile_sigma_vector(eq)}")

    if matrix == "Hbar" and first_row:
        code.extend(
            [
                "    for key in first_row.keys():",
                "        tmp = first_row[key] * c['first'][..., np.newaxis]",
                "        sigma[key] += tmp",
                "    c_vec = dict_to_vec(c, c[list(c.keys())[0]].shape[0])",
                "    first_row_vec = dict_to_vec(first_row, 1)",
                "    sigma['first'] += np.einsum('ik, ij->jk', first_row_vec, c_vec[1:, :], optimize=True)",
            ]
        )
    elif matrix == "s":
        code.append("    sigma['first'] = c['first'].copy()")

    code.append("    return sigma")
    return "\n".join(code)


def generate_template_c(block_list, ket_name="c"):
    index_dict = {
        "c": "nocc",
        "a": "nact",
        "v": "nvir",
        "C": "nocc",
        "A": "nact",
        "V": "nvir",
        "i": "ncore",
        "I": "ncore",
    }

    code = [f"def get_template_c(nlow, ncore, nocc, nact, nvir):", "    c = {"]

    for i in block_list:
        shape_strings = ["nlow"] + [f"{index_dict[item]}" for item in i]
        shape_formatted = ", ".join(shape_strings)
        code.append(f"         '{i}': np.zeros(({shape_formatted})),")

    code.append("        }")
    code.append("    return c")
    code = "\n".join(code)
    return code


def generate_first_row(mbeq, algo="normal"):
    code = [
        f"def build_first_row(einsum, einsum_type, c, Hbar, gamma1, eta1, lambda2, lambda3, lambda4):",
        "    sigma = {key: np.zeros((1, *tensor.shape[1:])) for key, tensor in c.items() if key != 'first'}",
    ]
    if algo == "normal":
        for eq in mbeq["|"]:
            if not any(
                t.label() in ["lambda4", "lambda5", "lambda6"]
                for t in eq.rhs().tensors()
            ):
                code.append(f"    {compile_first_row(eq, ket_name='c')}")
    elif algo == "ee":
        for eq in mbeq["|"]:
            if not any(t.label() in ["lambda5", "lambda6"] for t in eq.rhs().tensors()):
                code.append(f"    {compile_first_row(eq, ket_name='c')}")

    code.append("    return sigma")
    funct = "\n".join(code)
    return funct


def generate_transition_dipole(mbeq, algo="normal"):
    code = [
        f"def build_transition_dipole(einsum, einsum_type, c, Hbar, gamma1, eta1, lambda2, lambda3, lambda4):",
        "    sigma = 0.0",
    ]
    if algo == "normal":
        for eq in mbeq["|"]:
            if not any(
                t.label() in ["lambda4", "lambda5", "lambda6"]
                for t in eq.rhs().tensors()
            ):
                code.append(f"    {w.compile_einsum(eq)}")
    elif algo == "ee":
        for eq in mbeq["|"]:
            if not any(t.label() in ["lambda5", "lambda6"] for t in eq.rhs().tensors()):
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


def generate_block_contraction(
    block_str,
    mbeq,
    block_type="single",
    indent="once",
    bra_name="bra",
    ket_name="c",
    algo="normal",
):
    indent_spaces = {"once": "    ", "twice": "        "}
    space = indent_spaces.get(indent, "    ")
    code = []

    for eq in mbeq["|"]:
        no_print = False
        correct_contraction = False
        rhs = eq.rhs()
        for t in rhs.tensors():
            if (
                t.label() in ["lambda4", "lambda5", "lambda6"] and algo == "normal"
            ) or (t.label() in ["lambda5", "lambda6"] and algo == "ee"):
                no_print = True
                break
            elif t.label() == bra_name:
                bra_label = "".join([str(_)[0] for _ in t.lower()]) + "".join(
                    [str(_)[0] for _ in t.upper()]
                )
            elif t.label() == ket_name:
                ket_label = "".join([str(_)[0] for _ in t.upper()]) + "".join(
                    [str(_)[0] for _ in t.lower()]
                )

        if not no_print:
            if block_type == "single":
                correct_contraction = bra_label == ket_label == block_str
            elif block_type == "composite":
                correct_contraction = bra_label in block_str and ket_label in block_str

            if correct_contraction:
                code.append(
                    f"{space}{compile_sigma_vector(eq, bra_name=bra_name, ket_name=ket_name)}"
                )

    code.append(f"{space}sigma = antisymmetrize(sigma)")

    func = "\n".join(code)
    return func


def generate_S_12(mbeq, single_space, composite_space, tol=1e-4, algo="normal"):
    """
    single_space: a list of strings.
    composite_space: a list of lists of strings.
    """
    code = [
        f"def get_S_12(einsum, einsum_type, template_c, gamma1, eta1, lambda2, lambda3, lambda4, sym_dict, target_sym, act_sym, tol={tol}):",
        "    sigma = {}",
        "    c = {}",
        "    S_12 = []",
        "    sym_space = {}",
    ]

    def one_active_two_virtual(key):
        code_block = [
            f"    # {key} block (one active, two virtual)",
            f'    print("Starts {key} block", flush=True)',
            "    gv_half_dict = {}",
        ]
        space_order = {}
        for i_space in range(2):
            if key[i_space] not in ["a", "A"]:
                space_order["noact"] = i_space
            else:
                space_order["act"] = i_space
        reorder_temp = (space_order["noact"], 2, 3, space_order["act"])
        reorder_back = np.argsort(reorder_temp).tolist()

        if key[space_order["act"]] == "A":
            temp_rdm = "gamma1['AA']"
        else:
            temp_rdm = "gamma1['aa']"

        if key[2] == key[3]:  # Should be antisymmetrized
            anti = True
        else:
            anti = False

        code_block.extend(
            [
                f"    anti = {anti}",
                f"    sym_space['{key}'] = sym_dict['{key}'].transpose(*{reorder_temp})",
                f"    max_sym = np.max(sym_space['{key}'])",
                f"    nocc, nact, nvir = template_c['{key}'].shape[{space_order['noact']+1}], template_c['{key}'].shape[{space_order['act']+1}], template_c['{key}'].shape[3]",
                f"    ge, gv = np.linalg.eigh({temp_rdm})",
                f"    trunc_indices = np.where(ge > tol)[0]",
                f"    gv_half = gv[:, trunc_indices] / np.sqrt(ge[trunc_indices])",
                # f"    rows, cols = gv_half.shape",
                f"    for i_sym in range(max_sym+1):",
                f"        temp = {temp_rdm}.copy()",
                f"        mask = (act_sym != i_sym)",
                f"        temp[:, mask] = 0",
                f"        temp[mask, :] = 0",
                f"        ge, gv = np.linalg.eigh(temp)",
                f"        trunc_indices = np.where(ge > tol)[0]",
                f"        gv_half = gv[:, trunc_indices] / np.sqrt(ge[trunc_indices])",
                f"        gv_half_dict[i_sym] = gv_half",
                f"    rows = gv_half.shape[0]",
                f"    cols = max(arr.shape[1] for arr in gv_half_dict.values())",
                f"    num = nocc * nvir * nvir",
                f"    if anti:",
                f"        zero_idx = [i * nvir * nvir + a * nvir + a for i in range(nocc) for a in range(nvir)]",
                f"    else:",
                f"        zero_idx = []",
                f"    X = np.zeros((num * rows, (num - len(zero_idx)) * cols))",
                f"    current_shape = (nocc, nvir, nvir)",
                f"    start_col = 0",
                f"    for i in range(num):",
                f"        if i in zero_idx:",
                f"            continue",
                f"        unravel_idx = np.unravel_index(i, current_shape)",
                f"        possible_sym = sym_space['{key}'][unravel_idx]",
                f"        find_target_sym = np.where(possible_sym == target_sym)[0]",
                f"        if len(find_target_sym) == 0:",
                f"            continue",
                f"        else:",
                f"            act_sym_needed = act_sym[find_target_sym[0]]",
                f"        current_col = gv_half_dict[act_sym_needed].shape[1]",
                f"        if (unravel_idx[1] > unravel_idx[2] and anti) or (not anti):",
                f"            X[i * rows: (i+1) * rows, start_col:(start_col + current_col)] = gv_half_dict[act_sym_needed]",
                f"            start_col += current_col",
                f"    X = X[:, :start_col]",
                f"    nlow = X.shape[1]",
                f"    X = X.reshape(nocc, nvir, nvir, nact, nlow)",
                f"    X = np.transpose(X, axes=(*{reorder_back}, 4))",
                f"    X = X.reshape(nocc*nact*nvir*nvir, nlow)",
                f"    sym_space.clear()",
                f"    S_12.append(X)",
                f"    del X",
            ]
        )
        return code_block

    def two_active_two_virtual(key):
        # aavv, AAVV, aAvV
        code_block = [
            f"    # {key} block (two active, two virtual)",
            f'    print("Starts {key} block", flush=True)',
            "    gv_half_dict = {}",
        ]

        if key[2] == key[3]:  # Should be antisymmetrized
            anti = True
            if key[0] == key[1] == "a":
                temp_rdm = "gamma1['aa']"
                temp_lambda = "lambda2['aaaa']"
            elif key[0] == key[1] == "A":
                temp_rdm = "gamma1['AA']"
                temp_lambda = "lambda2['AAAA']"
        else:
            anti = False
            temp_lambda = "lambda2['aAaA']"

        code_block.extend(
            [
                f"    anti = {anti}",
                f"    sym_space['{key}'] = sym_dict['{key}'].transpose(2,3,0,1)",
                f"    max_sym = np.max(sym_space['{key}'])",
                f"    nact, nvir = template_c['{key}'].shape[1], template_c['{key}'].shape[3]",
            ]
        )
        if anti:
            code_block.extend(
                [
                    f"    overlap = np.einsum('ar,ob->oabr', {temp_rdm}, {temp_rdm}, optimize=True)",
                    f"    overlap -= np.einsum('ab,or->oabr', {temp_rdm}, {temp_rdm}, optimize=True)",
                ]
            )
        else:
            code_block.extend(
                [
                    f"    overlap = np.einsum('ar,ob->oabr', gamma1['AA'], gamma1['aa'], optimize=True)",
                ]
            )
        code_block.extend(
            [
                f"    overlap += {temp_lambda}",
                f"    overlap = overlap.reshape(nact*nact, nact*nact)",
                f"    if anti:",
                f"        zero_anti = np.array([u * nact + v for u in range(nact) for v in range(u, nact)])",
                f"        overlap[zero_anti, :] = 0",
                f"        overlap[:, zero_anti] = 0",
                f"    tol_act_sym = (act_sym[:, None] ^ act_sym[None, :]).flatten()",
                f"    ge, gv = np.linalg.eigh(overlap)",
                f"    trunc_indices = np.where(ge > tol)[0]",
                f"    gv_half = gv[:, trunc_indices] / np.sqrt(ge[trunc_indices])",
                # f"    rows, cols = gv_half.shape",
                f"    for i_sym in range(max_sym+1):",
                f"        temp = overlap.copy()",
                f"        mask = (tol_act_sym != i_sym)",
                f"        temp[:, mask] = 0",
                f"        temp[mask, :] = 0",
                f"        ge, gv = np.linalg.eigh(temp)",
                f"        trunc_indices = np.where(ge > tol)[0]",
                f"        gv_half = gv[:, trunc_indices] / np.sqrt(ge[trunc_indices])",
                f"        gv_half_dict[i_sym] = gv_half",
                f"    rows = gv_half.shape[0]",
                f"    cols = max(arr.shape[1] for arr in gv_half_dict.values())",
                f"    num = nvir * nvir",
                f"    if anti:",
                f"        zero_idx = [a * nvir + a for a in range(nvir)]",
                f"    else:",
                f"        zero_idx = []",
                f"    X = np.zeros((num * rows, (num - len(zero_idx)) * cols))",
                f"    current_shape = (nvir, nvir)",
                f"    start_col = 0",
                f"    for i in range(num):",
                f"        if i in zero_idx:",
                f"            continue",
                f"        unravel_idx = np.unravel_index(i, current_shape)",
                f"        possible_sym = sym_space['{key}'][unravel_idx].flatten()",
                f"        find_target_sym = np.where(possible_sym == target_sym)[0]",
                f"        if len(find_target_sym) == 0:",
                f"            continue",
                f"        else:",
                f"            act_sym_needed = tol_act_sym[find_target_sym[0]]",
                f"        current_col = gv_half_dict[act_sym_needed].shape[1]",
                f"        if (unravel_idx[0] > unravel_idx[1] and anti) or (not anti):",
                f"            X[i * rows: (i+1) * rows, start_col:(start_col + current_col)] = gv_half_dict[act_sym_needed]",
                f"            start_col += current_col",
                f"    X = X[:, :start_col]",
                f"    nlow = X.shape[1]",
                f"    X = X.reshape(nvir, nvir, nact, nact, nlow)",
                f"    X = np.transpose(X, axes=(2,3,0,1,4))",
                f"    X = X.reshape(nact * nact * nvir * nvir, nlow)",
                f"    sym_space.clear()",
                f"    S_12.append(X)",
                f"    del X",
            ]
        )
        return code_block

    def no_active(key):
        anti_up, anti_down = False, False
        if len(key) == 4:
            if key[0] == key[1] and key[2] == key[3]:
                anti_up, anti_down = True, True
            elif key[0] == key[1] and key[2] != key[3]:
                anti_up, anti_down = True, False
            elif key[0] != key[1] and key[2] == key[3]:
                anti_up, anti_down = False, True

        code_block = [
            f"    # {key} block",
            f'    print("Starts {key} block", flush=True)',
            f"    shape_block = template_c['{key}'].shape[1:]",
            f"    shape_size = np.prod(shape_block)",
            f"    sym_space['{key}'] = sym_dict['{key}']",
            f"    sym_vec = dict_to_vec(sym_space, 1).flatten()",
            f"    sym_space.clear()",
            f"    zero_up, zero_down = [], []",
            f"    if {anti_down}:",
            f"        zero_down = [i * shape_block[1] * shape_block[2] * shape_block[3] + j * shape_block[2] * shape_block[3] + a * shape_block[3] + b for i in range(shape_block[0]) for j in range(shape_block[1]) for a in range(shape_block[2]) for b in range(a, shape_block[3])]",
            f"    if {anti_up}:",
            f"        zero_up = [i * shape_block[1] * shape_block[2] * shape_block[3] + j * shape_block[2] * shape_block[3] + a * shape_block[3] + b for i in range(shape_block[0]) for j in range (i, shape_block[1]) for a in range(shape_block[2]) for b in range(shape_block[3])]",
            f"    mask = np.where(sym_vec != target_sym)[0]",
            f"    tol_mask = list(set(zero_down) | set(zero_up) | set(mask))",
            f"    X = np.identity(shape_size) ",
            f"    X = np.delete(X, tol_mask, axis=1)",
            f"    S_12.append(X)",
            f"    del tol_mask, sym_vec, zero_down, zero_up, X",
        ]

        return code_block

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
            f"    sym_space.clear()",
            f"    c_vec = dict_to_vec(c, shape_size)",
            f"    np.fill_diagonal(c_vec, 1)",
            f"    c = vec_to_dict(c, c_vec)",
            f"    del c_vec",
            f"    c = antisymmetrize(c)",
            f"    print('Starts contraction', flush=True)",
            generate_block_contraction(
                key, mbeq, block_type="single", indent="once", algo=algo
            ),
            f"    c.clear()",
            f"    vec = dict_to_vec(sigma, shape_size)",
            f"    sigma.clear()",
            f"    x_index, y_index = np.ogrid[:vec.shape[0], :vec.shape[1]]",
            f"    mask = (sym_vec[x_index] == target_sym) & (sym_vec[y_index] == target_sym)",
            f"    vec[~mask] = 0",
            f"    print('Starts diagonalization', flush=True)",
            f"    sevals, sevecs = scipy.linalg.eigh(vec)",
            f"    del sym_vec, vec, x_index, y_index, mask",
            f"    print('Diagonalization done', flush=True)",
            f"    trunc_indices = np.where(sevals > tol)[0]",
            f"    X = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])",
            "    S_12.append(X)",
            "    del sevals, sevecs, trunc_indices, X",
        ]

    def add_composite_space_code(space):
        code_block = [
            f"    # {space} composite block",
            f'    print("Starts {space} composite block", flush=True)',
            f"    shape_size = 0",
        ]
        for key in space:
            code_block.extend(
                [
                    f"    shape_block = template_c['{key}'].shape[1:]",
                    f"    shape_size += np.prod(shape_block)",
                    f"    sym_space['{key}'] = sym_dict['{key}']",
                ]
            )
        code_block.extend(
            [
                f"    for key in {space}:",
                f"        shape_block = template_c[key].shape[1:]",
                f"        c[key] = np.zeros((shape_size, *shape_block))",
                f"        sigma[key] = np.zeros((shape_size, *shape_block))",
                f"    sym_vec = dict_to_vec(sym_space, 1).flatten()",
                f"    sym_space.clear()",
                f"    c_vec = dict_to_vec(c, shape_size)",
                f"    np.fill_diagonal(c_vec, 1)",
                f"    c = vec_to_dict(c, c_vec)",
                f"    del c_vec",
                f"    c = antisymmetrize(c)",
                f"    print('Starts contraction', flush=True)",
            ]
        )
        code_block.append(
            generate_block_contraction(
                space, mbeq, block_type="composite", indent="once", algo=algo
            )
        )
        code_block.extend(
            [
                f"    c.clear()",
                f"    vec = dict_to_vec(sigma, shape_size)",
                f"    sigma.clear()",
                f"    x_index, y_index = np.ogrid[:vec.shape[0], :vec.shape[1]]",
                f"    mask = (sym_vec[x_index] == target_sym) & (sym_vec[y_index] == target_sym)",
                f"    vec[~mask] = 0",
                f"    print('Starts diagonalization', flush=True)",
                f"    sevals, sevecs = scipy.linalg.eigh(vec)",
                f"    del sym_vec, vec, x_index, y_index, mask",
                f"    print('Diagonalization done', flush=True)",
            ]
        )
        # if space == ['aa', 'AA', 'aaaa', 'AAAA', 'aAaA']:
        #     code_block.append(f"    trunc_indices = np.where(sevals > tol_act)[0]")
        # else:
        code_block.append(f"    trunc_indices = np.where(sevals > tol)[0]")
        code_block.extend(
            [
                f"    X = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])",
                "    S_12.append(X)",
                "    del sevals, sevecs, trunc_indices, X ",
            ]
        )
        return code_block

    # Add single space code blocks
    for key in single_space:
        if (
            len(key) == 4
            and key[2] in ["v", "V"]
            and key[3] in ["v", "V"]
            and (key[0] in ["a", "A"] or key[1] in ["a", "A"])
        ):
            if not (key[0] in ["a", "A"] and key[1] in ["a", "A"]):
                # One active, two virtual
                code.extend(one_active_two_virtual(key))
            else:
                # Two active, two virtual
                code.extend(two_active_two_virtual(key))
        elif "a" not in key and "A" not in key:
            code.extend(no_active(key))
        else:
            code.extend(add_single_space_code(key))
        code.append("")  # Blank line for separation

    # Add composite space code blocks
    for space in composite_space:
        code.extend(add_composite_space_code(space))
        code.append("")  # Blank line for separation

    code.append("    return S_12")
    return "\n".join(code)


def generate_preconditioner(
    mbeq, single_space, composite_space, diagonal_type="exact", algo="normal"
):
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
            generate_block_contraction(
                key, mbeq, block_type="single", indent="twice", algo=algo
            ),
            f"        c.clear()",
            f"        vec = dict_to_vec(sigma, northo)",
            f"        sigma.clear()",
            f"        vmv = tensor.T @ vec",
            f"        diagonal.append(vmv.diagonal())",
            f"        del vec, vmv",
        ]

    def add_code_only_H(key, i_key):
        return [
            f"    # {key} block",
            f'    print("Starts {key} block precond", flush=True)',
            f"    shape_block = template_c['{key}'].shape[1:]",
            f"    shape_size = np.prod(shape_block)",
            f"    c['{key}'] = np.zeros((shape_size, *shape_block))",
            f"    sigma['{key}'] = np.zeros((shape_size, *shape_block))",
            f"    c = vec_to_dict(c, np.identity(shape_size))",
            f"    c = antisymmetrize(c)",
            generate_block_contraction(
                key, mbeq, block_type="single", indent="once", algo=algo
            ),
            f"    c.clear()",
            f"    vec = dict_to_vec(sigma, shape_size)",
            f"    sigma.clear()",
            f"    diagonal.append(vec.diagonal())",
            f"    del vec",
        ]

    def add_composite_space_code(space, start):
        code_block = [
            f"    # {space} composite block",
            f'    print("Starts {space} composite block precond", flush=True)',
            f"    tensor = S_12[{start}]",
            f"    northo = tensor.shape[1]",
            f"    if northo != 0:",
            f"        vmv = np.zeros((northo, northo))",
        ]

        if diagonal_type == "exact":
            code_block.extend(
                [
                    f"        for key in {space}:",
                    f"            shape_block = template_c[key].shape[1:]",
                    f"            c[key] = np.zeros((northo, *shape_block))",
                    f"            sigma[key] = np.zeros((northo, *shape_block))",
                    f"        c = vec_to_dict(c, tensor)",
                    f"        c = antisymmetrize(c)",
                    generate_block_contraction(
                        space, mbeq, block_type="composite", indent="twice", algo=algo
                    ),
                    f"        c.clear()",
                    f"        vec = dict_to_vec(sigma, northo)",
                    f"        sigma.clear()",
                    f"        vmv = tensor.T @ vec",
                    f"        diagonal.append(vmv.diagonal())",
                    f"        del vec, vmv",
                ]
            )
        elif diagonal_type == "block":
            code_block.extend([f"        slice_tensor = 0"])
            for key_space in space:
                code_block.extend(
                    [
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
                        generate_block_contraction(
                            key_space,
                            mbeq,
                            block_type="single",
                            indent="twice",
                            algo=algo,
                        ),
                        f"        c.clear()",
                        f"        del c_vec",
                        f"        H_temp = dict_to_vec(sigma, shape_size)",
                        f"        sigma.clear()",
                        f"        S_temp = tensor[slice_tensor:slice_tensor+shape_size, :]",
                        f"        vmv += S_temp.T @ H_temp @ S_temp",
                        f"        slice_tensor += shape_size",
                        f"        del H_temp, S_temp",
                    ]
                )
        code_block.append(f"        diagonal.append(vmv.diagonal())")
        # code_block.extend([
        #     "    sigma.clear()",
        #     "    c.clear()"
        # ])
        return code_block

    if composite_space is None:
        code = [
            f"def compute_preconditioner_only_H(einsum, einsum_type, template_c, Hbar, gamma1, eta1, lambda2, lambda3, lambda4):",
            "    sigma = {}",
            "    c = {}",
            "    diagonal = [np.array([0.0])]",
        ]
        for i_key, key in enumerate(single_space):
            code.extend(add_code_only_H(key, i_key))
            code.append("")  # Blank line for separation
    else:
        code = [
            f"def compute_preconditioner_{diagonal_type}(einsum, einsum_type, template_c, S_12, Hbar, gamma1, eta1, lambda2, lambda3, lambda4):",
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


def antisymmetrize_tensor_2_1(Rccav, nlow, nocc, nact, nvir, method="ee"):
    # antisymmetrize the tensor
    if method == "ee":
        Rccav_anti = np.zeros((nlow, nocc, nocc, nact, nvir))
        Rccav_anti += np.einsum("pijab->pijab", Rccav)
        Rccav_anti -= np.einsum("pijab->pjiab", Rccav)
    elif method == "ip":
        Rccav_anti = np.zeros((nlow, nocc, nocc, nact))
        Rccav_anti += np.einsum("pija->pija", Rccav)
        Rccav_anti -= np.einsum("pija->pjia", Rccav)
    return Rccav_anti


def antisymmetrize_tensor_1_2(Rcavv, nlow, nocc, nact, nvir, method="ee"):
    # antisymmetrize the tensor
    Rcavv_anti = np.zeros((nlow, nocc, nact, nvir, nvir))
    Rcavv_anti += np.einsum("pijab->pijab", Rcavv)
    Rcavv_anti -= np.einsum("pijab->pijba", Rcavv)
    return Rcavv_anti


def antisymmetrize(input_dict, method="ee"):
    if type(input_dict) is dict:
        if method == "ee":
            for key in input_dict.keys():
                if len(key) == 4:
                    if key[0] == key[1] and key[2] != key[3]:
                        tensor = input_dict[key]
                        input_dict[key] = antisymmetrize_tensor_2_1(
                            tensor,
                            tensor.shape[0],
                            tensor.shape[1],
                            tensor.shape[3],
                            tensor.shape[4],
                        )
                    elif key[0] != key[1] and key[2] == key[3]:
                        tensor = input_dict[key]
                        input_dict[key] = antisymmetrize_tensor_1_2(
                            tensor,
                            tensor.shape[0],
                            tensor.shape[1],
                            tensor.shape[2],
                            tensor.shape[3],
                        )
                    elif key[0] == key[1] and key[2] == key[3]:
                        tensor = input_dict[key]
                        input_dict[key] = antisymmetrize_tensor_2_2(
                            tensor, tensor.shape[0], tensor.shape[1], tensor.shape[3]
                        )
                    else:
                        continue
        elif method == "ip":
            for key in input_dict.keys():
                if len(key) == 3:
                    if key[0] == key[1]:
                        tensor = input_dict[key]
                        input_dict[key] = antisymmetrize_tensor_2_1(
                            tensor,
                            tensor.shape[0],
                            tensor.shape[1],
                            tensor.shape[3],
                            None,
                            method="ip",
                        )
                    else:
                        continue
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
        raise ValueError(
            f"Matrix S is not Hermitian. Max non-Hermicity is {np.max(np.abs(S - S.conj().T))}"
        )
    if not is_hermitian(A):
        raise ValueError(
            f"Matrix A is not Hermitian. Max non-Hermicity is {np.max(np.abs(A - A.conj().T))}"
        )
    sevals, sevecs = np.linalg.eigh(S)
    trunc_indices = np.where(sevals > eta)[0]
    X = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])
    Ap = X.T @ A @ X
    eigval, eigvec = np.linalg.eigh(Ap)
    eigvec = X @ eigvec
    return eigval, eigvec


def is_antisymmetric_tensor_2_2(tensor):
    return (
        np.allclose(tensor, -tensor.transpose(0, 2, 1, 3, 4))
        and np.allclose(tensor, -tensor.transpose(0, 1, 2, 4, 3))
        and np.allclose(tensor, tensor.transpose(0, 2, 1, 4, 3))
    )


def is_antisymmetric_tensor_2_1(tensor):
    return np.allclose(tensor, -tensor.transpose(0, 2, 1, 3, 4))


def is_antisymmetric_tensor_1_2(tensor):
    return np.allclose(tensor, -tensor.transpose(0, 1, 2, 4, 3))


def is_antisymmetric(tensor_dict):
    for key, tensor in tensor_dict.items():
        if len(key) == 4:
            if key[0] == key[1] and key[2] != key[3]:
                if not is_antisymmetric_tensor_2_1(tensor):
                    raise ValueError("Subspace is not antisymmetric.")
            elif key[0] != key[1] and key[2] == key[3]:
                if not is_antisymmetric_tensor_1_2(tensor):
                    raise ValueError("Subspace is not antisymmetric.")
            elif key[0] == key[1] and key[2] == key[3]:
                if not is_antisymmetric_tensor_2_2(tensor):
                    raise ValueError("Subspace is not antisymmetric.")
    return True


def sym_dir(c, core_sym, occ_sym, act_sym, vir_sym):
    out_dir = {}
    dir = {
        "c": occ_sym,
        "C": occ_sym,
        "a": act_sym,
        "A": act_sym,
        "v": vir_sym,
        "V": vir_sym,
        "i": core_sym,
        "I": core_sym,
    }
    for key in c.keys():
        if len(key) == 2:
            if len(dir[key[0]]) == 0 or len(dir[key[1]]) == 0:
                out_dir[key] = np.zeros_like(c[key])
            else:
                out_dir[key] = dir[key[0]][:, None] ^ dir[key[1]][None, :]
        elif len(key) == 4:
            if (
                len(dir[key[0]]) == 0
                or len(dir[key[1]]) == 0
                or len(dir[key[2]]) == 0
                or len(dir[key[3]]) == 0
            ):
                out_dir[key] = np.zeros_like(c[key])
            else:
                out_dir[key] = (
                    dir[key[0]][:, None, None, None]
                    ^ dir[key[1]][None, :, None, None]
                    ^ dir[key[2]][None, None, :, None]
                    ^ dir[key[3]][None, None, None, :]
                )
        else:
            out_dir[key] = np.array([0])  # First
    return out_dir


def slice_H_core(Hbar_old, core_sym, occ_sym):
    if len(core_sym) == 0:
        raise ValueError("No core orbitals found.")
    elif len(occ_sym) == 0:
        print("No occupied orbitals. Just change 'C' to 'I' and 'c' to 'i'.")
        Hbar = {
            key.replace("c", "i").replace("C", "I"): value
            for key, value in Hbar_old.items()
        }
        return Hbar

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
        "i": core_indices,
        "c": occ_indices,
        "a": ...,
        "v": ...,
        "I": core_indices,
        "C": occ_indices,
        "A": ...,
        "V": ...,
    }

    # Initialize the new dictionary to hold the sliced tensors
    Hbar = {}

    for key, tensor in Hbar_old.items():
        indices_c = [index for index, char in enumerate(key) if char == "c"]
        indices_C = [index for index, char in enumerate(key) if char == "C"]
        count_c = len(indices_c)
        count_C = len(indices_C)

        # Generate all combinations of 'c'/'i' and 'C'/'I' replacements
        combinations_c = (
            list(itertools.product(["c", "i"], repeat=count_c)) if count_c > 0 else [[]]
        )
        combinations_C = (
            list(itertools.product(["C", "I"], repeat=count_C)) if count_C > 0 else [[]]
        )

        # Iterate through all combinations of 'c' and 'C' replacements
        for comb_c in combinations_c:
            for comb_C in combinations_C:
                new_key = list(key)
                for i, char in zip(indices_c, comb_c):
                    new_key[i] = char
                for i, char in zip(indices_C, comb_C):
                    new_key[i] = char
                new_key = "".join(new_key)

                if len(new_key) == 2:
                    first_dim_indices = indices_dict[new_key[0]]
                    second_dim_indices = indices_dict[new_key[1]]
                    Hbar[new_key] = tensor[first_dim_indices, :][:, second_dim_indices]
                elif len(new_key) == 4:
                    first_dim_indices = indices_dict[new_key[0]]
                    second_dim_indices = indices_dict[new_key[1]]
                    third_dim_indices = indices_dict[new_key[2]]
                    fourth_dim_indices = indices_dict[new_key[3]]
                    Hbar[new_key] = tensor[first_dim_indices, :, :, :][
                        :, second_dim_indices, :, :
                    ][:, :, third_dim_indices, :][:, :, :, fourth_dim_indices]

    return Hbar


def normalize(input_obj):
    if type(input_obj) is dict:
        vec = dict_to_vec(input_obj, input_obj[list(input_obj.keys())[0]].shape[0])
    elif type(input_obj) is np.ndarray:
        vec = input_obj

    out_array = np.zeros_like(vec)

    for i in range(vec.shape[1]):
        vec_i = vec[:, i]
        norm = np.linalg.norm(vec_i)
        if norm < 1e-6:
            continue
        else:
            out_array[:, i] = vec_i / np.linalg.norm(vec_i)

    if type(input_obj) is dict:
        output_obj = vec_to_dict(input_obj, out_array)
    elif type(input_obj) is np.ndarray:
        output_obj = out_array

    return output_obj


def orthonormalize(vectors, num_orthonormals=1, eps=1e-6):
    ortho_normals = vectors
    count_orthonormals = num_orthonormals
    # Skip unchanged ones.
    for i in range(num_orthonormals, vectors.shape[1]):
        vector_i = vectors[:, i]
        # Makes sure vector_i is orthogonal to all processed vectors.
        for j in range(i):
            vector_i -= ortho_normals[:, j] * np.dot(
                ortho_normals[:, j].conj(), vector_i
            )

        # Makes sure vector_i is normalized.
        if np.max(np.abs(vector_i)) < eps:
            continue
        ortho_normals[:, count_orthonormals] = vector_i / np.linalg.norm(vector_i)
        count_orthonormals += 1
    return ortho_normals[:, :count_orthonormals]


def filter_list(element_list, ncore, nocc, nact, nvir):
    return [
        element
        for element in element_list
        if (ncore != 0 or ("i" not in element and "I" not in element))
        and (nocc != 0 or ("c" not in element and "C" not in element))
        and (nact != 0 or ("a" not in element and "A" not in element))
        and (nvir != 0 or ("v" not in element and "V" not in element))
    ]
