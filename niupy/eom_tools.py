import wicked as w
import numpy as np
import copy
import itertools
import re


def op_to_tensor_label(op):
    wop = w.op("o", [op])
    op_canon = wop.__str__().split(" {")[1].split(" }")[0]
    indices = op_canon.split(" ")
    if len(indices) == 1:
        return op if "+" not in op else op.replace("+", "")
    cre = []
    ann = []
    for i in indices:
        if "+" in i:
            cre.append(i.replace("+", ""))
        else:
            ann.append(i)

    return "".join(ann[::-1]) + "".join(cre)


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


def increment_index(index):
    """
    increment a sting like 'a128' to 'a129' using regex
    """
    return re.sub(r"(\d+)", lambda x: str(int(x.group(0)) + 1), index)


def get_matrix_elements(bra, op, ket, inter_general=False, double_comm=False):
    """
    This function calculates the matrix elements of an operator
    between two internally contracted configurations.

    This is achieved by taking the second derivative of the fully
    contracted expression with respect to the fictitious bra and
    ket tensor elements.
    """
    wt = w.WickTheorem()  # Temporary fix
    if op is None:
        if double_comm:
            expr = w.commutator(bra.adjoint(), ket)
        else:
            expr = bra.adjoint() @ ket
    else:
        # This option will not be used in two virtual cases.
        if double_comm:
            expr = w.commutator(bra.adjoint() @ w.commutator(op, ket)) + w.commutator(
                w.commutator(bra.adjoint(), op) @ ket
            )
        else:
            expr = bra.adjoint() @ op @ ket
    label = "S" if op is None else "H"
    try:
        mbeq = wt.contract(
            expr, 0, 0, inter_general=inter_general
        ).to_manybody_equations(label)["|"]
    except KeyError:
        return
    mbeq_new = []
    for i in mbeq:
        eqdict = w.equation_to_dict(i)
        newdict = {"factor": eqdict["factor"], "lhs": [label, [], []], "rhs": []}
        lhs_indices = set()
        rhs_indices = set()
        for j in eqdict["rhs"]:
            # The fictitious tensor elements are
            # {p+ q+ s r} bra^{pq}_{rs}
            if j[0] == "bra":
                bra_indices = j[2][::1] + j[1]
                lhs_indices.update(bra_indices)
            elif j[0] == "ket":
                ket_indices = j[1] + j[2][::1]
                lhs_indices.update(ket_indices)
            else:
                newdict["rhs"].append(j)
                rhs_indices.update(j[1] + j[2])

        if len(bra_indices + ket_indices) > len(lhs_indices):
            lhs = bra_indices + ket_indices
            index_pool = lhs_indices | rhs_indices
            for i in lhs_indices:
                if lhs.count(i) == 2:
                    test_index = increment_index(i)
                    while True:
                        if test_index not in index_pool:
                            ket_indices[ket_indices.index(i)] = test_index
                            index_pool.update([test_index])
                            newdict["rhs"].append(["delta", [i], [test_index]])
                            break
                        test_index = increment_index(test_index)

        newdict["lhs"][1] = ket_indices
        newdict["lhs"][2] = bra_indices
        mbeq_new.append(w.dict_to_equation(newdict))
    return mbeq_new


def matrix_elements_to_diag(mbeq, indent="once"):
    indent_spaces = {"once": "    ", "twice": "        "}
    space = indent_spaces.get(indent, "    ")
    einsums = []
    for eq in mbeq:
        eqdict = w.equation_to_dict(eq)
        eqdict_new = {
            "factor": eqdict["factor"],
            "lhs": [eqdict["lhs"][0], [], []],
            "rhs": [],
        }
        deltas = {}
        for l, u in zip(eqdict["lhs"][1], eqdict["lhs"][2]):
            if not ("a" in l or "A" in l):
                deltas[l] = u
                eqdict_new["lhs"][2].append(u)
            else:
                eqdict_new["lhs"][1].append(u)
                eqdict_new["lhs"][1].append(l)
        eqdict_new["lhs"][2] = eqdict_new["lhs"][2][2:] + eqdict_new["lhs"][2][:2]
        eqdict_new["lhs"][2] += eqdict_new["lhs"][1]
        eqdict_new["lhs"][1] = []

        for t in eqdict["rhs"]:
            for i, l in enumerate(t[1]):
                if l in deltas:
                    t[1][i] = deltas[l]
            for i, l in enumerate(t[2]):
                if l in deltas:
                    t[2][i] = deltas[l]
            eqdict_new["rhs"].append(t)

        einsum = w.dict_to_equation(eqdict_new).compile("einsum")
        lhs = einsum.split(" +=")[0]
        einsum = einsum.replace(lhs, lhs[0])
        einsums.append(f"{space}{einsum}")

    func = "\n".join(einsums)
    return func


def generate_sigma_build(mbeq, matrix, first_row=True):
    code = [
        f"def build_sigma_vector_{matrix}(einsum, einsum_type, c, Hbar, gamma1, eta1, lambda2, lambda3, lambda4, first_row):",
        "    sigma = {key: np.zeros(c[key].shape) for key in c.keys()}",
    ]

    for eq in mbeq["|"]:
        code.append(f"    {compile_sigma_vector(eq)}")

    if matrix == "Hbar" and first_row:
        code.extend(
            [
                "    for key in first_row.keys():",
                "        if len(key) == 2:",
                "            tmp = first_row[key] * c['first'][:, :, np.newaxis]",
                "        elif len(key) == 4:",
                "            tmp = first_row[key] * c['first'][:, :, np.newaxis, np.newaxis, np.newaxis]",
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


def generate_first_row(mbeq):
    code = [
        f"def build_first_row(einsum, einsum_type, c, Hbar, gamma1, eta1, lambda2, lambda3, lambda4):",
        "    sigma = {key: np.zeros((1, *tensor.shape[1:])) for key, tensor in c.items() if key != 'first'}",
    ]
    for eq in mbeq["|"]:
        code.append(f"    {compile_first_row(eq, ket_name='c')}")

    code.append("    return sigma")
    funct = "\n".join(code)
    return funct


def generate_transition_dipole(mbeq):  # redundant
    code = [
        f"def build_transition_dipole(einsum, einsum_type, c, Hbar, gamma1, eta1, lambda2, lambda3, lambda4):",
        "    sigma = 0.0",
    ]
    for eq in mbeq["|"]:
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
):
    indent_spaces = {"once": "    ", "twice": "        "}
    space = indent_spaces.get(indent, "    ")
    code = []

    for eq in mbeq["|"]:
        correct_contraction = False
        rhs = eq.rhs()
        for t in rhs.tensors():
            if t.label() == bra_name:
                bra_label = "".join([str(_)[0] for _ in t.lower()]) + "".join(
                    [str(_)[0] for _ in t.upper()]
                )
            elif t.label() == ket_name:
                ket_label = "".join([str(_)[0] for _ in t.upper()]) + "".join(
                    [str(_)[0] for _ in t.lower()]
                )

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


def generate_S12(mbeq, single_space, composite_space):
    """
    single_space: a list of strings.
    composite_space: a list of lists of strings.
    tol: tolerance for truncating singular values.
    tol_semi: tolerance for truncating singular values for semi-internals.
    einsum, einsum_type, template_c, gamma1, eta1, lambda2, lambda3, lambda4, tol={tol}, tol_semi={tol_semi}
    """
    code = [
        f"def get_S12(eom_dsrg):",
        "    einsum = eom_dsrg.einsum",
        "    einsum_type = eom_dsrg.einsum_type",
        "    template_c = eom_dsrg.template_c",
        "    gamma1 = eom_dsrg.gamma1",
        "    eta1 = eom_dsrg.eta1",
        "    lambda2 = eom_dsrg.lambda2",
        "    lambda3 = eom_dsrg.lambda3",
        "    lambda4 = eom_dsrg.lambda4",
        "    tol = eom_dsrg.tol_s",
        "    tol_semi = eom_dsrg.tol_semi",
        "    num_ortho = 0",
        "    sigma = {}",
        "    c = {}",
    ]

    def one_active_two_virtual(key):
        code_block = [
            f"    # {key} block (one active, two virtual)",
            f'    print("Starts {key} block")',
        ]
        space_order = {}
        for i_space in range(2):
            if key[i_space] not in ["a", "A"]:
                space_order["noact"] = i_space
            else:
                space_order["act"] = i_space

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
                f"    ge, gv = np.linalg.eigh({temp_rdm})",
                f"    trunc_indices = np.where(ge > tol)[0]",
                f"    eom_dsrg.S12.{key} = gv[:, trunc_indices] / np.sqrt(ge[trunc_indices])",
                f"    nocc, nact, nvir = template_c['{key}'].shape[{space_order['noact']+1}], template_c['{key}'].shape[{space_order['act']+1}], template_c['{key}'].shape[3]",
                f"    if anti:",
                f"        zero_idx = [i * nvir * nvir + a * nvir + a for i in range(nocc) for a in range(nvir)]",
                f"    else:",
                f"        zero_idx = []",
                f"    eom_dsrg.S12.position_{key} = np.ones(nocc * nvir * nvir)",
                f"    current_shape = (nocc, nvir, nvir)",
                f"    for i in range(len(eom_dsrg.S12.position_{key})):",
                f"        if i in zero_idx:",
                f"            eom_dsrg.S12.position_{key}[i] = 0",
                f"        else:",
                f"            unravel_idx = np.unravel_index(i, current_shape)",
                f"            if (unravel_idx[1] > unravel_idx[2] and anti) or (not anti):",
                f"                continue",
                f"            else:",
                f"                eom_dsrg.S12.position_{key}[i] = 0",
                f"    num_ortho += np.sum(eom_dsrg.S12.position_{key} == 1)",
            ]
        )
        return code_block

    # SL: Two_active_two_virtual has been removed from the code. EE is disabled.

    # I think single excitation operators are too small to be considered here.
    def no_active(key):
        anti_up, anti_down = False, False
        if key[0] == key[1] and key[2] == key[3]:
            anti_up, anti_down = True, True
        elif key[0] == key[1] and key[2] != key[3]:
            anti_up, anti_down = True, False
        elif key[0] != key[1] and key[2] == key[3]:
            anti_up, anti_down = False, True
        return [
            f"    # {key} block (no active)",
            f"    shape_block = template_c['{key}'].shape[1:]",
            f"    shape_size = np.prod(shape_block)",
            f"    eom_dsrg.S12.{key} = np.ones(shape_size)",
            f"    zero_up, zero_down = [], []",
            f"    if {anti_down}:",
            f"        zero_down = [i * shape_block[1] * shape_block[2] * shape_block[3] + j * shape_block[2] * shape_block[3] + a * shape_block[3] + b for i in range(shape_block[0]) for j in range(shape_block[1]) for a in range(shape_block[2]) for b in range(a, shape_block[3])]",
            f"    if {anti_up}:",
            f"        zero_up = [i * shape_block[1] * shape_block[2] * shape_block[3] + j * shape_block[2] * shape_block[3] + a * shape_block[3] + b for i in range(shape_block[0]) for j in range (i, shape_block[1]) for a in range(shape_block[2]) for b in range(shape_block[3])]",
            f"    mask = list(set(zero_down) | set(zero_up))",
            f"    eom_dsrg.S12.{key}[mask] = 0",
            f"    num_ortho += np.sum(eom_dsrg.S12.{key} == 1)",
            f"    del mask, zero_down, zero_up",
        ]

    def add_single_space_code(key):
        return [
            f"    # {key} block",
            f'    print("Starts {key} block")',
            f"    shape_block = template_c['{key}'].shape[1:]",
            f"    shape_size = np.prod(shape_block)",
            f"    c['{key}'] = np.zeros((shape_size, *shape_block))",
            f"    sigma['{key}'] = np.zeros((shape_size, *shape_block))",
            f"    c_vec = dict_to_vec(c, shape_size)",
            f"    np.fill_diagonal(c_vec, 1)",
            f"    c = vec_to_dict(c, c_vec)",
            f"    del c_vec",
            f"    c = antisymmetrize(c)",
            f"    print('Starts contraction')",
            generate_block_contraction(key, mbeq, block_type="single", indent="once"),
            f"    c.clear()",
            f"    vec = dict_to_vec(sigma, shape_size)",
            f"    sigma.clear()",
            f"    print('Starts diagonalization', flush = True)",
            f"    sevals, sevecs = scipy.linalg.eigh(vec)",
            f"    del vec",
            f"    print('Diagonalization done')",
            f"    trunc_indices = np.where(sevals > tol)[0]",
            f"    eom_dsrg.S12.{key} = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])",
            f"    num_ortho += eom_dsrg.S12.{key}.shape[1]",
            "    del sevals, sevecs, trunc_indices",
        ]

    def add_composite_space_block(space):
        code_block = [
            f"    # {space} composite block",
            f'    print("Starts {space} composite block")',
            f"    shape_size = 0",
        ]

        for key in space:
            code_block.extend(
                [
                    f"    shape_block = template_c['{key}'].shape[1:]",
                    f"    shape_size += np.prod(shape_block)",
                ]
            )
        code_block.extend(
            [
                f"    for key in {space}:",
                f"        shape_block = template_c[key].shape[1:]",
                f"        c[key] = np.zeros((shape_size, *shape_block))",
                f"        sigma[key] = np.zeros((shape_size, *shape_block))",
                f"    c_vec = dict_to_vec(c, shape_size)",
                f"    np.fill_diagonal(c_vec, 1)",
                f"    c = vec_to_dict(c, c_vec)",
                f"    del c_vec",
                f"    c = antisymmetrize(c)",
                f"    print('Starts contraction')",
            ]
        )
        code_block.append(
            generate_block_contraction(
                space, mbeq, block_type="composite", indent="once"
            )
        )
        code_block.extend(
            [
                f"    c.clear()",
                f"    vec = dict_to_vec(sigma, shape_size)",
                f"    sigma.clear()",
            ]
        )
        return code_block

    def add_composite_space_code(space):
        code_block = add_composite_space_block(space)
        code_block.extend(
            [
                f"    print('Starts diagonalization', flush = True)",
                f"    sevals, sevecs = scipy.linalg.eigh(vec)",
                f"    del vec",
                f"    print('Diagonalization done')",
                f"    trunc_indices = np.where(sevals > tol)[0]",
                f"    eom_dsrg.S12.{space[0]} = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])",
                f"    num_ortho += eom_dsrg.S12.{space[0]}.shape[1]",
                f"    del sevals, sevecs, trunc_indices",
            ]
        )
        return code_block

    def sequential_orthogonalization(space):
        # Singles first!!
        code_block = add_composite_space_block(space)
        singles = []
        for key in space:
            if len(key) == 2:
                singles.append(key)

        code_block.extend(
            [
                f"    singles_size = 0",
                f"    for key in {singles}:",
                f"        shape_block = template_c[key].shape[1:]",
                f"        singles_size += np.prod(shape_block)",
                f"    S11 = vec[:singles_size, :singles_size].copy()",
                f"    S12 = vec[:singles_size, singles_size:].copy()",
                f"    sevals, sevecs = scipy.linalg.eigh(S11)",
                f"    trunc_indices = np.where(sevals > tol_semi)[0]",
                f"    S_inv_eval = 1.0/(sevals[trunc_indices])",
                f"    sevecs = sevecs[:, trunc_indices]",
                f"    S11inv = reduce(np.dot, (sevecs,np.diag(S_inv_eval),sevecs.T))",
                f"    Y12 = -np.matmul(S11inv, S12)",
                f"    Y = np.identity(vec.shape[0])",
                f"    Y[:singles_size, singles_size:] = Y12",
                f"    vec_proj = reduce(np.dot, (Y.T, vec, Y))",
                f"    del vec, S11, S12, S11inv, S_inv_eval",
                f"    print('Starts diagonalization (after projection))', flush = True)",
                f"    sevals, sevecs = scipy.linalg.eigh(vec_proj)",
                f"    del vec_proj",
                f"    print('Diagonalization done')",
                f"    trunc_indices = np.where(sevals > tol_semi)[0]",
                f"    X = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])",
                f"    eom_dsrg.S12.{space[0]} = np.matmul(Y, X)",
                f"    num_ortho += eom_dsrg.S12.{space[0]}.shape[1]",
                f"    del sevals, sevecs, trunc_indices, X, Y, Y12",
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
            # else:
            #     # Two active, two virtual
            #     code.extend(two_active_two_virtual(key))
        elif "a" not in key and "A" not in key and len(key) == 4:
            code.extend(no_active(key))
        else:
            code.extend(add_single_space_code(key))
        code.append("")  # Blank line for separation

    # Add composite space code blocks
    for space in composite_space:
        if any(len(key) == 2 for key in space):
            code.extend(sequential_orthogonalization(space))
        else:
            code.extend(add_composite_space_code(space))
        code.append("")  # Blank line for separation

    code.append("    print(f'Number of orthogonalized operators: {num_ortho}')")

    return "\n".join(code)


def generate_preconditioner(
    mbeq, mbeqs_one_active, mbeqs_no_active, single_space, composite_space
):
    """
    mbeqs_one_active and mbeqs_no_active are dictionaries.
    einsum, einsum_type, template_c, Hbar, gamma1, eta1, lambda2, lambda3, lambda4
    """
    code = [
        f"def compute_preconditioner(eom_dsrg):",
        "    einsum = eom_dsrg.einsum",
        "    einsum_type = eom_dsrg.einsum_type",
        "    template_c = eom_dsrg.template_c",
        "    Hbar = eom_dsrg.Hbar",
        "    gamma1 = eom_dsrg.gamma1",
        "    eta1 = eom_dsrg.eta1",
        "    lambda2 = eom_dsrg.lambda2",
        "    lambda3 = eom_dsrg.lambda3",
        "    lambda4 = eom_dsrg.lambda4",
        "    sigma = {}",
        "    c = {}",
        "    delta = {'ii': np.identity(eom_dsrg.ncore), 'II': np.identity(eom_dsrg.ncore), 'cc': np.identity(eom_dsrg.nocc), 'CC': np.identity(eom_dsrg.nocc), 'aa': np.identity(eom_dsrg.nact), 'AA': np.identity(eom_dsrg.nact), 'vv': np.identity(eom_dsrg.nvir), 'VV': np.identity(eom_dsrg.nvir)}",
        "    diagonal = [np.array([0.0])]",
    ]

    def add_single_space_code(key):
        return [
            f"    # {key} block",
            f'    print("Starts {key} block precond")',
            f"    shape_block = template_c['{key}'].shape[1:]",
            f"    northo = eom_dsrg.S12.{key}.shape[1]",
            f"    if northo != 0:",
            f"        c['{key}'] = np.zeros((northo, *shape_block))",
            f"        sigma['{key}'] = np.zeros((northo, *shape_block))",
            f"        c = vec_to_dict(c, eom_dsrg.S12.{key})",
            f"        c = antisymmetrize(c)",
            generate_block_contraction(key, mbeq, block_type="single", indent="twice"),
            f"        c.clear()",
            f"        vec = dict_to_vec(sigma, northo)",
            f"        sigma.clear()",
            f"        vmv = eom_dsrg.S12.{key}.T @ vec",
            f"        diagonal.append(vmv.diagonal())",
            f"        del vec, vmv",
        ]

    def add_composite_space_code(space):
        code_block = [
            f"    # {space} composite block",
            f'    print("Starts {space} composite block precond")',
            f"    northo = eom_dsrg.S12.{space[0]}.shape[1]",
            f"    if northo != 0:",
            f"        vmv = np.zeros((northo, northo))",
        ]

        code_block.extend(
            [
                f"        for key in {space}:",
                f"            shape_block = template_c[key].shape[1:]",
                f"            c[key] = np.zeros((northo, *shape_block))",
                f"            sigma[key] = np.zeros((northo, *shape_block))",
                f"        c = vec_to_dict(c, eom_dsrg.S12.{space[0]})",
                f"        c = antisymmetrize(c)",
                generate_block_contraction(
                    space, mbeq, block_type="composite", indent="twice"
                ),
                f"        c.clear()",
                f"        vec = dict_to_vec(sigma, northo)",
                f"        sigma.clear()",
                f"        vmv = eom_dsrg.S12.{space[0]}.T @ vec",
                f"        diagonal.append(vmv.diagonal())",
                f"        del vec, vmv",
            ]
        )

        return code_block

    def one_active_two_virtual(key):
        code_block = [
            f"    # {key} block (one active, two virtual)",
            f'    print("Starts {key} block precond")',
        ]

        space_order = {}
        for i_space in range(2):
            if key[i_space] not in ["a", "A"]:
                space_order["noact"] = i_space
            else:
                space_order["act"] = i_space

        code_block.extend(
            [
                f"    nocc, nact, nvir = template_c['{key}'].shape[{space_order['noact']+1}], template_c['{key}'].shape[{space_order['act']+1}], template_c['{key}'].shape[3]",
                "    H = np.zeros((nocc, nvir, nvir, nact, nact))",
                matrix_elements_to_diag(mbeqs_one_active[key]),
            ]
        )

        code_block.extend(
            [
                f"    H = np.einsum('xu, MeFxy, yu -> MeFu', eom_dsrg.S12.{key}, H, eom_dsrg.S12.{key}, optimize=True)",
                f"    H = H.reshape(-1, eom_dsrg.S12.{key}.shape[1])",
                f"    zero_mask = eom_dsrg.S12.position_{key} == 0",
                f"    H = np.delete(H, zero_mask, axis=0)",
                f"    diagonal.append(H.flatten())",
            ]
        )

        return code_block

    def no_active(key):
        code_block = [
            f"    # {key} block (no active)",
            f'    print("Starts {key} block precond")',
        ]
        code_block.extend(
            [
                f"    shape_block = template_c['{key}'].shape[1:]",
                "    H = np.zeros(shape_block)",
                matrix_elements_to_diag(mbeqs_no_active[key]),
            ]
        )

        code_block.extend(
            [
                f"    H = H.flatten()",
                f"    temp = H[eom_dsrg.S12.{key}==1]",
                f"    diagonal.append(temp)",
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
            and not (key[0] in ["a", "A"] and key[1] in ["a", "A"])
        ):
            # One active, two virtual
            code.extend(one_active_two_virtual(key))
        elif "a" not in key and "A" not in key and len(key) == 4:
            code.extend(no_active(key))
        else:
            code.extend(add_single_space_code(key))
        code.append("")  # Blank line for separation

    # Add composite space code blocks
    for space in composite_space:
        code.extend(add_composite_space_code(space))
        code.append("")  # Blank line for separation

    code.append("    full_diag = np.concatenate(diagonal)")
    code.append("    return full_diag")

    return "\n".join(code)


def generate_apply_S12(single_space, composite_space):
    code_block = [
        f"def apply_S12(eom_dsrg, ndim, t, transpose=False):",
        f"    Xt = np.zeros((ndim, 1))",
        f"    i_start_xt = 1",
        f"    i_start_t = 1",
        f"    Xt[0, 0] = t[0]",
        f"    template = eom_dsrg.template_c",
    ]
    for key in single_space:
        # No active, double excitation
        if "a" not in key and "A" not in key and len(key) == 4:
            code_block.extend(
                [
                    f"    num_op, num_ortho = len(eom_dsrg.S12.{key}), np.sum(eom_dsrg.S12.{key}==1)",
                    f"    i_end_xt, i_end_t = i_start_xt + (num_op if not transpose else num_ortho), i_start_t + (num_ortho if not transpose else num_op)",
                    f"    if not transpose:",
                    f"        temp = eom_dsrg.S12.{key}.copy()",
                    f"        temp[eom_dsrg.S12.{key} == 1] = t[i_start_t:i_end_t]",
                    f"        Xt[i_start_xt:i_end_xt, :] = temp.reshape(-1, 1).copy()",
                    f"    else:",
                    f"        temp = t[i_start_t:i_end_t][eom_dsrg.S12.{key}==1]",
                    f"        Xt[i_start_xt:i_end_xt, :] = temp.reshape(-1, 1).copy()",
                    f"    i_start_xt, i_start_t = i_end_xt, i_end_t",
                ]
            )
        # One active, two virtuals
        elif (
            len(key) == 4
            and key[2] in ["v", "V"]
            and key[3] in ["v", "V"]
            and (key[0] in ["a", "A"] or key[1] in ["a", "A"])
            and not (key[0] in ["a", "A"] and key[1] in ["a", "A"])
        ):
            space_order = {}
            for i_space in range(2):
                if key[i_space] not in ["a", "A"]:
                    space_order["noact"] = i_space
                else:
                    space_order["act"] = i_space
            reorder_temp = (space_order["noact"], 2, 3, space_order["act"])
            reorder_back = np.argsort(reorder_temp).tolist()
            code_block.extend(
                [
                    f"    nocc, nact, nvir = template['{key}'].shape[{space_order['noact']+1}], template['{key}'].shape[{space_order['act']+1}], template['{key}'].shape[3]",
                    f"    row, col = eom_dsrg.S12.{key}.shape[0], eom_dsrg.S12.{key}.shape[1]",
                    f"    num_positions = len(eom_dsrg.S12.position_{key})",
                    f"    num_op, num_ortho = row * num_positions, col * np.sum(eom_dsrg.S12.position_{key}==1)",
                    f"    i_end_xt, i_end_t = i_start_xt + (num_op if not transpose else num_ortho), i_start_t + (num_ortho if not transpose else num_op)",
                    f"    temp_t = t[i_start_t:i_end_t].copy()",
                    f"    temp_xt = np.zeros((num_positions, row)) if not transpose else np.zeros((np.sum(eom_dsrg.S12.position_{key}==1), col))",
                    # f"    temp_t = temp_t.reshape(np.sum(eom_dsrg.S12.position_{key}==1), col) if not transpose else temp_t.reshape(num_positions, row)",
                    f"    if not transpose:",
                    f"        temp_t = temp_t.reshape(np.sum(eom_dsrg.S12.position_{key}==1), col)",
                    f"        non_zero_mask = eom_dsrg.S12.position_{key} != 0",
                    f"        temp_xt[non_zero_mask] = np.dot(eom_dsrg.S12.{key}, temp_t.T).T",
                    f"        temp_xt = temp_xt.flatten()",
                    f"        temp_xt = temp_xt.reshape(nocc, nvir, nvir, nact, 1)",
                    f"        temp_xt = np.transpose(temp_xt, axes=(*{reorder_back}, 4))",
                    f"        temp_xt = temp_xt.reshape(-1, 1)",
                    f"    else:",
                    f"        zero_mask = eom_dsrg.S12.position_{key} == 0",
                    f"        temp_t = temp_t.reshape(*template['{key}'].shape[1:])",
                    f"        temp_t = temp_t.transpose(*{reorder_temp})",
                    f"        temp_t = temp_t.reshape(-1, row)",
                    f"        temp_t = np.delete(temp_t, zero_mask, axis=0)",
                    f"        temp_xt = np.dot(eom_dsrg.S12.{key}.T, temp_t.T).T",
                    f"    Xt[i_start_xt:i_end_xt, :] = temp_xt.reshape(-1, 1).copy()",
                    f"    i_start_xt, i_start_t = i_end_xt, i_end_t",
                ]
            )
        else:
            code_block.extend(
                [
                    f"    num_op, num_ortho = eom_dsrg.S12.{key}.shape",
                    f"    i_end_xt, i_end_t = i_start_xt + (num_op if not transpose else num_ortho), i_start_t + (num_ortho if not transpose else num_op)",
                    f"    Xt[i_start_xt:i_end_xt, :] += (eom_dsrg.S12.{key} @ t[i_start_t:i_end_t].reshape(-1, 1) if not transpose else eom_dsrg.S12.{key}.T @ t[i_start_t:i_end_t].reshape(-1, 1))",
                    f"    i_start_xt, i_start_t = i_end_xt, i_end_t",
                ]
            )
    for space in composite_space:
        code_block.extend(
            [
                f"    num_op, num_ortho = eom_dsrg.S12.{space[0]}.shape",
                f"    i_end_xt, i_end_t = i_start_xt + (num_op if not transpose else num_ortho), i_start_t + (num_ortho if not transpose else num_op)",
                f"    Xt[i_start_xt:i_end_xt, :] += (eom_dsrg.S12.{space[0]} @ t[i_start_t:i_end_t].reshape(-1, 1) if not transpose else eom_dsrg.S12.{space[0]}.T @ t[i_start_t:i_end_t].reshape(-1, 1))",
                f"    i_start_xt, i_start_t = i_end_xt, i_end_t",
            ]
        )
    code_block.append("    return Xt")
    return "\n".join(code_block)


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
