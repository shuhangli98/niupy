import wicked as w
import numpy as np
import copy
import itertools
import re
import scipy.constants

eh_to_ev = scipy.constants.value("Hartree energy in eV")
irrep_table = {
               "c1": {0:"A"},
               "ci": {0:"Ag", 1:"Au"},
               "c2": {0:"A", 1:"B"},
               "cs": {0:"A'", 1:"A''"},
               "d2": {0:"A", 1:"B1", 2:"B2", 3:"B3"},
               "c2v": {0:"A1", 1:"A2", 2:"B1", 3:"B2"}, 
               "c2h": {0:"Ag", 1:"Bg", 2:"Au", 3:"Bu"},
               "d2h": {0:"A1g", 1:"B1g", 2:"B2g", 3:"B3g", 4:"A1u", 5:"B1u", 6:"B2u", 7:"B3u"},
               }
for v in irrep_table.values():
    v.update({"Incorrect symmetry": "Incorrect symmetry"})


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


def compile_sigma_vector(equation, bra_name="bra", ket_name="c", optimize="True"):
    eq, d = w.compile_einsum(equation, return_eq_dict=True)
    for idx, t in enumerate(d["rhs"]):
        if t[0] == bra_name:
            bra_idx = idx
        if t[0] == ket_name:
            ket_idx = idx

    # this is for the edge case of scaling the 'aAa' operator in spin integrated IP theory
    factor = 1.0
    if d["rhs"][bra_idx][1] == "aaA":
        factor *= np.sqrt(2)
    if d["rhs"][ket_idx][1] == "aAa":
        factor *= np.sqrt(2)
        
    # This is for cvs_ee
    bra_key = d["rhs"][bra_idx][1]
    ket_key = d["rhs"][ket_idx][1]
    if len(bra_key) == 4 and bra_key[0].islower() and bra_key[1].isupper():
        if (bra_key.count('a') + bra_key.count('A') > 0):
            if bra_key[0].lower() == bra_key[1].lower():
                factor *= np.sqrt(2)
            if bra_key[2].lower() == bra_key[3].lower():
                factor *= np.sqrt(2)
            
    if len(ket_key) == 4 and ket_key[0].islower() and ket_key[1].isupper():
        if (ket_key.count('a') + ket_key.count('A') > 0):
            if ket_key[0].lower() == ket_key[1].lower():
                factor *= np.sqrt(2)
            if ket_key[2].lower() == ket_key[3].lower():
                factor *= np.sqrt(2)

    d["factor"] = float(d["factor"]) * factor
    d["rhs"][ket_idx][2] = "p" + d["rhs"][ket_idx][2]
    bra = d["rhs"].pop(bra_idx)
    nbody = len(bra[2]) // 2
    bra[0] = "sigma"
    bra[2] = "p" + bra[2][nbody:] + bra[2][:nbody]
    bra[1] = bra[1][nbody:] + bra[1][:nbody]
    d["lhs"] = [bra]
    return w.dict_to_einsum(d, optimize=optimize)

def compile_sigma_vector_singles(equation, bra_name="bra", ket_name="c", optimize="True"):
    eq, d = w.compile_einsum(equation, return_eq_dict=True)
    for idx, t in enumerate(d["rhs"]):
        if t[0] == bra_name:
            bra_idx = idx
        if t[0] == ket_name:
            ket_idx = idx
    bra_true = False
    ket_true = False
    
    bra = d["rhs"][bra_idx][1]
    ket = d["rhs"][ket_idx][1]
    if len(bra) == 2:
        bra_true = True
    if len(ket) == 2:
        ket_true = True
    if bra_true and ket_true:
        return compile_sigma_vector(equation, bra_name=bra_name, ket_name=ket_name, optimize=optimize)

def compile_first_row(equation, ket_name="c", optimize="True"):
    factor = 1.0
    eq, d = w.compile_einsum(equation, return_eq_dict=True)
    for idx, t in enumerate(d["rhs"]):
        if t[0] == ket_name:
            ket_idx = idx
    
    # This is for cvs_ee
    ket_key = d["rhs"][ket_idx][1]
    if len(ket_key) == 4 and ket_key[0].islower() and ket_key[1].isupper():
        if (ket_key.count('a') + ket_key.count('A') > 0):
            if ket_key[0].lower() == ket_key[1].lower():
                factor *= np.sqrt(2)
            if ket_key[2].lower() == ket_key[3].lower():
                factor *= np.sqrt(2)
                
    d["factor"] = float(d["factor"]) * factor    
    ket = d["rhs"].pop(ket_idx)
    ket[0] = "sigma"
    d["lhs"] = [ket]
    return w.dict_to_einsum(d, optimize=optimize)


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


def matrix_elements_to_diag(mbeq, indent="once", optimize="True"):
    def _get_space(indices):
        # return 'aAaC' for input ['a4', 'A1', 'a5', 'C1']
        return "".join([i[0] for i in indices])
    indent_spaces = {"once": "    ", "twice": "        "}
    space = indent_spaces.get(indent, "    ")
    einsums = []
    for eq in mbeq:
        eqdict = w.equation_to_dict(eq)
        
        factor_scaled = float(eqdict["factor"])
        bra = _get_space(eqdict["lhs"][1])
        ket = _get_space(eqdict["lhs"][2])
        if len(bra) == 4 and bra.count('a')+bra.count('A') > 0:
            if bra[0].islower() and bra[1].isupper():
                if bra[0].lower() == bra[1].lower():
                    factor_scaled *= np.sqrt(2)
            if bra[2].islower() and bra[3].isupper():
                if bra[2].lower() == bra[3].lower():
                    factor_scaled *= np.sqrt(2)
        if len(ket) == 4 and ket.count('a')+ket.count('A') > 0:
            if ket[0].islower() and ket[1].isupper():
                if ket[0].lower() == ket[1].lower():
                    factor_scaled *= np.sqrt(2)
            if ket[2].islower() and ket[3].isupper():
                if ket[2].lower() == ket[3].lower():
                    factor_scaled *= np.sqrt(2)

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

        einsum = w.compile_einsum(w.dict_to_equation(eqdict_new), optimize="True")
        lhs = einsum.split(" +=")[0]
        factor = einsum.split(" += ")[1].split(" * ")[0]
        einsum = einsum.replace(lhs, lhs[0])
        einsum = einsum.replace(factor, f"{factor_scaled:.8f}")
        einsums.append(f"{space}{einsum}")

    func = "\n".join(einsums)
    return func


def generate_sigma_build(mbeq, matrix, first_row=True, optimize="True"):
    code = [
        f"def build_sigma_vector_{matrix}(einsum, c, Hbar, gamma1, eta1, lambda2, lambda3, lambda4, first_row):",
        "    sigma = {key: np.zeros(c[key].shape) for key in c.keys()}",
    ]

    for eq in mbeq["|"]:
        code.append(f"    {compile_sigma_vector(eq, optimize=optimize)}")

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
    elif matrix == "s" and first_row:
        code.append("    sigma['first'] = c['first'].copy()")

    code.append("    return sigma")
    return "\n".join(code)

def generate_sigma_build_singles(mbeq, matrix, optimize="True"):
    code = [
        f"def build_sigma_vector_{matrix}_singles(einsum, c, Hbar, gamma1, eta1, lambda2, lambda3, lambda4):",
        "    sigma = {key: np.zeros(c[key].shape) for key in c.keys() if len(key) == 2 or key == 'first'}",
    ]

    for eq in mbeq["|"]:
        line = compile_sigma_vector_singles(eq, optimize=optimize)
        if line is not None:
            code.append(f"    {line}")

    if matrix == "Hbar":
        code.append("    sigma['first'] = np.zeros_like(c['first'])")
    elif matrix == "s":
        code.append("    sigma['first'] = c['first'].copy()")

    code.append("    return sigma")
    return "\n".join(code)


def generate_template_c(block_list, index_dict, function_args):
    code = [f"def get_template_c({function_args}):", "    c = {"]

    for i in block_list:
        shape_strings = ["nlow"] + [f"{index_dict[item]}" for item in i]
        shape_formatted = ", ".join(shape_strings)
        code.append(f"         '{i}': np.zeros(({shape_formatted})),")

    code.append("        }")
    code.append("    return c")
    code = "\n".join(code)
    return code


def generate_first_row(mbeq, optimize="True"):
    code = [
        f"def build_first_row(einsum, c, Hbar, gamma1, eta1, lambda2, lambda3, lambda4):",
        "    sigma = {key: np.zeros((1, *tensor.shape[1:])) for key, tensor in c.items() if key != 'first'}",
    ]
    for eq in mbeq["|"]:
        code.append(f"    {compile_first_row(eq, ket_name='c', optimize=optimize)}")

    code.append("    return sigma")
    funct = "\n".join(code)
    return funct


def generate_transition_dipole(mbeq, ket_name='c',optimize="True"): 
    code = [
        f"def build_transition_dipole(einsum, c, Hbar, gamma1, eta1, lambda2, lambda3, lambda4):",
        "    sigma = 0.0",
    ]
    for eq in mbeq["|"]:
        factor = 1.0
        _, d = w.compile_einsum(eq, return_eq_dict=True)
        for idx, t in enumerate(d["rhs"]):
            if t[0] == ket_name:
                ket_idx = idx
        ket_key = d["rhs"][ket_idx][1]
        if len(ket_key) == 4 and ket_key[0].islower() and ket_key[1].isupper():
            if (ket_key.count('a') + ket_key.count('A') > 0):
                if ket_key[0].lower() == ket_key[1].lower():
                    factor *= np.sqrt(2)
                if ket_key[2].lower() == ket_key[3].lower():
                    factor *= np.sqrt(2)
        d["factor"] = float(d["factor"]) * factor 
        code.append(f"    {w.dict_to_einsum(d, optimize=optimize)}")
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
    optimize="True",
    ea=False,
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
                f"{space}{compile_sigma_vector(eq, bra_name=bra_name, ket_name=ket_name, optimize=optimize)}"
            )

    code.append(f"{space}sigma = antisymmetrize(sigma, ea={ea})")

    func = "\n".join(code)
    return func


def generate_S12(mbeq, single_space, composite_space, sequential = True, ea=False):
    """
    single_space: a list of strings.
    composite_space: a list of lists of strings.
    tol: tolerance for truncating singular values.
    tol_semi: tolerance for truncating singular values for semi-internals.
    """
    code = [
        f"def get_S12(eom_dsrg):",
        "    einsum = eom_dsrg.einsum",
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
            f'    if eom_dsrg.verbose: print("Starts {key} block")',
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
            
        scale = False
        if key[2].islower() and key[3].isupper():
            scale = True

        code_block.extend(
            [
                f"    anti = {anti}",
                f"    ge, gv = np.linalg.eigh({temp_rdm} *2 if {scale} else {temp_rdm})",
                f"    if np.any(ge < -tol):",
                f"        raise ValueError('Negative overlap eigenvalues found in {key} block')",
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
            f'    if eom_dsrg.verbose: print("Starts {key} block")',
            f"    shape_block = template_c['{key}'].shape[1:]",
            f"    shape_size = np.prod(shape_block)",
            f"    c['{key}'] = np.zeros((shape_size, *shape_block))",
            f"    sigma['{key}'] = np.zeros((shape_size, *shape_block))",
            f"    c_vec = dict_to_vec(c, shape_size)",
            f"    np.fill_diagonal(c_vec, 1)",
            f"    c = vec_to_dict(c, c_vec)",
            f"    del c_vec",
            f"    c = antisymmetrize(c, ea={ea})",
            f"    if eom_dsrg.verbose: print('Starts contraction')",
            generate_block_contraction(key, mbeq, block_type="single", indent="once", ea=ea),
            f"    c.clear()",
            f"    vec = dict_to_vec(sigma, shape_size)",
            f"    sigma.clear()",
            f"    if eom_dsrg.verbose: print('Starts diagonalization', flush = True)",
            f"    sevals, sevecs = np.linalg.eigh(vec)",
            f"    if np.any(sevals < -tol):",
            f"        raise ValueError('Negative overlap eigenvalues found in {key} block')",
            f"    del vec",
            f"    if eom_dsrg.verbose: print('Diagonalization done')",
            f"    trunc_indices = np.where(sevals > tol)[0]",
            f"    eom_dsrg.S12.{key} = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])",
            f"    num_ortho += eom_dsrg.S12.{key}.shape[1]",
            "    del sevals, sevecs, trunc_indices",
        ]

    def add_composite_space_block(space):
        code_block = [
            f"    # {space} composite block",
            f'    if eom_dsrg.verbose: print("Starts {space} composite block", flush = True)',
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
                f"    c = antisymmetrize(c, ea={ea})",
                f"    if eom_dsrg.verbose: print('Starts contraction')",
            ]
        )
        code_block.append(
            generate_block_contraction(
                space, mbeq, block_type="composite", indent="once", ea=ea,
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
                f"    if eom_dsrg.verbose: print('Starts diagonalization', flush = True)",
                "    print(f'Symmetric: {np.allclose(vec, vec.T)}', flush = True)",
                f"    sevals, sevecs = scipy.linalg.eigh(vec)",
                f"    if np.any(sevals < -tol):",
                f'        raise ValueError("Negative overlap eigenvalues found in {space} space")',
                f"    del vec",
                f"    if eom_dsrg.verbose: print('Diagonalization done', flush = True)",
                f"    trunc_indices = np.where(sevals > tol)[0]",
                # "    print(f'Number of orthogonalized operators: {len(trunc_indices)}', flush = True)",
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
            if len(key) == 1 or len(key) == 2:
                singles.append(key)

        code_block.extend(
            [
                f"    singles_size = 0",
                f"    for key in {singles}:",
                f"        shape_block = template_c[key].shape[1:]",
                f"        singles_size += np.prod(shape_block)",
                f"    S11 = vec[:singles_size, :singles_size].copy()",
                f"    S12 = vec[:singles_size, singles_size:].copy()",
                f"    sevals, sevecs = np.linalg.eigh(S11)",
                f"    trunc_indices = np.where(sevals > tol_semi)[0]",
                f"    S_inv_eval = 1.0/(sevals[trunc_indices])",
                f"    sevecs = sevecs[:, trunc_indices]",
                f"    S11inv = reduce(np.dot, (sevecs,np.diag(S_inv_eval),sevecs.T))",
                f"    Y12 = -np.matmul(S11inv, S12)",
                f"    Y = np.identity(vec.shape[0])",
                f"    Y[:singles_size, singles_size:] = Y12",
                f"    vec_proj = reduce(np.dot, (Y.T, vec, Y))",
                f"    del vec, S11, S12, S11inv, S_inv_eval",
                f"    if eom_dsrg.verbose: print('Starts diagonalization (after projection))', flush = True)",
                f"    sevals, sevecs = np.linalg.eigh(vec_proj)",
                f"    if np.any(sevals < -tol_semi):",
                f'        raise ValueError("Negative overlap eigenvalues found in {space} space")',
                f"    del vec_proj",
                f"    if eom_dsrg.verbose: print('Diagonalization done')",
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
        if any((len(key) == 1 or len(key) == 2) for key in space):
            if sequential:
                code.extend(sequential_orthogonalization(space))
            else:
                code.extend(add_composite_space_code(space))
        else:
            code.extend(add_composite_space_code(space))
        code.append("")  # Blank line for separation

    code.append(
        "    if eom_dsrg.verbose: print(f'Number of orthogonalized operators: {num_ortho}')"
    )

    return "\n".join(code)


def generate_preconditioner(
    mbeq, mbeqs_one_active, mbeqs_no_active, single_space, composite_space, first_row=True, ea=False
):
    """
    mbeqs_one_active and mbeqs_no_active are dictionaries.
    """
    code = [
        f"def compute_preconditioner(eom_dsrg):",
        "    einsum = eom_dsrg.einsum",
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
        f"    diagonal = [{'np.array([0.0])' if first_row else ''}]",
    ]

    def add_single_space_code(key):
        return [
            f"    # {key} block",
            f'    if eom_dsrg.verbose: print("Starts {key} block precond")',
            f"    shape_block = template_c['{key}'].shape[1:]",
            f"    northo = eom_dsrg.S12.{key}.shape[1]",
            f"    if northo != 0:",
            f"        c['{key}'] = np.zeros((northo, *shape_block))",
            f"        sigma['{key}'] = np.zeros((northo, *shape_block))",
            f"        c = vec_to_dict(c, eom_dsrg.S12.{key})",
            f"        c = antisymmetrize(c, ea={ea})",
            generate_block_contraction(key, mbeq, block_type="single", indent="twice", ea=ea),
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
            f'    if eom_dsrg.verbose: print("Starts {space} composite block precond")',
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
                f"        c = antisymmetrize(c, ea={ea})",
                generate_block_contraction(
                    space, mbeq, block_type="composite", indent="twice", ea=ea
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
            f'    if eom_dsrg.verbose: print("Starts {key} block precond")',
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
            f'    if eom_dsrg.verbose: print("Starts {key} block precond")',
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


def generate_apply_S12(single_space, composite_space, first_row=True):
    code_block = [
        f"def apply_S12(eom_dsrg, ndim, t, transpose=False):",
        f"    Xt = np.zeros((ndim, 1))",
        f"    i_start_xt = {'1' if first_row else '0'}",
        f"    i_start_t = {'1' if first_row else '0'}",
        f"{'    Xt[0, 0] = t[0]' if first_row else ''}",
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


def antisymmetrize(input_dict, ea=False):
    if not isinstance(input_dict, dict):
        return input_dict
    
    def _antisym(tensor, key):
        if len(key) <= 2:
            return tensor
        transpositions = []
        if len(key) == 3:
            if ea:
                if key[1] == key[2]:
                    transpositions.append(([0,1,3,2],-1))
            else:
                if key[0] == key[1]:
                    transpositions.append(([0,2,1,3],-1))
        elif len(key) == 4:
            do_lower = (key[0] == key[1])
            do_upper = (key[2] == key[3])
            if do_lower:
                transpositions.append(([0,2,1,3,4], -1))
            if do_upper:
                transpositions.append(([0,1,2,4,3], -1))
            if do_lower and do_upper:
                transpositions.append(([0,2,1,4,3], 1))
        
        tensor_new = tensor.copy()
        for trans, fact in transpositions:
            tensor_new += fact * tensor.transpose(trans)
        return tensor_new
    
    for key, tensor in input_dict.items():
        input_dict[key] = _antisym(tensor, key)

    return input_dict


def is_hermitian(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        print("Matrix is not square")
        return False

    # Check if the matrix is equal to its conjugate transpose
    # print(matrix)
    return np.allclose(matrix, matrix.conj().T)


def sym_dir(c, core_sym, occ_sym, act_sym, vir_sym):
    """
    Generate the symmetry of the direct product of two spaces.
    """
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
        if key == "first":
            out_dir[key] = np.array([0])
            continue

        if any(len(dir[key[i]]) == 0 for i in range(len(key))):
            out_dir[key] = np.zeros_like(c[key])
        else:
            if len(key) == 1:
                out_dir[key] = dir[key]
            elif len(key) == 2:
                out_dir[key] = dir[key[0]][:, None] ^ dir[key[1]][None, :]
            elif len(key) == 3:
                out_dir[key] = (
                    dir[key[0]][:, None, None]
                    ^ dir[key[1]][None, :, None]
                    ^ dir[key[2]][None, None, :]
                )
            elif len(key) == 4:
                out_dir[key] = (
                    dir[key[0]][:, None, None, None]
                    ^ dir[key[1]][None, :, None, None]
                    ^ dir[key[2]][None, None, :, None]
                    ^ dir[key[3]][None, None, None, :]
                )

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


def filter_list(element_list, ncore, nocc, nact, nvir):
    return [
        element
        for element in element_list
        if (ncore != 0 or ("i" not in element and "I" not in element))
        and (nocc != 0 or ("c" not in element and "C" not in element))
        and (nact != 0 or ("a" not in element and "A" not in element))
        and (nvir != 0 or ("v" not in element and "V" not in element))
    ]


def op_to_ms(op):
    ms2 = 0
    for i in op.split(" "):
        ms2 += ((i.islower()) * 2 - 1) * (("+" in i) * 2 - 1)
    return int(ms2)


def filter_ops_by_ms(ops, ms2):
    """
    Filter operators by Ms*2 (safe integer comparison)
    """
    return [op for op in ops if op_to_ms(op) == ms2]

def eigh_gen(A, S, eta=1e-10):
    if not is_hermitian(S):
        raise ValueError(
            f"Matrix S is not Hermitian. Max non-Hermicity is {np.max(np.abs(S - S.conj().T))}")
    if not is_hermitian(A):
        raise ValueError(
            f"Matrix A is not Hermitian. Max non-Hermicity is {np.max(np.abs(A - A.conj().T))}")
    sevals, sevecs = np.linalg.eigh(S)
    trunc_indices = np.where(sevals > eta)[0]
    X = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])
    Ap = X.T @ A @ X
    eigval, eigvec = np.linalg.eigh(Ap)
    eigvec = X @ eigvec
    return eigval, eigvec

