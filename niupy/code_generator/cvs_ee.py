import wicked as w
import itertools
import os
from niupy.eom_tools import *


def generator_full(abs_path, blocked_ortho=True):
    w.reset_space()
    # alpha
    w.add_space("i", "fermion", "occupied", list("cdij"))
    w.add_space("c", "fermion", "occupied", list("klmn"))
    w.add_space("v", "fermion", "unoccupied", list("efgh"))
    w.add_space("a", "fermion", "general", list("oabrstuvwxyz"))
    # #Beta
    w.add_space("I", "fermion", "occupied", list("CDIJ"))
    w.add_space("C", "fermion", "occupied", list("KLMN"))
    w.add_space("V", "fermion", "unoccupied", list("EFGH"))
    w.add_space("A", "fermion", "general", list("OABRSTUVWXYZ"))
    wt = w.WickTheorem()

    s = [""] # first row
    s += w.gen_op("bra", 1, "avAV", "ciaCIA", only_terms=True)
    s += w.gen_op("bra", 2, "avAV", "ciaCIA", only_terms=True)
    s = [_.strip() for _ in s]
    s = filter_ops_by_ms(s, 0)
    s = [_ for _ in s if ("I" in _ or "i" in _)]

    single_space, composite_space, block_list = get_subspaces(wt, s)
    if not blocked_ortho:
        single_space = []
        composite_space = [block_list]
    print("Single space:", single_space)
    print("Composite spaces:", composite_space)

    ops = [tensor_label_to_op(_) for _ in block_list]

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

    function_args = "nlow, ncore, nocc, nact, nvir"
    func_template_c = generate_template_c(block_list, index_dict, function_args)

    H = w.gen_op_ms0("Hbar", 1, "ciav", "ciav") + w.gen_op_ms0(
        "Hbar", 2, "ciav", "ciav"
    )

    Hmbeq = {}
    Smbeq = {}
    for ibra in range(len(ops)):
        bop = ops[ibra]
        bra = w.op("bra", [bop])
        braind = op_to_index(bop)
        for iket in range(ibra + 1):
            kop = ops[iket]
            ket = w.op("ket", [kop])
            ketind = op_to_index(kop)
            S_mbeq = get_matrix_elements(
                wt, bra, None, ket, inter_general=True, double_comm=False
            )
            if S_mbeq:
                Smbeq[f"{braind}|{ketind}"] = S_mbeq
            double_comm = (kop.count("a") + kop.count("A") == 3) and (
                bop.count("a") + bop.count("A") == 3
            )
            H_mbeq = get_matrix_elements(
                wt, bra, H, ket, inter_general=True, double_comm=double_comm
            )
            if H_mbeq:
                Hmbeq[f"{braind}|{ketind}"] = H_mbeq

    print(f"Code generator: Writing to {abs_path}")

    with open(os.path.join(abs_path, "cvs_ee_eom_dsrg_full.py"), "w") as f:
        f.write(f"import numpy as np\n")
        f.write(f"from niupy.eom_tools import *\n\n")
        f.write(f"{func_template_c}\n\n")
        f.write(make_driver(Hmbeq, Smbeq))
        for k, v in Hmbeq.items():
            if v:
                f.write(make_function(k, v, "H") + "\n")
        for k, v in Smbeq.items():
            if v:
                f.write(make_function(k, v, "S") + "\n")
    return ops, single_space, composite_space


def generator_subspace(abs_path, ncore, nocc, nact, nvir, blocked_ortho=True):
    w.reset_space()
    # alpha
    w.add_space("i", "fermion", "occupied", list("cdij"))
    w.add_space("c", "fermion", "occupied", list("klmn"))
    w.add_space("v", "fermion", "unoccupied", list("efgh"))
    w.add_space("a", "fermion", "general", list("oabrstuvwxyz"))
    # #Beta
    w.add_space("I", "fermion", "occupied", list("CDIJ"))
    w.add_space("C", "fermion", "occupied", list("KLMN"))
    w.add_space("V", "fermion", "unoccupied", list("EFGH"))
    w.add_space("A", "fermion", "general", list("OABRSTUVWXYZ"))
    wt = w.WickTheorem()

    s = w.gen_op("bra", 1, "avAV", "ciaCIA", only_terms=True)
    # s.extend(["a+ a+ a i", "a+ A+ A i", "A+ A+ A I", "a+ A+ I a"])
    # + w.gen_op(
    #     "bra", 2, "avAV", "ciaCIA", only_terms=True
    # )
    s = [_.strip() for _ in s]
    s = filter_ops_by_ms(s, 0)
    s = [_ for _ in s if ("I" in _ or "i" in _)]
    s = filter_list(s, ncore, nocc, nact, nvir)

    single_space, composite_space, block_list = get_subspaces(wt, s)
    single_space = filter_list(single_space, ncore, nocc, nact, nvir)
    composite_space = [filter_list(_, ncore, nocc, nact, nvir) for _ in composite_space]
    block_list = single_space + list(itertools.chain(*composite_space))

    if not blocked_ortho:
        single_space = []
        composite_space = [block_list]
    print("Single space:", single_space)
    print("Composite spaces:", composite_space)

    ops = [tensor_label_to_op(_) for _ in block_list]
    print("Operators:", ops)

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

    function_args = "nlow, ncore, nocc, nact, nvir"
    func_template_c = generate_template_c(block_list, index_dict, function_args)

    H = w.gen_op_ms0("Hbar", 1, "ciav", "ciav") + w.gen_op_ms0(
        "Hbar", 2, "ciav", "ciav"
    )

    Hmbeq = {}
    Smbeq = {}
    for ibra in range(len(ops)):
        bop = ops[ibra]
        bra = w.op("bra", [bop])
        braind = op_to_index(bop)
        for iket in range(ibra + 1):
            kop = ops[iket]
            ket = w.op("ket", [kop])
            ketind = op_to_index(kop)
            S_mbeq = get_matrix_elements(
                wt, bra, None, ket, inter_general=True, double_comm=False
            )
            if S_mbeq:
                Smbeq[f"{braind}|{ketind}"] = S_mbeq
            double_comm = (kop.count("a") + kop.count("A") == 3) and (
                bop.count("a") + bop.count("A") == 3
            )
            H_mbeq = get_matrix_elements(
                wt, bra, H, ket, inter_general=True, double_comm=double_comm
            )
            if H_mbeq:
                Hmbeq[f"{braind}|{ketind}"] = H_mbeq

    print(f"Code generator: Writing to {abs_path}")

    with open(os.path.join(abs_path, "cvs_ee_eom_dsrg_full.py"), "w") as f:
        f.write(f"import numpy as np\n")
        f.write(f"from niupy.eom_tools import *\n\n")
        f.write(f"{func_template_c}\n\n")
        f.write(make_driver(Hmbeq, Smbeq))
        for k, v in Hmbeq.items():
            if v:
                f.write(make_function(k, v, "H") + "\n")
        for k, v in Smbeq.items():
            if v:
                f.write(make_function(k, v, "S") + "\n")
    return ops, single_space, composite_space


def generator(
    abs_path,
    ncore,
    nocc,
    nact,
    nvir,
    einsum_type,
    sequential_ortho=True,
    blocked_ortho=True,
):
    w.reset_space()
    # alpha
    w.add_space("i", "fermion", "occupied", list("cdij"))
    w.add_space("c", "fermion", "occupied", list("klmn"))
    w.add_space("v", "fermion", "unoccupied", list("efgh"))
    w.add_space("a", "fermion", "general", list("oabrstuvwxyz"))
    # #Beta
    w.add_space("I", "fermion", "occupied", list("CDIJ"))
    w.add_space("C", "fermion", "occupied", list("KLMN"))
    w.add_space("V", "fermion", "unoccupied", list("EFGH"))
    w.add_space("A", "fermion", "general", list("OABRSTUVWXYZ"))
    wt = w.WickTheorem()

    # Define operators
    s = w.gen_op("bra", 1, "avAV", "ciaCIA", only_terms=True) + w.gen_op(
        "bra", 2, "avAV", "ciaCIA", only_terms=True
    )
    s = [_.strip() for _ in s]
    s = filter_ops_by_ms(s, 0)
    s = [_ for _ in s if ("I" in _ or "i" in _)]
    s = filter_list(s, ncore, nocc, nact, nvir)

    single_space, composite_space, block_list = get_subspaces(wt, s)
    single_space = filter_list(single_space, ncore, nocc, nact, nvir)
    composite_space = [filter_list(_, ncore, nocc, nact, nvir) for _ in composite_space]
    block_list = single_space + list(itertools.chain(*composite_space))

    if not blocked_ortho:
        single_space = []
        composite_space = [block_list]
    print("Single space:", single_space)
    print("Composite spaces:", composite_space)

    # # These blocks are computed with the commutator trick.
    s_comm = [_ for _ in s if _.count("a") + _.count("A") >= 3]
    print("Commutator trick:", s_comm)

    T_adj = w.op("bra", s, unique=True).adjoint()
    T = w.op("c", s, unique=True)

    for i in s_comm:
        s.remove(i)

    T_comm_adj = w.op("bra", s_comm, unique=True).adjoint()
    T_comm = w.op("c", s_comm, unique=True)

    T_original_adj = w.op("bra", s, unique=True).adjoint()
    T_original = w.op("c", s, unique=True)

    # Define Hbar
    Hbar_op = w.gen_op_ms0("Hbar", 1, "ciav", "ciav") + w.gen_op_ms0(
        "Hbar", 2, "ciav", "ciav"
    )

    # ============================================================================

    one_active_two_virtual = []
    no_active = []

    for i in s:
        test_s = i.lower()
        num_v = test_s.count("v")
        num_a = test_s.count("a")
        if num_a == 1 and num_v == 2:
            one_active_two_virtual.append(i)
        elif num_a == 0 and num_v == 2:
            no_active.append(i)
        else:
            continue

    mbeqs_one_active_two_virtual = {}
    mbeqs_no_active = {}

    for i in one_active_two_virtual:
        bra = w.op("bra", [i])
        ket = w.op("ket", [i])
        inter_general = any(char.isupper() for char in i) and any(
            char.islower() for char in i
        )
        label = op_to_tensor_label(i)
        mbeqs_one_active_two_virtual[label] = get_matrix_elements(
            wt, bra, Hbar_op, ket, inter_general=inter_general, double_comm=False
        )
    for i in no_active:
        bra = w.op("bra", [i])
        ket = w.op("ket", [i])
        inter_general = any(char.isupper() for char in i) and any(
            char.islower() for char in i
        )
        label = op_to_tensor_label(i)
        mbeqs_no_active[label] = get_matrix_elements(
            wt, bra, Hbar_op, ket, inter_general=inter_general, double_comm=False
        )

    # Template C
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
    function_args = "nlow, ncore, nocc, nact, nvir"
    func_template_c = generate_template_c(block_list, index_dict, function_args)

    # Hbar
    THT_comm = w.rational(1, 2) * (
        T_comm_adj @ w.commutator(Hbar_op, T_comm)
        + w.commutator(T_comm_adj, Hbar_op) @ T_comm
    )
    THT_original = T_original_adj @ Hbar_op @ T_original
    THT_coupling = T_original_adj @ Hbar_op @ T_comm
    THT_coupling_2 = T_comm_adj @ Hbar_op @ T_original
    THT = THT_comm + THT_original + THT_coupling + THT_coupling_2
    expr = wt.contract(THT, 0, 0, inter_general=True)
    mbeq = expr.to_manybody_equation("sigma")

    # First row/column
    HT = Hbar_op @ T
    expr_first = wt.contract(HT, 0, 0, inter_general=True)
    mbeq_first = expr_first.to_manybody_equation("sigma")

    # S
    TT = T_adj @ T
    expr_s = wt.contract(TT, 0, 0, inter_general=True)
    mbeq_s = expr_s.to_manybody_equation("sigma")

    # Generate wicked contraction
    funct = generate_sigma_build(
        mbeq, "Hbar", first_row=True, einsum_type=einsum_type
    )  # HC
    funct_s = generate_sigma_build(
        mbeq_s, "s", first_row=True, einsum_type=einsum_type
    )  # SC
    funct_first = generate_first_row(
        mbeq_first, einsum_type=einsum_type
    )  # First row/column
    funct_dipole = generate_transition_dipole(mbeq_first, einsum_type=einsum_type)
    funct_S12 = generate_S12(
        mbeq_s,
        single_space,
        composite_space,
        sequential=sequential_ortho,
        einsum_type=einsum_type,
    )
    funct_preconditioner = generate_preconditioner(
        mbeq,
        mbeqs_one_active_two_virtual,
        mbeqs_no_active,
        single_space,
        composite_space,
        first_row=True,
        einsum_type=einsum_type,
    )
    funct_apply_S12 = generate_apply_S12(single_space, composite_space, first_row=True)

    # script_dir = os.path.dirname(__file__)
    # rel_path = "../cvs_ee_eom_dsrg.py"
    # abs_file_path = os.path.join(script_dir, rel_path)
    # print(f"Code generator: Writing to {abs_file_path}")
    print(f"Code generator: Writing to {abs_path}")

    with open(os.path.join(abs_path, "cvs_ee_eom_dsrg.py"), "w") as f:
        f.write(
            "import numpy as np\nimport scipy\nimport time\n\nfrom functools import reduce\n\nfrom niupy.eom_tools import *\n\n"
        )
        f.write(f"{func_template_c}\n\n")
        f.write(f"{funct_S12}\n\n")
        f.write(f"{funct_preconditioner}\n\n")
        f.write(f"{funct_apply_S12}\n\n")
        f.write(f"{funct}\n\n")
        f.write(f"{funct_s}\n\n")
        f.write(f"{funct_first}\n\n")
        f.write(f"{funct_dipole}\n\n")
