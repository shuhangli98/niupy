import wicked as w
import itertools
import os
from niupy.eom_tools import *


def generator_full(log, abs_path, ncore, nocc, nact, nvir, blocked_ortho=True):

    log.info("\nCode generator starts\n")

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
    log.debug(f"Single space: {single_space}")
    log.debug(f"Composite spaces: {composite_space}")
    ops = [tensor_label_to_op(_) for _ in block_list]
    log.debug(f"Operators: {ops}")

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

    log.info(f"Code generator: Writing to {abs_path}")

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


def generator_subspace(log, abs_path, ncore, nocc, nact, nvir, blocked_ortho=True):

    log.info("\nCode generator starts\n")

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
    log.debug(f"Single space: {single_space}")
    log.debug(f"Composite spaces: {composite_space}")
    ops = [tensor_label_to_op(_) for _ in block_list]
    log.debug(f"Operators: {ops}")

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

    log.info(f"Code generator: Writing to {abs_path}")

    with open(os.path.join(abs_path, "cvs_ee_eom_dsrg_subspace.py"), "w") as f:
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
    log,
    abs_path,
    ncore,
    nocc,
    nact,
    nvir,
    einsum_type,
    sequential_ortho=True,
    blocked_ortho=True,
    first_row=False,
):

    log.info("\nCode generator starts\n")

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

    s_small = [_ for _ in s if ("v" not in _ and "V" not in _) or len(_) == 4]
    s_large = [_ for _ in s if ("v" in _ or "V" in _) and len(_) == 9]

    s = filter_list(s, ncore, nocc, nact, nvir)
    s_small = filter_list(s_small, ncore, nocc, nact, nvir)
    s_large = filter_list(s_large, ncore, nocc, nact, nvir)
    log.debug(f"small blocks: {s_small}")
    log.debug(f"large blocks: {s_large}")
    log.debug(f"all: {s}")

    single_space, composite_space, block_list = get_subspaces(wt, s)
    single_space = filter_list(single_space, ncore, nocc, nact, nvir)
    composite_space = [filter_list(_, ncore, nocc, nact, nvir) for _ in composite_space]
    block_list = single_space + list(itertools.chain(*composite_space))

    small_space = [op_to_tensor_label(op) for op in s_small]

    if not blocked_ortho:
        single_space = []
        composite_space = [block_list]
    log.debug(f"Single space: {single_space}")
    log.debug(f"Composite spaces: {composite_space}")
    log.debug(f"Small space: {small_space}")

    # # These blocks are computed with the commutator trick.
    s_comm = [_ for _ in s if _.count("a") + _.count("A") >= 3]
    log.debug(f"Commutator trick: {s_comm}")

    # ============================================================================
    # Initialize operators
    T = w.op("c", s, unique=True)  # Used in first_row function

    for i in s_comm:
        s.remove(i)

    T_comm_adj = w.op("bra", s_comm, unique=True).adjoint()
    T_comm = w.op("c", s_comm, unique=True)

    T_original_adj = w.op("bra", s, unique=True).adjoint()
    T_original = w.op("c", s, unique=True)

    T_small_adj = w.op("bra", s_small, unique=True).adjoint()
    T_small = w.op("c", s_small, unique=True)

    T_large_adj = w.op("bra", s_large, unique=True).adjoint()
    T_large = w.op("c", s_large, unique=True)

    # Define Hbar
    Hbar_op = w.gen_op_ms0("Hbar", 1, "ciav", "ciav") + w.gen_op_ms0(
        "Hbar", 2, "ciav", "ciav"
    )

    # ============================================================================
    # Generate block S functions for single and composite spaces.
    single_ops = [tensor_label_to_op(_) for _ in single_space]
    Smbeq = {}
    for iop in range(len(single_ops)):
        sop = single_ops[iop]
        bra = w.op("bra", [sop])
        braind = op_to_index(sop)
        ket = w.op("ket", [sop])
        ketind = op_to_index(sop)
        S_mbeq = get_matrix_elements(
            wt,
            bra,
            None,
            ket,
            inter_general=True,
            double_comm=False,
        )
        if S_mbeq:
            Smbeq[f"{braind}|{ketind}"] = S_mbeq

    drivers = []
    for icomp in range(len(composite_space)):
        Smbeq_comp = {}
        composite_ops = [tensor_label_to_op(_) for _ in composite_space[icomp]]
        for ibra in range(len(composite_ops)):
            bop = composite_ops[ibra]
            bra = w.op("bra", [bop])
            braind = op_to_index(bop)
            for iket in range(ibra + 1):
                kop = composite_ops[iket]
                ket = w.op("ket", [kop])
                ketind = op_to_index(kop)
                S_mbeq = get_matrix_elements(
                    wt,
                    bra,
                    None,
                    ket,
                    inter_general=True,
                    double_comm=False,
                )
                if S_mbeq:
                    Smbeq[f"{braind}|{ketind}"] = S_mbeq
                    Smbeq_comp[f"{braind}|{ketind}"] = S_mbeq
        drivers.append(make_driver_composite(Smbeq_comp, composite_space[icomp], "S"))

    Hmbeq = {}
    for ibra in range(len(s_small)):
        bop = s_small[ibra]
        bra = w.op("bra", [bop])
        braind = op_to_index(bop)
        for iket in range(ibra + 1):
            kop = s_small[iket]
            ket = w.op("ket", [kop])
            ketind = op_to_index(kop)
            double_comm = (kop.count("a") + kop.count("A") == 3) and (
                bop.count("a") + bop.count("A") == 3
            )
            H_mbeq = get_matrix_elements(
                wt,
                bra,
                Hbar_op,
                ket,
                inter_general=True,
                double_comm=double_comm,
            )
            if H_mbeq:
                Hmbeq[f"{braind}|{ketind}"] = H_mbeq

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
    THT_large = T_large_adj @ Hbar_op @ T_large
    THT_coupling = T_large_adj @ Hbar_op @ T_small
    THT_coupling_2 = T_small_adj @ Hbar_op @ T_large
    THT = THT_large + THT_coupling + THT_coupling_2
    expr = wt.contract(THT, 0, 0, inter_general=True)
    mbeq = expr.to_manybody_equation("sigma")

    # First row/column
    HT = Hbar_op @ T
    expr_first = wt.contract(HT, 0, 0, inter_general=True)
    mbeq_first = expr_first.to_manybody_equation("sigma")

    # Generate wicked contraction
    funct = generate_sigma_build(
        mbeq, "Hbar", first_row=first_row, einsum_type=einsum_type
    )  # HC
    if first_row:
        funct_first = generate_first_row(
            mbeq_first, einsum_type=einsum_type
        )  # First row/column
    funct_dipole = generate_transition_dipole(mbeq_first, einsum_type=einsum_type)
    funct_S12 = generate_S12(
        single_space,
        composite_space,
        sequential=sequential_ortho,
    )

    # The preconditioner function needs full mbeq. TODO: read from small H.
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

    funct_preconditioner = generate_preconditioner(
        mbeq,
        mbeqs_one_active_two_virtual,
        mbeqs_no_active,
        single_space,
        composite_space,
        first_row=first_row,
        einsum_type=einsum_type,
    )
    funct_apply_S12 = generate_apply_S12(
        single_space, composite_space, first_row=first_row
    )

    log.info(f"Code generator: Writing to {abs_path}")

    with open(os.path.join(abs_path, "cvs_ee_eom_dsrg.py"), "w") as f:
        f.write(
            "import numpy as np\nimport scipy\nimport time\n\nfrom functools import reduce\n\nfrom niupy.eom_tools import *\n\n"
        )
        f.write(f"{func_template_c}\n\n")
        for k, v in Smbeq.items():
            if v:
                if len(k) == 19:
                    trans = (2, 3, 0, 1, 6, 7, 4, 5)
                elif len(k) == 14:
                    trans = (2, 3, 0, 1, 5, 4)
                elif len(k) == 9:
                    trans = (1, 0, 3, 2)
                f.write(
                    make_function(
                        k,
                        v,
                        "S",
                        trans=trans,
                        transpose=True,
                    )
                    + "\n"
                )
        for k, v in Hmbeq.items():
            if v:
                if len(k) == 19:
                    trans = (2, 3, 0, 1, 6, 7, 4, 5)
                elif len(k) == 14:
                    trans = (2, 3, 0, 1, 5, 4)
                elif len(k) == 9:
                    trans = (1, 0, 3, 2)
                f.write(
                    make_function(
                        k,
                        v,
                        "H",
                        trans=trans,
                        transpose=True,
                    )
                    + "\n"
                )
        for i in range(len(drivers)):
            f.write(drivers[i] + "\n")
        f.write(f'{make_driver_composite(Hmbeq, small_space, "H", "small")}\n\n')
        f.write(f"{funct_S12}\n\n")
        f.write(f"{funct_preconditioner}\n\n")
        f.write(f"{funct_apply_S12}\n\n")
        f.write(f"{funct}\n\n")
        if first_row:
            f.write(f"{funct_first}\n\n")
        else:
            f.write(f"build_first_row = NotImplemented\n")
        f.write(f"{funct_dipole}\n\n")

        return s_small, small_space
