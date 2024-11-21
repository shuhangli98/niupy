import wicked as w
import itertools
import os
from niupy.eom_tools import *


def generator(abs_path, ncore, nocc, nact, nvir):
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
    s = []

    for i in itertools.product(["v+", "a+"], ["i"]):
        s.append(" ".join(i))
    for i in itertools.product(["V+", "A+"], ["I"]):
        s.append(" ".join(i))
    for i in itertools.product(["v+", "a+"], ["v+", "a+"], ["a", "c", "i"], ["i"]):
        s.append(" ".join(i))
    for i in itertools.product(["V+", "A+"], ["V+", "A+"], ["A", "C", "I"], ["I"]):
        s.append(" ".join(i))
    for i in itertools.product(["v+", "a+"], ["V+", "A+"], ["a", "c", "i"], ["I"]):
        s.append(" ".join(i))
    for i in itertools.product(["v+", "a+"], ["V+", "A+"], ["A", "C", "I"], ["i"]):
        s.append(" ".join(i))

    s = filter_list(s, ncore, nocc, nact, nvir)

    T_adj = w.op("bra", s, unique=True).adjoint()
    T = w.op("c", s, unique=True)

    # These blocks are computed with the commutator trick.
    s_comm = ["a+ a+ a i", "A+ A+ A I", "a+ A+ a I", "a+ A+ A i"]
    for i in s_comm:
        s.remove(i)

    T_comm_adj = w.op("bra", s_comm, unique=True).adjoint()
    T_comm = w.op("c", s_comm, unique=True)

    T_original_adj = w.op("bra", s, unique=True).adjoint()
    T_original = w.op("c", s, unique=True)

    # Define Hbar
    Hops = []
    for i in itertools.product(["v+", "a+", "c+", "i+"], ["v", "a", "c", "i"]):
        Hops.append(" ".join(i))
    for i in itertools.product(["V+", "A+", "C+", "I+"], ["V", "A", "C", "I"]):
        Hops.append(" ".join(i))
    for i in itertools.product(
        ["v+", "a+", "c+", "i+"],
        ["v+", "a+", "c+", "i+"],
        ["v", "a", "c", "i"],
        ["v", "a", "c", "i"],
    ):
        Hops.append(" ".join(i))
    for i in itertools.product(
        ["V+", "A+", "C+", "I+"],
        ["V+", "A+", "C+", "I+"],
        ["V", "A", "C", "I"],
        ["V", "A", "C", "I"],
    ):
        Hops.append(" ".join(i))
    for i in itertools.product(
        ["v+", "a+", "c+", "i+"],
        ["V+", "A+", "C+", "I+"],
        ["v", "a", "c", "i"],
        ["V", "A", "C", "I"],
    ):
        Hops.append(" ".join(i))
    Hbar_op = w.op("Hbar", Hops, unique=True)

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
            bra, Hbar_op, ket, inter_general=inter_general, double_comm=False
        )
    for i in no_active:
        bra = w.op("bra", [i])
        ket = w.op("ket", [i])
        inter_general = any(char.isupper() for char in i) and any(
            char.islower() for char in i
        )
        label = op_to_tensor_label(i)
        mbeqs_no_active[label] = get_matrix_elements(
            bra, Hbar_op, ket, inter_general=inter_general, double_comm=False
        )

    # Define subspaces. Single first!
    S_half_0 = [
        "iv",
        "IV",
        "iAaV",
        "aIvA",
        "icvv",
        "iCvV",
        "cIvV",
        "ICVV",
        "iivv",
        "iIvV",
        "IIVV",
    ]  #        "iv", "IV"
    S_half_1 = [
        "icva",
        "iCvA",
        "iCaV",
        "cIvA",
        "cIaV",
        "ICVA",
        "iiva",
        "iIvA",
        "iIaV",
        "IIVA",
    ]
    S_half_minus_1 = ["iavv", "iAvV", "aIvV", "IAVV"]
    S_half_2 = ["icaa", "iCaA", "cIaA", "ICAA", "iiaa", "iIaA", "IIAA"]

    S_half_0_com_iv = ["iava", "iAvA"]
    S_half_0_com_IV = ["aIaV", "IAVA"]

    S_half_1_com_i = ["ia", "iaaa", "iAaA"]  # ia
    S_half_1_com_I = ["IA", "aIaA", "IAAA"]  # IA

    S_half_0 = filter_list(S_half_0, ncore, nocc, nact, nvir)
    S_half_1 = filter_list(S_half_1, ncore, nocc, nact, nvir)
    S_half_minus_1 = filter_list(S_half_minus_1, ncore, nocc, nact, nvir)
    S_half_2 = filter_list(S_half_2, ncore, nocc, nact, nvir)

    S_half_0_com_iv = filter_list(S_half_0_com_iv, ncore, nocc, nact, nvir)
    S_half_0_com_IV = filter_list(S_half_0_com_IV, ncore, nocc, nact, nvir)
    S_half_1_com_i = filter_list(S_half_1_com_i, ncore, nocc, nact, nvir)
    S_half_1_com_I = filter_list(S_half_1_com_I, ncore, nocc, nact, nvir)

    block_list = (
        S_half_0
        + S_half_1
        + S_half_minus_1
        + S_half_2
        + S_half_0_com_iv
        + S_half_0_com_IV
        + S_half_1_com_i
        + S_half_1_com_I
    )
    single_space = S_half_0 + S_half_1 + S_half_minus_1 + S_half_2
    composite_space = [S_half_0_com_iv, S_half_0_com_IV, S_half_1_com_i, S_half_1_com_I]

    # Template C
    func_template_c = generate_template_c(block_list)

    # Hbar
    THT_comm = w.rational(1, 2) * (
        T_comm_adj @ w.commutator(Hbar_op, T_comm)
        + w.commutator(T_comm_adj, Hbar_op) @ T_comm
    )
    THT_original = T_original_adj @ Hbar_op @ T_original
    THT_coupling = T_original_adj @ Hbar_op @ T_comm
    THT_coupling_2 = T_comm_adj @ Hbar_op @ T_original
    THT = THT_comm + THT_original + THT_coupling + THT_coupling_2
    # THT = T_adj @ Hbar_op @ T
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
    funct = generate_sigma_build(mbeq, "Hbar")  # HC
    funct_s = generate_sigma_build(mbeq_s, "s")  # SC
    funct_first = generate_first_row(mbeq_first)  # First row/column
    funct_dipole = generate_transition_dipole(mbeq_first)
    funct_S12 = generate_S12(mbeq_s, single_space, composite_space)
    funct_preconditioner = generate_preconditioner(
        mbeq,
        mbeqs_one_active_two_virtual,
        mbeqs_no_active,
        single_space,
        composite_space,
    )
    funct_apply_S12 = generate_apply_S12(single_space, composite_space)
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
