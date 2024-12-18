import wicked as w
import itertools
import os
from niupy.eom_tools import *

def generator_full_hbar(abs_path):
    w.reset_space()
    w.add_space('c', 'fermion', 'occupied', list('ijklmn'))
    w.add_space('v', 'fermion', 'unoccupied', list('abcdef'))
    w.add_space('a', 'fermion', 'general', list('stuvwxyz'))
    w.add_space('C', 'fermion', 'occupied', list('IJKLMN'))
    w.add_space('V', 'fermion', 'unoccupied', list('ABCDEF'))
    w.add_space('A', 'fermion', 'general', list('STUVWXYZ'))
    wt = w.WickTheorem()
    wt.set_max_cumulant(4)
    
    single_space = [
        "C",
        "cCa",
        "cAa",
        "cCv",
        "aCv",
        "cAv",
        "aAv",
        "CCA",
        "CCV",
        "CAV",
        "AAV",
    ]
    aac = ["aCa", "CAA"]
    active = ["A", "AAA", "aAa"]
    composite_space = [aac, active]
    block_list = single_space + aac + active

    ops = [tensor_label_to_op(_) for _ in block_list]

    index_dict = {
        "c": "nocc",
        "a": "nact",
        "v": "nvir",
        "C": "nocc",
        "A": "nact",
        "V": "nvir",
    }

    function_args = "nlow, ncore, nocc, nact, nvir"
    func_template_c = generate_template_c(block_list, index_dict, function_args)
    
    H = w.gen_op_ms0('Hbar', 1, 'cav', 'cav') +  w.gen_op_ms0('Hbar', 2, 'cav', 'cav')

    Hmbeq = {}
    Smbeq = {}
    for ibra in range(len(ops)):
        bop = ops[ibra]
        bra = w.op('bra', [bop])
        braind = op_to_index(bop)
        for iket in range(ibra+1):
            kop = ops[iket]
            ket = w.op('ket', [kop])
            ketind = op_to_index(kop)
            S_mbeq = get_matrix_elements(wt, bra, None, ket, inter_general=True, double_comm=False)
            if S_mbeq: 
                Smbeq[f'{braind}|{ketind}'] = S_mbeq
            double_comm = (kop.count('a') + kop.count('A') == 3) and (bop.count('a') + bop.count('A') == 3)
            H_mbeq = get_matrix_elements(wt, bra, H, ket, inter_general=True, double_comm=double_comm)
            if H_mbeq: Hmbeq[f'{braind}|{ketind}'] = H_mbeq

    singles = w.gen_op("bra", (0, 1), "avAV", "caCA", only_terms=True)
    singles = [_.strip() for _ in singles]
    singles = filter_ops_by_ms(singles, 1)
    T = w.op("c", ops, unique=True)
    P_adj = w.op("bra", singles, unique=True).adjoint()
    PT = P_adj @ T
    expr_p = wt.contract(PT, 0, 0, inter_general=True)
    mbeq_p = expr_p.to_manybody_equation("sigma")
    funct_p = generate_sigma_build(mbeq_p, "p", first_row=False, optimize="True")

    print(f"Code generator: Writing to {abs_path}")

    with open(os.path.join(abs_path, "ip_eom_dsrg_full.py"), "w") as f:
        f.write(f'import numpy as np\n')
        f.write(f'from niupy.eom_tools import *\n\n')
        f.write(f"{func_template_c}\n\n")
        f.write(f"{funct_p}\n\n")
        f.write(make_driver(Hmbeq, Smbeq))
        for k, v in Hmbeq.items():
            if v: f.write(make_function(k, v, 'H')+ '\n')
        for k, v in Smbeq.items():
            if v: f.write(make_function(k, v, 'S')+ '\n')
    return ops

def generator(abs_path):
    w.reset_space()
    # alpha
    w.add_space("c", "fermion", "occupied", list("klmn"))
    w.add_space("v", "fermion", "unoccupied", list("efgh"))
    w.add_space("a", "fermion", "general", list("oabrstuvwxyz"))
    # #Beta
    w.add_space("C", "fermion", "occupied", list("KLMN"))
    w.add_space("V", "fermion", "unoccupied", list("EFGH"))
    w.add_space("A", "fermion", "general", list("OABRSTUVWXYZ"))
    wt = w.WickTheorem()
    wt.set_max_cumulant(4)

    # Define operators
    s = w.gen_op("bra", (0, 1), "avAV", "caCA", only_terms=True) \
        + w.gen_op("bra", (1, 2), "avAV", "caCA", only_terms=True)
    s = [_.strip() for _ in s]
    s = filter_ops_by_ms(s, 1)
    s_comm = [_ for _ in s if _.count("a") + _.count("A") >= 3]
    print('Commutator trick:', s_comm)

    # used in spectroscopic amplitudes
    singles = w.gen_op("bra", (0, 1), "avAV", "caCA", only_terms=True)
    singles = [_.strip() for _ in singles]
    singles = filter_ops_by_ms(singles, 1)
    P_adj = w.op("bra", singles, unique=True).adjoint()

    T_adj = w.op("bra", s, unique=True).adjoint()
    T = w.op("c", s, unique=True)
    for i in s_comm:
        s.remove(i)

    T_comm_adj = w.op("bra", s_comm, unique=True).adjoint()
    T_comm = w.op("c", s_comm, unique=True)

    T_original_adj = w.op("bra", s, unique=True).adjoint()
    T_original = w.op("c", s, unique=True)

    # Define subspaces. Single first!
    single_space = [
        "C",
        "cCa",
        "cAa",
        "cCv",
        "aCv",
        "cAv",
        "aAv",
        "CCA",
        "CCV",
        "CAV",
        "AAV",
    ]
    aac = ["aCa", "CAA"]
    active = ["A", "AAA", "aAa"]
    composite_space = [aac, active]
    block_list = single_space + aac + active

    # Define Hbar
    Hbar_op = w.gen_op_ms0("Hbar", 1, "cav", "cav") + w.gen_op_ms0(
        "Hbar", 2, "cav", "cav"
    )

    # Template C
    index_dict = {
        "c": "nocc",
        "a": "nact",
        "v": "nvir",
        "C": "nocc",
        "A": "nact",
        "V": "nvir",
    }

    function_args = "nlow, ncore, nocc, nact, nvir"
    func_template_c = generate_template_c(block_list, index_dict, function_args)

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

    # S
    TT = T_adj @ T
    expr_s = wt.contract(TT, 0, 0, inter_general=True)
    mbeq_s = expr_s.to_manybody_equation("sigma")

    PT = P_adj @ T
    expr_p = wt.contract(PT, 0, 0, inter_general=True)
    mbeq_p = expr_p.to_manybody_equation("sigma")

    # Generate wicked contraction
    funct = generate_sigma_build(mbeq, "Hbar", first_row=False, optimize="True")  # HC
    funct_s = generate_sigma_build(mbeq_s, "s", first_row=False, optimize="True")  # SC
    funct_p = generate_sigma_build(mbeq_p, "p", first_row=False, optimize="True")
    funct_S_12 = generate_S12(mbeq_s, single_space, composite_space)
    funct_preconditioner = generate_preconditioner(
        mbeq, {}, {}, single_space, composite_space, first_row=False
    )
    funct_apply_S12 = generate_apply_S12(single_space, composite_space, first_row=False)

    # abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"Code generator: Writing to {abs_path}")

    with open(os.path.join(abs_path, "ip_eom_dsrg.py"), "w") as f:
        f.write(
            "import numpy as np\nimport scipy\nimport time\n\nfrom functools import reduce\n\nfrom niupy.eom_tools import *\n\n"
        )
        f.write(f"{func_template_c}\n\n")
        f.write(f"{funct_S_12}\n\n")
        f.write(f"{funct_preconditioner}\n\n")
        f.write(f"{funct}\n\n")
        f.write(f"{funct_s}\n\n")
        f.write(f"{funct_p}\n\n")
        f.write(f"{funct_apply_S12}\n\n")
        f.write(f"build_first_row = NotImplemented\n")
        f.write(f"build_transition_dipole = NotImplemented\n")


if __name__ == "__main__":
    generator(abs_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
