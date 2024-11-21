import wicked as w
import itertools
import os
from niupy.eom_tools import *

def generator():
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
    s = w.gen_op('bra', (0,1), 'avAV', 'caCA', only_terms=True) + w.gen_op('bra', (1,2), 'avAV', 'caCA', only_terms=True)
    s = [_.strip() for _ in s]
    s = filter_ops_by_ms(s, 1)

    T_adj = w.op("bra", s, unique=True).adjoint()
    T = w.op("c", s, unique=True)

    # Define subspaces. Single first!
    S_half_0 = ["c","ccv","caa","cAA","cCV","aCA"]
    S_half_1 = ["cca","cCA"]
    S_half_minus_1 = ["cav", "cAV", "aCV", "aAV", "a", "aaa", "aAA"]
    S_half_minus_2 = ["aav"]

    block_list = (
        S_half_0 + S_half_1 + S_half_minus_1 + S_half_minus_2
    )
    single_space = S_half_0 + S_half_1 + S_half_minus_1 + S_half_minus_2
    composite_space = []

    # Define Hbar
    Hbar_op = w.gen_op_ms0('Hbar', 1, 'cav', 'cav') +  w.gen_op_ms0('Hbar', 2, 'cav', 'cav')

    # Template C
    index_dict = {
        "c": "ncore",
        "a": "nact",
        "v": "nvir",
        "C": "ncore",
        "A": "nact",
        "V": "nvir",
    }

    function_args = 'nlow, ncore, nact, nvir'
    func_template_c = generate_template_c(block_list, index_dict, function_args)

    THT = T_adj @ Hbar_op @ T
    expr = wt.contract(THT, 0, 0, inter_general=True)
    mbeq = expr.to_manybody_equation("sigma")

    # S
    TT = T_adj @ T
    expr_s = wt.contract(TT, 0, 0, inter_general=True)
    mbeq_s = expr_s.to_manybody_equation("sigma")

    # Generate wicked contraction
    funct = generate_sigma_build(mbeq, "Hbar", optimize='True')  # HC
    funct_s = generate_sigma_build(mbeq_s, "s", optimize='True')  # SC
    funct_S_12 = generate_S_12(mbeq_s, single_space, composite_space, method='ip')
    funct_preconditioner_exact = generate_preconditioner(
        mbeq, single_space, composite_space, diagonal_type="exact", method='ip'
    )

    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"Code generator: Writing to {abs_path}")

    with open(os.path.join(abs_path, "ip_eom_dsrg.py"), "w") as f:
        f.write(
            "import numpy as np\nimport scipy\nimport time\n\nfrom functools import reduce\n\nfrom niupy.eom_tools import *\n\n"
        )
        f.write(f"{func_template_c}\n\n")
        f.write(f"{funct_S_12}\n\n")
        f.write(f"{funct_preconditioner_exact}\n\n")
        f.write(f"{funct}\n\n")
        f.write(f"{funct_s}\n\n")
        f.write(f"build_first_row = NotImplemented\n")
        f.write(f"build_transition_dipole = NotImplemented\n")
        f.write(f"compute_preconditioner_block = NotImplemented\n")
        f.write(f"compute_preconditioner_only_H = NotImplemented\n")

if __name__ == "__main__":
    generator()