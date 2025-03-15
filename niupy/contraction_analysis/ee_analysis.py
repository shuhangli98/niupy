import wicked as w
import itertools
import os

from niupy.eom_tools import *

w.reset_space()
w.add_space("c", "fermion", "occupied", list("klmnKLMN"))
w.add_space("v", "fermion", "unoccupied", list("efghEFGH"))
w.add_space("a", "fermion", "general", list("oabrstuvwxyzOABRSTUVWXYZ"))
wt = w.WickTheorem()

Hop = w.gen_op("H", 1, "cav", "cav") + w.gen_op("H", 2, "cav", "cav")

s = w.gen_op("bra", 1, "av", "ca", only_terms=True) + w.gen_op(
    "bra", 2, "av", "ca", only_terms=True
)
s = [_.strip() for _ in s]
all_act = ["a+ a", "a+ a+ a a"]
s = filter_ops_by_ms(s, 0)
s_no_active = [_ for _ in s if _ not in all_act]
print(s)
print(s_no_active)

# ic-MRUCC part
T = w.op("ket_t", s_no_active, unique=True)
T_adj = w.op("bra_t", s_no_active, unique=True).adjoint()
Hbar = w.bch_series(Hop, (T - T_adj), 1)
print(Hbar)

# EOM part
for op in s:
    T_adj = w.op("bra", [op], unique=True).adjoint()
    T = w.op("c", [op], unique=True)
    # THT_comm = w.commutator(T_adj, w.commutator(Hbar, T))
    # THT = T_adj @ Hbar @ T
    # expr = wt.contract(THT, 0, 0, inter_general=False)
    THT_comm = T_adj @ w.commutator(Hbar, T)
    expr = wt.contract(THT_comm, 0, 0, inter_general=False)
    mbeq = expr.to_manybody_equation("sigma")
    funct = generate_sigma_build(mbeq, "foo")  # HC
    with open("contractions.dat", "a") as f:
        f.write(f"{funct}\n\n")
