import wicked as w
import itertools
import os

from niupy.eom_tools import *


w.reset_space()
# alpha
w.add_space('c', 'fermion', 'occupied', list('klmn'))
w.add_space('v', 'fermion', 'unoccupied', list('efgh'))
w.add_space('a', 'fermion', 'general', list('oabrstuvwxyz'))
# Beta
# w.add_space('C', 'fermion', 'occupied', list('KLMN'))
# w.add_space('V', 'fermion', 'unoccupied', list('EFGH'))
# w.add_space('A', 'fermion', 'general', list('OABRSTUVWXYZ'))
wt = w.WickTheorem()

Hops = []
for i in itertools.product(['v+', 'a+', 'c+'], ['v', 'a', 'c']):
    Hops.append(' '.join(i))
# for i in itertools.product(['V+', 'A+', 'C+'], ['V', 'A', 'C']):
#     Hops.append(' '.join(i))
for i in itertools.product(['v+', 'a+', 'c+'], ['v+', 'a+', 'c+'], ['v', 'a', 'c'], ['v', 'a', 'c']):
    Hops.append(' '.join(i))
# for i in itertools.product(['V+', 'A+', 'C+'], ['V+', 'A+', 'C+'], ['V', 'A', 'C'], ['V', 'A', 'C']):
#     Hops.append(' '.join(i))
# for i in itertools.product(['v+', 'a+', 'c+'], ['V+', 'A+', 'C+'], ['v', 'a', 'c'], ['V', 'A', 'C']):
#     Hops.append(' '.join(i))
Hbar_op = w.op("Hbar", Hops, unique=True)

s = []

for i in itertools.product(['v+', 'a+'], ['a', 'c']):
    s.append(' '.join(i))
# for i in itertools.product(['V+', 'A+'], ['A', 'C']):
#     s.append(' '.join(i))
for i in itertools.product(['v+', 'a+'], ['v+', 'a+'], ['a', 'c'], ['a', 'c']):
    s.append(' '.join(i))
# for i in itertools.product(['V+', 'A+'], ['V+', 'A+'], ['A', 'C'], ['A', 'C']):
#     s.append(' '.join(i))
# for i in itertools.product(['v+', 'a+'], ['V+', 'A+'], ['a', 'c'], ['A', 'C']):
#     s.append(' '.join(i))

for op in s:
    T_adj = w.op("bra", [op], unique=True).adjoint()
    T = w.op("c", [op], unique=True)

    # Hbar
    THT = T_adj @ Hbar_op @ T
    expr = wt.contract(THT, 0, 0)
    mbeq = expr.to_manybody_equation('sigma')

    # Generate wicked contraction
    funct = generate_sigma_build(mbeq, 'foo')  # HC

    with open('ee_contractions_analysis.py', 'a') as f:
        f.write(f'{funct}\n\n')
