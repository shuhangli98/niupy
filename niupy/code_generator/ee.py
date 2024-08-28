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
w.add_space('C', 'fermion', 'occupied', list('KLMN'))
w.add_space('V', 'fermion', 'unoccupied', list('EFGH'))
w.add_space('A', 'fermion', 'general', list('OABRSTUVWXYZ'))
wt = w.WickTheorem()

# Define operators
s = []
for i in itertools.product(['v+', 'a+'], ['a', 'c']):
    s.append(' '.join(i))
for i in itertools.product(['V+', 'A+'], ['A', 'C']):
    s.append(' '.join(i))
for i in itertools.product(['v+', 'a+'], ['v+', 'a+'], ['a', 'c'], ['a', 'c']):
    s.append(' '.join(i))
for i in itertools.product(['V+', 'A+'], ['V+', 'A+'], ['A', 'C'], ['A', 'C']):
    s.append(' '.join(i))
for i in itertools.product(['v+', 'a+'], ['V+', 'A+'], ['a', 'c'], ['A', 'C']):
    s.append(' '.join(i))


T_adj = w.op("bra", s, unique=True).adjoint()
T = w.op("c", s, unique=True)

# Define subspaces
S_half_0 = ['cv', 'CV', 'ccvv', 'CCVV', 'cCvV']
S_half_1 = ['ccva', 'CCVA', 'cCvA', 'cCaV']
S_half_minus_1 = ['cavv', 'CAVV', 'cAvV', 'aCvV']
S_half_2 = ['ccaa', 'CCAA', 'cCaA']
S_half_minus_2 = ['aavv', 'AAVV', 'aAvV']

S_half_0_com = ['cava', 'CAVA', 'cAvA', 'cAaV', 'aCvA', 'aCaV']
S_half_1_com = ['ca', 'CA', 'caaa', 'CAAA', 'cAaA', 'aCaA']
S_half_minus_1_com = ['av', 'AV', 'aava', 'AAVA', 'aAvA', 'aAaV']

S_half_act = ['aa', 'AA', 'aaaa', 'AAAA', 'aAaA']

block_list = S_half_0 + S_half_1 + S_half_minus_1 + S_half_2 + S_half_minus_2 + \
    S_half_0_com + S_half_1_com + S_half_minus_1_com + S_half_act
single_space = S_half_0 + S_half_1 + S_half_minus_1 + S_half_2 + S_half_minus_2
composite_space = [S_half_0_com, S_half_1_com, S_half_minus_1_com, S_half_act]

# Define Hbar
Hops = []
for i in itertools.product(['v+', 'a+', 'c+'], ['v', 'a', 'c']):
    Hops.append(' '.join(i))
for i in itertools.product(['V+', 'A+', 'C+'], ['V', 'A', 'C']):
    Hops.append(' '.join(i))
for i in itertools.product(['v+', 'a+', 'c+'], ['v+', 'a+', 'c+'], ['v', 'a', 'c'], ['v', 'a', 'c']):
    Hops.append(' '.join(i))
for i in itertools.product(['V+', 'A+', 'C+'], ['V+', 'A+', 'C+'], ['V', 'A', 'C'], ['V', 'A', 'C']):
    Hops.append(' '.join(i))
for i in itertools.product(['v+', 'a+', 'c+'], ['V+', 'A+', 'C+'], ['v', 'a', 'c'], ['V', 'A', 'C']):
    Hops.append(' '.join(i))
Hbar_op = w.op("Hbar", Hops, unique=True)

# Template C
func_template_c = generate_template_c(block_list)

# Hbar
THT = T_adj @ Hbar_op @ T
expr = wt.contract(THT, 0, 0, inter_general=True)
mbeq = expr.to_manybody_equation('sigma')

# First row/column
HT = Hbar_op @ T
expr_first = wt.contract(HT, 0, 0, inter_general=True)
mbeq_first = expr_first.to_manybody_equation('sigma')

# S
TT = T_adj @ T
expr_s = wt.contract(TT, 0, 0, inter_general=True)
mbeq_s = expr_s.to_manybody_equation('sigma')

# Generate wicked contraction
funct = generate_sigma_build(mbeq, 'Hbar')  # HC
funct_s = generate_sigma_build(mbeq_s, 's')  # SC
funct_first = generate_first_row(mbeq_first)  # First row/column
funct_S_12 = generate_S_12(mbeq_s, single_space, composite_space, tol=1e-4)
funct_preconditioner_exact = generate_preconditioner(mbeq, single_space, composite_space, diagonal_type='exact')
funct_preconditioner_block = generate_preconditioner(mbeq, single_space, composite_space, diagonal_type='block')

script_dir = os.path.dirname(__file__)
# rel_path = "../wicked_contraction/ee_wicked.py"
rel_path = "../ee_eom_dsrg.py"
abs_file_path = os.path.join(script_dir, rel_path)

with open(abs_file_path, 'w') as f:
    f.write('import numpy as np\nimport time\n\nfrom niupy.eom_tools import *\n\n')
    f.write(f'{func_template_c}\n\n')
    f.write(f'{funct_S_12}\n\n')
    f.write(f'{funct_preconditioner_exact}\n\n')
    f.write(f'{funct_preconditioner_block}\n\n')
    f.write(f'{funct}\n\n')
    f.write(f'{funct_s}\n\n')
    f.write(f'{funct_first}\n\n')
