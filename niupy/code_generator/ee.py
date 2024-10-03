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

# This is a test for single internal.
# for i in s:
#     if i in ['a+ a+ a a', 'A+ A+ A A', 'a+ A+ a A']:
#         s.pop(s.index(i))

T_adj = w.op("bra", s, unique=True).adjoint()
T = w.op("c", s, unique=True)

# Define subspaces
S_half_0 = ['cv', 'CV', 'ccvv', 'CCVV', 'cCvV', 'aCvA', 'cAaV']
S_half_1 = ['ccva', 'CCVA', 'cCvA', 'cCaV']
S_half_minus_1 = ['cavv', 'CAVV', 'cAvV', 'aCvV']
S_half_2 = ['ccaa', 'CCAA', 'cCaA']
S_half_minus_2 = ['aavv', 'AAVV', 'aAvV']

S_half_0_com_cv = ['cava', 'cAvA']
S_half_0_com_CV = ['CAVA', 'aCaV']
S_half_1_com_c = ['ca', 'caaa', 'cAaA']
S_half_1_com_C = ['CA', 'CAAA', 'aCaA']
S_half_minus_1_com_v = ['av', 'aava', 'aAvA']
S_half_minus_1_com_V = ['AV', 'AAVA', 'aAaV']
S_half_act = ['aa', 'AA', 'aaaa', 'AAAA', 'aAaA']

block_list = S_half_0 + S_half_1 + S_half_minus_1 + S_half_2 + S_half_minus_2 + S_half_0_com_cv + \
    S_half_0_com_CV + S_half_1_com_c + S_half_1_com_C + S_half_minus_1_com_v + S_half_minus_1_com_V + S_half_act
single_space = S_half_0 + S_half_1 + S_half_minus_1 + S_half_2 + S_half_minus_2
composite_space = [S_half_0_com_cv, S_half_0_com_CV, S_half_1_com_c,
                   S_half_1_com_C, S_half_minus_1_com_v, S_half_minus_1_com_V, S_half_act]

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
funct = generate_sigma_build(mbeq, 'Hbar', algo='ee')  # HC
funct_s = generate_sigma_build(mbeq_s, 's', algo='ee')  # SC
funct_first = generate_first_row(mbeq_first, algo='ee')  # First row/column
funct_dipole = generate_transition_dipole(mbeq_first, algo='ee')
funct_S_12 = generate_S_12(mbeq_s, single_space, composite_space, algo='ee')
funct_preconditioner_exact = generate_preconditioner(mbeq, single_space, composite_space, diagonal_type='exact', algo='ee')
funct_preconditioner_block = generate_preconditioner(mbeq, single_space, composite_space, diagonal_type='block', algo='ee')
funct_preconditioner_only_H = generate_preconditioner(mbeq, block_list, None, algo='ee')

script_dir = os.path.dirname(__file__)
rel_path = "../ee_eom_dsrg.py"
abs_file_path = os.path.join(script_dir, rel_path)

with open(abs_file_path, 'w') as f:
    f.write('import numpy as np\nimport scipy\nimport time\n\nfrom niupy.eom_tools import *\n\n')
    f.write(f'{func_template_c}\n\n')
    f.write(f'{funct_S_12}\n\n')
    f.write(f'{funct_preconditioner_exact}\n\n')
    f.write(f'{funct_preconditioner_block}\n\n')
    f.write(f'{funct_preconditioner_only_H}\n\n')
    f.write(f'{funct}\n\n')
    f.write(f'{funct_s}\n\n')
    f.write(f'{funct_first}\n\n')
    f.write(f'{funct_dipole}\n\n')
