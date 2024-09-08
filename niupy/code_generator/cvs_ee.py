import wicked as w
import itertools
import os

from niupy.eom_tools import *


w.reset_space()
# alpha
w.add_space('i', 'fermion', 'occupied', list('cdij'))
w.add_space('c', 'fermion', 'occupied', list('klmn'))
w.add_space('v', 'fermion', 'unoccupied', list('efgh'))
w.add_space('a', 'fermion', 'general', list('oabrstuvwxyz'))
# #Beta
w.add_space('I', 'fermion', 'occupied', list('CDIJ'))
w.add_space('C', 'fermion', 'occupied', list('KLMN'))
w.add_space('V', 'fermion', 'unoccupied', list('EFGH'))
w.add_space('A', 'fermion', 'general', list('OABRSTUVWXYZ'))
wt = w.WickTheorem()

# Define operators
s = []

for i in itertools.product(['v+', 'a+'], ['i']):
    s.append(' '.join(i))
for i in itertools.product(['V+', 'A+'], ['I']):
    s.append(' '.join(i))
for i in itertools.product(['v+', 'a+'], ['v+', 'a+'], ['a', 'c'], ['i']):
    s.append(' '.join(i))
for i in itertools.product(['V+', 'A+'], ['V+', 'A+'], ['A', 'C'], ['I']):
    s.append(' '.join(i))
for i in itertools.product(['v+', 'a+'], ['V+', 'A+'], ['a', 'c'], ['I']):
    s.append(' '.join(i))
for i in itertools.product(['v+', 'a+'], ['V+', 'A+'], ['A', 'C'], ['i']):
    s.append(' '.join(i))

T_adj = w.op("bra", s, unique=True).adjoint()
T = w.op("c", s, unique=True)

# Define subspaces
S_half_0 = ['iv', 'IV', 'iAaV', 'aIvA', 'icvv', 'iCvV', 'cIvV', 'ICVV']
S_half_1 = ['icva', 'iCvA', 'iCaV', 'cIvA', 'cIaV', 'ICVA']
S_half_minus_1 = ['iavv', 'iAvV', 'aIvV', 'IAVV']
S_half_2 = ['icaa', 'iCaA', 'cIaA', 'ICAA']

S_half_0_com_iv = ['iava', 'iAvA']
S_half_0_com_IV = ['aIaV', 'IAVA']
S_half_1_com_i = ['ia', 'iaaa', 'iAaA']
S_half_1_com_I = ['IA', 'aIaA', 'IAAA']


block_list = S_half_0 + S_half_1 + S_half_minus_1 + S_half_2 + \
    S_half_0_com_iv + S_half_0_com_IV + S_half_1_com_i + S_half_1_com_I
single_space = S_half_0 + S_half_1 + S_half_minus_1 + S_half_2
composite_space = [S_half_0_com_iv, S_half_0_com_IV, S_half_1_com_i, S_half_1_com_I]

# Define Hbar
Hops = []
for i in itertools.product(['v+', 'a+', 'c+', 'i+'], ['v', 'a', 'c', 'i']):
    Hops.append(' '.join(i))
for i in itertools.product(['V+', 'A+', 'C+', 'I+'], ['V', 'A', 'C', 'I']):
    Hops.append(' '.join(i))
for i in itertools.product(['v+', 'a+', 'c+', 'i+'], ['v+', 'a+', 'c+', 'i+'], ['v', 'a', 'c', 'i'], ['v', 'a', 'c', 'i']):
    Hops.append(' '.join(i))
for i in itertools.product(['V+', 'A+', 'C+', 'I+'], ['V+', 'A+', 'C+', 'I+'], ['V', 'A', 'C', 'I'], ['V', 'A', 'C', 'I']):
    Hops.append(' '.join(i))
for i in itertools.product(['v+', 'a+', 'c+', 'i+'], ['V+', 'A+', 'C+', 'I+'], ['v', 'a', 'c', 'i'], ['V', 'A', 'C', 'I']):
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
funct_S_12 = generate_S_12(mbeq_s, single_space, composite_space)
funct_preconditioner_exact = generate_preconditioner(mbeq, single_space, composite_space, diagonal_type='exact')
funct_preconditioner_block = generate_preconditioner(mbeq, single_space, composite_space, diagonal_type='block')

script_dir = os.path.dirname(__file__)
rel_path = "../cvs_ee_eom_dsrg.py"
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
