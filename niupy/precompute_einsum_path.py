import numpy as np
import wicked as w

def precompute_file(fname, norbs, memory_limit=None):
    osi = w.osi().to_dict()
    index_dict = {'p':'p'}
    for key in osi.keys():
        for index in osi[key]:
            index_dict[index] = key

    sizes_dict = {k: norbs[index_dict[k]] for k in index_dict.keys()}

    with open(fname, 'r') as f:
        lines = f.readlines()

    with open(fname.replace('.py', '_path.py'), 'w') as f:
        for line in lines:
            f.write(w.precompute_path(line, sizes_dict, memory_limit))


if __name__ == "__main__":
    w.reset_space()
    w.add_space('c', 'fermion', 'occupied', list('klmn'))
    w.add_space('v', 'fermion', 'unoccupied', list('efgh'))
    w.add_space('a', 'fermion', 'general', list('oabrstuvwxyz'))
    w.add_space('C', 'fermion', 'occupied', list('KLMN'))
    w.add_space('V', 'fermion', 'unoccupied', list('EFGH'))
    w.add_space('A', 'fermion', 'general', list('OABRSTUVWXYZ'))

    norbs = {'c':10, 'a':6, 'v':20, 'C':10, 'A':6, 'V':20, 'p':5}
    precompute_file('ee_eom_dsrg.py', norbs)
