import forte, forte.utils
from niupy.ic_mrci_utils import *
from niupy.eom_tools import *
import itertools
import wicked as w

class ICMRCI:
    def __init__(self, nelec, nmos, scalar, h1, h2):
        self.nelec = nelec
        self.core = nmos['c']
        self.actv = nmos['a']
        self.virt = nmos['v']
        self.ncore = self.core.stop
        self.nactv = self.actv.stop-self.actv.start
        self.nvirt = self.virt.stop-self.virt.start
        self.nmos = self.virt.stop
        self.nelecas = self.nelec - self.ncore*2
        self.ham = make_hamiltonian(h1, h2, self.nmos, scalar)

        if 'i' in nmos:
            self.ninnr = nmos['i'].stop
            self.ndocc = self.ncore - self.ninnr
            self.innr = nmos['i']
            self.docc = slice(self.ninnr, self.ncore)
        else:
            self.ninnr = 0
            self.ndocc = self.ncore
            self.innr = slice(0,0)
            self.docc = slice(0,self.ncore)

        self.ops = None
    
    def do_casci(self, ncas=None, nelecas=None):
        if ncas is None:
            ncas = self.nactv
        if nelecas is None:
            nelecas = self.nelecas
        ncore = (self.nelec - nelecas) // 2
        self.cas_dets = make_hilbert_space(ncas, nelecas//2, nelecas//2, ncore=ncore)
        cas_ham = make_hamiltonian_matrix(self.cas_dets, self.ham)
        e, c = np.linalg.eigh(cas_ham)
        print('CASCI energy:', e[:10])
        return e, c
    
    def gen_ops_from_list(self, ops):
        self.ops = []
        ind = {'i':[str(_) for _ in list(range(self.ninnr))],
               'c':[str(_) for _ in list(range(self.ninnr, self.ninnr+self.ndocc))],
               'a':[str(_) for _ in list(range(self.ninnr+self.ndocc, self.ninnr+self.ndocc+self.nactv))],
               'v':[str(_) for _ in list(range(self.ninnr+self.ndocc+self.nactv, self.nmos))]}
        
        for op in ops:
            creann = op.split(' ')
            spaces = [ind[_[0]] for _ in creann]
            affix = [_[1:] for _ in creann]
            for i in itertools.product(*spaces):
                try:
                    t = forte.SparseOperator()
                    opstr = ' '.join([''.join(_) for _ in list(zip(i,affix))])
                    t.add(opstr)
                    self.ops.append(t)
                except:
                    pass
    
    def gen_ic_basis(self):
        assert isinstance(self.ops, list), 'Generate operators first'
        self.ic_basis = []
        for i in self.ops:
            self.ic_basis.append(forte.apply_op(i, self.psi))
    
    def gen_ic_basis_comm(self):
        assert isinstance(self.ops, list), 'Generate operators first'
        self.ic_basis_comm = []
        for i in self.ops:
            self.ic_basis_comm.append(forte.apply_op(self.ham.commutator(i), self.psi))

    def make_eigval_problem_ic(self):
        nbasis = len(self.ic_basis)
        H = np.zeros((nbasis,nbasis), dtype=np.complex128)
        S = np.zeros((nbasis,nbasis), dtype=np.complex128)
        for i in range(nbasis):
            for j in range(i+1):
                H[i,j] = forte.overlap(self.ic_basis[i],forte.apply_op(self.ham,self.ic_basis[j]))
                H[j,i] = H[i,j]
                S[i,j] = forte.overlap(self.ic_basis[i],self.ic_basis[j])
                S[j,i] = S[i,j]
                # make a progress bar
                if j == i:
                    print(f'\r{i+1}/{nbasis}', end='')
        return H, S
    
    def make_eigval_problem_ic_comm(self):
        nbasis = len(self.ic_basis)
        H = np.zeros((nbasis,nbasis), dtype=np.complex128)
        S = np.zeros((nbasis,nbasis), dtype=np.complex128)
        for i in range(nbasis):
            for j in range(nbasis):
                H[i,j] = forte.overlap(self.ic_basis[i], self.ic_basis_comm[j])
                S[i,j] = forte.overlap(self.ic_basis[i], self.ic_basis[j])
                # make a progress bar
                if j == i:
                    print(f'\r{i+1}/{nbasis}', end='')
        return H, S

if __name__ == "__main__":
    x = 3.0
    mol = pyscf.gto.M(atom=f"""
    Be 0.0   0.0             0.0
    H  {x}   {2.54-0.46*x}   0.0
    H  {x}  -{2.54-0.46*x}   0.0
    """, basis='sto-6g', symmetry='c2v', unit='bohr')
    # mol = pyscf.gto.M(atom=f"""
    # O  0.0000  0.0000  0.1173
    # H  0.0000  0.7572 -0.4692
    # H  0.0000 -0.7572 -0.4692
    # """, basis='6-31g', symmetry='c2v', unit='bohr')

    mf = pyscf.scf.RHF(mol).run()
    mc = pyscf.mcscf.CASSCF(mf, 2, 2)
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-8
    mc.kernel()
    fci = pyscf.fci.FCI(mf)
    fci.nroots = 10
    e_gs = fci.kernel()[0]
    print('GS energy:', e_gs)

    mol_cat = pyscf.gto.M(atom=f"""
    Be 0.0   0.0             0.0
    H  {x}   {2.54-0.46*x}   0.0
    H  {x}  -{2.54-0.46*x}   0.0
    """, basis='sto-6g', symmetry='c2v', unit='bohr',charge=1,spin=1)
    # mol_cat = pyscf.gto.M(atom=f"""
    # O  0.0000  0.0000  0.1173
    # H  0.0000  0.7572 -0.4692
    # H  0.0000 -0.7572 -0.4692
    # """, basis='6-31g', symmetry='c2v', unit='bohr', charge=1, spin=1)

    mf_cat = pyscf.scf.RHF(mol_cat).run()
    fci_cat = pyscf.fci.FCI(mf_cat)
    fci_cat.nroots = 10
    e_cat = fci_cat.kernel()[0]
    print('Cation energy:', e_cat)
    print('IP:', e_cat-e_gs[0])
    nao = mol.nao
    
    nactv = 2
    ncore = (mol.nelectron - nactv) // 2
    nvirt = nao - ncore - nactv
    print(f'Number of active orbitals: {nactv}')
    print(f'Number of core orbitals: {ncore}')
    print(f'Number of virtual orbitals: {nvirt}')
    nmo = {'c':slice(0,ncore), 'a':slice(ncore,ncore+nactv), 'v':slice(ncore+nactv,nao)}

    scalar, h1, h2 = get_sa_ints('temp/beh-3.0/save_Hbar_degno_bare.npz', nmo, nao)
    h2_asym = h2 - h2.swapaxes(2,3)

    ic_cas = ICMRCI(7, nmo, scalar, (h1,h1), (h2_asym,h2,h2_asym))
    e,c = ic_cas.do_casci()
    psi = form_psi(ic_cas.cas_dets, c[:,1])
    e_bare = forte.overlap(psi, forte.apply_op(ic_cas.ham, psi)).real
    print('Bare reference energy: ', e_bare)

    scalar, h1, h2 = get_sa_ints('temp/beh-3.0/save_Hbar_degno.npz', nmo, nao)
    h2_asym = h2 - h2.swapaxes(2,3)
    
    ic = ICMRCI(7, nmo, scalar, (h1,h1), (h2_asym,h2,h2_asym))
    ic.psi = psi
    e_dsrg = forte.overlap(psi, forte.apply_op(ic.ham, psi)).real
    print('DSRG reference energy: ', e_dsrg)
    w.reset_space()
    w.add_space('c', 'fermion', 'occupied', list('ijklmn'))
    w.add_space('v', 'fermion', 'unoccupied', list('abcdef'))
    w.add_space('a', 'fermion', 'general', list('stuvwxyz'))
    w.add_space('C', 'fermion', 'occupied', list('IJKLMN'))
    w.add_space('V', 'fermion', 'unoccupied', list('ABCDEF'))
    w.add_space('A', 'fermion', 'general', list('STUVWXYZ'))

    ops = w.gen_op('bra', (0,1), 'avAV', 'caCA', only_terms=True) + w.gen_op('bra', (1,2), 'avAV', 'caCA', only_terms=True)
    ops = [_.strip() for _ in ops]
    ops = filter_ops_by_ms(ops, 1)

    ip_ops = wicked_ops_to_forte_ops(ops)
    ic.gen_ops_from_list(ip_ops)
    print(f'Number of operators: {len(ic.ops)}')
    ic.gen_ic_basis()
    # ic.gen_ic_basis_comm()
    # H,S = ic.make_eigval_problem_ic_comm()
    # e,c = eigh(H, S, tol=1e-5)
    # print(e)

    H,S = ic.make_eigval_problem_ic()
    e,c = eigh(H, S, tol=1e-5)
    print(e-e_dsrg)
    print(e-e_gs[0])