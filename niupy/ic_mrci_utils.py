import forte, forte.utils
import numpy as np
import scipy.linalg as la
import pyscf
import wicked as w
import niupy.eom_tools
import copy

def form_psi(cas_dets, civec):
    psi = forte.SparseState(dict(zip(cas_dets, civec)))
    return psi

def de_normal_order(scalar, oei, tei, gamma1, lambda2, nmo):
    nhole = nmo['a'].stop
    hole = slice(0,nhole)
    actv = nmo['a']
    h1a, h1b = oei
    h2aa, h2ab, h2bb = tei
    g1a_h = np.eye(nhole)
    g1b_h = np.eye(nhole)
    g1a_h[actv,actv] = gamma1['aa']
    g1b_h[actv,actv] = gamma1['AA']

    scalar1 = .0
    scalar1 -= np.einsum('ji,ij->', h1a[hole,hole], g1a_h)
    scalar1 -= np.einsum('JI,IJ->', h1b[hole,hole], g1b_h) 
    scalar2 = .0
    scalar2 += 0.5 * np.einsum('ij,jlik,kl->', g1a_h, h2aa[hole,hole,hole,hole], g1a_h)
    scalar2 += 0.5 * np.einsum('IJ,JLIK,KL->', g1b_h, h2bb[hole,hole,hole,hole], g1b_h)
    scalar2 += np.einsum('ij,jLiK,KL->', g1a_h, h2ab[hole,hole,hole,hole], g1b_h)

    scalar2 -= 0.25 * np.einsum('xyuv,uvxy->', h2aa[actv,actv,actv,actv], lambda2['aaaa'])
    scalar2 -= 0.25 * np.einsum('XYUV,UVXY->', h2bb[actv,actv,actv,actv], lambda2['AAAA'])
    scalar2 -= np.einsum('xYuV,uVxY->', h2ab[actv,actv,actv,actv], lambda2['aAaA'])
    scalar += scalar1 + scalar2
    h1a -= np.einsum('piqj,ji->pq', h2aa[:,hole,:,hole], g1a_h)
    h1a -= np.einsum('pIqJ,JI->pq', h2ab[:,hole,:,hole], g1b_h)
    h1b -= np.einsum('iPjQ,ji->PQ', h2ab[hole,:,hole,:], g1a_h)
    h1b -= np.einsum('PIQJ,JI->PQ', h2bb[:,hole,:,hole], g1b_h)

    return scalar, (h1a, h1b), tei

def re_normal_order(scalar, oei, tei, gamma1, lambda2, nmo):
    nhole = nmo['a'].stop
    hole = slice(0,nhole)
    actv = nmo['a']
    h1a, h1b = oei
    h2aa, h2ab, h2bb = tei
    g1a_h = np.eye(nhole)
    g1b_h = np.eye(nhole)
    g1a_h[actv,actv] = gamma1['aa']
    g1b_h[actv,actv] = gamma1['AA']

    h1a += np.einsum('piqj,ji->pq', h2aa[:,hole,:,hole], g1a_h)
    h1a += np.einsum('pIqJ,JI->pq', h2ab[:,hole,:,hole], g1b_h)
    h1b += np.einsum('iPjQ,ji->PQ', h2ab[hole,:,hole,:], g1a_h)
    h1b += np.einsum('PIQJ,JI->PQ', h2bb[:,hole,:,hole], g1b_h)

    scalar1 = .0
    scalar1 += np.einsum('ji,ij->', h1a[hole,hole], g1a_h)
    scalar1 += np.einsum('JI,IJ->', h1b[hole,hole], g1b_h) 

    scalar2 = .0
    scalar2 -= 0.5 * np.einsum('ij,jlik,kl->', g1a_h, h2aa[hole,hole,hole,hole], g1a_h)
    scalar2 -= 0.5 * np.einsum('IJ,JLIK,KL->', g1b_h, h2bb[hole,hole,hole,hole], g1b_h)
    scalar2 -= np.einsum('ij,jLiK,KL->', g1a_h, h2ab[hole,hole,hole,hole], g1b_h)

    scalar2 += 0.25 * np.einsum('xyuv,uvxy->', h2aa[actv,actv,actv,actv], lambda2['aaaa'])
    scalar2 += 0.25 * np.einsum('XYUV,UVXY->', h2bb[actv,actv,actv,actv], lambda2['AAAA'])
    scalar2 += np.einsum('xYuV,uVxY->', h2ab[actv,actv,actv,actv], lambda2['aAaA'])
    scalar += scalar1 + scalar2

    return scalar, (h1a, h1b), tei

def get_si_ints(fname, mos, norbs):
    mos_loc = copy.deepcopy(mos)
    if 'i' in mos:
        mos_loc['c'] = slice(0,mos['c'].stop)
        mos_loc['C'] = slice(0,mos['C'].stop)
    h1a = np.zeros((norbs,)*2)
    h1b = np.zeros((norbs,)*2)
    h2aa = np.zeros((norbs,)*4)
    h2ab = np.zeros((norbs,)*4)
    h2bb = np.zeros((norbs,)*4)
    ints = np.load(fname)
    scalar = ints['Hbar0']

    for k,v in ints.items():
        if len(k)==2:
            if k.islower():
                h1a[mos_loc[k[0]],mos_loc[k[1]]] = v
            else:
                h1b[mos_loc[k[0]],mos_loc[k[1]]] = v
        elif len(k)==4:
            if k.islower():
                h2aa[mos_loc[k[0]],mos_loc[k[1]],mos_loc[k[2]],mos_loc[k[3]]] = v
            elif k.isupper():
                h2bb[mos_loc[k[0]],mos_loc[k[1]],mos_loc[k[2]],mos_loc[k[3]]] = v
            else:
                h2ab[mos_loc[k[0]],mos_loc[k[1]],mos_loc[k[2]],mos_loc[k[3]]] = v
    return scalar, (h1a, h1b), (h2aa, h2ab, h2bb) 

def make_hamiltonian(oei, tei, nmo, scalar):
    oei_a, oei_b = oei
    tei_aa, tei_ab, tei_bb = tei
    ham = forte.SparseOperatorList()
    ham.add(f"[]", scalar)
    for p in range(nmo):
        for q in range(nmo):
            ham.add(f"[{p}a+ {q}a-]",oei_a[p,q])
            ham.add(f"[{p}b+ {q}b-]",oei_b[p,q])

    for p in range(nmo):
        for q in range(p + 1,nmo):
            for r in range(nmo):
                for s in range(r + 1,nmo):
                    ham.add(f"[{p}a+ {q}a+ {s}a- {r}a-]",tei_aa[p,q,r,s])
                    ham.add(f"[{p}b+ {q}b+ {s}b- {r}b-]",tei_bb[p,q,r,s])

    for p in range(nmo):
        for q in range(nmo):
            for r in range(nmo):
                for s in range(nmo):
                    ham.add(f"[{p}a+ {q}b+ {s}b- {r}a-]",tei_ab[p,q,r,s])

    ham_op = ham.to_operator()
    return ham_op

def make_hilbert_space(nmo,na,nb,ncore=0):
    import itertools
    dets = []
    astr = list(itertools.combinations(range(nmo), na))
    bstr = list(itertools.combinations(range(nmo), nb))
    for a in astr:
        for b in bstr:
            d = forte.Determinant()
            for i in range(ncore):
                d.set_alfa_bit(i,1)
                d.set_beta_bit(i,1)
            for i in a:
                d.set_alfa_bit(i+ncore,1)
            for i in b:
                d.set_beta_bit(i+ncore,1)                
            dets.append(d)
    print(f'Orbitals: {nmo}')
    print(f'Electrons: {na}a/{nb}b')
    print(f'Size of Hilbert space: {len(dets)}')
    return dets

def make_hamiltonian_matrix(dets, ham):
    N = len(dets)
    H = np.zeros((N,N),dtype=np.complex128)
    for i,I in enumerate(dets):
        refI = forte.SparseState({I:1.})
        HrefI = forte.apply_op(ham,refI)
        for j,J in enumerate(dets):
            refJ = forte.SparseState({J:1.})
            H[i,j] = forte.overlap(refJ,HrefI)
    return H

def eigh(H, S, tol=1e-6):
    seval, sevec = la.eigh(S)
    if np.any(seval < -tol):
        raise ValueError('Overlap matrix is not positive semidefinite')
    s = seval[np.abs(seval) > tol]
    sevec = sevec[:,np.abs(seval) > tol]
    X = sevec / np.sqrt(s)
    H = X.T @ H @ X
    Heval, Hevec = la.eigh(H)
    return Heval, X @ Hevec

def wicked_ops_to_forte_ops(ops):
    def process_op_str(opstr):
        res = opstr[0].lower()
        res += 'a' if opstr[0].islower() else 'b'
        res += '+' if '+' in opstr else '-'
        return res
    forte_ops = []
    for op in ops:
        forte_ops.append(' '.join([process_op_str(_) for _ in op.split(' ')]))

    return forte_ops

def get_pyscf_ints(mol, mf):
    nao = mf.mo_coeff.shape[0]
    h1 = np.einsum('ui,uv,vj->ij', mf.mo_coeff, mf.get_hcore(), mf.mo_coeff)
    h2 = pyscf.ao2mo.full(mol, mf.mo_coeff, compact=False).reshape(nao,nao,nao,nao)
    h2 = h2.swapaxes(1,2)
    return h1, h2

def generate_operator_manifold(manifold='ip', truncation=2, cvs=False):
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

    cre = 'avAV'
    ann = 'icaICA' if cvs else 'caCA'
    if manifold == 'ip':
        nops = [(i, i+1) for i in range(truncation)]
    elif manifold == 'ea':
        nops = [(i+1, i) for i in range(truncation)]
    elif manifold == 'ee':
        nops = [i+1 for i in range(truncation)]
    
    ops = []
    for i in nops:
        ops += w.gen_op('dummy', i, cre, ann, only_terms=True)
    ops = [_.strip() for _ in ops]
    ops = niupy.eom_tools.filter_ops_by_ms(ops, 1 if manifold in ['ip','ea'] else 0)
    if cvs:
        ops = [_ for _ in ops if ("I" in _ or "i" in _)]

    ip_ops = wicked_ops_to_forte_ops(ops)
    return ip_ops
