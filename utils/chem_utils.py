'''
beta version
'''
import math
from copy import copy
from tqdm import tqdm

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import calinski_harabasz_score

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import Descriptors

MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':5, 'O':2, 'P':5, 'S':6} #, 'Se':4, 'Si':4}
Bond_List = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


class AtomVocab:
    def __init__(self):
        # atom
        self.idx2atom = list(MAX_VALENCE.keys())
        self.atom2idx = { atom: i for i, atom in enumerate(self.idx2atom) }
        # bond
        self.idx2bond = copy(Bond_List)
        self.bond2idx = { bond: i for i, bond in enumerate(self.idx2bond) }
        
    def idx_to_atom(self, idx):
        return self.idx2atom[idx]
    
    def atom_to_idx(self, atom):
        return self.atom2idx[atom]

    def idx_to_bond(self, idx):
        return self.idx2bond[idx]
    
    def bond_to_idx(self, bond):
        return self.bond2idx[bond]
    
    def num_atom_type(self):
        return len(self.idx2atom)
    
    def num_bond_type(self):
        return len(self.idx2bond)

    

def smi2mol(smiles: str, kekulize=False, sanitize=True):
    '''turn smiles to molecule'''
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol


def mol2smi(mol):
    return Chem.MolToSmiles(mol)


def canonical_order(smiles: str):
    mol = smi2mol(smiles)
    smi = Chem.MolToSmiles(mol, canonical=True)
    return smi


def fingerprint(mol):
    return AllChem.GetMorganFingerprint(mol, 2)


def similarity(mol1, mol2):
    if isinstance(mol1, str):
        mol1 = smi2mol(mol1)
    if isinstance(mol2, str):
        mol2 = smi2mol(mol2)
    fps1 = fingerprint(mol1)
    fps2 = fingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fps1, fps2)


def fingerprint2numpy(fingerprint):
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint,arr)
    return arr


def pkd(kd):
    return -math.log(kd, 10)


def draw_molecules(*mols, canvas_size=(500, 500), useSVG=False, molsPerRow=3,
                   highlightAtomLists=None, highlightBondLists=None, legends=None,
                   save_path=None):
    rdkit_mols = []
    for mol in mols:
        if isinstance(mol, str): # smiles
            mol = smi2mol(mol)
        rdkit_mols.append(mol)
    image = Draw.MolsToGridImage(rdkit_mols, molsPerRow=molsPerRow, subImgSize=canvas_size, useSVG=useSVG,
                                 highlightAtomLists=highlightAtomLists, highlightBondLists=highlightBondLists,
                                 legends=legends)
    if save_path is not None:
        img_type = 'SVG' if useSVG else 'PNG'
        image.save(save_path, img_type)
    return image


def diversity(mols):
    '''
    \frac{2}{n(n-1)} \sum_{i < j} (1 - similarity(mol_i, mol_j))
    '''
    mols = [smi2mol(mol) if isinstance(mol, str) else mol for mol in mols]
    n = len(mols)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(similarity(mols[i], mols[j]))
    assert len(sims) == n * (n - 1) / 2
    diversity = 1 - sum(sims) / len(sims)
    return diversity


def calculate_1dqsar_repr(mol):
    # 从 SMILES 字符串创建分子对象
    if isinstance(mol, str):
        mol = smi2mol(mol)
    # 计算分子的分子量
    mol_weight = Descriptors.MolWt(mol)
    # 计算分子的 LogP 值
    log_p = Descriptors.MolLogP(mol)
    # 计算分子中的氢键供体数量
    num_h_donors = Descriptors.NumHDonors(mol)
    # 计算分子中的氢键受体数量
    num_h_acceptors = Descriptors.NumHAcceptors(mol)
    # 计算分子的表面积极性
    tpsa = Descriptors.TPSA(mol)
    # 计算分子中的可旋转键数量
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    # 计算分子中的芳香环数量
    num_aromatic_rings = Descriptors.NumAromaticRings(mol)
    # 计算分子中的脂环数量
    num_aliphatic_rings = Descriptors.NumAliphaticRings(mol)
    # 计算分子中的饱和环数量
    num_saturated_rings = Descriptors.NumSaturatedRings(mol)
    # 计算分子中的杂原子数量
    num_heteroatoms = Descriptors.NumHeteroatoms(mol)
    # 计算分子中的价电子数量
    num_valence_electrons = Descriptors.NumValenceElectrons(mol)
    # 计算分子中的自由基电子数量
    num_radical_electrons = Descriptors.NumRadicalElectrons(mol)
    # 计算分子的 QED 值
    qed = Descriptors.qed(mol)
    # 返回所有计算出的属性值
    return [mol_weight, log_p, num_h_donors, num_h_acceptors, tpsa, num_rotatable_bonds, num_aromatic_rings,
            num_aliphatic_rings, num_saturated_rings, num_heteroatoms, num_valence_electrons, num_radical_electrons,qed]


def find_cliques(mol):
    '''
    Cluster:
      1. a rotatable bond which is not in a ring
      2. rings in the smallest set of smallest rings (SSSR)
    '''
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1: #special case
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( (a1,a2) )

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for i in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls


def clustering(smiles_list, K=None, sim_mat=None, random_state=None):

    if sim_mat is None:
        mols = [smi2mol(smi) for smi in smiles_list]
        fps = [fingerprint(mol) for mol in mols]

        sim_mat = [[0 for _ in range(len(fps))] for _ in range(len(fps))]

        # 1D similarity
        qsars = np.array([calculate_1dqsar_repr(mol) for mol in mols])  # [N, n_feat]
        # normalize
        mean, std = np.mean(qsars, axis=0), np.std(qsars, axis=0) # [n_feat]
        qsars = (qsars - mean[None, :]) / (1e-16 + std[None, :]) # [N, n_feat]
        sim_qsar = cosine_similarity(qsars, qsars)

        # 2D similarity
        for i in tqdm(range(len(fps))):
            sim_mat[i][i] = 1.0
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                sim_mat[i][j] = sim_mat[j][i] = sim

        sim_mat = (np.array(sim_mat) + sim_qsar) / 2  # average
    
    if K is None:
        # try to find the best K
        best_K, best_score, best_labels = 0, 0, None
        K, patient = 2, 20
        eigen_values, eigen_vectors = np.linalg.eigh(sim_mat)
        vecs = eigen_vectors[:, -100:]
        while True:
            cls_labels = SpectralClustering(K, affinity='precomputed', random_state=random_state).fit_predict(sim_mat)
            score = calinski_harabasz_score(vecs, cls_labels)
            print(f'Trying K = {K}, CH-score = {score}, patience = {patient}')
            if score > best_score:
                best_K, best_score, best_labels = K, score, cls_labels
                patient = 20
            else:
                patient -= 1
            if patient < 0:
                break
            K += 1
        print(f'Best K = {best_K}, score = {best_score}')
        cls_labels = best_labels
    else:
        cls_labels = SpectralClustering(K, affinity='precomputed', random_state=random_state).fit_predict(sim_mat)

    # if visualize:
    #     eigen_values, eigen_vectors = np.linalg.eigh(sim_mat)
    #     high_dim_scatterplot({'vec': eigen_vectors[:, -20:], 'cluster': [str(c) for c in cls_labels]},
    #                          vec='vec', hue='cluster', save_path=os.path.join(out_dir, 'cluster_vis.png'),
    #                          hue_order=[str(i) for i in range(10)])

    return cls_labels, sim_mat


def get_R_N_sub(smiles, beta_aa=False):
    mol = smi2mol(smiles)
    frame = 'OC(=O)CCN' if beta_aa else 'OC(=O)CN'
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(frame))
    assert len(matches) == 1, f'{smiles}: number of amino-bond {len(matches)}'
    
    # find N and R carbon
    frame_atom_idx = { atom_idx: True for atom_idx in matches[0] }
    N_idx, R_C_idx = None, None
    for atom_idx in matches[0]:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() == 7: # is N
            N_idx = atom_idx
            break
    
    for atom_idx in matches[0]:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() == 6: # is C
            bonds = atom.GetBonds()
            for bond in bonds:
                if bond.GetBeginAtomIdx() == N_idx or bond.GetEndAtomIdx() == N_idx:
                    R_C_idx = atom_idx
                    break
            if R_C_idx is not None:
                break
    
    # N substitute
    N = mol.GetAtomWithIdx(N_idx)
    N_sub_bond = None
    for bond in N.GetBonds():
        if bond.GetBeginAtomIdx() != R_C_idx and bond.GetEndAtomIdx() != R_C_idx:
            assert N_sub_bond is None
            N_sub_bond = bond

    cycle = False
    if N_sub_bond is None: # no substituent on Nitrogen
        N_sub = None
    else:
        frags = Chem.GetMolFrags(Chem.FragmentOnBonds(Chem.Mol(mol), [N_sub_bond.GetIdx()], addDummies=False), asMols=True)
        if len(frags) == 1:
            cycle = True
        else:
            matches = frags[0].GetSubstructMatches(Chem.MolFromSmarts(frame))
            N_sub = frags[0] if len(matches) == 0 else frags[1]
    
    # R group
    RC = mol.GetAtomWithIdx(R_C_idx)
    R_bonds = []
    for bond in RC.GetBonds():
        if bond.GetBeginAtomIdx() not in frame_atom_idx or bond.GetEndAtomIdx() not in frame_atom_idx:
            R_bonds.append(bond)

    if len(R_bonds) == 0: # no R group
        R = None
    else:
        if not cycle:
            frags = Chem.GetMolFrags(Chem.FragmentOnBonds(Chem.Mol(mol), [b.GetIdx() for b in R_bonds], addDummies=False), asMols=True)
            matches = frags[0].GetSubstructMatches(Chem.MolFromSmarts(frame))
            R = frags[0] if len(matches) == 0 else frags[1]
    if cycle and len(R_bonds) == 1: # one cycle
        frags = Chem.GetMolFrags(Chem.FragmentOnBonds(Chem.Mol(mol), [N_sub_bond.GetIdx()] + [b.GetIdx() for b in R_bonds], addDummies=False), asMols=True)
        assert len(frags) == 2, smiles
        matches = frags[0].GetSubstructMatches(Chem.MolFromSmarts(frame))
        N_sub = R = (frags[0] if len(matches) == 0 else frags[1])
    elif cycle and len(R_bonds) > 1: # one cycle and an R group
        frags = Chem.GetMolFrags(Chem.FragmentOnBonds(Chem.Mol(mol), [b.GetIdx() for b in R_bonds], addDummies=False), asMols=True)
        matches = frags[0].GetSubstructMatches(Chem.MolFromSmarts(frame))
        R = (frags[0] if len(matches) == 0 else frags[1])
        frags = Chem.GetMolFrags(Chem.FragmentOnBonds(Chem.Mol(mol), [N_sub_bond.GetIdx()] + [b.GetIdx() for b in R_bonds], addDummies=False), asMols=True)
        R_smi = mol2smi(R)
        R_idx = None
        for i, frag in enumerate(frags):
            if mol2smi(frag) == R_smi:
                R_idx = i
                break
        frags = [f for i, f in enumerate(frags) if i != R_idx]
        assert len(frags) == 2
        matches = frags[0].GetSubstructMatches(Chem.MolFromSmarts(frame))
        N_sub = (frags[0] if len(matches) == 0 else frags[1])

    return (N_sub_bond, N_sub), (R_bonds, R)
