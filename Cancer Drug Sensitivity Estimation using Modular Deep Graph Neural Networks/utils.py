import rdkit
from rdkit import Chem
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os 
import json
import uuid
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
# Pytorch and Pytorch Geometric
import torch
# from torch_geometric.data import Data, Batch
# import torch_geometric
import pickle
import sys
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from matplotlib import pyplot as plt

class SmileEmbedder():
    def __init__(self):
        self.replace = "LMWERTYUZXVLQqAaDdGgJjlmwertyuzxv<>?&"
    def randomize_smiles(self, smiles, canonical=True):
            """Perform a randomization of a SMILES string
            must be RDKit sanitizable.
            From https://github.com/EBjerrum/SMILES-enumeration/blob/master/SmilesEnumerator.py"""
            m = Chem.MolFromSmiles(smiles)
            ans = list(range(m.GetNumAtoms()))
            np.random.shuffle(ans)
            nm = Chem.RenumberAtoms(m,ans)
            return Chem.MolToSmiles(nm, canonical=canonical)
    def fit(self, smile_series):
        smiles = []
        smile_series = smile_series.dropna()
        self.seq_length = smile_series.apply(len).max()
        smiles.append(smile_series.apply(self.randomize_smiles).str.extract("(\[.\])").dropna().loc[:,0].unique())
        smiles.append(smile_series.apply(self.randomize_smiles).str.extract("(\[..\])").dropna().loc[:,0].unique())
        smiles.append(smile_series.apply(self.randomize_smiles).str.extract("(\[...\])").dropna().loc[:,0].unique())
        smiles.append(smile_series.apply(self.randomize_smiles).str.extract("(\[....\])").dropna().loc[:,0].unique())
        smiles.append(smile_series.apply(self.randomize_smiles).str.extract("(\[.....\])").dropna().loc[:,0].unique())
        smile_brackets = np.concatenate(smiles)
        smile_patterns = []
        for smile in smile_brackets:
            smile_patterns.append(f"\[{smile[1:-1]}\]")
        smiles = []
        smiles.append(smile_series.apply(self.randomize_smiles).str.extract("(\(.\))").dropna().loc[:,0].unique())
        smiles.append(smile_series.apply(self.randomize_smiles).str.extract("(\(..\))").dropna().loc[:,0].unique())
        smile_par = np.concatenate(smiles)
        smile_patterns = []
        for smile in smile_par:
            smile_patterns.append(f"\({smile[1:-1]}\)")
        self.smile_patterns = smile_patterns
        replace = self.replace
        all_symbols = []
        symbols = list(smile_series.apply(self.randomize_smiles).apply(lambda x:list(np.unique(list(x)))))
        for sym in symbols:
            if type(sym) == list:
                all_symbols += sym
            else:
                all_symbols.append(sym)
        unique_symbols = np.unique(all_symbols)
        i = 1
        embeddings = {}
        for sym in unique_symbols:
            embeddings[sym] = i
            i+=1
        for sym in replace[0:len(smile_patterns)]:
            embeddings[sym] = i
            i+=1
        self.vocabulary = i
        self.embeddings = embeddings
    def transform(self, smiles, cannonical=False):
        string = self.randomize_smiles(smiles, cannonical)
        for i in range(0, len(self.smile_patterns)):
            string = regex.sub(self.smile_patterns[i], self.replace[i], string)
        vec = list(string)
        vec = pd.Series(vec).map(self.embeddings).replace(np.nan, self.vocabulary+3).to_numpy()
        padding_l = self.seq_length - len(vec)
        padding = np.zeros(padding_l)
        seq = np.concatenate([padding, np.array([self.vocabulary+1]), vec, np.array([self.vocabulary+2]),])
        return seq
    def transform_batch(self, smile_batch):
        smiles = smile_batch.apply(self.randomize_smiles)
        vectorized_smiles = []
        for smile in smiles:
            string = smile
            for i in range(0, len(self.smile_patterns)):
                string = regex.sub(self.smile_patterns[i], self.replace[i], string)
            vec = list(string)
            vec = pd.Series(vec).map(self.embeddings).to_numpy()
            padding_l = self.seq_length - len(vec)
            padding = np.zeros(padding_l)
            seq = np.concatenate([padding, np.array([self.vocabulary+1]), vec, np.array([self.vocabulary+2]),])
            vectorized_smiles.append(seq)
            
        return np.vstack(vectorized_smiles)
    
def preprocess_lines(lines, include, norm_range=1):
    filtered_lines = lines.loc[include.to_numpy().squeeze()].T
    mms = MinMaxScaler([-norm_range, norm_range])
    mms.fit(filtered_lines)
    filtered_lines.iloc[:,:] = mms.transform(filtered_lines)
    return filtered_lines

class KFoldGen():
    def __init__(self, data=None, partition_column="COSMIC_ID", k = 16, **kwargs):
        self.k = k
        if data is None:
            data = pd.read_csv("data/ic50_processed_windex.csv")
        self.data = data
        self.levels = data.loc[:,partition_column].unique()
        self.partition_column = partition_column
        self.n_levels = len(self.levels)
        self.k_masks(self.n_levels)
        self.stratified_partition()
        self.train_test_val()
    def k_masks(self, length, shuffle = True, seed=3558):
        k = self.k
        np.random.seed(seed)
        index = np.arange(0, length)
        if shuffle:
            index = np.random.choice(index, length, replace=False)
        masks = np.array_split(index, k)
        self.masks = masks
        
    def stratified_partition(self, shuffle=True):
        self.k_folds = [self.data[self.data[self.partition_column].isin(self.levels[self.masks[i]])] for i in range(self.k)]
    
    def train_test_val(self):
        index = np.arange(self.k)
        not_valid = True
        while not_valid:
            mask_1 = np.random.choice(index, self.k, replace=False)
            mask_2 = np.random.choice(index, self.k, replace=False)
            not_valid = np.any(mask_1 == mask_2)
        self.mask_test = mask_1
        self.mask_val = mask_2
    
    def __iter__(self):
        self.a = 0
        return self

    def __next__(self):
        i = self.a
        self.a += 1
        nums = list(range(self.k))
        test_i = self.mask_test[i]
        val_i = self.mask_val[i]
        holdout = [test_i, val_i]
        split = {"train":pd.concat([self.k_folds[i] for i in nums if i not in holdout]),
                "test":self.k_folds[test_i],
                "val":self.k_folds[val_i]}
        return split
    def __getitem__(self, idx):
        i = idx
        nums = list(range(self.k))
        test_i = self.mask_test[i]
        val_i = self.mask_val[i]
        holdout = [test_i, val_i]
        split = {"train":pd.concat([self.k_folds[i] for i in nums if i not in holdout]),
                "test":self.k_folds[test_i],
                "val":self.k_folds[val_i]}
        return split
    
class JSONLogger():
    def __init__(self, path = None):
        log = {}
        if path is None:
            path = str(uuid.uuid4()) + ".json"
        if os.path.exists(path):
            with open(path, "r") as f:
                log = json.load(f)
        else:
            with open(path, "w") as f:
                json.dump(log, f)
        self.path = path
        self.log = log
    def __call__(self, epoch, **kwargs):
        key_args = self.nums_as_strings(kwargs)
        epoch_log = {}
        self.log[epoch] = key_args
        with open(self.path, "w") as f:
                json.dump(self.log, f)
    def nums_as_strings(self, args):
        key_args = {}
        for arg in list(args.keys()):
            key_args[arg] = str(args[arg])
        return key_args
    
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    
    data_list = []
    
    for (smiles, y_val) in zip(x_smiles, y):
        try:
            # convert SMILES to RDKit mol object
            mol = Chem.MolFromSmiles(smiles)
            # get feature dimensions
            n_nodes = mol.GetNumAtoms()
            n_edges = 2*mol.GetNumBonds()
            unrelated_smiles = "O=O"
            unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
            n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
            n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
            # construct node feature matrix X of shape (n_nodes, n_node_features)
            X = np.zeros((n_nodes, n_node_features))
            for atom in mol.GetAtoms():
                X[atom.GetIdx(), :] = get_atom_features(atom)

            X = torch.tensor(X, dtype = torch.float)

            # construct edge index array E of shape (2, n_edges)
            (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
            torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
            torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
            E = torch.stack([torch_rows, torch_cols], dim = 0)

            # construct edge feature array EF of shape (n_edges, n_edge_features)
            EF = np.zeros((n_edges, n_edge_features))

            for (k, (i,j)) in enumerate(zip(rows, cols)):

                EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))

            EF = torch.tensor(EF, dtype = torch.float)

            # construct label tensor
            y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)

            # construct Pytorch Geometric data object and append to data list
            data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))
        except:
            pass
    return data_list

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
        
class KFoldGraphs():
    def __init__(self, graphs=None, k = 16, **kwargs):
        self.k = k
        graphs = torch.load("data/graphs_ranking.pkl")
        length = len(graphs)
        self.graphs = graphs
        self.k_masks(length)
        self.generate_partitions()
        self.train_test_val()
    def k_masks(self, length, shuffle = True, seed=3558):
        k = self.k
        np.random.seed(seed)
        index = np.arange(0, length)
        if shuffle:
            index = np.random.choice(index, length, replace=False)
        masks = np.array_split(index, k)
        self.masks = masks
        
    def generate_partitions(self, shuffle=True):
        self.k_folds = []
        for i in range(self.k):
            mask = self.masks[i]
            self.k_folds.append([self.graphs[idx] for idx in mask])
    
    def train_test_val(self):
        index = np.arange(self.k)
        not_valid = True
        while not_valid:
            mask_1 = np.random.choice(index, self.k, replace=False)
            mask_2 = np.random.choice(index, self.k, replace=False)
            not_valid = np.any(mask_1 == mask_2)
        self.mask_test = mask_1
        self.mask_val = mask_2
    
    def __iter__(self):
        self.a = 0
        return self

    def __next__(self):
        i = self.a
        self.a += 1
        nums = list(range(self.k))
        test_i = self.mask_test[i]
        val_i = self.mask_val[i]
        holdout = [test_i, val_i]
        train_partition = []
        for j in nums:
            if j not in holdout:
                train_partition+=self.k_folds[j]
        split = {"train":train_partition,
                "test":self.k_folds[test_i],
                "val":self.k_folds[val_i]}
        return split
    def __getitem__(self, idx):
        i = idx
        nums = list(range(self.k))
        test_i = self.mask_test[i]
        val_i = self.mask_val[i]
        holdout = [test_i, val_i]
        train_partition = []
        for j in nums:
            if j not in holdout:
                train_partition+=self.k_folds[j]
        split = {"train_graphs":train_partition,
                "test_graphs":self.k_folds[test_i],
                "val_graphs":self.k_folds[val_i]}
        return split

class KFoldLenient():
    def __init__(self, data, k = 16, **kwargs):
        self.k = k
        length = data.shape[0]
        self.data = data
        self.k_masks(length)
        self.generate_partitions()
        self.train_test_val()
    def k_masks(self, length, shuffle = True, seed=3558):
        k = self.k
        np.random.seed(seed)
        index = np.arange(0, length)
        if shuffle:
            index = np.random.choice(index, length, replace=False)
        masks = np.array_split(index, k)
        self.masks = masks
        
    def generate_partitions(self):
        self.k_folds = []
        for i in range(self.k):
            self.k_folds.append(self.data.iloc[self.masks[i]])
    
    def train_test_val(self):
        index = np.arange(self.k)
        not_valid = True
        while not_valid:
            mask_1 = np.random.choice(index, self.k, replace=False)
            mask_2 = np.random.choice(index, self.k, replace=False)
            not_valid = np.any(mask_1 == mask_2)
        self.mask_test = mask_1
        self.mask_val = mask_2
    
    def __iter__(self):
        self.a = 0
        return self

    def __next__(self):
        i = self.a
        self.a += 1
        print(i)
        return self[i]
    def __getitem__(self, idx):
        i = idx
        nums = list(range(self.k))
        test_i = self.mask_test[i]
        val_i = self.mask_val[i]
        holdout = [test_i, val_i]
        train_partition = []
        for j in nums:
            if j not in holdout:
                train_partition.append(self.k_folds[j])
        split = {"train":pd.concat(train_partition, axis=0),
                "test":self.k_folds[test_i],
                "val":self.k_folds[val_i]}
        return split

def get_paccmann_metrics(model, blind):
    assert model in ["nn_baseline", "paccmann"]
    assert blind in ["drug_wise", "cell_wise"]
    if blind == "drug_wise":
        blind_ = "_drug_wise"
    else:
        blind_ = ""
    for i in range(5):
        try:
            target = pd.read_csv(f"data/cv_splits/cv_5{blind_}/test_cv_5_fold_{i}.csv", index_col=0)
            min_v = target.min().min()
            max_v = target.max().max()
        except FileNotFoundError:
            pass
    mms = MinMaxScaler([0, 1])
    mms.fit([[min_v], [max_v]])
    pearsons = []
    r2s = []
    evs = []
    mses = []
    for i in range(5):
        try:
            target = pd.read_csv(f"data/cv_splits/cv_5{blind_}/test_cv_5_fold_{i}.csv", index_col=0).unstack().to_numpy()

            pred = pd.read_csv(f"data/results/preds/pred_test_{model}_mse_paccmann__{blind}20210617_{i}.csv", index_col=0).unstack().to_numpy()
            isna = np.isnan(pred)
            preds_f = mms.fit_transform(-pred[~isna][:,None]).squeeze()
            targets = mms.fit_transform(target[~isna][:,None]).squeeze()
            r2s.append(metrics.r2_score(target[~isna], -pred[~isna]))
            pearsons.append(pearsonr(target[~isna], -pred[~isna])[0])

            evs.append(metrics.explained_variance_score(targets, preds_f))

            mses.append(metrics.mean_squared_error(targets, preds_f))
        except FileNotFoundError:
            pass
    return {"R":np.nanmean(pearsons), "Explained variance": np.nanmean(evs), "MSE": np.nanmean(mses), "R2": np.nanmean(r2s),
           "R_std":np.nanstd(pearsons), "Explained variance_std": np.nanstd(evs), "MSE_std": np.nanstd(mses), "R2_std": np.nanstd(r2s)}

def get_av_performance(log, epoch):
    perfs = [metrics  for metrics in eval(log[str(epoch)]["eval_m"])]
    av_perfs = {}
    for i, dict_metrics in enumerate(perfs):
        for metric in dict_metrics.keys():
            if metric not in av_perfs.keys():
                av_perfs[metric] = [dict_metrics[metric]]
            else:
                av_perfs[metric] += [dict_metrics[metric]]
    out = {}
    for metric in av_perfs.keys():
        out[metric + "_std"] = np.nanstd(av_perfs[metric])
        out[metric] = np.nanmean(av_perfs[metric])
    return out

def analyze_log(log, best_epoch, max_epoch=23, plot=True):
    with open(log, "r") as f:
        log = json.load(f)
    if plot:
        palette = sns.color_palette("viridis", n_colors=max_epoch)
        plt.figure(figsize=[18, 12])
        axes = plt.axes()
        for k in range(max_epoch):
            rs = [metrics["R"] for metrics in eval(log[str(k)]["eval_m"])]
            sns.kdeplot(rs, ax=axes, legend=True, color=palette[k], bw_adjust=0.8)
        plt.show()
        palette
    return get_av_performance(log, best_epoch)

class GraphCreator():
    def __init__(self):
        pass
    def one_hot_encoding(self, x, permitted_list):
        """
        Maps input elements x which are not in the permitted list to the last element
        of the permitted list.
        """
        if x not in permitted_list:
            x = permitted_list[-1]
        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
        return binary_encoding


    def get_atom_features(self, atom, 
                          use_chirality = True, 
                          hydrogens_implicit = True):
        """
        Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
        """
        # define list of permitted atoms

        permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']

        if hydrogens_implicit == False:
            permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

        # compute atom features

        atom_type_enc = self.one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

        n_heavy_neighbors_enc = self.one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

        formal_charge_enc = self.one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

        hybridisation_type_enc = self.one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

        is_in_a_ring_enc = [int(atom.IsInRing())]

        is_aromatic_enc = [int(atom.GetIsAromatic())]

        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]

        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]

        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

        if use_chirality == True:
            chirality_type_enc = self.one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
            atom_feature_vector += chirality_type_enc

        if hydrogens_implicit == True:
            n_hydrogens_enc = self.one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
            atom_feature_vector += n_hydrogens_enc
        return np.array(atom_feature_vector)

    def get_bond_features(self, bond, 
                          use_stereochemistry = True):
        """
        Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
        """
        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_type_enc = self.one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

        bond_is_conj_enc = [int(bond.GetIsConjugated())]

        bond_is_in_ring_enc = [int(bond.IsInRing())]

        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

        if use_stereochemistry == True:
            stereo_type_enc = self.one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
            bond_feature_vector += stereo_type_enc
        return np.array(bond_feature_vector)

    def __call__(self, smiles_list, drugs = None, use_supernode = False, **kwargs):
        """
        Inputs:

       smiles_list = [smiles_1, smiles_2, ....] ... a list of SMILES strings
        y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)

        Outputs:

        data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning

        """
        if drugs is None:
            drugs = np.arange(0, len(smiles_list))
        data_dict = {}

        for x, drug in enumerate(drugs):
            try:
                # convert SMILES to RDKit mol object
                smiles = smiles_list[x]
                mol = Chem.MolFromSmiles(smiles)
                # get feature dimensions
                n_nodes = mol.GetNumAtoms()
                n_edges = 2*mol.GetNumBonds()
                unrelated_smiles = "O=O"
                unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
                n_node_features = len(self.get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
                n_edge_features = len(self.get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
                # construct node feature matrix X of shape (n_nodes, n_node_features)
                X = np.zeros((n_nodes, n_node_features))
                for atom in mol.GetAtoms():
                    X[atom.GetIdx(), :] = self.get_atom_features(atom)

                X = torch.tensor(X, dtype = torch.float)

                # construct edge index array E of shape (2, n_edges)
                (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
                torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
                torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
                E = torch.stack([torch_rows, torch_cols], dim = 0)

                # construct edge feature array EF of shape (n_edges, n_edge_features)
                EF = np.zeros((n_edges, n_edge_features))

                for (k, (i,j)) in enumerate(zip(rows, cols)):

                    EF[k] = self.get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))

                EF = torch.tensor(EF, dtype = torch.float)
                add_f = {kwarg: torch.Tensor([kwargs[kwarg][x]]) for kwarg in kwargs.keys()}
                if use_supernode:
                    super_node = torch.zeros([1, X.shape[1]])
                    trgt_supernode = X.shape[0]
                    extra_indices = torch.cat([torch.arange(0, X.shape[0])[:, None], 
                                         torch.full([X.shape[0], 1],  trgt_supernode)], axis=1).T
                    extra_f = torch.zeros([extra_indices.shape[1], EF.shape[1]])
                    indicator_indices = torch.cat([torch.zeros([E.shape[1], 1]),
                                               torch.ones([extra_indices.shape[1], 1])], axis=0)
                    X = torch.cat([X, super_node], axis=0)
                    E = torch.cat([E, extra_indices], axis=1)
                    EF = torch.cat([indicator_indices, torch.cat([EF,
                                                        extra_f], axis=0)], axis=1)
                data_dict[drug] = Data(x = X, edge_index = E, edge_attr = EF, **add_f)
            except Exception as e:
                print(e)
        return data_dict