import pandas as pd
from rdkit import Chem
import numpy as np
from utils import GraphCreator
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# create folder structure
os.mkdir("data")
os.mkdir("feature_importances")
os.mkdir("logs")
os.mkdir("data")
os.mkdir("plots")
os.mkdir("trained_models")

gc = GraphCreator()
catmos = pd.read_excel("data/CATMoS.xlsx", 1, header=1)
smiles = catmos.loc[:, "Canonical_QSARr"].to_numpy()
gdsc = pd.read_csv("data/SMILES_gdsc.csv")
smiles_gdsc = gdsc.loc[:, "canonical_smiles"].to_numpy()
pd.read_csv("data/met_features.csv")["CID"].drop_duplicates().to_csv("CIDS.txt", header=None, index=None)
smiles_features = pd.read_csv("data/smiles_features.txt.gz", sep = "\t", header = None).groupby(0).first().drop_duplicates()
# check for overlap
mols = [Chem.CanonSmiles(d) for d in smiles]
mols_gdsc = [Chem.CanonSmiles(d) for d in smiles_gdsc]
in_gdsc = np.array([np.array([mols[i] == m for m in mols_gdsc]).any() for i in range(len(mols))])
catmos = catmos.assign(canonical_smiles = mols).loc[~in_gdsc].loc[:, ["canonical_smiles", "very_toxic", "nontoxic"]].dropna().astype({"very_toxic":int, "nontoxic":int})
graphs_catmos = gc(catmos.loc[:, "canonical_smiles"].to_numpy(), catmos.index.to_numpy())
drugs_catmos = list(graphs_catmos.keys())
for drug in drugs_catmos:
    if catmos.loc[drug]["very_toxic"]:
        y = 2
    elif catmos.loc[drug]["nontoxic"]: 
        y = 0
    else:
        y = 1
    graphs_catmos[drug]["y"] = torch.Tensor([y])
torch.save(list(graphs_catmos.values()), "data/graphs_ranking.pkl") # save graphs and targets
torch.save(graphs_catmos, "data/graphs_catmos.pt") # save graphs alone
catmos.drop("canonical_smiles", axis=1).to_csv("data/catmos.csv") # save data
# create and save GDSC graphs
graphs_gdsc = gc(smiles_gdsc, gdsc["drug_name"].to_numpy())
torch.save(graphs_gdsc, "data/graphs.pkl")
# create and save graphs for feature prediction
graphs_f = gc(smiles_features.to_numpy().squeeze(), smiles_features.index.to_numpy().squeeze())
all_cids = list(graphs_f.keys())
features = pd.read_csv("data/met_features.csv").set_index("CID").iloc[:, [2, 3, 4, 5, 7, 8, 9]]
mms = MinMaxScaler((-1, 1))
mms2 = MinMaxScaler((0, 1))
features.loc[:] = mms.fit_transform(features.to_numpy())
feature_cids = features.loc[all_cids]
for cid in all_cids:
    graphs_f[cid]["y"] = torch.Tensor(feature_cids.loc[[cid]].iloc[0].to_numpy())
torch.save(list(graphs_f.values()), "data/graph_batch_features.pkl")
screening_gdsc = pd.read_csv("data/GDSC1_fitted_dose_response_25Feb20.csv").loc[:, ["COSMIC_ID", "DRUG_NAME", "LN_IC50"]]
screening_gdsc["LN_IC50"] = mms2.fit_transform(screening_gdsc["LN_IC50"][:, None]).squeeze()
# processing the expression data
cell_lines_exp = pd.read_csv("data/Cell_line_RMA_proc_basalExp.txt.zip", sep = "\t")
genes = cell_lines_exp.T.loc["GENE_SYMBOLS"]
cell_lines_exp = cell_lines_exp.T.iloc[2:]
cell_lines_exp.columns = genes
cell_lines_exp.index = cell_lines_exp.index.str.extract("DATA.0?([0-9]+)").squeeze()
paccmann_genes = pd.read_csv("https://raw.githubusercontent.com/prassepaul/mlmed_ranking/main/data/gdsc_data/paccmann_gene_list.txt", header=None).to_numpy().squeeze()
gene_exists = np.isin(paccmann_genes, genes)
cell_lines_exp = cell_lines_exp.loc[:, paccmann_genes[gene_exists]]
cell_lines_exp = cell_lines_exp.reset_index().astype({0:int}).set_index(0).sort_index()
cell_lines_exp.loc[:] = mms.fit_transform(cell_lines_exp)
cell_lines_exp.to_csv("data/lines_processed.csv")
screening_gdsc.query("DRUG_NAME in @graphs_gdsc.keys() & COSMIC_ID in @cell_lines_exp.index.to_numpy()").to_csv("data/ic50_processed_windex.csv")
