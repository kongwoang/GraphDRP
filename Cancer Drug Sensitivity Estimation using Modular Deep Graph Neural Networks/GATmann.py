import rdkit
from rdkit import Chem
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils import *
import sys
from GATF import GATF
import pickle
from torch_geometric.data import Batch
import rdkit
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_max_pool,  GraphMultisetTransformer, GATConv
from torchmetrics.functional import pearson_corrcoef, r2_score, mean_squared_error, explained_variance
from torchmetrics import PearsonCorrCoef, R2Score, MeanSquaredError, ExplainedVariance
import gc
import torch_geometric.transforms as T


class CancerDrugGraphDataset(Dataset):
    def __init__(self, ic50_df, cellines, drug_dict, graphs, transform=None):
        """
            
        """
        self.ic50_df = ic50_df
        self.graphs = graphs
        self.cellines = cellines
        self.drug_dict = drug_dict
        self.transform = transform
    def __len__(self):
        return self.ic50_df.shape[0]

    def __getitem__(self, idx):
        item = self.ic50_df.iloc[idx]
        drug_data = self.graphs[self.drug_dict[item.iloc[1]]]
        score = item.iloc[2]
        index = item.iloc[4]
        cell_line = item.iloc[0]
        cell_data = self.cellines.loc[cell_line].to_numpy()
        return (drug_data,
                cell_data,
                np.array([score]),
            np.array([index]))
    
class DrugGraphDataset(Dataset):
    def __init__(self, drug_dict, graphs, transform=None):
        """
            
        """
        self.graphs = graphs
        self.drug_dict = drug_dict
        self.transform = transform
    def __len__(self):
        return len(self.drug_dict.keys())

    def __getitem__(self, idx):
        drug = list(self.drug_dict.keys())[idx]
        drug_data = self.graphs[self.drug_dict[drug]]
        return drug_data

class PrismDataset(Dataset):
    def __init__(self, ec50_df, cellines, graphs, transform=None):
        """
            
        """
        self.ic50_df = ec50_df
        self.graphs = graphs
        self.cellines = cellines
        self.transform = transform
    def __len__(self):
        return self.ic50_df.shape[0]

    def __getitem__(self, idx):
        item = self.ic50_df.iloc[idx]
        drug_data = self.graphs[item.iloc[1]]
        score = item.iloc[2]
        index = item.iloc[4]
        cell_line = item.iloc[0]
        cell_data = self.cellines.loc[cell_line].to_numpy()
        return drug_data, cell_data, np.array([score]), np.array([index])
    
    
def collate_batch(data):
    genes = []
    graphs = []
    targets = []
    indices = []
    for gr, g, t, i in data:
        genes.append(g)
        graphs.append(gr)
        targets.append(t)
        indices.append(i)
    genes = torch.Tensor(np.vstack(genes))
    targets = torch.Tensor(np.hstack(targets))
    graphs = Batch.from_data_list(graphs)
    indices = torch.Tensor(np.hstack(indices))
    return genes, graphs, targets, indices

def prepare_dataloaders(train, test, val, train_batch=1024, test_batch=512, num_workers=4, no_val=False):
    with open("data/graphs.pkl", "rb") as f:
        graphs = pickle.load(f)
    lines_preprocessed = pd.read_csv("data/lines_processed.csv", index_col=0)
    drugs = pd.read_csv("data/drugs_processed.csv", index_col=0)
    drugs["index"] = np.arange(0, drugs.shape[0])
    drug_dict = {drugs.index[i]:drugs["index"].iloc[i] for i in range(drugs.shape[0])}
    if no_val:
        train_d = CancerDrugGraphDataset(pd.concat([train, test], axis=0), lines_preprocessed, drug_dict, graphs)
        val_d = CancerDrugGraphDataset(val, lines_preprocessed, drug_dict, graphs)
        train_loader = DataLoader(train_d, batch_size=train_batch, num_workers=num_workers, collate_fn=collate_batch, shuffle=True)
        val_loader = DataLoader(val_d, batch_size=test_batch, num_workers=num_workers, collate_fn=collate_batch)
        return train_loader, None, val_loader
    else:
        train_d = CancerDrugGraphDataset(train, lines_preprocessed, drug_dict, graphs)
        test_d = CancerDrugGraphDataset(test, lines_preprocessed, drug_dict, graphs)
        val_d = CancerDrugGraphDataset(val, lines_preprocessed, drug_dict, graphs)
        train_loader = DataLoader(train_d, batch_size=train_batch, num_workers=num_workers, collate_fn=collate_batch, shuffle=True)
        test_loader = DataLoader(test_d, batch_size=test_batch, num_workers=num_workers, collate_fn=collate_batch)
        val_loader = DataLoader(val_d, batch_size=test_batch, num_workers=num_workers, collate_fn=collate_batch)
        return train_loader, test_loader, val_loader

def prepare_dataloaders_corrupted(train, test, val, train_batch=1024, test_batch=512, num_workers=8, no_val=False, corruption="total"):
    with open(f"/u/pealo/mcac/graphs_total.pkl", "rb") as f:
        graphs = pickle.load(f)
    lines_preprocessed = pd.read_csv("data/lines_processed.csv", index_col=0)
    drugs = pd.read_csv("data/drugs_processed.csv", index_col=0)
    drugs["index"] = np.arange(0, drugs.shape[0])
    drug_dict = {drugs.index[i]:drugs["index"].iloc[i] for i in range(drugs.shape[0])}
    if no_val:
        train_d = CancerDrugGraphDataset(pd.concat([train, test], axis=0), lines_preprocessed, drug_dict, graphs)
        val_d = CancerDrugGraphDataset(val, lines_preprocessed, drug_dict, graphs)
        train_loader = DataLoader(train_d, batch_size=train_batch, num_workers=num_workers, collate_fn=collate_batch, shuffle=True)
        val_loader = DataLoader(val_d, batch_size=test_batch, num_workers=num_workers, collate_fn=collate_batch)
        return train_loader, None, val_loader
    else:
        train_d = CancerDrugGraphDataset(train, lines_preprocessed, drug_dict, graphs)
        test_d = CancerDrugGraphDataset(test, lines_preprocessed, drug_dict, graphs)
        val_d = CancerDrugGraphDataset(val, lines_preprocessed, drug_dict, graphs)
        train_loader = DataLoader(train_d, batch_size=train_batch, num_workers=num_workers, collate_fn=collate_batch, shuffle=True)
        test_loader = DataLoader(test_d, batch_size=test_batch, num_workers=num_workers, collate_fn=collate_batch)
        val_loader = DataLoader(val_d, batch_size=test_batch, num_workers=num_workers, collate_fn=collate_batch)
        return train_loader, test_loader, val_loader

def prepare_met_dataloader(drop_mets, num_workers=8, no_val=False):
    with open("data/graphs.pkl", "rb") as f:
        graphs = pickle.load(f)
    drugs = pd.read_csv("data/drugs_processed.csv", index_col=0)
    drugs["index"] = np.arange(0, drugs.shape[0])
    used_in_training = drugs["drug_name"].isin(drop_mets)
    drugs = drugs[~used_in_training]
    drug_dict = {drugs.index[i]:drugs["index"].iloc[i] for i in range(drugs.shape[0])}
    return DataLoader(DrugGraphDataset(drug_dict, graphs), batch_size=drugs.shape[0], num_workers=num_workers, collate_fn=Batch.from_data_list)

    
    
def prepare_dataloaders_allgenes(train, test, val, train_batch=1024, test_batch=512, num_workers=8, no_val=False):
    with open("/u/pealo/mcac/graphs.pkl", "rb") as f:
        graphs = pickle.load(f)
    lines_preprocessed = pd.read_csv("data/lines_altprocessed.csv", index_col=0)
    drugs = pd.read_csv("data/drugs_processed.csv", index_col=0)
    drugs["index"] = np.arange(0, drugs.shape[0])
    drug_dict = {drugs.index[i]:drugs["index"].iloc[i] for i in range(drugs.shape[0])}
    if no_val:
        train_d = CancerDrugGraphDataset(pd.concat([train, test], axis=0), lines_preprocessed, drug_dict, graphs)
        val_d = CancerDrugGraphDataset(val, lines_preprocessed, drug_dict, graphs)
        train_loader = DataLoader(train_d, batch_size=train_batch, num_workers=num_workers, collate_fn=collate_batch, shuffle=True)
        val_loader = DataLoader(val_d, batch_size=test_batch, num_workers=num_workers, collate_fn=collate_batch)
        return train_loader, None, val_loader
    else:
        train_d = CancerDrugGraphDataset(train, lines_preprocessed, drug_dict, graphs)
        test_d = CancerDrugGraphDataset(test, lines_preprocessed, drug_dict, graphs)
        val_d = CancerDrugGraphDataset(val, lines_preprocessed, drug_dict, graphs)
        train_loader = DataLoader(train_d, batch_size=train_batch, num_workers=num_workers, collate_fn=collate_batch, shuffle=True)
        test_loader = DataLoader(test_d, batch_size=test_batch, num_workers=num_workers, collate_fn=collate_batch)
        val_loader = DataLoader(val_d, batch_size=test_batch, num_workers=num_workers, collate_fn=collate_batch, shuffle=False)
        return train_loader, test_loader, val_loader

def prepare_dataloaders_prism(train, test, val, train_batch=1024, test_batch=512, num_workers=8, no_val=False):
    with open("data/prism_graphs.pkl", "rb") as f:
        graphs = pickle.load(f)
    lines_preprocessed = pd.read_csv("data/prism_expression_processed.csv", index_col=0)
    if no_val:
        train_d = PrismDataset(pd.concat([train, test], axis=0), lines_preprocessed, graphs)
        val_d = PrismDataset(val, lines_preprocessed, graphs)
        train_loader = DataLoader(train_d, batch_size=train_batch, num_workers=num_workers, collate_fn=collate_batch, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_d, batch_size=test_batch, num_workers=num_workers, collate_fn=collate_batch)
        return train_loader, None, val_loader
    else:
        train_d = PrismDataset(train, lines_preprocessed, graphs)
        test_d = PrismDataset(test, lines_preprocessed, graphs)
        val_d = PrismDataset(val, lines_preprocessed, graphs)
        train_loader = DataLoader(train_d, batch_size=train_batch, num_workers=num_workers, collate_fn=collate_batch, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_d, batch_size=test_batch, num_workers=num_workers, collate_fn=collate_batch)
        val_loader = DataLoader(val_d, batch_size=test_batch, num_workers=num_workers, collate_fn=collate_batch)
        return train_loader, test_loader, val_loader



class GATmann(nn.Module):
    def __init__(self,
                 node_features=79,
                 edge_features=10,
                 fc_hidden = 1024,
                 gene_hidden = 4096,
                 n_conv_layers = 3,
                 n_heads=2,
                 p_dropout_gene = 0.1,
                 p_dropout_gat = 0.001,
                 p_dropout_attn = 0.1,
                 p_dropout_fc = 0.25,
                 embed_dim=32,
                 hidden_dim = 512,
                 output_dim=256,
                 n_genes = 2089,
                 pooling = "gated",
                 **kwargs):
        super().__init__()
        # self.gene_attention = nn.Sequential(nn.Linear(n_genes, n_genes),
        #                                    nn.Softmax(2))
        self.gene = nn.Sequential(nn.Linear(n_genes, gene_hidden),
                                  nn.Dropout(p_dropout_gene),
                                  nn.ReLU(),
                                  nn.Linear(gene_hidden, embed_dim*n_heads))
        self.gat_init = GATv2Conv(node_features, embed_dim, heads= n_heads, edge_dim=edge_features, dropout=p_dropout_gat)
        self.gat_layers = nn.ModuleList([GATv2Conv(embed_dim*n_heads,
                                             embed_dim*n_heads,
                                             heads= n_heads,
                                             edge_dim=edge_features,
                                             concat=False, dropout=p_dropout_gat) for g in range(n_conv_layers-1)])
        if pooling == "gated":
            self.global_attention = GlobalAttention(nn.Sequential(nn.Linear(embed_dim*n_heads, hidden_dim),
                                                                 nn.ReLU(),
                                                                 nn.Dropout(p_dropout_attn),
                                                                 nn.Linear(hidden_dim, 1)),
                                                   nn.Sequential(nn.Linear(embed_dim*n_heads, hidden_dim),
                                                                 nn.ReLU(),
                                                                 nn.Dropout(p_dropout_attn),
                                                                 nn.Linear(hidden_dim, output_dim)))
            self.req_edg_idx = False
        elif pooling =="transformer":
            self.global_attention = GraphMultisetTransformer(in_channels = embed_dim*n_heads,
                                                         hidden_channels = hidden_dim,
                                                         out_channels= output_dim,
                                                        )
            self.req_edg_idx = True
        else:
            raise NotImplementedError
        self.fc = nn.Sequential(nn.ReLU(),
                               nn.Dropout(p=p_dropout_fc),
                               nn.Linear(output_dim, fc_hidden),
                               # nn.BatchNorm1d(512),
                               nn.Sigmoid(),
                               nn.Linear(fc_hidden, 1))
        # self.w_g = (nn.Linear(1, embed_dim*n_heads))
    def forward(self, exp, x, edge_index, edge_attr, batch, *args, **kwargs):
        exp = self.gene(exp).squeeze()
        exp = torch.repeat_interleave(exp, torch.bincount(batch), 0)
        x = self.gat_init(x, edge_index, edge_attr)
        for gat_layer in self.gat_layers:
            x = gat_layer(F.leaky_relu(x), edge_index, edge_attr)
        if self.req_edg_idx:
            x = self.global_attention(x + exp, batch, edge_index)
        else:
            x = self.global_attention(x + exp, batch)
        return self.fc(x)

class GATmannP(nn.Module):
    def __init__(self,
                 GATmann_encoder,
                 node_features=79,
                 edge_features=10,
                 fc_hidden = 1024,
                 gene_hidden = 4096,
                 p_dropout_gene = 0.1,
                 p_dropout_attn = 0.1,
                 p_dropout_fc = 0.25,
                 embed_dim=1024,
                 hidden_dim = 512,
                 output_dim=256,
                 n_genes = 2089,
                 pooling = "gated",
                 **kwargs):
        super().__init__()
        # self.gene_attention = nn.Sequential(nn.Linear(n_genes, n_genes),
        #                                    nn.Softmax(2))
        self.gene = nn.Sequential(nn.Linear(n_genes, gene_hidden),
                                  nn.Dropout(p_dropout_gene),
                                  nn.ReLU(),
                                  nn.Linear(gene_hidden, embed_dim))
        self.gat_encoder = GATmann_encoder
        if pooling == "gated":
            self.global_attention = GlobalAttention(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                                 nn.ReLU(),
                                                                 nn.Dropout(p_dropout_attn),
                                                                 nn.Linear(hidden_dim, 1)),
                                                   nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                                 nn.ReLU(),
                                                                 nn.Dropout(p_dropout_attn),
                                                                 nn.Linear(hidden_dim, output_dim)))
            self.req_edg_idx = False
        elif pooling =="transformer":
            self.global_attention = GraphMultisetTransformer(in_channels = embed_dim,
                                                         hidden_channels = hidden_dim,
                                                         out_channels= output_dim,
                                                        )
            self.req_edg_idx = True
        else:
            raise NotImplementedError
        self.fc = nn.Sequential(nn.ReLU(),
                               nn.Dropout(p=p_dropout_fc),
                               nn.Linear(output_dim, fc_hidden),
                               # nn.BatchNorm1d(512),
                               nn.Sigmoid(),
                               nn.Linear(fc_hidden, 1))
        # self.w_g = (nn.Linear(1, embed_dim*n_heads))
    def set_cold(self):
        for p in self.gat_encoder.parameters():
            p.requires_grad=False
            
    def set_warm(self):
        self.gat_encoder.set_warm()
    
    def forward(self, exp, x, edge_index, edge_attr, batch, *args, **kwargs):
        # exp_att = self.gene_attention(exp)
        # ga = torch.bmm(exp_att, exp.transpose(2, 1)).squeeze(1)
        ga = self.gene(exp).squeeze()
        x_ga = torch.repeat_interleave(ga, torch.bincount(batch), 0)
        x = self.gat_encoder(x, edge_index, edge_attr)
        if self.req_edg_idx:
            x = self.global_attention(x + x_ga, batch, edge_index)
        else:
            x = self.global_attention(x + x_ga, batch)
        return self.fc(x)


def get_model(config={}):
    model = GATmann(**config)
    return model

def get_pretrained_model(config={}):
    with open("params/optimal_config_GATF.json", "r") as f:
        config_gatf = json.load(f)
    try:
        config_gatf["p_dropout_gat"] = config["p_dropout_gat"]
    except KeyError:
        pass
    pretrained_GATF = GATF(**config_gatf)
    pretrained_GATF.load_state_dict(torch.load("trained_models/GATR_pretrained.pth",  map_location=torch.device("cpu")))
    encoder = pretrained_GATF.gat_encoder
    model = GATmannP(encoder, **config)
    return model

def get_pretrained1_model(config={}):
    with open("params/optimal_config_GATF.json", "r") as f:
        config_gatf = json.load(f)
    with open("params/optimal_config_GATR.json", "r") as f:
        config_gatr = json.load(f)
    try:
        config_gatf["p_dropout_gat"] = config["p_dropout_gat"]
        config_gatr["p_dropout_attn"] = config["p_dropout_attn"]
    except KeyError:
        pass
    pretrained_GATF = GATF(**config_gatf)
    pretrained_GATF.load_state_dict(torch.load("trained_models/best_GATF.pth",  map_location=torch.device("cpu")))
    encoder = pretrained_GATF.gat_encoder
    model = GATmannP(encoder, **config)
    return model

def train_epoch(device, loss_fn, train_dataloader, model, optimizer, test_dataloader = None, val_dataloader = None, **kwargs):
    optimizer.zero_grad()
    model.train()
    losses = []
    for x, data in enumerate(train_dataloader):
        loss = loss_fn()
        expression, drugs, target, _ = data
        expression,target = expression.float().to(device),\
                target.float().to(device)
        expression = expression.unsqueeze(1)
        node_features = drugs["x"].float().to(device)
        edge_index = drugs["edge_index"].long().to(device)
        edge_attr = drugs["edge_attr"].float().to(device)
        batch = drugs["batch"].long().to(device)
        preds = model(expression, node_features, edge_index, edge_attr, batch)
        mse = loss(target.squeeze(), preds.squeeze())
        mse.backward()
        optimizer.step()
        optimizer.zero_grad()
        mean_loss = mse.detach().item()
        losses.append(mean_loss)
    del mean_loss, mse, loss, target, preds, batch, drugs, edge_attr, edge_index, node_features, expression, data
    gc.collect()
    return np.mean(losses)

def test_epoch(device, loss_fn, test_dataloader, model, optimizer=None, train_dataloader = None, val_dataloader = None, **kwargs):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, data in enumerate(test_dataloader):
            loss = loss_fn()
            expression, drugs, target, _ = data
            expression,target = expression.float().to(device),\
                    target.float().to(device)
            expression = expression.unsqueeze(1)
            node_features = drugs["x"].float().to(device)
            edge_index = drugs["edge_index"].long().to(device)
            edge_attr = drugs["edge_attr"].float().to(device)
            batch = drugs["batch"].long().to(device)
            preds = model(expression, node_features, edge_index, edge_attr, batch)
            mse = loss(target.squeeze(), preds.squeeze())
            mean_loss = mse.detach().item()
            losses.append(mean_loss)
    del mean_loss, mse, loss, target, preds, batch, drugs, edge_attr, edge_index, node_features, expression, data
    gc.collect()
    return np.mean(losses)

def eval_metrics(device, val_dataloader, model, loss_fn = None, optimizer=None, train_dataloader = None, test_dataloader = None, **kwargs):
    model.eval()
    rs = []
    r2s = []
    mses = []
    evs = []
    with torch.no_grad():
        for x, data in enumerate(val_dataloader):
            loss = loss_fn()
            expression, drugs, target = data
            expression,target = expression.float().to(device),\
                    target.float().to(device)
            expression = expression.unsqueeze(1)
            node_features = drugs["x"].float().to(device)
            edge_index = drugs["edge_index"].long().to(device)
            edge_attr = drugs["edge_attr"].float().to(device)
            batch = drugs["batch"].long().to(device)
            preds = model(expression, node_features, edge_index, edge_attr, batch)
            preds, target = preds.squeeze(), target.squeeze()
            r = pearson_corrcoef(preds, target).cpu().numpy()
            r2 = r2_score(preds, target).cpu().numpy()
            mse = mean_squared_error(preds, target).cpu().numpy()
            ev = explained_variance(preds, target).cpu().numpy()
            r2s.append(r2)
            rs.append(r)
            mses.append(mse)
            evs.append(ev)
    metrics = {"R":np.mean(rs),
              "R2":np.mean(r2s),
              "MSE":np.mean(mses),
              "Explained variance":np.mean(evs)}
    return metrics

def eval_metrics2(device, val_dataloader, model, **kwargs):
    model.eval()
    r2 = R2Score().to(device)
    r = PearsonCorrCoef().to(device)
    mse = MeanSquaredError().to(device)
    ev = ExplainedVariance().to(device)
    with torch.no_grad():
        for x, data in enumerate(val_dataloader):
            expression, drugs, target, _ = data
            expression,target = expression.float().to(device),\
                    target.float().to(device)
            expression = expression.unsqueeze(1)
            node_features = drugs["x"].float().to(device)
            edge_index = drugs["edge_index"].long().to(device)
            edge_attr = drugs["edge_attr"].float().to(device)
            batch = drugs["batch"].long().to(device)
            preds = model(expression, node_features, edge_index, edge_attr, batch)
            preds, target = preds.squeeze(), target.squeeze()
            r.update(preds, target)
            r2.update(preds, target)
            mse.update(preds, target)
            ev.update(preds, target)
    metrics = {"R":r.compute().item(),
              "R2":r2.compute().item(),
              "MSE":mse.compute().item(),
              "Explained variance":ev.compute().item()}
    return metrics

def prepare_cross_validation(config=None, dataset = "GDSC1", k=16,  use_slopes = False, **kwargs):
    partitions = {}
    try:
        use_log_scale = config["use_log_scale"]
    except (KeyError, TypeError):
        use_log_scale = True
    try:
        use_split = config["use_split"]
    except (KeyError, TypeError):
        use_split = "blind_lines"
    if dataset == "GDSC1":
        n_genes = 2089
        if use_slopes:
            data = pd.read_csv("data/data_slopes.csv", index_col=0)
        elif use_log_scale:
            data = pd.read_csv("data/ic50_processed_windex.csv")
        else:
            data = pd.read_csv("data/ic50_processed_nolog.csv")
        if use_split == "blind_lines":
            partition_col = "COSMIC_ID"
        elif use_split == "blind_chems":
            partition_col = "DRUG_NAME"
        prepare_dataloaders_fn = prepare_dataloaders
    elif dataset == "PRISM":
        n_genes = 2055
        if use_log_scale:
            data = pd.read_csv("data/prism_screening_processed.csv", index_col=0)
        else:
            raise NotImplementedError
        if use_split == "blind_lines":
            partition_col = "depmap_id"
        elif use_split == "blind_chems":
            partition_col = "name"
        prepare_dataloaders_fn = prepare_dataloaders_prism
    else:
        raise NotImplementedError
    if use_split in ["blind_lines", "blind_chems"]:
        kf = KFoldGen(data, partition_col, k=k)
    elif use_split == "lenient":
        kf = KFoldLenient(data, partition_col, k=k)
    else:
        raise NotImplementedError
    if config is not None:
        config["n_genes"] = n_genes
        learning_rate = config["learning_rate"]
        betas =  (config["beta_1"], config["beta_2"])
        weight_decay = config["weight_decay"]
        train_batch = config["train_batch"]
        for i in range(k):
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(train_batch = train_batch, **kf[i])
            partitions[i] = {"model":get_model(config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters(),
                                                          lr=learning_rate,
                                                         betas = betas,
                                                         weight_decay = weight_decay)
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=0.99)
    else:
        config = {}
        config["n_genes"] = n_genes
        for i in range(k):
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(**kf[i])
            partitions[i] = {"model":get_model(config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters())
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=0.99)
    return partitions

def prepare_cross_validation_P(config=None, dataset = "GDSC1", k=16, **kwargs):
    partitions = {}
    try:
        use_log_scale = config["use_log_scale"]
    except (KeyError, TypeError):
        use_log_scale = True
    try:
        use_split = config["use_split"]
    except (KeyError, TypeError):
        use_split = "blind_lines"
    if dataset == "GDSC1":
        n_genes = 2089
        if use_log_scale:
            data = pd.read_csv("data/ic50_processed_windex.csv")
        else:
            data = pd.read_csv("data/ic50_processed_nolog.csv")
        if use_split == "blind_lines":
            partition_col = "COSMIC_ID"
        elif use_split == "blind_chems":
            partition_col = "DRUG_NAME"
        prepare_dataloaders_fn = prepare_dataloaders
    elif dataset == "PRISM":
        n_genes = 2055
        if use_log_scale:
            data = pd.read_csv("data/prism_screening_processed.csv", index_col=0)
        else:
            raise NotImplementedError
        if use_split == "blind_lines":
            partition_col = "depmap_id"
        elif use_split == "blind_chems":
            partition_col = "name"
        prepare_dataloaders_fn = prepare_dataloaders_prism
    else:
        raise NotImplementedError
    if use_split in ["blind_lines", "blind_chems"]:
        kf = KFoldGen(data, partition_col, k=k)
    elif use_split == "lenient":
        kf = KFoldLenient(data, partition_col, k=k)
    else:
        raise NotImplementedError
    if config is not None:
        config["n_genes"] = n_genes
        learning_rate = config["learning_rate"]
        betas =  (config["beta_1"], config["beta_2"])
        weight_decay = config["weight_decay"]
        train_batch = config["train_batch"]
        for i in range(k):
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(train_batch = train_batch, **kf[i])
            partitions[i] = {"model":get_pretrained_model(config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters(),
                                                          lr=learning_rate,
                                                         betas = betas,
                                                         weight_decay = weight_decay)
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=0.99)
    else:
        config = {}
        config["n_genes"] = n_genes
        for i in range(k):
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(**kf[i])
            partitions[i] = {"model":get_pretrained_model(config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters())
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=0.99)
    return partitions

def prepare_cross_validation_P1(config=None, dataset = "GDSC1", k=16, **kwargs):
    partitions = {}
    try:
        use_log_scale = config["use_log_scale"]
    except (KeyError, TypeError):
        use_log_scale = True
    try:
        use_split = config["use_split"]
    except (KeyError, TypeError):
        use_split = "blind_lines"
    if dataset == "GDSC1":
        n_genes = 2089
        if use_log_scale:
            data = pd.read_csv("data/ic50_processed_windex.csv")
        else:
            data = pd.read_csv("data/ic50_processed_nolog.csv")
        if use_split == "blind_lines":
            partition_col = "COSMIC_ID"
        elif use_split == "blind_chems":
            partition_col = "DRUG_NAME"
        prepare_dataloaders_fn = prepare_dataloaders
    elif dataset == "PRISM":
        n_genes = 2055
        if use_log_scale:
            data = pd.read_csv("data/prism_screening_processed.csv", index_col=0)
        else:
            raise NotImplementedError
        if use_split == "blind_lines":
            partition_col = "depmap_id"
        elif use_split == "blind_chems":
            partition_col = "name"
        prepare_dataloaders_fn = prepare_dataloaders_prism
    else:
        raise NotImplementedError
    if use_split in ["blind_lines", "blind_chems"]:
        kf = KFoldGen(data, partition_col, k=k)
    elif use_split == "lenient":
        kf = KFoldLenient(data, partition_col, k=k)
    else:
        raise NotImplementedError
    if config is not None:
        config["n_genes"] = n_genes
        learning_rate = config["learning_rate"]
        betas =  (config["beta_1"], config["beta_2"])
        weight_decay = config["weight_decay"]
        train_batch = config["train_batch"]
        for i in range(k):
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(train_batch = train_batch, **kf[i])
            partitions[i] = {"model":get_pretrained1_model(config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters(),
                                                          lr=learning_rate,
                                                         betas = betas,
                                                         weight_decay = weight_decay)
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=0.99)
    else:
        config = {}
        config["n_genes"] = n_genes
        for i in range(k):
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(**kf[i])
            partitions[i] = {"model":get_pretrained1_model(config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters())
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=0.99)
    return partitions
