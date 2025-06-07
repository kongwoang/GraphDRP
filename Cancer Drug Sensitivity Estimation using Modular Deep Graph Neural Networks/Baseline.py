import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sys
from GATmann import prepare_dataloaders, prepare_dataloaders_prism, prepare_dataloaders_allgenes
from utils import init_weights, KFoldGen, KFoldLenient
import pickle
from GATF import CancerDrugGraphFeaturesDataset, GATmannEncoder, GATF
from GATF import get_model as get_pretrained_encoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.nn import GlobalAttention, GraphMultisetTransformer, GATConv, GCNConv, global_mean_pool
import json
from torchmetrics.functional import confusion_matrix, accuracy, auroc
from GATF import GATmannEncoder
from GATR import GATR
from torchmetrics.functional import pearson_corrcoef, r2_score, mean_squared_error, explained_variance
from torchmetrics import PearsonCorrCoef, R2Score, MeanSquaredError, ExplainedVariance
from GAMMA import FCBlock
import gc

class BaselineEncoder(nn.Module):
    def __init__(self,
                 node_features=79,
                 n_conv_layers = 3,
                 embed_dim=32,
                 **kwargs):
        super().__init__()
        self.n_conv_layers = n_conv_layers
        self.gat_init = GCNConv(node_features, embed_dim)
        self.gat_layers = nn.ModuleList([GCNConv(embed_dim,
                                             embed_dim) for g in range(n_conv_layers-1)])
        self.gat_init.apply(init_weights)
        [layer.apply(init_weights) for layer in self.gat_layers]
    
    def forward(self, x, edge_index, edge_attr, *args, **kwargs):
        x = self.gat_init(x, edge_index)
        for gat_layer in self.gat_layers:
            x = gat_layer(F.leaky_relu(x), edge_index)
        return x

class Baseline(nn.Module):
    def __init__(self,
                 p_dropout_attn = 0.3577,
                 p_dropout_fc = 0.30,
                 embed_dim=1024,
                 encoded_gene_dim = 512,
                 hidden_factor = 1,
                 hidden_factor_attn = 1,
                 output_dim=256,
                 n_layers = 3, 
                 n_genes=2089,
                 attn_pooling=False,
                 node_features=79,
                 **kwargs):
        super().__init__()
        self.gene_encoder = nn.Linear(n_genes, encoded_gene_dim)
        self.w_g = nn.Sequential(nn.ReLU(),
                                nn.Linear(encoded_gene_dim, embed_dim))
        self.gcn_encoder = BaselineEncoder(n_conv_layers = n_layers, embed_dim = embed_dim)
        if attn_pooling:
            self.global_pooling = GlobalAttention(nn.Sequential(nn.Linear(embed_dim, int(hidden_factor_attn * 512)),
                                                                 nn.ReLU(),
                                                                 nn.Dropout(p_dropout_attn),
                                                                 nn.Linear(int(hidden_factor_attn * 512), 1)),
                                                   nn.Sequential(nn.Linear(embed_dim, int(hidden_factor_attn * 512)),
                                                                 nn.ReLU(),
                                                                 nn.Dropout(p_dropout_attn),
                                                                 nn.Linear(int(hidden_factor_attn * 512), output_dim)))
            self.global_pooling.apply(init_weights)
        else:
            self.global_pooling = global_mean_pool
        self.fc = FCBlock(input_dim = embed_dim,
                               hidden_factor = hidden_factor,
                               p_dropout=p_dropout_fc)
        
            
    def forward(self, exp, x, edge_index, edge_attr, batch, *args, **kwargs):
        exp = self.gene_encoder(exp).squeeze()
        node_embeddings = self.gcn_encoder(x, edge_index, edge_attr)
        x_gatt = self.w_g(exp)
        x_gatt = torch.repeat_interleave(x_gatt, torch.bincount(batch), 0)
        x_int = self.global_pooling(node_embeddings + x_gatt, batch)
        return self.fc(x_int)    

def get_model(config = {}, **kwargs):
    model = Baseline()
    return model

def prepare_cross_validation(config=None, dataset = "GDSC1", k=16, **kwargs):
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