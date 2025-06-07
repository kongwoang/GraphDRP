import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sys
from os.path import dirname, realpath
from GATmann import prepare_dataloaders, prepare_dataloaders_prism, prepare_dataloaders_allgenes, prepare_dataloaders_corrupted
from utils import create_pytorch_geometric_graph_data_list_from_smiles_and_labels, init_weights, KFoldGen, KFoldLenient
import pickle
from GATF import CancerDrugGraphFeaturesDataset, GATmannEncoder, GATF
from GATF import get_model as get_pretrained_encoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.nn import GlobalAttention, GraphMultisetTransformer, GATConv
import json
from torchmetrics.functional import confusion_matrix, accuracy, auroc
from GATF import GATmannEncoder
from GATR import GATR
from torchmetrics.functional import pearson_corrcoef, r2_score, mean_squared_error, explained_variance
from torchmetrics import PearsonCorrCoef, R2Score, MeanSquaredError, ExplainedVariance
import gc

class FCBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_factor,
                 p_dropout,
                 **kwargs):
        super().__init__()
        self.nw = nn.Sequential(nn.ReLU(),
                               nn.Dropout(p=p_dropout),
                               nn.Linear(input_dim, int(hidden_factor * 512)),
                               nn.Sigmoid(),
                               nn.Linear(int(hidden_factor * 512), 1))
        self.nw.apply(init_weights)
    def forward(self, x):
        return self.nw(x)
    
class AttnDropout(nn.Module):
    def __init__(self,
                 p_dropout = 0.1,
                 **kwargs):
        super().__init__()
        self.id = nn.Identity()
        self.bern = torch.distributions.bernoulli.Bernoulli(torch.Tensor([p_dropout]))
    def forward(self, x):
        x = self.id(x)
        if self.training:
            mask = self.bern.sample([x.shape[0], x.shape[1]]).squeeze()
            incorrect = mask.all(axis=1).any()
            while incorrect:
                mask = self.bern.sample([x.shape[0], x.shape[1]]).squeeze()
                incorrect = mask.all(axis=1).any()
            x[mask.bool()] = float("-inf")
        return x
        
class GAMMA(nn.Module):
    def __init__(self,
                 GATmann_encoder,
                 graph_selfattention,
                 p_dropout_attn = 0.3577,
                 p_dropout_fc = 0.30,
                 embed_dim=1024,
                 encoded_gene_dim = 512,
                 hidden_factor_attn = 1,
                 hidden_factor_drug = 1,
                 hidden_factor_interaction = 1,
                 hidden_factor_line = 1,
                 output_dim=256,
                 n_genes=2089,
                 pooling="gated",
                 **kwargs):
        super().__init__()
        self.gene_encoder = nn.Linear(n_genes, encoded_gene_dim)
        self.w_g = nn.Sequential(nn.ReLU(),
                                nn.Linear(encoded_gene_dim, embed_dim))
        self.gat_encoder = GATmann_encoder
        self.graph_selfattention = graph_selfattention
        self.global_attention = GlobalAttention(nn.Sequential(nn.Linear(embed_dim, int(hidden_factor_attn * 512)),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(int(hidden_factor_attn * 512), 1)),
                                               nn.Sequential(nn.Linear(embed_dim, int(hidden_factor_attn * 512)),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(int(hidden_factor_attn * 512), output_dim)))
        if pooling == "gated":
            self.global_attention = GlobalAttention(nn.Sequential(nn.Linear(embed_dim, int(hidden_factor_attn * 512)),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(int(hidden_factor_attn * 512), 1)),
                                               nn.Sequential(nn.Linear(embed_dim, int(hidden_factor_attn * 512)),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(int(hidden_factor_attn * 512), output_dim)))
            self.req_edg_idx = False
        elif pooling =="transformer":
            self.global_attention = GraphMultisetTransformer(in_channels = embed_dim,
                                                         hidden_channels = int(hidden_factor_attn * 512),
                                                         out_channels= output_dim,
                                                         
                                                        )
            self.req_edg_idx = True
        else:
            raise NotImplementedError
        self.fc_drug = FCBlock(input_dim = output_dim,
                               hidden_factor = hidden_factor_drug,
                               p_dropout=p_dropout_fc)
        self.fc_line = FCBlock(input_dim = encoded_gene_dim,
                               hidden_factor = hidden_factor_line,
                               p_dropout=p_dropout_fc)
        self.fc_interaction = FCBlock(input_dim = output_dim,
                               hidden_factor = hidden_factor_interaction,
                               p_dropout=p_dropout_fc)
        
        self.global_attention.apply(init_weights)
                     
    def set_cold(self):
        for p in self.gat_encoder.parameters():
            p.requires_grad=False
        for p in self.graph_selfattention.parameters():
            p.requires_grad=False
            
    def set_warm(self):
        self.gat_encoder.set_warm()
        for p in self.graph_selfattention.parameters():
            p.requires_grad=True
            
    def forward(self, exp, x, edge_index, edge_attr, batch, *args, **kwargs):
        exp = self.gene_encoder(exp).squeeze()
        node_embeddings = self.gat_encoder(x, edge_index, edge_attr)
        x_drug = self.graph_selfattention(node_embeddings, batch)
        x_gatt = self.w_g(exp)
        x_gatt = torch.repeat_interleave(x_gatt, torch.bincount(batch), 0)
        if self.req_edg_idx:
            x_int = self.global_attention(node_embeddings + x_gatt, batch, edge_index)
        else:
            x_int = self.global_attention(node_embeddings + x_gatt, batch)
        return torch.cat([self.fc_drug(x_drug)[:, None], self.fc_line(exp)[:, None], self.fc_interaction(x_int)[:, None]], axis=1).squeeze()
    
class GAMMAMLP(GAMMA):
    def __init__(self,
                 GATmann_encoder,
                 graph_selfattention,
                 p_dropout_attn = 0.3577,
                 p_dropout_fc = 0.30,
                 embed_dim=1024,
                 encoded_gene_dim = 512,
                 hidden_factor_attn = 1,
                 hidden_factor_drug = 1,
                 hidden_factor_interaction = 1,
                 hidden_factor_line = 1,
                 output_dim=256,
                 n_genes=2089,
                 pooling="gated",
                 **kwargs):
        super().__init__(GATmann_encoder,
                 graph_selfattention,
                 p_dropout_attn = p_dropout_attn,
                 p_dropout_fc = p_dropout_fc,
                 embed_dim=embed_dim,
                 encoded_gene_dim = encoded_gene_dim,
                 hidden_factor_attn = hidden_factor_attn,
                 hidden_factor_drug = hidden_factor_drug,
                 hidden_factor_interaction = hidden_factor_interaction,
                 hidden_factor_line = hidden_factor_line,
                 output_dim=output_dim,
                 n_genes=n_genes,
                 pooling=pooling)
        self.fc_interaction = FCBlock(input_dim = output_dim+encoded_gene_dim,
                               hidden_factor = hidden_factor_interaction,
                               p_dropout=p_dropout_fc)
    def forward(self, exp, x, edge_index, edge_attr, batch, *args, **kwargs):
        exp = self.gene_encoder(exp).squeeze()
        node_embeddings = self.gat_encoder(x, edge_index, edge_attr)
        x_drug = self.graph_selfattention(node_embeddings, batch)
        x_gatt = self.w_g(exp)
        return torch.cat([self.fc_drug(x_drug)[:, None], self.fc_line(exp)[:, None], self.fc_interaction(torch.cat([x_drug, exp], axis = -1))[:, None]], axis=1).squeeze()    

def get_model_FS(**config):
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
    encoder = pretrained_GATF.gat_encoder
    tox_attn = GATR(encoder, **config_gatr)
    tox_attn.load_state_dict(torch.load("trained_models/GATR_pretrained.pth",  map_location=torch.device("cpu")))
    model = GAMMAMLP(tox_attn.gat_encoder, tox_attn.global_attention, output_dim = config_gatr["output_dim"], **config)
    model.set_cold()
    return model

def get_model_without_tox(**config):
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
    tox_attn = GATR(encoder, **config_gatr)
    model = GAMMA(tox_attn.gat_encoder, tox_attn.global_attention, output_dim = config_gatr["output_dim"], **config)
    model.set_cold()
    return model

def get_model(**config):
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
    encoder = pretrained_GATF.gat_encoder
    tox_attn = GATR(encoder, **config_gatr)
    tox_attn.load_state_dict(torch.load("trained_models/GATR_pretrained.pth",  map_location=torch.device("cpu")))
    model = GAMMA(tox_attn.gat_encoder, tox_attn.global_attention, output_dim = config_gatr["output_dim"], **config)
    model.set_cold()
    return model

def get_model_NP(**config):
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
    encoder = pretrained_GATF.gat_encoder
    tox_attn = GATR(encoder, **config_gatr)
    model = GAMMA(tox_attn.gat_encoder, tox_attn.global_attention, **config)
    model.set_cold()
    return model

def train_epoch(device, loss_fn, train_dataloader, model, optimizer, test_dataloader = None, val_dataloader = None, l2_weight = 0.01, **kwargs):
    optimizer.zero_grad()
    model.train()
    losses = []
    loss = loss_fn()
    reg = loss_fn()
    for x, data in enumerate(train_dataloader):
        one_out = torch.ones(data[0].shape[0], 3, device=device)
        expression, drugs, target, _ = data
        expression,target = expression.float().to(device),\
                target.float().to(device)
        expression = expression.unsqueeze(1)
        node_features = drugs["x"].float().to(device)
        edge_index = drugs["edge_index"].long().to(device)
        edge_attr = drugs["edge_attr"].float().to(device)
        batch = drugs["batch"].long().to(device)
        preds = model(expression, node_features, edge_index, edge_attr, batch)
        variances = torch.Tensor()
        reg_ = reg(one_out, preds.squeeze())
        preds = preds[:, 0] + preds[:, 1] + preds[:, 2]
        mse = loss(target.squeeze(), preds.squeeze())
        loss_ = mse + l2_weight*reg_
        loss_.backward()
        optimizer.step()
        optimizer.zero_grad()
        mean_loss = mse.detach().item()
        losses.append(mean_loss)
    del mean_loss, loss_, mse, reg_, preds, reg, batch, edge_attr, edge_index, node_features, expression, target, drugs, one_out, data
    gc.collect()
    return np.mean(losses)

def test_epoch(device, loss_fn, test_dataloader, model, optimizer=None, train_dataloader = None, val_dataloader = None, **kwargs):
    model.eval()
    losses = []
    loss = loss_fn()
    with torch.no_grad():
        for x, data in enumerate(test_dataloader):
            expression, drugs, target, _ = data
            expression,target = expression.float().to(device),\
                    target.float().to(device)
            expression = expression.unsqueeze(1)
            node_features = drugs["x"].float().to(device)
            edge_index = drugs["edge_index"].long().to(device)
            edge_attr = drugs["edge_attr"].float().to(device)
            batch = drugs["batch"].long().to(device)
            preds = model(expression, node_features, edge_index, edge_attr, batch)
            preds = preds[:, 0] + preds[:, 1] + preds[:, 2]
            mse = loss(target.squeeze(), preds.squeeze())
            mean_loss = mse.detach().item()
            losses.append(mean_loss)
    del mean_loss, mse, preds, batch, edge_attr, edge_index, node_features, expression, target, drugs, data
    gc.collect()
    return np.mean(losses)

def eval_metrics(device, val_dataloader, model, **kwargs):
    model.eval()
    rs = []
    r2s = []
    mses = []
    evs = []
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
            preds = preds[:, 0] + preds[:, 1] + preds[:, 2]
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
              "RMSE": np.sqrt(np.mean(mses)),
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
            preds = preds[:, 0] + preds[:, 1] + preds[:, 2]
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

def predict(device, val_dataloader, model, loss_fn = None, optimizer=None, train_dataloader = None, test_dataloader = None, **kwargs):
    model.eval()
    predictions = []
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
            try:
                preds = preds + model.bias
            except:
                pass
            predictions.append(preds.cpu().numpy())
    return np.vstack(predictions)

def prepare_cross_validation_FS(config=None, dataset = "GDSC1", k=16, no_val = False, all_genes=False, use_slopes = False, corruption = None, **kwargs):
    partitions = {}
    try:
        use_log_scale = config["use_log_scale"]
    except (KeyError, TypeError):
        use_log_scale = True
    try:
        use_split = config["use_split"]
    except (KeyError, TypeError):
        use_split = "blind_chems"
    if dataset == "GDSC1":
        if all_genes:
            n_genes = 17419
            prepare_dataloaders_fn = prepare_dataloaders_allgenes
        else:
            n_genes = 2089
            prepare_dataloaders_fn = prepare_dataloaders
        
        if corruption is not None:
            prepare_dataloaders_fn = prepare_dataloaders_corrupted
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
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(train_batch = train_batch, no_val=no_val,**kf[i])
            partitions[i] = {"model":get_model_FS(**config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters(),
                                                          lr=learning_rate,
                                                         betas = betas,
                                                         weight_decay = weight_decay)
            partitions[i]["model"].set_cold()
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=config["lr_decay"])
    else:
        config = {}
        config["n_genes"] = n_genes
        for i in range(k):
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(no_val=no_val,**kf[i])
            partitions[i] = {"model":get_model_FS(**config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters())
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=0.99)
    return partitions

def prepare_cross_validation_P1(config=None, dataset = "GDSC1", k=16, no_val = False, all_genes=False, use_slopes = False, corruption = None, **kwargs):
    partitions = {}
    try:
        use_log_scale = config["use_log_scale"]
    except (KeyError, TypeError):
        use_log_scale = True
    try:
        use_split = config["use_split"]
    except (KeyError, TypeError):
        use_split = "blind_chems"
    if dataset == "GDSC1":
        if all_genes:
            n_genes = 17419
            prepare_dataloaders_fn = prepare_dataloaders_allgenes
        else:
            n_genes = 2089
            prepare_dataloaders_fn = prepare_dataloaders
        
        if corruption is not None:
            prepare_dataloaders_fn = prepare_dataloaders_corrupted
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
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(train_batch = train_batch, no_val=no_val,**kf[i])
            partitions[i] = {"model":get_model_without_tox(**config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters(),
                                                          lr=learning_rate,
                                                         betas = betas,
                                                         weight_decay = weight_decay)
            partitions[i]["model"].set_cold()
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=config["lr_decay"])
    else:
        config = {}
        config["n_genes"] = n_genes
        for i in range(k):
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(no_val=no_val,**kf[i])
            partitions[i] = {"model":get_model_without_tox(**config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters())
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=0.99)
    return partitions

def prepare_cross_validation_P(config=None, dataset = "GDSC1", k=16, no_val = False, all_genes=False, use_slopes = False, corruption = None, **kwargs):
    partitions = {}
    try:
        use_log_scale = config["use_log_scale"]
    except (KeyError, TypeError):
        use_log_scale = True
    try:
        use_split = config["use_split"]
    except (KeyError, TypeError):
        use_split = "blind_chems"
    if dataset == "GDSC1":
        if all_genes:
            n_genes = 17419
            prepare_dataloaders_fn = prepare_dataloaders_allgenes
        else:
            n_genes = 2089
            prepare_dataloaders_fn = prepare_dataloaders
        
        if corruption is not None:
            prepare_dataloaders_fn = prepare_dataloaders_corrupted
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
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(train_batch = train_batch, no_val=no_val,**kf[i])
            partitions[i] = {"model":get_model(**config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters(),
                                                          lr=learning_rate,
                                                         betas = betas,
                                                         weight_decay = weight_decay)
            partitions[i]["model"].set_cold()
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=config["lr_decay"])
    else:
        config = {}
        config["n_genes"] = n_genes
        for i in range(k):
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(no_val=no_val,**kf[i])
            partitions[i] = {"model":get_model(**config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters())
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=0.99)
    return partitions

def prepare_cross_validation(config=None, dataset = "GDSC1", k=16, no_val = False, all_genes=False,**kwargs):
    partitions = {}
    try:
        use_log_scale = config["use_log_scale"]
    except (KeyError, TypeError):
        use_log_scale = True
    try:
        use_split = config["use_split"]
    except (KeyError, TypeError):
        use_split = "blind_chems"
    if dataset == "GDSC1":
        if all_genes:
            n_genes = 17419
            prepare_dataloaders_fn = prepare_dataloaders_allgenes
        else:
            n_genes = 2089
            prepare_dataloaders_fn = prepare_dataloaders
        if use_log_scale:
            data = pd.read_csv("data/ic50_processed_windex.csv")
        else:
            data = pd.read_csv("data/ic50_processed_nolog.csv")
        if use_split == "blind_lines":
            partition_col = "COSMIC_ID"
        elif use_split == "blind_chems":
            partition_col = "DRUG_NAME"
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
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(train_batch = train_batch, no_val=no_val,**kf[i])
            partitions[i] = {"model":get_model_NP(**config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters(),
                                                          lr=learning_rate,
                                                         betas = betas,
                                                         weight_decay = weight_decay)
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=config["lr_decay"])
    else:
        config = {}
        config["n_genes"] = n_genes
        for i in range(k):
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_fn(no_val=no_val,**kf[i])
            partitions[i] = {"model":get_model_NP(**config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters())
            partitions[i]["scheduler"] = torch.optim.lr_scheduler.ExponentialLR(partitions[i]["optimizer"], gamma=0.99)
    return partitions
