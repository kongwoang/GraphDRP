import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import create_pytorch_geometric_graph_data_list_from_smiles_and_labels, init_weights, KFoldGraphs
import pickle
from GATF import CancerDrugGraphFeaturesDataset, GATmannEncoder, GATF, train_epoch, test_epoch
from GATF import get_model as get_pretrained_encoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.nn import GlobalAttention
import json
from torchmetrics.functional import confusion_matrix, accuracy, auroc
import gc

def prepare_dataloaders(train_batch=1024, test_batch=1024, num_workers=16, **kwargs):
    graphs = torch.load("data/graphs_ranking.pkl")
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.1, random_state=3558, shuffle=True)
    train_dataset = CancerDrugGraphFeaturesDataset(train_graphs)
    test_dataset = CancerDrugGraphFeaturesDataset(test_graphs)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch,
                                  num_workers=num_workers,
                                  collate_fn=Batch.from_data_list,
                                  shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=train_batch,
                              num_workers=num_workers,
                              collate_fn=Batch.from_data_list,
                              shuffle=False)
    return train_dataloader, test_dataloader

def prepare_train_dataloader(train_batch=1024, num_workers=16, **kwargs):
    graphs = torch.load("data/graphs_ranking.pkl")
    train_dataset = CancerDrugGraphFeaturesDataset(graphs)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch,
                                  num_workers=num_workers,
                                  collate_fn=Batch.from_data_list,
                                  shuffle=False)
    return train_dataloader

def prepare_dataloaders_from_split(train_graphs, test_graphs, val_graphs, train_batch=1024, test_batch=1024, num_workers=16, **kwargs):
    train_dataset = CancerDrugGraphFeaturesDataset(train_graphs)
    test_dataset = CancerDrugGraphFeaturesDataset(test_graphs)
    val_dataset = CancerDrugGraphFeaturesDataset(val_graphs)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch,
                                  num_workers=num_workers,
                                  collate_fn=Batch.from_data_list,
                                  shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=train_batch,
                              num_workers=num_workers,
                              collate_fn=Batch.from_data_list,
                              shuffle=False)
    val_dataloader = DataLoader(test_dataset, batch_size=train_batch,
                              num_workers=num_workers,
                              collate_fn=Batch.from_data_list,
                              shuffle=False)
    return train_dataloader, test_dataloader, val_dataloader

class GATR(nn.Module):
    def __init__(self,
                 GATmann_encoder,
                 fc_hidden = 512,
                 p_dropout_attn = 0.05,
                 p_dropout_fc = 0.43,
                 embed_dim=1024,
                 hidden_dim = 512,
                 output_dim=1024,
                 **kwargs):
        super().__init__()
        # self.gene_attention = nn.Sequential(nn.Linear(n_genes, n_genes),
        #                                    nn.Softmax(2))
        
        self.gat_encoder = GATmann_encoder
        self.global_attention = GlobalAttention(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(hidden_dim, 1)),
                                               nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(hidden_dim, output_dim)))
        self.fc = nn.Sequential(nn.ReLU(),
                               nn.Dropout(p=p_dropout_fc),
                               nn.Linear(output_dim, fc_hidden),
                               nn.Sigmoid(),
                               nn.Linear(fc_hidden, 3))
        self.fc.apply(init_weights)
        self.global_attention.apply(init_weights)
        
    def set_cold(self):
        for p in self.gat_encoder.parameters():
            p.requires_grad=False
            
    def set_warm(self):
        for p in self.gat_encoder.parameters():
            p.requires_grad=True
            
    def forward(self, x, edge_index, edge_attr, batch, *args, **kwargs):
        x = self.gat_encoder(x, edge_index, edge_attr)
        x = self.global_attention(x, batch)
        return self.fc(x)

def get_model(encoder_path = None, **config):
    with open("params/optimal_config_GATF.json", "r") as f:
        config_encoder = json.load(f)
    config_encoder["p_dropout_gat"] = config["p_dropout_gat"]
    encoder = get_pretrained_encoder(**config_encoder)
    if encoder_path is None:
        encoder.load_state_dict(torch.load("trained_models/best_GATF.pth"))
    else:
        encoder.load_state_dict(torch.load(encoder_path))
    encoder_embedder = encoder.gat_encoder
    model = GATR(encoder_embedder, **config)
    model.gat_encoder.set_cold()
    return model

def train_epoch(device, loss_fn, train_dataloader, model, optimizer, loss_weights = [0.681891, 0.778346, 4.020664], **kwargs):
    optimizer.zero_grad()
    model.train()
    losses = []
    loss_fn_ = loss_fn(torch.Tensor(loss_weights)).to(device)
    for x, batch in enumerate(train_dataloader):
        drugs = batch
        node_features = drugs["x"].float().to(device)
        edge_index = drugs["edge_index"].long().to(device)
        edge_attr = drugs["edge_attr"].float().to(device)
        target = drugs["y"].long().to(device)
        batch = drugs["batch"].long().to(device)
        preds = model(node_features, edge_index, edge_attr, batch)
        mse = loss_fn_(preds.squeeze(), target.squeeze())
        mse.backward()
        optimizer.step()
        optimizer.zero_grad()
        mean_loss = mse.data.cpu().numpy()
        losses.append(mean_loss)
    del drugs, node_features, edge_index, edge_attr, target, batch, preds, loss_fn_
    gc.collect()
    return np.mean(losses)

def test_epoch(device, loss_fn, test_dataloader, model, loss_weights = [0.681891, 0.778346, 4.020664], **kwargs):
    model.eval()
    losses = []
    loss_fn_ = loss_fn(torch.Tensor(loss_weights)).to(device)
    with torch.no_grad():
        for x, batch in enumerate(test_dataloader):
            drugs = batch
            node_features = drugs["x"].float().to(device)
            edge_index = drugs["edge_index"].long().to(device)
            edge_attr = drugs["edge_attr"].float().to(device)
            target = drugs["y"].long().to(device)
            batch = drugs["batch"].long().to(device)
            preds = model(node_features, edge_index, edge_attr, batch)
            mse = loss_fn_(preds.squeeze(), target.squeeze())
            mean_loss = mse.data.cpu().numpy()
            losses.append(mean_loss)
        del drugs, node_features, edge_index, edge_attr, target, batch, preds, loss_fn_
        gc.collect()
    return np.mean(losses)

def predict_batch(device, test_batch, model, **kwargs):
    model.eval()
    losses = []
    with torch.no_grad():
        drugs = test_batch
        node_features = drugs["x"].float().to(device)
        edge_index = drugs["edge_index"].long().to(device)
        edge_attr = drugs["edge_attr"].float().to(device)
        target = drugs["y"].float().to(device)
        batch = drugs["batch"].long().to(device)
        preds = model(node_features, edge_index, edge_attr, batch)
        preds = F.softmax(preds, dim=1)
    return preds.argmax(axis=1)

def eval_metrics(device, val_dataloader, model, **kwargs):
    model.eval()
    accs = []
    aurocs = []
    confusion_matrix_acc = None
    with torch.no_grad():
        for x, batch in enumerate(val_dataloader):
            drugs = batch
            node_features = drugs["x"].float().to(device)
            edge_index = drugs["edge_index"].long().to(device)
            edge_attr = drugs["edge_attr"].float().to(device)
            target = drugs["y"].long().to(device)
            batch = drugs["batch"].long().to(device)
            preds = model(node_features, edge_index, edge_attr, batch)
            preds = F.softmax(preds, dim=1)
            if confusion_matrix_acc is None:
                confusion_matrix_acc = confusion_matrix(preds, target, 3).cpu()
            else:
                confusion_matrix_acc += confusion_matrix(preds, target, 3).cpu()
            accs.append((accuracy(preds, target, average = 'macro', num_classes = 3)).cpu().numpy())
            aurocs.append((auroc(preds, target, 3)).cpu().numpy())
    metrics = {"acc": np.mean(accs),"auroc": np.mean(aurocs), "confusion_matrix":confusion_matrix_acc.numpy()}
    return metrics

def prepare_cross_validation(config=None, k=16, **kwargs):
    partitions = {}
    kf = KFoldGraphs(k=k)
    if config is not None:
        learning_rate = config["learning_rate"]
        betas =  (config["beta_1"], config["beta_2"])
        weight_decay = config["weight_decay"]
        train_batch = config["train_batch"]
        for i in range(k):
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_from_split(train_batch = train_batch, **kf[i])
            partitions[i] = {"model":get_model(**config),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters(),
                                                          lr=learning_rate,
                                                         betas = betas,
                                                         weight_decay = weight_decay)
    else:
        for i in range(k):
            train_dataloader, test_dataloader, val_dataloader = prepare_dataloaders_from_split(**kf[i])
            partitions[i] = {"model":get_model(),
                            "train_dataloader":train_dataloader,
                            "test_dataloader":test_dataloader,
                            "val_dataloader":val_dataloader}
            partitions[i]["optimizer"] = torch.optim.Adam(partitions[i]["model"].parameters())
    return partitions