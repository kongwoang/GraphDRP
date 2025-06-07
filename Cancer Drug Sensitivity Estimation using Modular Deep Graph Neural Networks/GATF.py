import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
# Pytorch and Pytorch Geometric
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import pickle
import torch_geometric
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from torch_geometric.data import Batch
import rdkit
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_max_pool, global_mean_pool
from torchmetrics.functional import pearson_corrcoef, r2_score, mean_squared_error, explained_variance

class GATmannEncoder(nn.Module):
    def __init__(self,
                 node_features=79,
                 edge_features=10,
                 n_conv_layers = 3,
                 n_heads=2,
                 p_dropout_gat = 0.001,
                 embed_dim=32,
                 **kwargs):
        super().__init__()
        self.n_conv_layers = n_conv_layers
        self.gat_init = GATv2Conv(node_features, embed_dim, heads= n_heads, edge_dim=edge_features, dropout=p_dropout_gat)
        self.gat_layers = nn.ModuleList([GATv2Conv(embed_dim*n_heads,
                                             embed_dim*n_heads,
                                             heads= n_heads,
                                             edge_dim=edge_features,
                                             concat=False, dropout=p_dropout_gat) for g in range(n_conv_layers-1)])
        self.gat_init.apply(init_weights)
        [layer.apply(init_weights) for layer in self.gat_layers]
        
    def set_warm(self):
        for p in self.gat_layers[self.n_conv_layers - 2].parameters():
            p.requires_grad=True
            
    def set_cold(self):
        for p in self.gat_layers.parameters():
            p.requires_grad=False
    
    def forward(self, x, edge_index, edge_attr, *args, **kwargs):
        x = self.gat_init(x, edge_index, edge_attr)
        for gat_layer in self.gat_layers:
            x = gat_layer(F.leaky_relu(x), edge_index, edge_attr)
        return x
    
class GATF(nn.Module):
    def __init__(self,
                 node_features=79,
                 edge_features=10,
                 fc_hidden = 1024,
                 n_conv_layers = 3,
                 n_heads=2,
                 p_dropout_gat = 0.001,
                 p_dropout_attn = 0.1,
                 p_dropout_fc = 0.25,
                 embed_dim=32,
                 hidden_dim = 512,
                 output_dim=256,
                 n_genes = 2089,
                 **kwargs):
        super().__init__()      
        self.gat_encoder = GATmannEncoder(node_features,
                 edge_features,
                 n_conv_layers,
                 n_heads,
                 p_dropout_gat,
                 embed_dim,)
        self.global_attention = GlobalAttention(nn.Sequential(nn.Linear(embed_dim*n_heads, hidden_dim),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(hidden_dim, 1)),
                                               nn.Sequential(nn.Linear(embed_dim*n_heads, hidden_dim),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(hidden_dim, output_dim)))
        self.fc = nn.Sequential(nn.ReLU(),
                               nn.Dropout(p=p_dropout_fc),
                               nn.Linear(output_dim, fc_hidden),
                               nn.Sigmoid(),
                               nn.Linear(fc_hidden, 7))
        self.fc.apply(init_weights)
        self.global_attention.apply(init_weights)
    def forward(self, x, edge_index, edge_attr, batch, *args, **kwargs):
        x = self.gat_encoder(x, edge_index, edge_attr)
        x = self.global_attention(x, batch)
        return self.fc(x)
    
class CancerDrugGraphFeaturesDataset(Dataset):
    def __init__(self, graphs, transform=None):
        """
            
        """
        self.graphs = graphs
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx].clone()
        graph["y"] = graph["y"].unsqueeze(0)
        return graph

def prepare_dataloaders(train_batch=1024, test_batch=1024, num_workers=16, **kwargs):
    graphs = torch.load ("data/graph_batch_features.pkl")
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

def train_epoch(device, loss_fn, train_dataloader, model, optimizer, **kwargs):
    optimizer.zero_grad()
    model.train()
    losses = []
    for x, batch in enumerate(train_dataloader):
        loss = loss_fn()
        drugs = batch
        node_features = drugs["x"].float().to(device)
        edge_index = drugs["edge_index"].long().to(device)
        edge_attr = drugs["edge_attr"].float().to(device)
        target = drugs["y"].float().to(device)
        batch = drugs["batch"].long().to(device)
        preds = model(node_features, edge_index, edge_attr, batch)
        mse = loss(target.squeeze()[~target.isnan()], preds.squeeze()[~target.isnan()])
        mse.backward()
        optimizer.step()
        optimizer.zero_grad()
        mean_loss = mse.data.cpu().numpy()
        losses.append(mean_loss)
    return np.mean(losses)

def test_epoch(device, loss_fn, test_dataloader, model, **kwargs):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, batch in enumerate(test_dataloader):
            loss = loss_fn()
            drugs = batch
            node_features = drugs["x"].float().to(device)
            edge_index = drugs["edge_index"].long().to(device)
            edge_attr = drugs["edge_attr"].float().to(device)
            target = drugs["y"].float().to(device)
            batch = drugs["batch"].long().to(device)
            preds = model(node_features, edge_index, edge_attr, batch)
            mse = loss(target.squeeze()[~target.isnan()], preds.squeeze()[~target.isnan()])
            mean_loss = mse.data.cpu().numpy()
            losses.append(mean_loss)
    return np.mean(losses)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
        
def predict_batch(device, test_batch, model, **kwargs):
    model.eval()
    losses = []
    with torch.no_grad():
        loss = loss_fn()
        drugs = test_batch
        node_features = drugs["x"].float().to(device)
        edge_index = drugs["edge_index"].long().to(device)
        edge_attr = drugs["edge_attr"].float().to(device)
        target = drugs["y"].float().to(device)
        batch = drugs["batch"].long().to(device)
        preds = model(node_features, edge_index, edge_attr, batch)
    return preds

def get_model(**config):
    model = GATF(**config)
    return model