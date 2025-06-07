import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sys
import argparse
import os
import gc
from utils import create_pytorch_geometric_graph_data_list_from_smiles_and_labels, init_weights, KFoldGen, JSONLogger
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.nn import GlobalAttention
import json
from GAMMA import predict
from GAMMA import train_epoch, test_epoch, eval_metrics, prepare_dataloaders, get_model
from GAMMA import prepare_cross_validation_P as prepare_cross_validation
from captum.attr import IntegratedGradients, DeepLift

def attribute_importance(device, ig, loss_fn = None, optimizer=None, train_dataloader = None, val_dataloader = None, **kwargs):
    predictions = []
    for x, data in enumerate(val_dataloader):
        expression, drugs, target, _ = data
        expression,target = expression.float().to(device),\
                target.float().to(device)
        expression = expression.unsqueeze(1)
        node_features = drugs["x"].float().to(device)
        edge_index = drugs["edge_index"].long().to(device)
        edge_attr = drugs["edge_attr"].float().to(device)
        batch = drugs["batch"].long().to(device)
        exp_imp = ig.attribute(expression, additional_forward_args=(node_features,
                                                                    edge_index,
                                                                    edge_attr,
                                                                    batch, ), internal_batch_size = 1, return_convergence_delta=True)
        predictions += [exp_imp[0].squeeze().detach().cpu()]
    return torch.cat(predictions, axis=0)

with open(f"params/best_params_GAMMA_pretrained_blind_lines_GDSC1.json", "r") as f:
    config = json.load(f)
partitions = prepare_cross_validation(config, "GDSC1", k=25)
device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")
preds = {}
import tqdm
trained_partitions = torch.load(f"trained_models/gamma_pretrained_blind_lines_25_s2.pkl")
for i in tqdm.tqdm(range(25)):
    partitions[i]["model"].load_state_dict(trained_partitions[i])
    partitions[i]["model"].eval()
    partitions[i]["model"].to(device)
    ig = IntegratedGradients(partitions[i]["model"])
    gc.collect()
    preds[i] = attribute_importance(ig = ig, **partitions[i], device=device)
    torch.save(preds[i], f"feature_importances/gamma_pretrained_blind_lines_25_partition_{i}_linescore.pkl")
    partitions[i]["model"].to(torch.device("cpu"))