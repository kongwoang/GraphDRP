#!/usr/bin/env python3.9

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

def train_model(config, k=25):
    partitions = prepare_cross_validation(config, "GDSC1", k=k)
    device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
    preds = {}
    if args.notox:
        train_suffix = "no_tox"
    for i in range(k):
        trained_partitions = torch.load(f"trained_models/{args.model}_{config['train_suffix']}_{config['blind_suffix']}_{args.k}.pkl")
        partitions[i]["model"].load_state_dict(trained_partitions[i])
        partitions[i]["model"].to(device)
        gc.collect()
        preds[i] = predict(**partitions[i], device=device)
        partitions[i]["model"].to(torch.device("cpu"))
    with open(f"predictions/{args.model}_{config['train_suffix']}_{config['blind_suffix']}_{args.k}.pkl", "wb") as f:
        pickle.dump(preds, f)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model to be trained (gamma, gatmann or baseline)")
    parser.add_argument(
        "--blind",
        type=str,
        required=True,
        help="The blind setting: (drugs or lines)")
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of folds to be employed")
    
    parser.add_argument(
        "--pretrained", action="store_true", help="Uses the pretrained model")
    parser.add_argument(
        "--noregularizer",
        action="store_true", help="Remove Regularization")
    parser.add_argument(
        "--notox",
        action="store_true", help="Remove toxicity pretraining")
    parser.add_argument(
        "--concat", action="store_true", help="Replaces cross-attention by concatenation")
    
    args, _ = parser.parse_known_args()
    if args.blind == "lines":
        blind_suffix = "blind_lines"
    elif args.blind == "drugs":
        blind_suffix = "blind_chems"
    else: 
        raise NotImplementedError
    if args.pretrained:
        train_suffix = "pretrained"
    else:
        train_suffix = "no_pretraining"
    if args.model == "gamma":
        from GAMMA import train_epoch, test_epoch, eval_metrics, prepare_dataloaders, get_model
        if args.pretrained:
            if args.notox:
                from GAMMA import prepare_cross_validation_P1 as prepare_cross_validation
            elif args.concat:
                from GAMMA import prepare_cross_validation_FS as prepare_cross_validation
            else:
                from GAMMA import prepare_cross_validation_P as prepare_cross_validation
        else:
            from GAMMA import prepare_cross_validation
        with open(f"params/best_params_GAMMA_{train_suffix}_{blind_suffix}_GDSC1.json", "r") as f:
            config = json.load(f)
        if not args.notox:
            train_suffix = "pretrained"
        else:
            train_suffix = "no_tox"
        if args.concat:
            train_suffix += "_concatenation"
        if args.noregularizer:
            config["l2_weight"] = 0.0
            train_suffix += "_noregularizer"
        config["pooling"] = "gated"
    elif args.model == "gatmann":
        from GATmann import train_epoch, test_epoch, eval_metrics, prepare_dataloaders, get_model
        if args.pretrained:
            if not args.notox:
                from GATmann import prepare_cross_validation_P as prepare_cross_validation
            else:
                from GATmann import prepare_cross_validation_P1 as prepare_cross_validation
        else:
            from GATmann import prepare_cross_validation
        with open(f"params/best_params_GATMANN_{train_suffix}_{blind_suffix}_GDSC1.json", "r") as f:
            config = json.load(f)
            config["lr_decay"] = 0.99
        if not args.notox:
            train_suffix = "pretrained"
        else:
            train_suffix = "no_tox"
    elif args.model == "baseline":
        if args.pretrained:
            raise NotImplementedError
        from GATmann import train_epoch, test_epoch, eval_metrics2, prepare_dataloaders
        from Baseline import get_model, prepare_cross_validation
        config = {
        "learning_rate": 7.472890340227499e-5,
        "beta_1": 0.7421965359678988,
        "beta_2": 0.9706232710650776,
        "train_batch": 512,
        "weight_decay": 1.4868271080888546e-8,
        }
        if args.blind == "drugs":
            config["learning_rate"] = 7.472890340227499e-7
        config ["use_split"] = blind_suffix
        config["eval_model"] = False
        config["train_length"] = 50
        config["lr_decay"] = 0.99
    config["train_suffix"] = train_suffix
    config["blind_suffix"] = blind_suffix
    train_model(config, k=args.k)
    gc.collect()
