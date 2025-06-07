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

def train_model(config, k=25):
    MAX_PATIENCE = [10]
    logger = JSONLogger(f"logs/{args.model}_{blind_suffix}_{train_suffix}.json")
    partitions = prepare_cross_validation(config, "GDSC1", k=k)
    loss_fn = nn.MSELoss
    patience = MAX_PATIENCE[0]
    best_loss = 1
    devices = [torch.device(f"cuda:{cuda_n}") if torch.cuda.is_available() else torch.device("cpu") for i in range(4)]
    evals = []
    for epoch in range(0, 100):
        train_losses = []
        test_losses = []
        for i in range(init, end):
            device = devices[i%4]
            partitions[i]["model"].to(device)
            if args.pretrained:
                try:
                    if epoch == config["pretraining_length"]:
                        partitions[i]["model"].set_warm()
                except KeyError:
                    if epoch == 2:
                        partitions[i]["model"].set_warm()
            train_loss = train_epoch(device, loss_fn, **partitions[i])
            test_loss = test_epoch(device, loss_fn, **partitions[i])
            metrics = eval_metrics2(device, **partitions[i])
            metrics["epoch"] = epoch
            metrics["p"] = i
            evals += [metrics]
            train_losses += [train_loss]
            test_losses += [test_loss]
            partitions[i]["scheduler"].step()
            gc.collect()
            partitions[i]["model"].to(torch.device("cpu"))
        pd.DataFrame(evals).to_csv("logs/new_log.csv")
        mean_train_loss = np.mean(train_losses)
        std_train_loss = np.std(train_losses)
        mean_test_loss = np.mean(test_losses)
        std_test_loss = np.std(test_losses)
        if (mean_test_loss + std_test_loss) <  best_loss:
            patience = MAX_PATIENCE[0]
            best_loss = mean_test_loss + std_test_loss
            best_models = [partitions[i]["model"].state_dict() for i in range(k)]
        else:
            patience -= 1
        if patience <= 0:
            break
    torch.save(best_models, f"trained_models/{args.model}_{train_suffix}_{blind_suffix}_{args.k}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--cuda",
        type=int,
        required=False,
        default=0,
        help="Selects cuda device")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model to be trained (gamma, gatmann or baseline)")
    parser.add_argument(
        "--concat", action="store_true", help="Replaces cross-attention by concatenation")
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
        "--noregularizer",
        action="store_true", help="Remove Regularization")
    parser.add_argument(
        "--notox",
        action="store_true", help="Remove toxicity pretraining")
    
    parser.add_argument(
        "--fold",
        type=int,
        required=False,
        default=-1,
        help="Optional, train in one fold")
    
    parser.add_argument(
        "--pretrained", action="store_true", help="Uses the pretrained model")
    
    args, _ = parser.parse_known_args()
    global cuda_n
    cuda_n = args.cuda
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
        from GAMMA import train_epoch, test_epoch, eval_metrics, prepare_dataloaders, get_model, eval_metrics2
        if args.pretrained:
            if args.notox:
                from GAMMA import prepare_cross_validation_P1 as prepare_cross_validation
            elif args.concat:
                from GAMMA import prepare_cross_validation_FS as prepare_cross_validation
            else:
                from GAMMA import prepare_cross_validation_P1 as prepare_cross_validation
        else:
            from GAMMA import prepare_cross_validation
        with open(f"params/best_params_GAMMA_{train_suffix}_{blind_suffix}_GDSC1.json", "r") as f:
            config = json.load(f)
            config["pooling"] = "gated"
        if not args.notox:
            train_suffix = "pretrained"
        else:
            train_suffix = "no_tox"
        if args.concat:
            train_suffix += "_concatenation"
        if args.noregularizer:
            config["l2_weight"] = 0.0
            train_suffix += "_noregularizer"
    elif args.model == "gatmann":
        from GATmann import train_epoch, test_epoch, eval_metrics, prepare_dataloaders, get_model, eval_metrics2
        if args.pretrained:
            if args.notox:
                from GATmann import prepare_cross_validation_P1 as prepare_cross_validation
            else:
                from GATmann import prepare_cross_validation_P as prepare_cross_validation
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
    global init, end
    if args.fold == -1:
        init = 0
        end = args.k
    else:
        init = args.fold
        end = args.fold + 1
    config ["pooling"] = "gated"
    train_model(config, k=args.k)
    gc.collect()
