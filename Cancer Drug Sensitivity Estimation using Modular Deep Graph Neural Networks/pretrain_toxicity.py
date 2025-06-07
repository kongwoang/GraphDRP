from GATR import prepare_cross_validation, get_model, train_epoch, test_epoch
import json
import torch
import numpy as np

if __name__ == '__main__':
    with open("params/optimal_config_GATR.json", "r") as f:
        config = json.load(f)
    partitions = prepare_cross_validation(config)
    device = torch.device('cuda')
    loss_fn = torch.nn.CrossEntropyLoss
    best_loss = 1000
    patience = 10
    best_epoch = -1
    for epoch in range(200):
        val_losses = []
        train_losses = []
        if patience <= 0:
            break
        for fold in range(len(partitions)):
            if fold > 3:
                break
            partitions[fold]["model"].to(device)
            train_loss = train_epoch(device,
                                     loss_fn,
                                     partitions[fold]["train_dataloader"], 
                                     partitions[fold]["model"],
                                     partitions[fold]["optimizer"],)
            val_loss = train_epoch(device,
                                     loss_fn,
                                     partitions[fold]["val_dataloader"], 
                                     partitions[fold]["model"],
                                     partitions[fold]["optimizer"],)
            train_losses += [train_loss]
            val_losses += [val_loss]
        partitions[fold]["model"].to(torch.device("cpu"))
        if (np.mean(val_losses) + np.std(val_losses)) < best_loss:
            best_loss = np.mean(val_losses) + np.std(val_losses)
            best_state = model.state_dict()
            patience = 10
            best_epoch = epoch
        else:
            patience -= 1
        if patience <= 0:
            break
        print(f"epoch {epoch} train_loss {np.mean(train_losses)} ± {np.std(train_losses)} test_loss {np.mean(val_losses)} ± {np.std(val_losses)}")
    torch.save(best_state, "trained_models/GATR_pretrained.pth")
    print(f"training completed, saving model from epoch {best_epoch}")