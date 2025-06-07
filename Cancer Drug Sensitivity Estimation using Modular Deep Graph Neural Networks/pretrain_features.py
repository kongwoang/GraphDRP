from GATF import prepare_dataloaders, get_model, train_epoch, test_epoch
import json
import torch

if __name__ == '__main__':
    train_dataloader, test_dataloader = prepare_dataloaders(train_batch = 1024) # can be reduced to make the model fit smaller CPUs, performance might change.
    with open("params/optimal_config_GATF.json", "r") as f:
        config = json.load(f)
    model = get_model(**config)
    device = torch.device('cuda')
    model.to(device)
    loss_fn = torch.nn.MSELoss
    optimizer = torch.optim.Adam(model.parameters(),
                                 config["learning_rate"],
                                 betas = [config["beta_1"], config["beta_1"]],
                                 weight_decay = config["weight_decay"])
    for epoch in range(1000):
        model.train()
        train_loss = train_epoch(device, loss_fn, train_dataloader, model, optimizer)
        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(device, loss_fn, test_dataloader, model)
        if (epoch+100)%1 == 0:
            print(f"epoch {epoch}: training loss: {train_loss}, test loss {test_loss}")
    print("training completed!")
    torch.save(model.state_dict(), "trained_models/best_GATF.pth")