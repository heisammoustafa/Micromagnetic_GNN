import torch
from torch.nn import HuberLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

def run_training(model, train_loader, val_loader, optimizer, scheduler, device, config):
    """
    Trains the model with early stopping and learning rate scheduling.
    """
    criterion = HuberLoss()
    patience = config.get("patience", 30)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    save_path = config["save_path"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = evaluate(model, train_loader, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model at epoch {epoch}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered. Restoring best model from epoch {epoch - patience + 1}")
            model.load_state_dict(torch.load(save_path))
            break

def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on a given dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.view(-1, 1))
            total_loss += loss.item()
    return total_loss / len(loader.dataset)
