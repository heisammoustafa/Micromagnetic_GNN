# main.py

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import GNNModel
from train import run_training
from evaluate import evaluate_model, plot_predictions_vs_true, get_r2

# load datasets
from data_loader_ood import dataset, train_loader, val_loader, test_loader, label_scaler

def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    torch.manual_seed(0)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = GNNModel(
        orig_node_fea_len=dataset[0].num_node_features,
        edge_fea_len=config["edge_fea_len"],
        node_fea_len=config["node_fea_len"],
        n_conv=config["n_conv"],
        h_fea_len=config["h_fea_len"],
        n_h=config["n_h"]
    ).to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4, verbose=True)

    # Train the model
    run_training(model, train_loader, val_loader, optimizer, scheduler, device, config)

    # Evaluate the model
    train_preds, train_true = evaluate_model(dataset, train_loader, label_scaler, config, device)
    test_preds, test_true = evaluate_model(dataset, test_loader, label_scaler, config, device)

    # Plot predictions
    plot_predictions_vs_true(train_preds, train_true, test_preds, test_true)

    get_r2(train_preds, train_true, test_preds, test_true)

if __name__ == "__main__":
    main()
