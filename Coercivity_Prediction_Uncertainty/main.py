import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import GNNModel  # Make sure this matches the model in evaluate.py
from train import run_training
from evaluate import (
    load_model,
    evaluate,
    evaluate_with_uncertainty,
    plot_predictions_with_uncertainty
)
from data_loader import dataset, train_loader as loader_tr, test_loader as loader_te, label_scaler
from sklearn.metrics import r2_score

def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set seed and device
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Train model ===
    model = GNNModel(
        orig_node_fea_len=dataset[0].num_node_features,
        edge_fea_len=config["edge_fea_len"],
        node_fea_len=config["node_fea_len"],
        n_conv=config["n_conv"],
        h_fea_len=config["h_fea_len"],
        n_h=config["n_h"]
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4, verbose=True)

    run_training(model, loader_tr, loader_te, optimizer, scheduler, device, config)

    # Save model
    torch.save(model.state_dict(), "weights_uncertainty.pt")

    # === Evaluation ===
    model = load_model("weights_uncertainty.pt", input_dim=dataset[0].num_node_features, edge_dim=config["edge_fea_len"])
    model.to(device)

    # Evaluate without uncertainty
    train_preds, train_true = evaluate(loader_tr, model, label_scaler)
    test_preds, test_true = evaluate(loader_te, model, label_scaler)
    print(f"R^2 Score (Train): {r2_score(train_true, train_preds):.4f}")
    print(f"R^2 Score (Test):  {r2_score(test_true, test_preds):.4f}")

    # Evaluate with uncertainty
    num_samples = config.get("mc_dropout_samples", 50)
    train_preds_u, train_true_u, train_epi, train_alea = evaluate_with_uncertainty(
        loader_tr, model, num_samples, device, label_scaler
    )
    test_preds_u, test_true_u, test_epi, test_alea = evaluate_with_uncertainty(
        loader_te, model, num_samples, device, label_scaler
    )

    # Plot uncertainty
    plot_predictions_with_uncertainty(train_preds_u, train_true_u, train_epi, train_alea,
                                      label='Train', save_path='train_uncertainty.svg')
    plot_predictions_with_uncertainty(test_preds_u, test_true_u, test_epi, test_alea,
                                      label='Test', save_path='test_uncertainty.svg')

if __name__ == "__main__":
    main()
