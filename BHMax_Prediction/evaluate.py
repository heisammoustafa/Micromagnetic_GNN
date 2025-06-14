# evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import GNNModel
from sklearn.metrics import r2_score

def evaluate_model(dataset, loader, label_scaler, config, device):
    """
    Loads the best model and evaluates it on the given loader.
    Returns predicted and true values (inverse transformed).
    """
    model = GNNModel(
        orig_node_fea_len=dataset[0].num_node_features,
        edge_fea_len=config["edge_fea_len"],
        node_fea_len=config["node_fea_len"],
        n_conv=config["n_conv"],
        h_fea_len=config["h_fea_len"],
        n_h=config["n_h"]
    ).to(device)

    model.load_state_dict(torch.load(config["save_path"], map_location=device))
    model.eval()

    # Enable dropout layers during evaluation if needed
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    predictions, true_values = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            out_original = label_scaler.inverse_transform(out.cpu().numpy())
            true_original = label_scaler.inverse_transform(data.y.view(-1, 1).cpu().numpy())
            predictions.append(out_original)
            true_values.append(true_original)

    return np.concatenate(predictions), np.concatenate(true_values)

def plot_predictions_vs_true(train_predictions, train_true, test_predictions, test_true):
    """
    Plots predicted vs true values for training and test sets.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(train_predictions, train_true, color="blue", label="Training Data", alpha=0.6)
    plt.scatter(test_predictions, test_true, color="orange", label="Test Data", alpha=0.6)
    min_val = min(train_true.min(), test_true.min())
    max_val = max(train_true.max(), test_true.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal Fit")
    plt.xlabel("Predicted BHMax (kJ/m³)")
    plt.ylabel("True BHMax (kJ/m³)")
    plt.title("True vs Predicted BHMax")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_r2(train_predictions, train_true, test_predictions, test_true):

    r2 = r2_score(train_true, train_predictions)
    print(f"R^2 Score for Train Set: {r2:.4f}")

    r2 = r2_score(test_true, test_predictions)
    print(f"R^2 Score for Test Set: {r2:.4f}")
