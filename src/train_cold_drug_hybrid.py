from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from dataset import GDSCGraphDataset
from model import HybridGNNModel


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5


def create_cell_line_mapping_for_cold_split():
    train_df = pd.read_csv(DATA_DIR / "cold_drug_train.csv")
    val_df = pd.read_csv(DATA_DIR / "cold_drug_val.csv")
    test_df = pd.read_csv(DATA_DIR / "cold_drug_test.csv")

    all_cell_lines = pd.concat([
        train_df["CELL_LINE_NAME"],
        val_df["CELL_LINE_NAME"],
        test_df["CELL_LINE_NAME"],
    ]).dropna().unique()

    all_cell_lines = sorted(all_cell_lines)
    return {cell_line: idx for idx, cell_line in enumerate(all_cell_lines)}


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        preds = model(batch)
        target = batch.y.view(-1, 1)

        loss = loss_fn(preds, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, loss_fn):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)

            preds = model(batch)
            target = batch.y.view(-1, 1)

            loss = loss_fn(preds, target)
            total_loss += loss.item() * batch.num_graphs

            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mse = total_loss / len(loader.dataset)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(all_preds - all_targets)))

    if len(all_preds) > 1:
        pearson = float(np.corrcoef(all_preds, all_targets)[0, 1])
    else:
        pearson = 0.0

    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "preds": all_preds,
        "targets": all_targets,
    }


def plot_predictions(targets, preds, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.4, s=10)
    plt.xlabel("Actual LN_IC50")
    plt.ylabel("Predicted LN_IC50")
    plt.title("Cold-drug Hybrid GNN: Predicted vs Actual LN_IC50")
    plt.grid(True)

    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_history(history, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Cold-drug Hybrid GNN Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    print("Using device:", DEVICE)

    cell_line_to_idx = create_cell_line_mapping_for_cold_split()
    num_cell_lines = len(cell_line_to_idx)

    train_dataset = GDSCGraphDataset(DATA_DIR / "cold_drug_train.csv", cell_line_to_idx)
    val_dataset = GDSCGraphDataset(DATA_DIR / "cold_drug_val.csv", cell_line_to_idx)
    test_dataset = GDSCGraphDataset(DATA_DIR / "cold_drug_test.csv", cell_line_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = HybridGNNModel(num_cell_lines=num_cell_lines).to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_mae": [],
        "val_pearson": [],
    }

    best_val_rmse = float("inf")
    best_model_path = RESULTS_DIR / "best_cold_drug_hybrid_gnn_model.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_metrics = evaluate(model, val_loader, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["mse"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_pearson"].append(val_metrics["pearson"])

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val Pearson: {val_metrics['pearson']:.4f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save(model.state_dict(), best_model_path)
            print("  Saved best model")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, loss_fn)

    print("\nFinal cold-drug test results:")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test Pearson: {test_metrics['pearson']:.4f}")

    metrics_to_save = {
        "model": "Cold-drug Hybrid GNN: GCN drug encoder + cell-line embedding",
        "split_type": "cold-drug",
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_cell_lines": num_cell_lines,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "best_val_rmse": best_val_rmse,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_pearson": test_metrics["pearson"],
        "history": history,
    }

    with open(RESULTS_DIR / "metrics_cold_drug_hybrid_gnn.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    plot_predictions(
        test_metrics["targets"],
        test_metrics["preds"],
        RESULTS_DIR / "cold_drug_hybrid_predicted_vs_actual_ln_ic50.png",
    )

    plot_training_history(
        history,
        RESULTS_DIR / "cold_drug_hybrid_training_history.png",
    )

    print("\nSaved results:")
    print("results/best_cold_drug_hybrid_gnn_model.pth")
    print("results/metrics_cold_drug_hybrid_gnn.json")
    print("results/cold_drug_hybrid_predicted_vs_actual_ln_ic50.png")
    print("results/cold_drug_hybrid_training_history.png")


if __name__ == "__main__":
    main()