from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

with open(RESULTS_DIR / "metrics_hybrid_gnn.json", "r") as f:
    hybrid = json.load(f)

with open(RESULTS_DIR / "metrics_gnn_only.json", "r") as f:
    gnn_only = json.load(f)

comparison = pd.DataFrame([
    {
        "Model": "GNN-only",
        "Description": "Drug molecular graph only",
        "Test RMSE": gnn_only["test_rmse"],
        "Test MAE": gnn_only["test_mae"],
        "Test Pearson": gnn_only["test_pearson"],
    },
    {
        "Model": "Hybrid GNN",
        "Description": "Drug molecular graph + cell-line embedding",
        "Test RMSE": hybrid["test_rmse"],
        "Test MAE": hybrid["test_mae"],
        "Test Pearson": hybrid["test_pearson"],
    },
])

comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

print(comparison)

# Plot comparison
metrics = ["Test RMSE", "Test MAE", "Test Pearson"]

for metric in metrics:
    plt.figure(figsize=(6, 4))
    plt.bar(comparison["Model"], comparison[metric])
    plt.ylabel(metric)
    plt.title(f"Model Comparison: {metric}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    filename = metric.lower().replace(" ", "_") + "_comparison.png"
    plt.savefig(RESULTS_DIR / filename, dpi=300)
    plt.close()

print("\nSaved:")
print("results/model_comparison.csv")
print("results/test_rmse_comparison.png")
print("results/test_mae_comparison.png")
print("results/test_pearson_comparison.png")