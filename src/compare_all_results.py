from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

files = {
    "Warm GNN-only": RESULTS_DIR / "metrics_gnn_only.json",
    "Warm Hybrid GNN": RESULTS_DIR / "metrics_hybrid_gnn.json",
    "Cold-drug Hybrid GNN": RESULTS_DIR / "metrics_cold_drug_hybrid_gnn.json",
}

rows = []

for model_name, path in files.items():
    with open(path, "r") as f:
        metrics = json.load(f)

    rows.append({
        "Model / setting": model_name,
        "Test RMSE": metrics["test_rmse"],
        "Test MAE": metrics["test_mae"],
        "Test Pearson": metrics["test_pearson"],
    })

comparison = pd.DataFrame(rows)

comparison.to_csv(RESULTS_DIR / "all_model_comparison.csv", index=False)

print("\nAll model comparison:")
print(comparison)

# Bar chart for RMSE
plt.figure(figsize=(8, 5))
plt.bar(comparison["Model / setting"], comparison["Test RMSE"])
plt.ylabel("Test RMSE")
plt.title("Model Comparison: Test RMSE")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "all_models_rmse_comparison.png", dpi=300)
plt.close()

# Bar chart for MAE
plt.figure(figsize=(8, 5))
plt.bar(comparison["Model / setting"], comparison["Test MAE"])
plt.ylabel("Test MAE")
plt.title("Model Comparison: Test MAE")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "all_models_mae_comparison.png", dpi=300)
plt.close()

# Bar chart for Pearson
plt.figure(figsize=(8, 5))
plt.bar(comparison["Model / setting"], comparison["Test Pearson"])
plt.ylabel("Test Pearson correlation")
plt.title("Model Comparison: Test Pearson Correlation")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "all_models_pearson_comparison.png", dpi=300)
plt.close()

print("\nSaved:")
print("results/all_model_comparison.csv")
print("results/all_models_rmse_comparison.png")
print("results/all_models_mae_comparison.png")
print("results/all_models_pearson_comparison.png")