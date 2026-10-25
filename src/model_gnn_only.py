import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNOnlyModel(nn.Module):
    """
    Baseline model:
    Drug molecule graph -> GCN encoder -> LN_IC50 prediction

    This model does NOT use cell-line information.
    It is used to compare against the hybrid GNN model.
    """

    def __init__(
        self,
        atom_feature_dim=5,
        hidden_dim=64,
        drug_embedding_dim=128,
        dropout=0.2,
    ):
        super().__init__()

        self.gcn1 = GCNConv(atom_feature_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, drug_embedding_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.predictor = nn.Sequential(
            nn.Linear(drug_embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.gcn3(x, edge_index)
        x = F.relu(x)

        drug_embedding = global_mean_pool(x, batch)

        prediction = self.predictor(drug_embedding)

        return prediction


if __name__ == "__main__":
    from pathlib import Path
    from torch_geometric.loader import DataLoader
    from dataset import GDSCGraphDataset, load_cell_line_mapping

    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"

    cell_line_to_idx = load_cell_line_mapping()
    train_dataset = GDSCGraphDataset(DATA_DIR / "train.csv", cell_line_to_idx)

    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = GNNOnlyModel()

    batch = next(iter(loader))
    output = model(batch)

    print("Batch target shape:", batch.y.shape)
    print("Model output shape:", output.shape)
    print("Example predictions:", output[:5].view(-1))