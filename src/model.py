import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class HybridGNNModel(nn.Module):
    """
    Hybrid model:
    Drug molecule graph -> GCN encoder
    Cell line ID -> embedding layer
    Drug embedding + cell-line embedding -> LN_IC50 prediction
    """

    def __init__(
        self,
        num_cell_lines,
        atom_feature_dim=5,
        hidden_dim=64,
        drug_embedding_dim=128,
        cell_embedding_dim=64,
        dropout=0.2,
    ):
        super().__init__()

        # GNN drug encoder
        self.gcn1 = GCNConv(atom_feature_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, drug_embedding_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Cell-line encoder
        self.cell_embedding = nn.Embedding(num_cell_lines, cell_embedding_dim)

        # Fusion + prediction
        fusion_dim = drug_embedding_dim + cell_embedding_dim

        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
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

        # Drug graph branch
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.gcn3(x, edge_index)
        x = F.relu(x)

        # Convert node embeddings to one graph embedding per molecule
        drug_embedding = global_mean_pool(x, batch)

        # Cell-line branch
        cell_idx = data.cell_line_idx.view(-1)
        cell_embedding = self.cell_embedding(cell_idx)

        # Hybrid fusion
        combined = torch.cat([drug_embedding, cell_embedding], dim=1)

        # Predict LN_IC50
        prediction = self.predictor(combined)

        return prediction


if __name__ == "__main__":
    from pathlib import Path
    from torch_geometric.loader import DataLoader
    from dataset import GDSCGraphDataset, load_cell_line_mapping

    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"

    cell_line_to_idx = load_cell_line_mapping()
    num_cell_lines = len(cell_line_to_idx)

    train_dataset = GDSCGraphDataset(DATA_DIR / "train.csv", cell_line_to_idx)

    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = HybridGNNModel(num_cell_lines=num_cell_lines)

    batch = next(iter(loader))
    output = model(batch)

    print("Number of cell lines:", num_cell_lines)
    print("Batch target shape:", batch.y.shape)
    print("Model output shape:", output.shape)
    print("Example predictions:", output[:5].view(-1))