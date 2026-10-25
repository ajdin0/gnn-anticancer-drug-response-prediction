from pathlib import Path
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def atom_features(atom):
    """
    Simple atom-level features for GNN.
    These are numerical features, not one-hot encoded yet.
    """
    return [
        atom.GetAtomicNum(),          # atomic number
        atom.GetDegree(),             # number of directly bonded atoms
        atom.GetFormalCharge(),       # formal charge
        atom.GetTotalNumHs(),         # number of hydrogens
        int(atom.GetIsAromatic()),    # aromatic atom or not
    ]


def smiles_to_graph(smiles):
    """
    Convert SMILES string into PyTorch Geometric graph.
    Returns node features x and edge_index.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))

    x = torch.tensor(x, dtype=torch.float)

    # Edges
    edge_index = []

    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()

        # Add both directions because molecular graph is undirected
        edge_index.append([start, end])
        edge_index.append([end, start])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return x, edge_index


def create_cell_line_mapping():
    """
    Create mapping from cell line name to integer ID.
    This is needed for the cell-line embedding layer.
    """
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    all_cell_lines = pd.concat(
        [
            train_df["CELL_LINE_NAME"],
            val_df["CELL_LINE_NAME"],
            test_df["CELL_LINE_NAME"],
        ]
    ).dropna().unique()

    all_cell_lines = sorted(all_cell_lines)

    mapping = {cell_line: idx for idx, cell_line in enumerate(all_cell_lines)}

    with open(DATA_DIR / "cell_line_to_idx.json", "w") as f:
        json.dump(mapping, f, indent=2)

    return mapping


def load_cell_line_mapping():
    mapping_path = DATA_DIR / "cell_line_to_idx.json"

    if not mapping_path.exists():
        return create_cell_line_mapping()

    with open(mapping_path, "r") as f:
        return json.load(f)


class GDSCGraphDataset(Dataset):
    """
    Dataset for hybrid GNN model.

    Each item contains:
    - molecular graph from SMILES
    - cell line index
    - target LN_IC50
    """

    def __init__(self, csv_path, cell_line_to_idx):
        self.df = pd.read_csv(csv_path)
        self.cell_line_to_idx = cell_line_to_idx
        self.graphs = []

        skipped = 0

        for _, row in self.df.iterrows():
            smiles = row["SMILES"]
            cell_line = row["CELL_LINE_NAME"]
            target = row["LN_IC50"]

            graph = smiles_to_graph(smiles)

            if graph is None:
                skipped += 1
                continue

            if cell_line not in self.cell_line_to_idx:
                skipped += 1
                continue

            x, edge_index = graph

            data = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([target], dtype=torch.float),
                cell_line_idx=torch.tensor(
                    [self.cell_line_to_idx[cell_line]],
                    dtype=torch.long
                ),
            )

            self.graphs.append(data)

        print(f"Loaded {csv_path.name}: {len(self.graphs)} graphs, skipped {skipped}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


if __name__ == "__main__":
    cell_line_to_idx = create_cell_line_mapping()

    print("Number of cell lines:", len(cell_line_to_idx))

    train_dataset = GDSCGraphDataset(DATA_DIR / "train.csv", cell_line_to_idx)
    val_dataset = GDSCGraphDataset(DATA_DIR / "val.csv", cell_line_to_idx)
    test_dataset = GDSCGraphDataset(DATA_DIR / "test.csv", cell_line_to_idx)

    sample = train_dataset[0]

    print("\nExample graph:")
    print("Node feature shape:", sample.x.shape)
    print("Edge index shape:", sample.edge_index.shape)
    print("Cell line index:", sample.cell_line_idx)
    print("Target LN_IC50:", sample.y)