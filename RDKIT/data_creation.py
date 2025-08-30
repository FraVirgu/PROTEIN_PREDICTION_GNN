# ==========================
# Imports
# ==========================
from tdc.multi_pred import DDI
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================
# Step 1. Load + binarize DDI
# ==========================
data = DDI(name='DrugBank')
split = data.get_split()

# 1 = has interaction, 0 = no interaction
for key in ['train', 'valid', 'test']:
    # If Y is already numeric, this keeps 0/1; if it's strings, this maps non-null to 1
    split[key]['Y'] = split[key]['Y'].apply(lambda x: int(x != 'null'))

print("Preview of the TRAIN split:")
print(split['train'].head())

# ==========================
# Step 2. SMILES → Graph (Adj + Node Feats)
# ==========================
def smiles_to_graph(smiles: str, max_nodes: int = 128):
    """
    Convert a SMILES string into:
    - padded adjacency matrix [max_nodes, max_nodes]
    - padded node feature matrix [max_nodes, F]
    
    If the molecule has more than `max_nodes` atoms → return (None, None).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Build adjacency
    adj = rdmolops.GetAdjacencyMatrix(mol).astype(np.float32)

    # Build features
    feats = []
    for atom in mol.GetAtoms():
        feats.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic())
        ])
    node_feats = np.asarray(feats, dtype=np.float32)

    n_nodes = adj.shape[0]
    feat_dim = node_feats.shape[1] if n_nodes > 0 else 4

    # Skip if too big
    if n_nodes > max_nodes:
        return None, None

    # Pad adjacency
    adj_padded = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    adj_padded[:n_nodes, :n_nodes] = adj

    # Pad features
    feat_padded = np.zeros((max_nodes, feat_dim), dtype=np.float32)
    feat_padded[:n_nodes, :] = node_feats

    return adj_padded, feat_padded

# Optional helper to quickly filter bad strings
def is_valid_smiles(s: str) -> bool:
    return Chem.MolFromSmiles(s) is not None

# ==========================
# Step 3. Build Pairs → Graphs
# ==========================

# Worker function (must be top-level so it can be pickled)
def process_row(row):
    s1 = row['Drug1']
    s2 = row['Drug2']
    y  = int(row['Y'])

    a_adj, a_feat = smiles_to_graph(s1)
    b_adj, b_feat = smiles_to_graph(s2)

    if a_adj is None or b_adj is None:
        return None  # skip invalid
    return [s1, s2, a_adj, a_feat, b_adj, b_feat, y]


def build_dataset_from_split(df: pd.DataFrame, max_rows: int | None = None, shuffle: bool = True, n_jobs: int = None):
    """
    Parallel dataset builder using ProcessPoolExecutor.
    """
    cols_needed = {'Drug1', 'Drug2', 'Y'}
    if not cols_needed.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns {cols_needed}, got {df.columns.tolist()}")

    work = df.copy()
    if shuffle:
        work = work.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if max_rows is not None:
        work = work.iloc[:max_rows].reset_index(drop=True)

    print(f"🔄 Building graphs for {len(work)} pairs in parallel...")

    dataset = []
    skipped = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(process_row, row): i for i, row in work.iterrows()}

        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is None:
                skipped += 1
                continue
            dataset.append(result)

            if (idx + 1) % 1000 == 0:
                print(f"  ...processed {idx+1}/{len(work)} rows")

    print(f"✅ Done. Built {len(dataset)} samples, skipped {skipped} invalid rows.")
    return dataset
# ==========================
# Step 4. Build train/valid/test datasets
# ==========================
train_data = build_dataset_from_split(split['train'], max_rows=5000, n_jobs=8)   # use 8 workers
valid_data = build_dataset_from_split(split['valid'], max_rows=2000, n_jobs=8)
test_data  = build_dataset_from_split(split['test'],  max_rows=2000, n_jobs=8)
print(f"Train: {len(train_data)} | Valid: {len(valid_data)} | Test: {len(test_data)}")


print(f"Train samples: {len(train_data)}")
print(f"Valid samples: {len(valid_data)}")
print(f"Test samples:  {len(test_data)}")

# ==========================
# (Optional) Example peek
# ==========================
if len(train_data) > 0:
    s1, s2, adj1, feat1, adj2, feat2, y = train_data[0]
    print("Example label:", y)
    print("Drug1 SMILES:", s1, " | adj:", adj1.shape, " feats:", feat1.shape)
    print("Drug2 SMILES:", s2, " | adj:", adj2.shape, " feats:", feat2.shape)
