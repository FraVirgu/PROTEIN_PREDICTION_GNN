# data_creation.py
from tdc.multi_pred import DDI
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_NUM_NODES = 128

data = DDI(name='DrugBank')
split = data.get_split()
for key in ['train', 'valid', 'test']:
    split[key]['Y'] = split[key]['Y'].apply(lambda x: int(x != 'null'))

def smiles_to_graph(smiles: str, max_nodes: int = MAX_NUM_NODES):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    adj = rdmolops.GetAdjacencyMatrix(mol).astype(np.float32)
    feats = [[a.GetAtomicNum(), a.GetDegree(), a.GetTotalNumHs(), int(a.GetIsAromatic())]
             for a in mol.GetAtoms()]
    feats = np.asarray(feats, dtype=np.float32)
    n = adj.shape[0]
    if n > max_nodes:
        return None, None
    adj_pad = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    adj_pad[:n, :n] = adj
    feat_pad = np.zeros((max_nodes, feats.shape[1] if n > 0 else 4), dtype=np.float32)
    feat_pad[:n, :] = feats
    return adj_pad, feat_pad

def process_row(row):
    s1, s2, y = row['Drug1'], row['Drug2'], int(row['Y'])
    a_adj, a_feat = smiles_to_graph(s1)
    b_adj, b_feat = smiles_to_graph(s2)
    if a_adj is None or b_adj is None:
        return None
    return [s1, s2, a_adj, a_feat, b_adj, b_feat, y]

def build_dataset_from_split(df: pd.DataFrame, max_rows=None, shuffle=True, n_jobs=None):
    need = {'Drug1','Drug2','Y'}
    if not need.issubset(df.columns):
        raise ValueError(f"DataFrame must contain {need}, got {df.columns.tolist()}")
    work = df.copy()
    if shuffle: work = work.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if max_rows is not None: work = work.iloc[:max_rows].reset_index(drop=True)

    print(f"🔄 Building graphs for {len(work)} pairs in parallel...")
    dataset, skipped = [], 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(process_row, row) for _, row in work.iterrows()]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r is None: skipped += 1
            else: dataset.append(r)
            if i % 1000 == 0 or i == len(futures):
                print(f"  ...processed {i}/{len(futures)} rows")
    print(f"✅ Done. Built {len(dataset)} samples, skipped {skipped} invalid rows.")
    return dataset

def get_datasets(max_train=5000, max_valid=2000, max_test=2000, n_jobs=8):
    train = build_dataset_from_split(split['train'], max_rows=max_train, n_jobs=n_jobs)
    valid = build_dataset_from_split(split['valid'], max_rows=max_valid, n_jobs=n_jobs)
    test  = build_dataset_from_split(split['test'],  max_rows=max_test,  n_jobs=n_jobs)
    return train, valid, test, MAX_NUM_NODES

if __name__ == "__main__":
    # CLI run (so you still get the immediate result when running the file directly)
    tr, va, te, _N = get_datasets(n_jobs=8)
    print(f"Train: {len(tr)} | Valid: {len(va)} | Test: {len(te)}")
