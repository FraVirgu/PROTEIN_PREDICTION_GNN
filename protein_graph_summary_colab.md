# 🧬 Protein Graph GNN Pipeline – Summary

---

## 🔹 What is a Protein Sequence?

A **protein sequence** is a string of letters, each representing an amino acid:

$$
\text{Example: } \texttt{MVTYDFGSDEMHD}
$$

Each letter (e.g., M, V, T) is an **amino acid**, which becomes a **node** in the graph.

---

## 🔹 Goal: Convert a Protein Sequence $\rightarrow$ Graph for GNN

A protein graph has:

- **Nodes**: Amino acids (residues)
- **Edges**: Proximity in 3D structure (from `.pdb`) or sequence-based
- **Node features**: A vector for each amino acid

---

## 🔹 Node Feature Options

### Option 1: One-Hot Encoding

Each amino acid becomes a vector of length 20 (one-hot):

$$
\texttt{A} \rightarrow [1, 0, 0, ..., 0] \in \mathbb{R}^{20}
$$

### Option 2: Physicochemical Properties (from `pcp_dict`)

Each amino acid maps to a 7D vector based on empirical properties:

$$
\texttt{M} \rightarrow [0.64, -0.59, ..., -0.51] \in \mathbb{R}^{7}
$$

### Option 3: SeqVec Embedding

Each amino acid gets a 1024D embedding from a language model:

$$
\texttt{embedding = SeqVecEmbedder().embed(seq)} \Rightarrow \text{shape } [L, 1024]
$$

---

## 🔹 Example: From Sequence to Node Features (with `pcp_dict`)

```python
sequence = "MVTYDFGSDEMHD"
features = [pcp_dict[res] for res in sequence]
node_features = torch.tensor(features, dtype=torch.float)  # shape [L, 7]
```

---

## 🔹 Example: Using SeqVec Embedding

```python
from bio_embeddings.embed import SeqVecEmbedder
sequence = "MVTYDFGSDEMHD"
embedder = SeqVecEmbedder()
embedding = embedder.embed(sequence)  # shape: [L, 1024]
node_features = torch.tensor(embedding, dtype=torch.float)
```

---

## 🔹 Create Graph Object (for PyTorch Geometric)

```python
from torch_geometric.data import Data

# Assume edge_index is precomputed or based on contact map
graph = Data(x=node_features, edge_index=edge_index)
```

- $x$ = node features $\in \mathbb{R}^{L \times d}$
- $edge\_index$ = edge list, shape $[2, num\_edges]$

---

## ✅ Summary Table

| Component       | What It Represents            | Shape           |
| --------------- | ----------------------------- | --------------- |
| `sequence`      | String of amino acids         | L (length)      |
| `node_features` | Feature matrix for nodes      | $[L, d]$        |
| `edge_index`    | Graph connectivity            | $[2, E]$        |
| `Data` object   | Final PyTorch Geometric graph | GNN-ready input |

---

> ℹ️ Use `.pt` files to save each protein graph with `torch.save(data, "path.pt")`, and load pairs with labels for training.

# 🧩 `LabelledDataset` — GNN Input for Protein Pairs

This custom dataset class is designed to **load pairs of protein graphs** and provide a **label** for each pair. It works perfectly with PyTorch's `DataLoader` for batching during GNN training.

---

## 🔹 Purpose

To support a GNN model that learns **interactions or relationships between two proteins** (e.g. binding, classification).

Each sample contains:

- A graph for **Protein 1**
- A graph for **Protein 2**
- A **label** (e.g., 0 or 1)

---

## 🔍 Class Breakdown

### `__init__`

```python
def __init__(self, npy_file, processed_dir):
    self.npy_ar = np.load(npy_file)  # Load [num_samples, features]
    self.processed_dir = processed_dir
    self.protein_1 = self.npy_ar[:,2]  # File names for Protein 1
    self.protein_2 = self.npy_ar[:,5]  # File names for Protein 2
    self.label = self.npy_ar[:,6].astype(float)  # Labels
    self.n_samples = self.npy_ar.shape[0]
```

---

### `__len__`

```python
def __len__(self):
    return self.n_samples
```

Returns the number of protein pairs in the dataset.

---

### `__getitem__`

```python
def __getitem__(self, index):
    prot_1 = os.path.join(self.processed_dir, self.protein_1[index] + ".pt")
    prot_2 = os.path.join(self.processed_dir, self.protein_2[index] + ".pt")
    prot_1 = torch.load(glob.glob(prot_1)[0])
    prot_2 = torch.load(glob.glob(prot_2)[0])
    return prot_1, prot_2, torch.tensor(self.label[index])
```

- Loads `.pt` files (saved earlier as PyTorch Geometric `Data` objects).
- Each contains:
  - `x`: Node features $\in \mathbb{R}^{L \times D}$
  - `edge_index`: Graph edges

---

## 🔁 Usage with `DataLoader`

```python
trainloader = DataLoader(dataset, batch_size=4)
for prot1, prot2, label in trainloader:
    # Pass into your GNN model
```

---

## 📦 Example Output Per Sample

| Element      | Shape or Type    | Description                |
| ------------ | ---------------- | -------------------------- |
| `prot_1.x`   | $[L_1, D]$       | Node features of protein 1 |
| `prot_2.x`   | $[L_2, D]$       | Node features of protein 2 |
| `edge_index` | $[2, E]$         | Graph edges                |
| `label`      | `float` or `int` | Ground-truth output        |

---

## ✅ Summary

- Loads **protein graphs** from `.pt` files
- Uses `.npy` file to link pairs and labels
- Perfect for training **GNNs on protein–protein tasks**
