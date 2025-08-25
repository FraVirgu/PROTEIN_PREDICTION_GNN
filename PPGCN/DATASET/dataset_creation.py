import os
import requests
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from esm.pretrained import esm2_t33_650M_UR50D
from Bio.PDB import PDBParser

# --------------------------
# CONFIGURATION
# --------------------------
POS_FILE = "PPGCN/DATASET/Real-Datasets/H. sapien/H. sapien_Positive_Real.xlsx"
NEG_FILE = "PPGCN/DATASET/Real-Datasets/H. sapien/H. sapien_Negative_Real.xlsx"
MAX_PAIRS = 500
UNIPROT_MIN_LEN = 10
SAVE_DIR = "PPGCN/DATASET/graphs"

PDB_DIR = "alphafold_pdbs"
DISTANCE_THRESHOLD = 8.0  # Ångstroms

# --------------------------
# LOAD UNIProt ID PAIRS
# --------------------------
def load_labeled_uniprot_pairs(pos_file, neg_file, max_pairs):
    pos_df = pd.read_excel(pos_file).head(max_pairs)
    neg_df = pd.read_excel(neg_file).head(max_pairs)

    pos_df.columns = ["gene_a", "gene_b"]
    neg_df.columns = ["gene_a", "gene_b"]

    pos_df["label"] = 1
    neg_df["label"] = 0

    df = pd.concat([pos_df, neg_df], ignore_index=True)
    uniprot_ids = set(df["gene_a"].astype(str)).union(df["gene_b"].astype(str))
    print(f"✅ Loaded {len(df)} pairs ({len(pos_df)} positive, {len(neg_df)} negative), {len(uniprot_ids)} unique UniProt IDs")
    return df, uniprot_ids

# --------------------------
# DOWNLOAD ALPHAFOLD PDB STRUCTURE
# --------------------------
def download_alphafold_pdb(uniprot_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{uniprot_id}.pdb")
    if os.path.exists(out_path):
        return out_path
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    r = requests.get(url)
    if r.status_code == 200:
        with open(out_path, "w") as f:
            f.write(r.text)
        print(f"📥 Downloaded AlphaFold structure for {uniprot_id}")
        return out_path
    else:
        print(f"[WARN] AlphaFold structure not found for {uniprot_id}")
        return None

# --------------------------
# EXTRACT Cα COORDINATES FROM PDB
# --------------------------
def extract_ca_coordinates(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
    return np.array(coords)

# --------------------------
# BUILD CONTACT EDGE INDEX BASED ON DISTANCE
# --------------------------
def build_contact_edges(ca_coords, distance_threshold=8.0):
    edge_index = []
    n = len(ca_coords)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if dist <= distance_threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
    return torch.tensor(edge_index).t().contiguous()

# --------------------------
# SEQUENCE TO STRUCTURE-AWARE GRAPH
# --------------------------
def sequence_to_graph(uniprot_id, sequence, model, batch_converter, pdb_dir):
    # Get ESM embeddings
    data = [(uniprot_id, sequence)]
    _, _, tokens = batch_converter(data)
    with torch.no_grad():
        results = model(tokens, repr_layers=[33])
    embeddings = results["representations"][33][0][1:-1]  # exclude CLS/EOS

    # Load AlphaFold structure
    pdb_file = download_alphafold_pdb(uniprot_id, pdb_dir)
    if not pdb_file:
        raise FileNotFoundError(f"Structure for {uniprot_id} not found.")

    # Extract Cα coordinates
    ca_coords = extract_ca_coordinates(pdb_file)
    if len(ca_coords) != embeddings.shape[0]:
        raise ValueError(f"Sequence/structure mismatch for {uniprot_id} — sequence {embeddings.shape[0]}, structure {len(ca_coords)}")

    edge_index = build_contact_edges(ca_coords, DISTANCE_THRESHOLD)
    return Data(x=embeddings, edge_index=edge_index, id=uniprot_id)

# --------------------------
# GET UNIPROT SEQUENCE
# --------------------------
def get_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        lines = r.text.strip().split("\n")
        return "".join(lines[1:])
    except Exception:
        return None

# --------------------------
# SAVE GRAPH TO TXT FILE
# --------------------------
def save_graph_to_txt(graph: Data, directory: str):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{graph.id}.txt")

    num_nodes = graph.num_nodes
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int)
    adj_matrix[graph.edge_index[0], graph.edge_index[1]] = 1
    embeddings = graph.x.cpu().numpy()

    with open(path, "w") as f:
        f.write("# Adjacency Matrix (NxN)\n")
        for row in adj_matrix.tolist():
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n# Node Embeddings (NxD)\n")
        for vec in embeddings:
            f.write(" ".join(f"{v:.6f}" for v in vec) + "\n")

    print(f"💾 Saved graph to {path}")

# --------------------------
# SAVE UNIProt PAIRS WITH LABEL
# --------------------------
def save_labeled_pairs(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    pos_path = os.path.join(output_dir, "positive_pairs.txt")
    neg_path = os.path.join(output_dir, "negative_pairs.txt")

    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]

    with open(pos_path, "w") as f_pos:
        for _, row in pos_df.iterrows():
            f_pos.write(f"{row['gene_a']} {row['gene_b']} 1\n")

    with open(neg_path, "w") as f_neg:
        for _, row in neg_df.iterrows():
            f_neg.write(f"{row['gene_a']} {row['gene_b']} 0\n")

    print(f"💾 Saved {len(pos_df)} positive pairs to {pos_path}")
    print(f"💾 Saved {len(neg_df)} negative pairs to {neg_path}")

# --------------------------
# MAIN EXECUTION
# --------------------------
def main():
    df, uniprot_ids = load_labeled_uniprot_pairs(POS_FILE, NEG_FILE, MAX_PAIRS)

    print("📦 Loading ESM-2 model...")
    model, alphabet = esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    graphs = []
    processed_ids = set()

    print("🧪 Building structure-aware graphs...")
    for uniprot in tqdm(uniprot_ids):
        if uniprot in processed_ids:
            continue

        sequence = get_uniprot_sequence(uniprot)
        if not sequence or len(sequence) < UNIPROT_MIN_LEN:
            continue

        try:
            graph = sequence_to_graph(uniprot, sequence, model, batch_converter, PDB_DIR)
            graphs.append(graph)
            processed_ids.add(uniprot)
            save_graph_to_txt(graph, SAVE_DIR)
            print(f"✅ Graph for {uniprot} ({len(sequence)} residues)")
        except Exception as e:
            print(f"[ERROR] {uniprot}: {e}")

    print(f"\n✅ Total graphs created: {len(graphs)}")

    # Save combined labeled pairs
    save_labeled_pairs(df, SAVE_DIR)

if __name__ == "__main__":
    main()
