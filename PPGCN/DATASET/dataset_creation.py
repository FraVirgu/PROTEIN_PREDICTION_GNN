import os
import gzip
import requests
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from esm.pretrained import esm2_t33_650M_UR50D

# --------------------------
# CONFIGURATION
# --------------------------
PPI_FILE = "DATASET/PP-Pathways_ppi.csv.gz"
MAX_PAIRS = 50
UNIPROT_MIN_LEN = 10
SAVE_DIR = "DATASET/graphs"

# --------------------------
# LOAD PPI FILE
# --------------------------
def load_ppi_data(ppi_file, max_pairs):
    with gzip.open(ppi_file, "rt") as f:
        df = pd.read_csv(f)
    df.columns = ["gene_a", "gene_b"]
    df = df.head(max_pairs)
    gene_ids = set(df["gene_a"].astype(str)).union(df["gene_b"].astype(str))
    print(f"✅ Loaded {len(df)} PPI pairs, {len(gene_ids)} unique gene IDs")
    return df, gene_ids

# --------------------------
# ENTREZ → UNIPROT VIA MYGENE.INFO
# --------------------------
def entrez_to_uniprot(entrez_ids):
    print("🔁 Mapping Entrez IDs to UniProt IDs via MyGene.info...")
    url = "https://mygene.info/v3/query"
    id_map = {}
    for entrez_id in entrez_ids:
        params = {
            "q": entrez_id,
            "scopes": "entrezgene",
            "fields": "uniprot",
            "species": "human"
        }
        try:
            r = requests.get(url, params=params)
            if r.status_code != 200:
                continue
            hits = r.json().get("hits", [])
            if not hits:
                continue
            hit = hits[0]
            if "uniprot" in hit:
                up = hit["uniprot"]
                if isinstance(up, dict):
                    uniprot_id = up.get("Swiss-Prot") or up.get("TrEMBL")
                    if isinstance(uniprot_id, list):
                        uniprot_id = uniprot_id[0]
                    if uniprot_id:
                        id_map[str(entrez_id)] = uniprot_id
        except Exception:
            continue
    print(f"✅ Mapped {len(id_map)} Entrez IDs to UniProt (via MyGene.info)")
    return id_map

# --------------------------
# DOWNLOAD SEQUENCE FROM UNIPROT
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
# BUILD GRAPH FROM SEQUENCE
# --------------------------
def sequence_to_graph(uniprot_id, sequence, model, batch_converter):
    data = [(uniprot_id, sequence)]
    _, _, tokens = batch_converter(data)
    with torch.no_grad():
        results = model(tokens, repr_layers=[33])
    embeddings = results["representations"][33][0][1:-1]  # exclude CLS and EOS

    edge_index = []
    for i in range(len(embeddings) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index = torch.tensor(edge_index).t().contiguous()

    return Data(x=embeddings, edge_index=edge_index, id=uniprot_id)

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
# SAVE PPI PAIRS TO FILE
# --------------------------
def save_uniprot_pairs(df, id_map, output_file):
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            a = id_map.get(str(row["gene_a"]))
            b = id_map.get(str(row["gene_b"]))
            if a and b:
                f.write(f"{a} {b}\n")
    print(f"💾 Saved UniProt pairs to {output_file}")

# --------------------------
# MAIN EXECUTION
# --------------------------
def main():
    df, gene_ids = load_ppi_data(PPI_FILE, MAX_PAIRS)
    id_map = entrez_to_uniprot(list(gene_ids))

    print("📦 Loading ESM-2 model...")
    model, alphabet = esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    graphs = []
    processed_ids = set()

    print("🧪 Building graphs from UniProt sequences...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        for gene_id in [str(row["gene_a"]), str(row["gene_b"])]:
            if gene_id not in id_map:
                continue
            uniprot = id_map[gene_id]
            if uniprot in processed_ids:
                continue

            sequence = get_uniprot_sequence(uniprot)
            if not sequence or len(sequence) < UNIPROT_MIN_LEN:
                continue

            try:
                graph = sequence_to_graph(uniprot, sequence, model, batch_converter)
                graphs.append(graph)
                processed_ids.add(uniprot)
                save_graph_to_txt(graph, SAVE_DIR)
                print(f"✅ Graph for {uniprot} ({len(sequence)} residues)")
            except Exception as e:
                print(f"[ERROR] {uniprot}: {e}")

    print(f"\n✅ Total graphs created: {len(graphs)}")

    # Save UniProt pairs file
    save_uniprot_pairs(df, id_map, os.path.join(SAVE_DIR, "pairs.txt"))

if __name__ == "__main__":
    main()
