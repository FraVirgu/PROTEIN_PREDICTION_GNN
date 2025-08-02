import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------------
# CONFIGURATION
# --------------------------
GRAPH_DIR = "PPGCN/DATASET/graphs"
PAIRS_FILE = os.path.join(GRAPH_DIR, "A_pairs.txt")
NEG_PAIRS_FILE = os.path.join(GRAPH_DIR, "A_non_interacting_pairs.txt")
TEST_SIZE = 0.2
RANDOM_SEED = 42

# --------------------------
# LOAD GRAPH FROM TXT FILE
# --------------------------
def load_graph(uniprot_id):
    path = os.path.join(GRAPH_DIR, f"{uniprot_id}.txt")
    if not os.path.isfile(path):
        return None, None

    with open(path, "r") as f:
        lines = f.read().splitlines()

    # Parse adjacency matrix
    adj_start = 1
    adj_end = lines.index("")  # empty line between sections
    adj_lines = lines[adj_start:adj_end]
    adj = np.array([[int(x) for x in line.split()] for line in adj_lines])

    # Parse node embeddings
    feat_lines = lines[adj_end + 2:]
    features = np.array([[float(x) for x in line.split()] for line in feat_lines])

    return adj, features

# --------------------------
# LOAD PAIRS AND LABEL
# --------------------------
def load_pairs(path, label):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            a, b = line.strip().split()
            pairs.append((a, b, label))
    return pairs

# --------------------------
# BUILD DATASET
# --------------------------
def build_dataset(pos_pairs, neg_pairs):
    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)
    dataset = []

    print(f"🔄 Loading {len(all_pairs)} total pairs...")

    for a, b, label in all_pairs:
        a_adj, a_feat = load_graph(a)
        b_adj, b_feat = load_graph(b)

        if a_adj is None or b_adj is None:
            print(f"[SKIPPED] Missing graph: {a} or {b}")
            continue

        sample = [a, b, a_adj, a_feat, b_adj, b_feat, label]
        dataset.append(sample)

    return dataset

# --------------------------
# RUN EVERYTHING IMMEDIATELY
# --------------------------

# Load pairs
pos_pairs = load_pairs(PAIRS_FILE, label=1)
neg_pairs = load_pairs(NEG_PAIRS_FILE, label=0)

# Build full dataset
full_dataset = build_dataset(pos_pairs, neg_pairs)

# Split into train/test
train_set, test_set = train_test_split(full_dataset, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# Now train_set and test_set are available globally
print(f"✅ Loaded {len(train_set)} training samples")
print(f"✅ Loaded {len(test_set)} testing samples")
