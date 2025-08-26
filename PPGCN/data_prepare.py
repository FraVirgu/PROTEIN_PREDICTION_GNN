import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------------
# CONFIGURATION
# --------------------------
GRAPH_DIR = "PPGCN/DATASET/graphs"
PAIRS_FILE = os.path.join(GRAPH_DIR, "A_positive_pairs.txt")
NEG_PAIRS_FILE = os.path.join(GRAPH_DIR, "A_negative_pairs.txt")
TEST_SIZE = 0.2
RANDOM_SEED = 42
MAX_FEATURES = 1024  # Max features per node


# --------------------------
# LOAD GRAPH FROM TXT FILE
# --------------------------

MAX_NUM_NODES = 0

def downsample_vector(features, out_len=64):
    """
    Downsample a 1D vector into out_len values by averaging blocks.
    Example: len(features)=1024, out_len=32 → returns shape (32,)
    """
    n = features.shape[0]
    block_size = n // out_len
    trimmed = features[:block_size * out_len]   # drop extra if not divisible
    return trimmed.reshape(out_len, block_size).mean(axis=1)

def load_graph(uniprot_id, max_features=MAX_FEATURES):
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
    features = np.array(
        [[float(x) for x in line.split()[:max_features]] for line in feat_lines]
    )
    for vector in features:
        downsample_vector(vector)
        
    global MAX_NUM_NODES
    if(features.shape[0] > MAX_NUM_NODES):
        MAX_NUM_NODES = features.shape[0]
        print(f"🔄 Updated MAX_NUM_NODES to {MAX_NUM_NODES}")

    return adj, features


# --------------------------
# LOAD PAIRS AND LABEL
# --------------------------
def load_pairs(path, label):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # Skip malformed lines
            a, b = parts[0], parts[1]
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
# MODIFY DATASET TO HAVE FIXED SIZE MATRICES 
# --------------------------
def pad_matrix(dataset, max_num_nodes):
    out = []  # avoid in-place surprises
    for sample in dataset:
        a_adj, a_feat = sample[2], sample[3]
        b_adj, b_feat = sample[4], sample[5]

        # dtypes (helps memory a lot)
        if a_adj.dtype != np.uint8 and set(np.unique(a_adj)).issubset({0,1}):
            a_adj = a_adj.astype(np.uint8, copy=False)
        if b_adj.dtype != np.uint8 and set(np.unique(b_adj)).issubset({0,1}):
            b_adj = b_adj.astype(np.uint8, copy=False)
        a_feat = a_feat.astype(np.float32, copy=False)
        b_feat = b_feat.astype(np.float32, copy=False)

        na, fa = a_feat.shape
        nb, fb = b_feat.shape
        N = max_num_nodes

        # Preallocate zeros and slice-copy (much faster than np.pad for big arrays)
        A_pad = np.zeros((N, N), dtype=a_adj.dtype)
        B_pad = np.zeros((N, N), dtype=b_adj.dtype)
        XA_pad = np.zeros((N, fa), dtype=a_feat.dtype)
        XB_pad = np.zeros((N, fb), dtype=b_feat.dtype)

        A_pad[:na, :na] = a_adj
        B_pad[:nb, :nb] = b_adj
        XA_pad[:na, :fa] = a_feat
        XB_pad[:nb, :fb] = b_feat

        # Rebuild sample
        out.append([sample[0], sample[1], A_pad, XA_pad, B_pad, XB_pad, sample[6]])
    return out

# --------------------------
# RUN EVERYTHING IMMEDIATELY
# --------------------------

# Load pairs
pos_pairs = load_pairs(PAIRS_FILE, label=1)
neg_pairs = load_pairs(NEG_PAIRS_FILE, label=0)

# Build full dataset
full_dataset = build_dataset(pos_pairs, neg_pairs)
# Pad matrices to have uniform size
print(f"🔄 Padding matrices to {MAX_NUM_NODES} nodes...")
#full_dataset = pad_matrix(full_dataset, MAX_NUM_NODES)

# Split into train/test
train_set, test_set = train_test_split(
    full_dataset, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# Now train_set and test_set are available globally
print(f"✅ Loaded {len(train_set)} training samples")
print(f"✅ Loaded {len(test_set)} testing samples")
