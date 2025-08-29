import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed

# --------------------------
# GLOBAL OUTPUTS
# --------------------------
TRAIN_SET = None
TEST_SET = None
MAX_NUM_NODES_COMPUTED = 0  # keep separate from CONFIG's MAX_FEATURES, etc.


# --------------------------
# CONFIGURATION
# --------------------------
GRAPH_DIR = "PPGCN/DATASET/graphs"
PAIRS_FILE = os.path.join(GRAPH_DIR, "A_positive_pairs.txt")
NEG_PAIRS_FILE = os.path.join(GRAPH_DIR, "A_negative_pairs.txt")
TEST_SIZE = 0.1
RANDOM_SEED = 42
MAX_FEATURES = 1024     # Max features per node to read
DOWNSAMPLE_TO = 512      # Final feature length per node (set None to disable)
USE_PARALLEL_PADDING = False  # True to also pad in parallel (trades memory for speed)

# --------------------------
# UTILITIES
# --------------------------
def downsample_vector(features, out_len=DOWNSAMPLE_TO):
    """
    Downsample a 1D vector into out_len values by averaging blocks.
    Example: len(features)=1024, out_len=DOWNSAMPLE_TO → returns shape (DOWNSAMPLE_TO,)
    """
    if out_len is None or features.shape[0] == out_len:
        return features
    n = features.shape[0]
    if n < out_len:
        # pad up to out_len (rare, but for safety)
        pad = np.zeros(out_len, dtype=features.dtype)
        pad[:n] = features
        return pad
    block_size = n // out_len
    trimmed = features[:block_size * out_len]
    return trimmed.reshape(out_len, block_size).mean(axis=1)

def load_graph(uniprot_id, max_features=MAX_FEATURES, down_to=DOWNSAMPLE_TO):
    path = os.path.join(GRAPH_DIR, f"{uniprot_id}.txt")
    if not os.path.isfile(path):
        return None, None

    with open(path, "r") as f:
        lines = f.read().splitlines()

    # Parse adjacency matrix
    # Expect: first line maybe header; adjacency lines start at 1 until blank line
    adj_start = 1
    try:
        adj_end = lines.index("")  # empty line between sections
    except ValueError:
        # No blank line => malformed
        return None, None

    adj_lines = lines[adj_start:adj_end]
    adj = np.array([[int(x) for x in line.split()] for line in adj_lines], dtype=np.uint8)

    # Parse node embeddings
    feat_lines = lines[adj_end + 2:]
    features_raw = np.array(
        [[float(x) for x in line.split()[:max_features]] for line in feat_lines],
        dtype=np.float32
    )

    # Apply downsampling PER ROW and build new array (also ensures fixed width)
    if down_to is not None:
        if features_raw.shape[1] < down_to:
            # If fewer features than target, pad rows
            padded = np.zeros((features_raw.shape[0], down_to), dtype=np.float32)
            padded[:, :features_raw.shape[1]] = features_raw
            features_raw = padded
        features = np.vstack([downsample_vector(row, out_len=down_to) for row in features_raw]).astype(np.float32)
    else:
        features = features_raw

    return adj, features

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

# ------------- PARALLEL WORKERS -------------

def _process_pair(sample_triplet):
    """
    Worker function run in child processes.
    Returns:
      None if missing graphs
      else a tuple:
        (a, b, a_adj, a_feat, b_adj, b_feat, label, na, nb)
    """
    a, b, label = sample_triplet
    a_adj, a_feat = load_graph(a)
    b_adj, b_feat = load_graph(b)

    if a_adj is None or b_adj is None:
        # Return a marker to skip in the main process
        return None

    na = a_feat.shape[0]
    nb = b_feat.shape[0]
    return (a, b, a_adj, a_feat, b_adj, b_feat, label, na, nb)

def _pad_one(sample, N):
    """
    Pad a single loaded sample to NxN and NxF in-place style (returns new tuple).
    Input sample layout: (a, b, a_adj, a_feat, b_adj, b_feat, label, na, nb)
    Output layout:       [a, b, A_pad, XA_pad, B_pad, XB_pad, label]
    """
    a, b, a_adj, a_feat, b_adj, b_feat, label, na, nb = sample

    # Cast for memory efficiency
    if a_adj.dtype != np.uint8:
        a_adj = a_adj.astype(np.uint8, copy=False)
    if b_adj.dtype != np.uint8:
        b_adj = b_adj.astype(np.uint8, copy=False)
    a_feat = a_feat.astype(np.float32, copy=False)
    b_feat = b_feat.astype(np.float32, copy=False)

    fa = a_feat.shape[1]
    fb = b_feat.shape[1]

    A_pad = np.zeros((N, N), dtype=a_adj.dtype)
    B_pad = np.zeros((N, N), dtype=b_adj.dtype)
    XA_pad = np.zeros((N, fa), dtype=a_feat.dtype)
    XB_pad = np.zeros((N, fb), dtype=b_feat.dtype)

    A_pad[:na, :na] = a_adj
    B_pad[:nb, :nb] = b_adj
    XA_pad[:na, :fa] = a_feat
    XB_pad[:nb, :fb] = b_feat

    return [a, b, A_pad, XA_pad, B_pad, XB_pad, label]

# --------------------------
# MAIN
# --------------------------
def main():
    global TRAIN_SET, TEST_SET, MAX_NUM_NODES_COMPUTED

    random.seed(RANDOM_SEED)

    # Load pairs
    pos_pairs = load_pairs(PAIRS_FILE, label=1)
    neg_pairs = load_pairs(NEG_PAIRS_FILE, label=0)

    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)
    print(f"🔄 Loading {len(all_pairs)} total pairs in parallel... (workers={os.cpu_count()})")

    # Phase 1: parallel load/parse
    results = []
    skipped = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(_process_pair, triplet) for triplet in all_pairs]
        for fut in as_completed(futures):
            out = fut.result()
            if out is None:
                skipped += 1
            else:
                results.append(out)

    if skipped:
        print(f"[SKIPPED] {skipped} pairs due to missing/malformed graphs")
    if not results:
        raise RuntimeError("No valid samples loaded. Check your input files and paths.")

    # Compute and save global max nodes
    MAX_NUM_NODES_COMPUTED = max(max(na, nb) for *_, na, nb in results)
    print(f"🔄 Padding matrices to {MAX_NUM_NODES_COMPUTED} nodes...")

    # Phase 2: padding
    if USE_PARALLEL_PADDING:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            futures = [ex.submit(_pad_one, sample, MAX_NUM_NODES_COMPUTED) for sample in results]
            full_dataset = [f.result() for f in as_completed(futures)]
    else:
        full_dataset = [_pad_one(sample, MAX_NUM_NODES_COMPUTED) for sample in results]

    TRAIN_SET, TEST_SET = train_test_split(
        full_dataset, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    print(f"✅ Loaded {len(TRAIN_SET)} training samples")
    print(f"✅ Loaded {len(TEST_SET)} testing samples")

    # Return as well, in case you want it programmatically
    return TRAIN_SET, TEST_SET, MAX_NUM_NODES_COMPUTED

if __name__ == "__main__":
    # The guard is important for Windows/macOS spawn semantics in multiprocessing.
    main()
