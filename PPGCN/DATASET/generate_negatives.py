import os
import random

# --------------------------
# CONFIGURATION
# --------------------------
PAIRS_FILE = "DATASET/graphs/pairs.txt"
NEGATIVE_PAIRS_FILE = "DATASET/graphs/non_interacting_pairs.txt"
NEGATIVE_RATIO = 1  # 1 = same number as positives, 2 = double, etc.

# --------------------------
# LOAD POSITIVE PAIRS
# --------------------------
def load_positive_pairs(path):
    pairs = []
    proteins = set()
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                a, b = parts
                pairs.append(tuple(sorted((a, b))))
                proteins.update([a, b])
    print(f"✅ Loaded {len(pairs)} positive pairs from {path}")
    return pairs, list(proteins)

# --------------------------
# GENERATE RANDOM NEGATIVE PAIRS
# --------------------------
def generate_negative_pairs(all_proteins, positive_pairs, n_negatives):
    negative_pairs = set()
    positive_set = set(positive_pairs)

    print(f"🎯 Generating {n_negatives} negative pairs...")
    while len(negative_pairs) < n_negatives:
        a, b = random.sample(all_proteins, 2)
        pair = tuple(sorted((a, b)))
        if pair not in positive_set and pair not in negative_pairs:
            negative_pairs.add(pair)

    return list(negative_pairs)

# --------------------------
# SAVE TO FILE
# --------------------------
def save_pairs(pairs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")
    print(f"💾 Saved {len(pairs)} negative pairs to {path}")

# --------------------------
# MAIN
# --------------------------
def main():
    positives, all_proteins = load_positive_pairs(PAIRS_FILE)
    n_negatives = len(positives) * NEGATIVE_RATIO
    negatives = generate_negative_pairs(all_proteins, positives, n_negatives)
    save_pairs(negatives, NEGATIVE_PAIRS_FILE)

if __name__ == "__main__":
    main()
