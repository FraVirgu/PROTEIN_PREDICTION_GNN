# parallel_data_prepare.py

from data_creation import build_dataset_from_split, split

MAX_NODES = 128  # or whatever cutoff you chose

def main():
    train_data = build_dataset_from_split(split['train'], max_rows=5000, n_jobs=32)
    valid_data = build_dataset_from_split(split['valid'], max_rows=2000, n_jobs=32)
    test_data  = build_dataset_from_split(split['test'],  max_rows=2000, n_jobs=32)

    # return also max node size so the model knows what to expect
    return train_data, test_data, MAX_NODES

# make these accessible for imports
TRAIN_SET, TEST_SET, MAX_NUM_NODES_COMPUTED = main()
