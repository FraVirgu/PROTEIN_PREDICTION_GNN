import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import math
import random
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import flax.serialization

from models import (
    GCNN, 
    model_forward,           # functional forward
    binary_cross_entropy_loss
)
import parallel_data_prepare as ds

# ---------------------------
# Data
# ---------------------------
if getattr(ds, "TRAIN_SET", None) is None or getattr(ds, "TEST_SET", None) is None:
    train_set, test_set, N = ds.main()
else:
    train_set, test_set, N = ds.TRAIN_SET, ds.TEST_SET, ds.MAX_NUM_NODES_COMPUTED

# ---------------------------
# Model init (same as training)
# ---------------------------
num_features = train_set[0][3].shape[1]
layers_size = {"num_features": num_features, "hidden_dim": 256, "output_dim": 128}

GCNN_model = GCNN(
    input_dim=layers_size["num_features"],
    hidden_dim=layers_size["hidden_dim"],
    output_dim=layers_size["output_dim"],
)
key = jax.random.PRNGKey(0)
params = GCNN_model.init_params(key)  # dummy init with same structure

# ---------------------------
# Load trained parameters
# ---------------------------

with open("PPGCN/PARAMETER/gcnn_params_29_08_2025.pkl", "rb") as f:
    params = flax.serialization.from_bytes(params, f.read())
    print("✅ Parameters loaded successfully")




# ---------------------------
# Testing
# ---------------------------
def evaluate_model(params, test_set):
    y_trues = []
    y_preds = []

    for sample in tqdm(test_set, desc = "Evaluating", ncols = 100):
        a_adj, a_feat = sample[2], sample[3]
        b_adj, b_feat = sample[4], sample[5]
        y = float(sample[6])
        y_pred = model_forward(params, a_adj, a_feat, b_adj, b_feat)
        y_trues.append(y)
        y_preds.append(y_pred)

    y_trues = jnp.array(y_trues)
    y_preds = jnp.array(y_preds)

    # Calculate accuracy
    y_pred_labels = (y_preds >= 0.5).astype(jnp.float32)
    accuracy = jnp.mean(y_pred_labels == y_trues)

    # Calculate loss
    loss = jnp.mean(binary_cross_entropy_loss(params, 
                                              jnp.zeros((N, N)), jnp.zeros((N, num_features)), 
                                              jnp.zeros((N, N)), jnp.zeros((N, num_features)), 
                                              y_trues))

    return accuracy.item(), loss.item(), y_trues, y_preds



accuracy, loss, y_trues, y_preds = evaluate_model(params, test_set)
print(f"Test Accuracy: {accuracy*100:.2f}%, Test Loss: {loss:.4f}")