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
# Hyperparameters
# ---------------------------
num_features = train_set[0][3].shape[1]
layers_size = {"num_features": num_features, "hidden_dim": 64, "output_dim": 8}

num_epochs   = 20
batch_size   = 64

# SGD + momentum + cosine decay schedule
momentum = 0.9
lr_init  = 5e-3          # starting LR
lr_min   = 5e-4          # final LR (10% of initial)

# ---------------------------
# Model init
# ---------------------------
GCNN_model = GCNN(
    input_dim=layers_size["num_features"],
    hidden_dim=layers_size["hidden_dim"],
    output_dim=layers_size["output_dim"],
)
key = jax.random.PRNGKey(0)
params = GCNN_model.init_params(key)

# ---------------------------
# JITed loss/grad fns
# ---------------------------
loss_jit = jax.jit(binary_cross_entropy_loss)
grad_jit = jax.jit(jax.grad(binary_cross_entropy_loss, argnums=0))

# ---------------------------
# Utilities
# ---------------------------
def cosine_decay(step, total_steps, lr0, lr_min=0.0):
    """Cosine LR schedule from lr0 → lr_min over total_steps."""
    t = min(step, total_steps)
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * t / total_steps))
    return lr_min + (lr0 - lr_min) * cos_factor

def batch_indices(n, bs, rng):
    idx = np.arange(n)
    rng.shuffle(idx)
    for s in range(0, n, bs):
        yield idx[s:s + bs]

# ---------------------------
# Train
# ---------------------------
n_train = len(train_set)
steps_per_epoch = math.ceil(n_train / batch_size)
total_steps = steps_per_epoch * num_epochs

history_train = []

# momentum buffer same tree-structure as params
velocity = jax.tree_util.tree_map(jnp.zeros_like, params)
global_step = 0

# initial loss for plotting
a_adj, a_feat = train_set[0][2], train_set[0][3]
b_adj, b_feat = train_set[0][4], train_set[0][5]
label0 = float(train_set[0][6])
history_train.append(loss_jit(params, a_adj, a_feat, b_adj, b_feat, label0))

print(f"Training on {n_train} samples with {num_features} features per node...")

rng = np.random.default_rng(0)

for epoch in tqdm(range(num_epochs)):
    epoch_loss_sum = 0.0

    for ids in batch_indices(n_train, batch_size, rng):
        batch = [train_set[i] for i in ids]

        # accumulate grads and loss over batch
        grads_sum = None
        loss_sum = 0.0

        for sample in batch:
            a_adj, a_feat = sample[2], sample[3]
            b_adj, b_feat = sample[4], sample[5]
            y = float(sample[6])

            g = grad_jit(params, a_adj, a_feat, b_adj, b_feat, y)
            grads_sum = g if grads_sum is None else jax.tree_util.tree_map(lambda x, y_: x + y_, grads_sum, g)

            loss_sum += float(loss_jit(params, a_adj, a_feat, b_adj, b_feat, y))

        # mean gradients / loss
        bsz = len(batch)
        grads_avg = jax.tree_util.tree_map(lambda g: g / bsz, grads_sum)
        batch_loss = loss_sum / bsz

        # cosine‑decayed LR for this step
        lr_t = cosine_decay(global_step, total_steps, lr_init, lr_min)

        # SGD with momentum (Polyak)
        velocity = jax.tree_util.tree_map(lambda v, g: momentum * v - lr_t * g, velocity, grads_avg)
        params   = jax.tree_util.tree_map(lambda p, v: p + v, params, velocity)

        global_step += 1
        epoch_loss_sum += batch_loss * bsz

    epoch_loss = epoch_loss_sum / n_train
    history_train.append(epoch_loss)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(10, 6))
plt.plot(history_train, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------
# Save params
# ---------------------------
os.makedirs("PPGCN/PARAMETER", exist_ok=True)
with open("PPGCN/PARAMETER/gcnn_params.pkl", "wb") as f:
    f.write(flax.serialization.to_bytes(params))
print("✅ Model parameters saved to gcnn_params.pkl")
