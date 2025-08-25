import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax
from models import (
    GCNN, 
    model_forward,  # functional forward
    binary_cross_entropy_loss
)
import jax.numpy as jnp
from data_prepare import train_set, test_set
from tqdm import tqdm
import matplotlib.pyplot as plt
import flax.serialization
import random

# Hyperparameters
num_features = train_set[0][3].shape[1]
layers_size = {"num_features": num_features, "hidden_dim": 32, "output_dim": 4}
num_epochs = 300
learning_rate = 1e-4
########################################

# Initialize GCNN model
GCNN_model = GCNN(input_dim=layers_size["num_features"], hidden_dim=layers_size["hidden_dim"], output_dim=layers_size["output_dim"])
key = jax.random.PRNGKey(0)
params = GCNN_model.init_params(key)
########################################

# JIT-compiled functions for training
grad = jax.jit(jax.grad(binary_cross_entropy_loss, argnums = 0 ))
loss_jit = jax.jit(binary_cross_entropy_loss)
grad_jit = jax.jit(grad)
########################################


n_samples =len(train_set)
history_train = list()
history_valid = list()
first_sample = train_set[0]
a_adj, a_feat = first_sample[2], first_sample[3]
b_adj, b_feat = first_sample[4], first_sample[5]
label = first_sample[6]
history_train.append(loss_jit(params, a_adj, a_feat, b_adj, b_feat,label))

print(f"Training on {n_samples} samples with {num_features} features per node...")

# Training loop
batch_size = 10

for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0.0

    # Sample random batch
    start_idx = random.randint(0, len(train_set) - batch_size)
    batch = train_set[start_idx : start_idx + batch_size]

    for sample in batch:
        a_adj, a_feat = sample[2], sample[3]
        b_adj, b_feat = sample[4], sample[5]
        label = float(sample[6])  # Ensure label is float


        grads = grad_jit(params, a_adj, a_feat, b_adj, b_feat, label)
        params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)

        loss_val = loss_jit(params, a_adj, a_feat, b_adj, b_feat, label)
        epoch_loss += loss_val

    epoch_loss /= batch_size
    history_train.append(epoch_loss)


plt.figure(figsize=(10, 6))
plt.plot(history_train, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


with open("PPGCN/PARAMETER/gcnn_params.pkl", "wb") as f:
    f.write(flax.serialization.to_bytes(params))
print("✅ Model parameters saved to gcnn_params.pkl")


