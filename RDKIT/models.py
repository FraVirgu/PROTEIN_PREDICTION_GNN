import jax
import jax.numpy as jnp
from functools import partial

# -----------------------------
# Activation and normalization
# -----------------------------

@jax.jit
def relu(x):
    return jnp.maximum(0, x)

@jax.jit
def normalize_adjacency(A):
    A_hat = A + jnp.eye(A.shape[0])
    degree = jnp.sum(A_hat, axis=1)
    D_inv_sqrt = jnp.where(degree > 0, 1.0 / jnp.sqrt(degree), 0.0)
    D_hat = jnp.diag(D_inv_sqrt)
    return D_hat @ A_hat @ D_hat

# -----------------------------
# Xavier Initialization
# -----------------------------

def xavier_init(key, in_dim, out_dim):
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    W = jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)
    b = jnp.zeros((out_dim,))
    return W, b

# -----------------------------
# GCN + Dense model functions
# -----------------------------

@jax.jit
def gcn_layer(A_norm, X, W, b):
    return relu(A_norm @ X @ W + b)

@jax.jit
def protein_forward(params, A, X):
    # Handle small or invalid graphs by returning a zero vector
    num_nodes, num_features = X.shape
    expected_input_dim = params[0][0].shape[0]
    if num_nodes < 2 or num_features != expected_input_dim:
        return jnp.zeros(params[-1][0].shape[1])
    
    A_norm = normalize_adjacency(A)
    (W1, b1), (W2, b2) = params
    h1 = gcn_layer(A_norm, X, W1, b1)
    h2 = gcn_layer(A_norm, h1, W2, b2)
    return jnp.mean(h2, axis=0)

@jax.jit
def dense_concat(combined_params, h1, h2):
    x = jnp.concatenate([h1, h2])
    for W, b in combined_params[:-1]:
        x = relu(x @ W + b)

    W_last, b_last = combined_params[-1]
    assert x.shape[0] == W_last.shape[0], f"Expected shape {W_last.shape[0]}, got {x.shape[0]}"
    logits = x @ W_last + b_last
    return jax.nn.sigmoid(logits)

@jax.jit
def model_forward(params, A1, X1, A2, X2):
    h1 = protein_forward(params[0], A1, X1)
    h2 = protein_forward(params[1], A2, X2)
    return dense_concat(params[2], h1, h2)

@jax.jit
def binary_cross_entropy_loss(params, A1, X1, A2, X2, y_true, eps=1e-8):
    y_pred = model_forward(params, A1, X1, A2, X2)
    loss = -(y_true * jnp.log(y_pred + eps) + (1 - y_true) * jnp.log(1 - y_pred + eps))
    return jnp.squeeze(loss)

# -----------------------------
# GCNN Class for Param Init
# -----------------------------

class GCNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def init_params(self, key):
        keys = jax.random.split(key, 7)

        # GCN parameters for protein 1
        W1a, b1a = xavier_init(keys[0], self.input_dim, self.hidden_dim)
        W2a, b2a = xavier_init(keys[1], self.hidden_dim, self.output_dim)
        protein1_params = [(W1a, b1a), (W2a, b2a)]

        # GCN parameters for protein 2
        W1b, b1b = xavier_init(keys[2], self.input_dim, self.hidden_dim)
        W2b, b2b = xavier_init(keys[3], self.hidden_dim, self.output_dim)
        protein2_params = [(W1b, b1b), (W2b, b2b)]

        # Dense layers for concatenated output
        Wc1, bc1 = xavier_init(keys[4], 2 * self.output_dim, 256)
        Wc2, bc2 = xavier_init(keys[5], 256, 64)
        Wc3, bc3 = xavier_init(keys[6], 64, 1)
        combined_params = [(Wc1, bc1), (Wc2, bc2), (Wc3, bc3)]

        return protein1_params, protein2_params, combined_params
