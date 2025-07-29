import jax
import jax.numpy as jnp


def relu(x):
    return jnp.maximum(0, x)


def normalize_adjacency(A):
    A_hat = A + jnp.eye(A.shape[0])  # Add self-loops
    D_hat = jnp.diag(1.0 / jnp.sqrt(jnp.sum(A_hat, axis=1) + 1e-5))
    return D_hat @ A_hat @ D_hat

class Model:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def init_params(self, key):
        """
        Initializes the parameters (weights and biases) for a 2-layer GCN followed by a classifier using Xavier/Glorot initialization.
        Args:
            key: jax.random.PRNGKey for random number generation.
        Returns:
            List of tuples [(W1, b1), (W2, b2), (W3, b3)]:
                - W1, b1: Weights and biases for GCN Layer 1 (input_dim → hidden_dim)
                - W2, b2: Weights and biases for GCN Layer 2 (hidden_dim → output_dim)
                - W3, b3: Weights and biases for the classifier (output_dim → 1)
        🧭 Dimensions Summary
        Layer         Input Dim    Output Dim    Weight Shape         Bias Shape
        GCN Layer 1   input_dim    hidden_dim    [input_dim, hidden_dim]   [hidden_dim]
        GCN Layer 2   hidden_dim   output_dim    [hidden_dim, output_dim]  [output_dim]
        Classifier    output_dim   1             [output_dim, 1]           [1]
        """
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

        # Xavier/Glorot initialization
        def xavier_init(k, in_dim, out_dim):
            limit = jnp.sqrt(6 / (in_dim + out_dim))
            W = jax.random.uniform(k, (in_dim, out_dim), minval=-limit, maxval=limit)
            b = jnp.zeros((out_dim,))
            return W, b
        

        W1, b1 = xavier_init(k1, self.input_dim, self.hidden_dim)
        W2, b2 = xavier_init(k2, self.hidden_dim, self.output_dim)
        W3, b3 = xavier_init(k3, self.output_dim, 1)

        return [(W1, b1), (W2, b2), (W3, b3)]

    def normalize_adjacency(self, A):
        A_hat = A + jnp.eye(A.shape[0])
        D_hat = jnp.diag(1.0 / jnp.sqrt(jnp.sum(A_hat, axis=1) + 1e-5))
        return D_hat @ A_hat @ D_hat

    def gcn_layer(self, A_norm, X, W, b):
        return relu(A_norm @ X @ W + b)

    def forward(self, params, A, X):
        A_norm = self.normalize_adjacency(A)
        (W1, b1), (W2, b2), (W3, b3) = params

        h1 = self.gcn_layer(A_norm, X, W1, b1)
        h2 = self.gcn_layer(A_norm, h1, W2, b2)
        graph_emb = jnp.mean(h2, axis=0)  # pool across nodes → 1 graph embedding
        logit = graph_emb @ W3 + b3       # final linear layer
