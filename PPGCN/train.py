import jax
from models import (
    GCNN, 
    model_forward  # functional forward
)
import jax.numpy as jnp
from data_prepare import train_set, test_set


# Hyperparameters
layers_size = []
num_epochs = 1000
learning_rate = 1e-3

num_features = train_set[0][3].shape[1]
print(f"Number of features: {num_features}")

GCNN_model = GCNN(input_dim=num_features, hidden_dim=64, output_dim=32)
key = jax.random.PRNGKey(0)
params = GCNN_model.init_params(key)

first_sample = train_set[0]
a_adj, a_feat = first_sample[2], first_sample[3]
b_adj, b_feat = first_sample[4], first_sample[5]


output = model_forward(params, a_adj, a_feat, b_adj, b_feat)
print(f"Predicted interaction probability: {output}")############################################




