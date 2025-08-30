import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import jax, jax.numpy as jnp
import flax.serialization
from models import GCNN, model_forward, binary_cross_entropy_loss
from tqdm import tqdm
import data_creation as dc   # ⬅️ import directly

# build datasets first to know num_features if you want a sanity check
train_set, valid_set, test_set, N = dc.get_datasets(max_train=5000, max_valid=2000, max_test=2000, n_jobs=8)
num_features = train_set[0][3].shape[1]

ckpt_path = "RDKIT/PARAMETER/gcnn_params.pkl"
with open(ckpt_path, "rb") as f:
    payload = flax.serialization.from_bytes(None, f.read())

cfg = payload["config"]
state_dict = payload["params"] 

# sanity: dims must match training
assert cfg["num_features"] == num_features, \
    f"Dataset features ({num_features}) != checkpoint ({cfg['num_features']})"

# instantiate model with SAME dims as training
model = GCNN(input_dim=cfg["num_features"], hidden_dim=cfg["hidden_dim"], output_dim=cfg["output_dim"])

print("✅ Parameters + config loaded:", cfg)

key = jax.random.PRNGKey(0)
target_params = model.init_params(key)  # dummy init with same structure

# ---------------------------
# Load trained parameters
# ---------------------------

# 4) Map state_dict → target pytree (preserves tuple/list structure)
params = flax.serialization.from_state_dict(target_params, state_dict)



# ---------------------------
# Testing
# ---------------------------
def evaluate_model(params, test_set, num_features):
    y_trues = []
    y_preds = []
    loss_sum = 0.0

    for sample in tqdm(test_set, desc="Evaluating", ncols=100):
        A1, X1 = sample[2], sample[3]
        A2, X2 = sample[4], sample[5]
        y      = float(sample[6])

        # forward
        y_pred = model_forward(params, A1, X1, A2, X2)
        y_pred = float(jnp.squeeze(y_pred))   # make sure it's a scalar
        

        # per-sample loss on the *real* graphs
        loss = float(binary_cross_entropy_loss(params, A1, X1, A2, X2, y))

        y_trues.append(y)
        y_preds.append(y_pred)
        loss_sum += loss

    y_trues = jnp.array(y_trues)
    y_preds = jnp.array(y_preds)

    # accuracy at 0.5 threshold
    y_hat = (y_preds >= 0.5).astype(jnp.float32)
    acc = float(jnp.mean(y_hat == y_trues))

    mean_loss = loss_sum / len(test_set)
    return acc, mean_loss, y_trues, y_preds


num_features = train_set[0][3].shape[1]
accuracy, loss, y_trues, y_preds = evaluate_model(params, test_set,num_features)
print(f"Test Accuracy: {accuracy*100:.2f}%, Test Loss: {loss:.4f}")
