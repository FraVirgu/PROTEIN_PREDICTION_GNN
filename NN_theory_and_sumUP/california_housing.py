import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time 
import jax.numpy as jnp
from jax import tree_util, flatten_util
from jax.scipy.optimize import minimize
import jax
from tqdm import tqdm



data = pd.read_csv("./california_housing_train.csv")
data = data[data['median_house_value'] < 500001] 

# Initial exploration of the dataset
'''
data.head()
data.info()
data.describe()  

sns.scatterplot(data = data, x='longitude' ,y='latitude', hue='median_house_value')
plt.show()


data.corr()

sns.heatmap(data.corr(), annot = True, cmap = 'vlag_r', vmin = -1, vmax = 1)
plt.show()



sns.scatterplot(data = data, x='latitude' ,y='median_house_value')
plt.show()

'''


#Data normalization
data_mean = data.mean()
data_std = data.std()
data_normalized = (data - data_mean) / data_std
'''
data_normalized.describe()
_, ax = plt.subplots(figsize=(16,6))
sns.violinplot(data = data_normalized, ax = ax)
plt.show()
'''



# TRAIN VALIDATION SPLIT
np.random.seed(42)
data_normalized_np  = data_normalized.to_numpy()
np.random.shuffle(data_normalized_np)

fraction_validation = 0.2
num_train = int(data_normalized_np.shape[0] * (1 - fraction_validation))
x_train = data_normalized_np[:num_train, :-1]
y_train = data_normalized_np[:num_train, -1:]
x_valid = data_normalized_np[num_train:, :-1]
y_valid = data_normalized_np[num_train:, -1:]


def initialize_params(layers_size):
    np.random.seed(42)
    params = []
    for i in range(len(layers_size) - 1):
        W = np.random.randn(layers_size[i + 1], layers_size[i]) * np.sqrt(
            2 / (layers_size[i + 1] + layers_size[i])
        )
        b = np.zeros((layers_size[i + 1], 1))
        params.append((W, b))
    return params 


params = initialize_params([8, 5, 5, 1])
params = [(jnp.array(W), jnp.array(b)) for W, b in params]  # convert to jnp arrays



activation = lambda x: jnp.maximum(0.0, x)


def ANN(x,params):
    layer = x.T
    for W, b in params[:-1]:
        layer = activation(jnp.dot(W, layer) + b)
    W, b = params[-1]
    return (jnp.dot(W, layer) + b).transpose()


def loss(x,y,params):
    y_pred = ANN(x, params)
    return jnp.mean((y_pred - y) ** 2)

# Hyperparameters
layers_size = [8, 20, 20, 1]
# Training options
num_epochs = 2000
learning_rate = 1e-1
########################################

params = initialize_params(layers_size)

grad = jax.jit(jax.grad(loss, argnums=2))
loss_jit = jax.jit(loss)
grad_jit = jax.jit(grad)

n_samples = x_train.shape[0]

history_train = list()
history_valid = list()
history_train.append(loss_jit(x_train, y_train, params))
history_valid.append(loss_jit(x_valid, y_valid, params))

t0 = time.time()
for epoch in tqdm(range(num_epochs)):
    grads = grad_jit(x_train, y_train, params)

    params = [(W - learning_rate * dW, b - learning_rate * db) 
          for (W, b), (dW, db) in zip(params, grads)]


    history_train.append(loss_jit(x_train, y_train, params))
    history_valid.append(loss_jit(x_valid, y_valid, params))

print("elapsed time: %f s" % (time.time() - t0))
print("loss train     : %1.3e" % history_train[-1])
print("loss validation: %1.3e" % history_valid[-1])

fig, axs = plt.subplots(1, figsize=(16, 8))
axs.loglog(history_train, label="train")
axs.loglog(history_valid, label="validation")
axs.legend()
plt.show()



# TESTING
data_test = pd.read_csv("./california_housing_test.csv")
data_test = data_test[data_test["median_house_value"] < 500001]
data_test_normalized = (data_test - data.mean()) / data.std()
x_test = data_test_normalized.drop("median_house_value", axis=1).to_numpy()
Y_test = data_test["median_house_value"].to_numpy()[:, None]

y_predicted = ANN(x_test, params)
Y_predicted = (y_predicted * data["median_house_value"].std()) + data[
    "median_house_value"
].mean()

test = pd.DataFrame({"predicted": Y_predicted[:, 0], "actual": Y_test[:, 0]})
fig = sns.jointplot(data=test, x="actual", y="predicted")
fig.ax_joint.plot([0, 500000], [0, 500000.0], "r")
plt.show()