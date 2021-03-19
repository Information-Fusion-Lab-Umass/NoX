# NuX - Normalizing Flows using JAX

[![.github/workflows/CI.yml](https://github.com/MilesCranmer/NuX/actions/workflows/CI.yml/badge.svg)](https://github.com/MilesCranmer/NuX/actions/workflows/CI.yml)

## What is NuX?
NuX is a library for building [normalizing flows](https://arxiv.org/pdf/1912.02762.pdf) using [JAX](https://github.com/google/jax).

## What are normalizing flows?
Normalizing flows learn a parametric model over an unknown probability density function using data.  We assume that data points are sampled i.i.d. from an unknown distribution p(x).  Normalizing flows learn a parametric approximation of the true data distribution using maximum likelihood learning.  The learned distribution can be efficiently sampled from and has a log likelihood that can be evaluated exactly.

## Why use NuX?
It is easy to build, train and evaluate normalizing flows with NuX

```python
import nux
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)

# Build a dummy dataset
x_train, x_test = jnp.ones((2, 100, 2))
train_inputs, test_inputs = {"x": x_train}, {"x": x_test}

# Build a simple normalizing flow
def create_flow():
  return nux.sequential(nux.RealNVP(), nux.AffineLDU(), nux.UnitGaussianPrior())

# Perform data-dependent initialization
flow = nux.Flow(create_flow, key, train_inputs, batch_axes=(0,))

# Run the flow on inputs
outputs = flow.apply(key, test_inputs)
finv_x, log_px = outputs["x"], outputs["log_px"]

# Generate reconstructions
outputs = flow.reconstruct(key, {"x": finv_x})
reconstr = outputs["x"]

# Sample from the flow
outputs = flow.sample(key, n_samples=8)
fz, log_pfz = outputs["x"], outputs["log_px"]

# Construct a maximum likelihood trainer for the flow
trainer = nux.MaximumLikelihoodTrainer(flow)

# Train the flow
keys = jax.random.split(key, 10)
for key in keys:
  trainer.grad_step(key, train_inputs)
```

## Get started
The easiest way to use NuX is to clone this repo and install the prerequisites with the "pip install ." command.  JAX needs to be manually installed (see [this](https://github.com/google/jax#installation)) because GPU support is system dependent.  The NuX package available on pip is outdated and does not have much of the functionality of the current code.

NuX is in active development, so expect the API to change over time.  Contributions are welcome!
