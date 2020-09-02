import os
import numpy as np
import jax.numpy as jnp
from jax import jit, random
from functools import partial
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from jax.tree_util import tree_flatten, tree_unflatten
import jax
import pickle
import haiku as hk
import pathlib

################################################################################################################

@jit
def xTAx(A, x):
    return jnp.einsum('i,ij,j', x, A, x)

################################################################################################################

def linear_warmup_lr_schedule(i, warmup=1000, lr_decay=1.0, lr=1e-4):
    return jnp.where(i < warmup,
                     lr*i/warmup,
                     lr*(lr_decay**(i - warmup)))

################################################################################################################

from haiku._src.data_structures import frozendict
from collections import OrderedDict

def dict_recurse(pytree, root_key=None):
    if(isinstance(pytree, dict) or
       isinstance(pytree, OrderedDict) or
       isinstance(pytree, frozendict)):

        return_list = []
        items = pytree.items()
        for key, val in items:
            joined_key = key if root_key is None else root_key+'/'+key
            ret_list = dict_recurse(val, joined_key)
            return_list.extend(ret_list)

        return return_list
    else:
        return [(root_key, pytree)]

################################################################################################################

def key_tree_like(key, pytree):
    # Figure out what the tree structure is
    flat_tree, treedef = jax.tree_util.tree_flatten(pytree)

    # Generate a tree of keys with the same structure as pytree
    n_keys = len(flat_tree)
    keys = random.split(key, n_keys)
    key_tree = jax.tree_util.tree_unflatten(treedef, keys)
    return key_tree

@partial(jit, static_argnums=(0,))
def tree_multimap_multiout(f, tree, *rest):
    # Like tree_multimap but expects f(leaves) to return a tuple.
    # This function will return trees for each tuple element.
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    new_leaves = [f(*xs) for xs in zip(*all_leaves)]
    return [treedef.unflatten(leaf) for leaf in zip(*new_leaves)]

@jit
def tree_shapes(pytree):
    return jax.tree_util.tree_map(lambda x:x.shape, pytree)

@jit
def tree_ndims(pytree):
    return jax.tree_util.tree_map(lambda x:x.ndim, pytree)

################################################################################################################

@jit
def whiten(x):
    U, s, VT = jnp.linalg.svd(x, full_matrices=False)
    return jnp.dot(U, VT)

################################################################################################################

class SimpleMLP(hk.Module):

    def __init__(self, out_shape, hidden_layer_sizes, is_additive, name=None):
        super().__init__(name=name)
        assert len(out_shape) == 1
        self.out_dim = out_shape[0]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.is_additive = is_additive

    def __call__(self, x, **kwargs):
        for dim in self.hidden_layer_sizes:
            x = hk.Linear(dim, hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'))(x)
            x = jax.nn.relu(x)
        mu = hk.Linear(self.out_dim, hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'))(x)
        if self.is_additive:
            return mu
        alpha = hk.Linear(self.out_dim, hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'))(x)
        alpha = jnp.tanh(alpha)
        return mu, alpha

class SimpleConv(hk.Module):

    def __init__(self, out_shape, n_hidden_channels, is_additive, name=None):
        super().__init__(name=name)
        _, _, out_channels = out_shape
        self.out_channels = out_channels
        self.n_hidden_channels = n_hidden_channels
        self.is_additive = is_additive

        self.last_channels = out_channels if is_additive else 2*out_channels

    def __call__(self, x, **kwargs):
        H, W, C = x.shape

        x = hk.Conv2D(output_channels=self.n_hidden_channels,
                      kernel_shape=(3, 3),
                      stride=(1, 1),
                      w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'))(x[None])[0]
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=self.n_hidden_channels,
                      kernel_shape=(1, 1),
                      stride=(1, 1),
                      w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'))(x[None])[0]
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=self.last_channels,
                      kernel_shape=(3, 3),
                      stride=(1, 1),
                      w_init=hk.initializers.Constant(0),
                      b_init=hk.initializers.Constant(0))(x[None])[0]

        if self.is_additive:
            return x

        mu, alpha = jnp.split(x, 2, axis=-1)
        alpha = jnp.tanh(alpha)
        return mu, alpha

################################################################################################################

def dilated_squeeze(x, filter_shape, dilation):
    H, W, C = x.shape

    fh, fw = filter_shape
    dh, dw = dilation

    assert H%(dh*fh) == 0
    assert W%(dw*fw) == 0

    # Rearrange for dilation
    x = x.reshape((H//dh, dh, W//dw, dw, C))
    x = x.transpose((1, 0, 3, 2, 4)) # (dh, H//dh, dw, W//dw, C)

    # Squeeze
    x = x.reshape((H//fh, fh, W//fw, fw, C))
    x = x.transpose((0, 2, 1, 3, 4)) # (H//fh, W//fw, fh, fw, C)
    x = x.reshape((H//fh, W//fw, C*fh*fw))
    return x

def dilated_unsqueeze(x, filter_shape, dilation):

    fh, fw = filter_shape
    dh, dw = dilation

    H_in, W_in, C_in = x.shape
    assert C_in%(fh*fw) == 0

    H, W, C = H_in*fh, W_in*fw, C_in//(fh*fw)

    assert H%(dh*fh) == 0
    assert W%(dw*fw) == 0

    # Un-squeeze
    x = x.reshape((H_in, W_in, fh, fw, C))
    x = x.transpose((0, 2, 1, 3, 4))

    # Un-dilate
    x = x.reshape((dh, H//dh, dw, W//dw, C))
    x = x.transpose((1, 0, 3, 2, 4))
    x = x.reshape((H, W, C))

    return x

################################################################################################################

def scaled_logsumexp(x, log_b, axis=0):
    """ logsumexp with scaling
    """
    x_max = jnp.amax(log_b + x, axis=axis, keepdims=True)
    y = jnp.sum(jnp.exp(log_b + x - x_max), axis=axis)
    sign_y = jnp.sign(y)
    abs_y = jnp.log(jnp.abs(y))
    return abs_y + jnp.squeeze(x_max, axis=axis)

################################################################################################################

@partial(jit, static_argnums=(0,))
def replicate(shape, pytree):
    replicate_fun = lambda x: jnp.broadcast_to(x, shape + x.shape)
    return tree_map(replicate_fun, pytree)

@jit
def unreplicate(pytree):
    return tree_map(lambda x:x[0], pytree)

################################################################################################################

def save_np_array_to_file(np_array, path):
    np.savetxt(path, np_array, delimiter=",")

def save_pytree_to_file(pytree, path):
    """ Save a pytree to file in pickle format"""
    dir_structure, file_name = os.path.split(path)
    assert file_name.endswith('.npz')

    # Create the path if it doesn't exist
    pathlib.Path(dir_structure).mkdir(parents=True, exist_ok=True)

    # Save the raw numpy parameters
    flat_pytree, _ = ravel_pytree(pytree)
    numpy_tree = np.array(flat_pytree)

    # Save the array to an npz file
    np.savez_compressed(path, flat_tree=numpy_tree)

def load_pytree_from_file(pytree, path):
    assert os.path.exists(path), '%s does not exist!'%path

    # Load the pytree structure
    _, unflatten = ravel_pytree(pytree)

    with np.load(path) as data:
        numpy_tree = data['flat_tree']

    return unflatten(numpy_tree)

################################################################################################################

@jit
def gaussian_chol_cov_logpdf(x, mean, cov_chol):
    dx = x - mean
    y = jax.lax_linalg.triangular_solve(cov_chol, dx, lower=True, transpose_a=True)
    log_px = -0.5*jnp.sum(y**2) - jnp.log(jnp.diag(cov_chol)).sum() - 0.5*x.shape[0]*jnp.log(2*jnp.pi)
    return log_px

@jit
def gaussian_full_cov_logpdf(x, mean, cov):
    dx = x - mean
    cov_inv = jnp.linalg.inv(cov)
    log_px = -0.5*jnp.sum(jnp.dot(dx, cov_inv.T)*dx, axis=-1)
    return log_px - 0.5*jnp.linalg.slogdet(cov)[1] - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

@jit
def gaussian_diag_cov_logpdf(x, mean, log_diag_cov):
    dx = x - mean
    log_px = -0.5*jnp.sum(dx*jnp.exp(-log_diag_cov)*dx, axis=-1)
    return log_px - 0.5*jnp.sum(log_diag_cov) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

@jit
def unit_gaussian_logpdf(x):
    if x.ndim > 1:
        return jax.vmap(unit_gaussian_logpdf)(x)
    return -0.5*jnp.dot(x, x) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

################################################################################################################

@jit
def lower_cho_solve(chol, x):
    return jax.scipy.linalg.cho_solve((chol, True), x)

def upper_triangular_indices(N):
    values = jnp.arange(N)
    padded_values = jnp.hstack([values, 0])

    idx = np.ogrid[:N,N:0:-1]
    idx = sum(idx) - 1

    mask = jnp.arange(N) >= jnp.arange(N)[:,None]
    return (idx + jnp.cumsum(values + 1)[:,None][::-1] - N + 1)*mask

def n_elts_upper_triangular(N):
    return N*(N + 1) // 2 - 1

def upper_triangular_from_values(vals, N):
    assert n_elts_upper_triangular(N) == vals.shape[-1]
    zero_padded_vals = jnp.pad(vals, (1, 0))
    return zero_padded_vals[upper_triangular_indices(N)]

tri_solve = jax.scipy.linalg.solve_triangular
L_solve = jit(partial(tri_solve, lower=True, unit_diagonal=True))
U_solve = jit(partial(tri_solve, lower=False, unit_diagonal=True))

################################################################################################################

@jit
def householder(x, v):
    return x - 2*jnp.einsum('i,j,j', v, v, x)/jnp.sum(v**2)

@jit
def householder_prod_body(carry, inputs):
    x = carry
    v = inputs
    return householder(x, v), 0

@jit
def householder_prod(x, vs):
    return jax.lax.scan(householder_prod_body, x, vs)[0]

@jit
def householder_prod_transpose(x, vs):
    return jax.lax.scan(householder_prod_body, x, vs[::-1])[0]

@jit
def householder_apply(U, log_s, VT, z):
    # Compute Az
    x = householder_prod(z, VT)
    x = x*jnp.exp(log_s)
    x = jnp.pad(x, (0, U.shape[1] - z.shape[0]))
    x = householder_prod(x, U)
    return x

@jit
def householder_pinv_apply(U, log_s, VT, x):
    # Compute A^+@x and also return U_perp^T@x
    UTx = householder_prod_transpose(x, U)
    z, UperpTx = jnp.split(UTx, jnp.array([log_s.shape[0]]))
    z = z*jnp.exp(-log_s)
    z = householder_prod_transpose(z, VT)
    return z, UperpTx

@jit
def householder_to_dense(U, log_s, VT):
    return jax.vmap(partial(householder_apply, U, log_s, VT))(jnp.eye(VT.shape[0])).T

@jit
def householder_pinv_to_dense(U, log_s, VT):
    return jax.vmap(partial(householder_pinv_apply, U, log_s, VT))(jnp.eye(U.shape[0]))[0].T

################################################################################################################