# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention core modules for Flax."""

from collections.abc import Iterable  # pylint: disable=g-importing-member

import warnings

from flax import jax_utils
from flax.nn.activation import softmax
from flax.nn.base import Collection, Module, collection_from_iterable, iterate_collection
from flax.nn.initializers import zeros
from flax.nn.stochastic import make_rng
from flax.nn.linear import DenseGeneral, default_kernel_init
from flax import struct

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

import numpy as np
import cmath
import abc
from collections.abc import Iterable  # pylint: disable=g-importing-member
import functools
from absl import logging
import gin


# Nonlinear mappings encoding different attention kernels.
gin.external_configurable(jnp.cos, 'jcos')
gin.external_configurable(jnp.sin, 'jsin')
gin.external_configurable(jnp.tanh, 'jtanh')
gin.external_configurable(jax.nn.sigmoid, 'jsigmoid')
gin.external_configurable(
    lambda x: jax.nn.gelu(x, approximate=False), 'jgelu'
)  # Needs to be exact, although might be slower. See https://github.com/google/jax/issues/4428.
gin.external_configurable(lambda x: x * x * (x > 0.0), 'jrequ')
gin.external_configurable(jnp.exp, 'jexp')
gin.external_configurable(lambda x: x, 'jidentity')
gin.external_configurable(
    lambda x: (jnp.exp(x)) * (x <= 0.0) + (x + 1.0) * (x > 0.0), 'jshiftedelu'
)  # Nonlinearity used in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (https://arxiv.org/abs/2006.16236).


def nonnegative_softmax_kernel_feature_creator(data,
                                               projection_matrix,
                                               attention_dims_t,
                                               batch_dims_t,
                                               precision,
                                               is_query,
                                               normalize_data=True,
                                               eps=0.0001):
  """Constructs nonnegative kernel features for fast softmax attention.
  Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    attention_dims_t: tuple of attention dimensions
    batch_dims_t: tuple of batch dimensions
    precision: precision parameter
    is_query: predicate indicating whether input data corresponds to queries or
      keys
    normalize_data: predicate indicating whether data should be normalized,
    eps: numerical stabilizer.
  Returns:
    Random features for fast softmax attention.
  """
  del attention_dims_t
  if normalize_data:
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1])))
  else:
    data_normalizer = 1.0
  ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
  data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix

  data_dash = lax.dot_general(
      data_normalizer * data,
      data_thick_random_matrix,
      (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)),
       (batch_dims_t, batch_dims_t)),
      precision=precision)

  diag_data = jnp.square(data)
  diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)

  if is_query:
    last_dims_t = (len(data_dash.shape) - 1,)
    data_dash = ratio * (
        jnp.exp(data_dash - diag_data -
                jnp.max(data_dash, axis=last_dims_t, keepdims=True)) + eps)
  else:
    data_dash = ratio * (
        jnp.exp(data_dash - diag_data - jnp.max(data_dash)) + eps)

  return data_dash


def sincos_softmax_kernel_feature_creator(data,
                                          projection_matrix,
                                          attention_dims_t,
                                          batch_dims_t,
                                          precision,
                                          normalize_data=True):
  """Constructs kernel sin-cos features for fast softmax attention.
  Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    attention_dims_t: tuple of attention dimensions
    batch_dims_t: tuple of batch dimensions
    precision: precision parameter
    normalize_data: predicate indicating whether data should be normalized.
  Returns:
    Random features for fast softmax attention.
  """
  if normalize_data:
    # We have: exp(qk^T/sqrt{d}) = exp(|q|^2/2sqrt{d}) * exp(|k|^2/2sqrt{d}) *
    # exp(-(|q*c-k*c|^2)/2), where c = 1.0 / sqrt{sqrt{d}}.
    data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1])))
  else:
    data_normalizer = 1.0
  ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
  data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix

  data_dash = lax.dot_general(
      data_normalizer * data,
      data_thick_random_matrix,
      (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)),
       (batch_dims_t, batch_dims_t)),
      precision=precision)
  data_dash_cos = ratio * jnp.cos(data_dash)
  data_dash_sin = ratio * jnp.sin(data_dash)
  data_dash = jnp.concatenate((data_dash_cos, data_dash_sin), axis=-1)

  # Constructing D_data and data^{'}
  diag_data = jnp.square(data)
  diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)
  # Additional renormalization for numerical stability
  data_renormalizer = jnp.max(diag_data, attention_dims_t, keepdims=True)
  diag_data -= data_renormalizer
  diag_data = jnp.exp(diag_data)
  data_prime = data_dash * diag_data
  return data_prime


def generalized_kernel_feature_creator(data, projection_matrix, batch_dims_t,
                                       precision, kernel_fn, kernel_epsilon,
                                       normalize_data):
  """Constructs kernel features for fast generalized attention.
  Args:
    data: input for which features are computes
    projection_matrix: matrix used to compute features
    batch_dims_t: tuple of batch dimensions
    precision: precision parameter
    kernel_fn: kernel function used
    kernel_epsilon: additive positive term added to every feature for numerical
      stability
    normalize_data: predicate indicating whether data should be normalized.
  Returns:
    Random features for fast generalized attention.
  """
  if normalize_data:
    data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1])))
  else:
    data_normalizer = 1.0
  if projection_matrix is None:
    return kernel_fn(data_normalizer * data) + kernel_epsilon
  else:
    data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
    data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix
    data_dash = lax.dot_general(
        data_normalizer * data,
        data_thick_random_matrix,
        (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)),
         (batch_dims_t, batch_dims_t)),
        precision=precision)
  data_prime = kernel_fn(data_dash) + kernel_epsilon
  return data_prime


@gin.configurable
def make_fast_softmax_attention(qkv_dim,
                                renormalize_attention=True,
                                numerical_stabilizer=0.000001,
                                nb_features=256,
                                ortho_features=True,
                                ortho_scaling=0.0,
                                redraw_features=True,
                                unidirectional=False,
                                nonnegative_features=True,
                                lax_scan_unroll=1):
  """Construct a fast softmax attention method."""
  logging.info(
      'Fast softmax attention: %s features and orthogonal=%s, renormalize=%s',
      nb_features, ortho_features, renormalize_attention)
  if ortho_features:
    matrix_creator = functools.partial(
        GaussianOrthogonalRandomMatrix,
        nb_features,
        qkv_dim,
        scaling=ortho_scaling)
  else:
    matrix_creator = functools.partial(GaussianUnstructuredRandomMatrix,
                                       nb_features, qkv_dim)
  if nonnegative_features:

    def kernel_feature_creator(data,
                               projection_matrix,
                               attention_dims_t,
                               batch_dims_t,
                               precision,
                               is_query,
                               normalize_data=True):
      return nonnegative_softmax_kernel_feature_creator(
          data, projection_matrix, attention_dims_t, batch_dims_t, precision,
          is_query, normalize_data, numerical_stabilizer)
  else:

    def kernel_feature_creator(data,
                               projection_matrix,
                               attention_dims_t,
                               batch_dims_t,
                               precision,
                               is_query,
                               normalize_data=True):
      del is_query
      return sincos_softmax_kernel_feature_creator(data, projection_matrix,
                                                   attention_dims_t,
                                                   batch_dims_t, precision,
                                                   normalize_data)

  attention_fn = FastAttentionviaLowRankDecomposition(
      matrix_creator,
      kernel_feature_creator,
      renormalize_attention=renormalize_attention,
      numerical_stabilizer=numerical_stabilizer,
      redraw_features=redraw_features,
      unidirectional=unidirectional,
      lax_scan_unroll=lax_scan_unroll).dot_product_attention
  return attention_fn


@gin.configurable
def make_fast_generalized_attention(qkv_dim,
                                    renormalize_attention=True,
                                    numerical_stabilizer=0.0,
                                    nb_features=256,
                                    features_type='deterministic',
                                    kernel_fn=jax.nn.relu,
                                    kernel_epsilon=0.001,
                                    redraw_features=False,
                                    unidirectional=False,
                                    lax_scan_unroll=1):
  """Construct a fast generalized attention menthod."""
  logging.info('Fast generalized attention.: %s features and renormalize=%s',
               nb_features, renormalize_attention)
  if features_type == 'ortho':
    matrix_creator = functools.partial(
        GaussianOrthogonalRandomMatrix, nb_features, qkv_dim, scaling=False)
  elif features_type == 'iid':
    matrix_creator = functools.partial(GaussianUnstructuredRandomMatrix,
                                       nb_features, qkv_dim)
  elif features_type == 'deterministic':
    matrix_creator = None
  else:
    raise ValueError('Unknown feature value type')

  def kernel_feature_creator(data,
                             projection_matrix,
                             attention_dims_t,
                             batch_dims_t,
                             precision,
                             is_query,
                             normalize_data=False):
    del attention_dims_t
    del is_query
    return generalized_kernel_feature_creator(data, projection_matrix,
                                              batch_dims_t, precision,
                                              kernel_fn, kernel_epsilon,
                                              normalize_data)

  attention_fn = FastAttentionviaLowRankDecomposition(
      matrix_creator,
      kernel_feature_creator,
      renormalize_attention=renormalize_attention,
      numerical_stabilizer=numerical_stabilizer,
      redraw_features=redraw_features,
      unidirectional=unidirectional,
      lax_scan_unroll=lax_scan_unroll).dot_product_attention
  return attention_fn


class RandomMatrix(object):
  r"""Abstract class providing a method for constructing 2D random arrays.
  Class is responsible for constructing 2D random arrays.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def get_2d_array(self):
    raise NotImplementedError('Abstract method')


class GaussianUnstructuredRandomMatrix(RandomMatrix):

  def __init__(self, nb_rows, nb_columns, key):
    self.nb_rows = nb_rows
    self.nb_columns = nb_columns
    self.key = key

  def get_2d_array(self):
    return random.normal(self.key, (self.nb_rows, self.nb_columns))


class GaussianOrthogonalRandomMatrix(RandomMatrix):
  r"""Class providing a method to create Gaussian orthogonal matrix.
  Class is responsible for constructing 2D Gaussian orthogonal arrays.
  """

  def __init__(self, nb_rows, nb_columns, key, scaling=0):
    self.nb_rows = nb_rows
    self.nb_columns = nb_columns
    self.key = key
    self.scaling = scaling

  def get_2d_array(self):
    nb_full_blocks = int(self.nb_rows / self.nb_columns)
    block_list = []
    rng = self.key
    for _ in range(nb_full_blocks):
      rng, rng_input = jax.random.split(rng)
      unstructured_block = random.normal(rng_input,
                                         (self.nb_columns, self.nb_columns))
      q, _ = jnp.linalg.qr(unstructured_block)
      q = jnp.transpose(q)
      block_list.append(q)
    remaining_rows = self.nb_rows - nb_full_blocks * self.nb_columns
    if remaining_rows > 0:
      rng, rng_input = jax.random.split(rng)
      unstructured_block = random.normal(rng_input,
                                         (self.nb_columns, self.nb_columns))
      q, _ = jnp.linalg.qr(unstructured_block)
      q = jnp.transpose(q)
      block_list.append(q[0:remaining_rows])
    final_matrix = jnp.vstack(block_list)

    if self.scaling == 0:
      multiplier = jnp.linalg.norm(
          random.normal(self.key, (self.nb_rows, self.nb_columns)), axis=1)
    elif self.scaling == 1:
      multiplier = jnp.sqrt(float(self.nb_columns)) * jnp.ones((self.nb_rows))
    else:
      raise ValueError('Scaling must be one of {0, 1}. Was %s' % self._scaling)

    return jnp.matmul(jnp.diag(multiplier), final_matrix)


class FastAttention(object):
  r"""Abstract class providing a method for fast attention.
  Class is responsible for providing a method <dot_product_attention> for fast
  approximate attention.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def dot_product_attention(self,
                            query,
                            key,
                            value,
                            dtype=jnp.float32,
                            bias=None,
                            axis=None,
                            broadcast_dropout=True,
                            dropout_rng=None,
                            dropout_rate=0.,
                            deterministic=False,
                            precision=None):
    """Computes dot-product attention given query, key, and value.
    This is the core function for applying fast approximate dot-product
    attention. It calculates the attention weights given query and key and
    combines the values using the attention weights. This function supports
    multi-dimensional inputs.
    Args:
      query: queries for calculating attention with shape of [batch_size, dim1,
        dim2, ..., dimN, num_heads, mem_channels].
      key: keys for calculating attention with shape of [batch_size, dim1, dim2,
        ..., dimN, num_heads, mem_channels].
      value: values to be used in attention with shape of [batch_size, dim1,
        dim2,..., dimN, num_heads, value_channels].
      dtype: the dtype of the computation (default: float32)
      bias: bias for the attention weights. This can be used for incorporating
        autoregressive mask, padding mask, proximity bias.
      axis: axises over which the attention is applied.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout.
      dropout_rate: dropout rate.
      deterministic: bool, deterministic or not (to apply dropout).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
    Returns:
      Output of shape [bs, dim1, dim2, ..., dimN,, num_heads, value_channels].
    """
    raise NotImplementedError('Abstract method')


def _numerator(z_slice_shape, precision, unroll=1):

  def fwd(qs, ks, vs):

    def body(p, qkv):
      (q, k, v) = qkv
      p += jnp.einsum('...m,...d->...md', k, v, precision=precision)
      X_slice = jnp.einsum('...m,...md->...d', q, p, precision=precision)
      return p, X_slice

    init_value = jnp.zeros(z_slice_shape)
    p, W = lax.scan(body, init_value, (qs, ks, vs), unroll=unroll)
    return W, (p, qs, ks, vs)

  def bwd(pqkv, W_ct):

    def body(carry, qkv_xct):
      p, p_ct = carry
      q, k, v, x_ct = qkv_xct
      q_ct = jnp.einsum('...d,...md->...m', x_ct, p, precision=precision)
      p_ct += jnp.einsum('...d,...m->...md', x_ct, q, precision=precision)
      k_ct = jnp.einsum('...md,...d->...m', p_ct, v, precision=precision)
      v_ct = jnp.einsum('...md,...m->...d', p_ct, k, precision=precision)
      p -= jnp.einsum('...m,...d->...md', k, v, precision=precision)
      return (p, p_ct), (q_ct, k_ct, v_ct)

    p, qs, ks, vs = pqkv
    _, (qs_ct, ks_ct, vs_ct) = lax.scan(
        body, (p, jnp.zeros_like(p)), (qs, ks, vs, W_ct),
        reverse=True,
        unroll=unroll)
    return qs_ct, ks_ct, vs_ct

  @jax.custom_vjp
  def _numerator_impl(qs, ks, vs):
    W, _ = fwd(qs, ks, vs)
    return W

  _numerator_impl.defvjp(fwd, bwd)

  return _numerator_impl


def _denominator(t_slice_shape, precision, unroll=1):

  def fwd(qs, ks):

    def body(p, qk):
      q, k = qk
      p += k
      x = jnp.einsum('...m,...m->...', q, p, precision=precision)
      return p, x

    p = jnp.zeros(t_slice_shape)
    p, R = lax.scan(body, p, (qs, ks), unroll=unroll)
    return R, (qs, ks, p)

  def bwd(qkp, R_ct):

    def body(carry, qkx):
      p, p_ct = carry
      q, k, x_ct = qkx
      q_ct = jnp.einsum('...,...m->...m', x_ct, p, precision=precision)
      p_ct += jnp.einsum('...,...m->...m', x_ct, q, precision=precision)
      k_ct = p_ct
      p -= k
      return (p, p_ct), (q_ct, k_ct)

    qs, ks, p = qkp
    _, (qs_ct, ks_ct) = lax.scan(
        body, (p, jnp.zeros_like(p)), (qs, ks, R_ct),
        reverse=True,
        unroll=unroll)
    return (qs_ct, ks_ct)

  @jax.custom_vjp
  def _denominator_impl(qs, ks):
    R, _ = fwd(qs, ks)
    return R

  _denominator_impl.defvjp(fwd, bwd)

  return _denominator_impl


class FastAttentionviaLowRankDecomposition(FastAttention):
  r"""Class providing a method for fast attention via low rank decomposition.
  Class is responsible for providing a method <dot_product_attention> for fast
  dot-product attention with the use of low rank decomposition (e.g. with
  random feature maps).
  """

  def __init__(self,
               matrix_creator,
               kernel_feature_creator,
               renormalize_attention,
               numerical_stabilizer,
               redraw_features,
               unidirectional,
               lax_scan_unroll=1):  # For optimal GPU performance, set to 16.
    rng = random.PRNGKey(0)
    self.matrix_creator = matrix_creator
    self.projection_matrix = self.draw_weights(rng)
    self.kernel_feature_creator = kernel_feature_creator
    self.renormalize_attention = renormalize_attention
    self.numerical_stabilizer = numerical_stabilizer
    self.redraw_features = redraw_features
    self.unidirectional = unidirectional
    self.lax_scan_unroll = lax_scan_unroll

  def draw_weights(self, key):
    if self.matrix_creator is None:
      return None
    matrixrng, _ = random.split(key)
    projection_matrix = self.matrix_creator(key=matrixrng).get_2d_array()
    return projection_matrix

  def dot_product_attention(self,
                            query,
                            key,
                            value,
                            dtype=jnp.float32,
                            bias=None,
                            axis=None,
                            broadcast_dropout=True,
                            dropout_rng=None,
                            dropout_rate=0.,
                            deterministic=False,
                            precision=None):

    assert key.shape[:-1] == value.shape[:-1]
    assert (query.shape[0:1] == key.shape[0:1] and
            query.shape[-1] == key.shape[-1])
    if axis is None:
      axis = tuple(range(1, key.ndim - 2))
    if not isinstance(axis, Iterable):
      axis = (axis,)
    assert key.ndim == query.ndim
    assert key.ndim == value.ndim
    for ax in axis:
      if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
        raise ValueError('Attention axis must be between the batch '
                         'axis and the last-two axes.')
    n = key.ndim

    # Constructing projection tensor.
    if self.redraw_features:
      # TODO(kchoro): Get rid of the constant below.
      query_seed = lax.convert_element_type(
          jnp.ceil(jnp.sum(query) * 10000000.0), jnp.int32)
      rng = random.PRNGKey(query_seed)
      self.projection_matrix = self.draw_weights(rng)

    # batch_dims is  <bs, <non-attention dims>, num_heads>
    batch_dims = tuple(np.delete(range(n), axis + (n - 1,)))
    # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
    qk_perm = batch_dims + axis + (n - 1,)
    k_extra_perm = axis + batch_dims + (n - 1,)
    key_extra = key.transpose(k_extra_perm)
    key = key.transpose(qk_perm)
    query = query.transpose(qk_perm)
    # v -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
    v_perm = batch_dims + axis + (n - 1,)
    value = value.transpose(v_perm)
    batch_dims_t = tuple(range(len(batch_dims)))
    attention_dims_t = tuple(
        range(len(batch_dims),
              len(batch_dims) + len(axis)))

    # Constructing tensors Q^{'} and K^{'}.
    query_prime = self.kernel_feature_creator(query, self.projection_matrix,
                                              attention_dims_t, batch_dims_t,
                                              precision, True)
    key_prime = self.kernel_feature_creator(key, self.projection_matrix,
                                            attention_dims_t, batch_dims_t,
                                            precision, False)

    if self.unidirectional:
      index = attention_dims_t[0]
      z_slice_shape = key_prime.shape[0:len(batch_dims_t)] + (
          key_prime.shape[-1],) + (value.shape[-1],)

      numerator_fn = _numerator(z_slice_shape, precision, self.lax_scan_unroll)
      W = numerator_fn(
          jnp.moveaxis(query_prime, index, 0),
          jnp.moveaxis(key_prime, index, 0), jnp.moveaxis(value, index, 0))

      # Constructing W = (Q^{'}(K^{'})^{T})_{masked}V
      W = jnp.moveaxis(W, 0, index)

      if not self.renormalize_attention:
        # Unidirectional, not-normalized attention.
        perm_inv = _invert_perm(qk_perm)
        result = W.transpose(perm_inv)
        return result
      else:
        # Unidirectional, normalized attention.
        thick_all_ones = jnp.zeros(key.shape[0:-1]) + jnp.ones(
            key_extra.shape[0:len(axis)])

        index = attention_dims_t[0]
        t_slice_shape = key_prime.shape[0:len(batch_dims_t)] + (
            key_prime.shape[-1],)
        denominator_fn = _denominator(t_slice_shape, precision,
                                      self.lax_scan_unroll)
        R = denominator_fn(
            jnp.moveaxis(query_prime, index, 0),
            jnp.moveaxis(key_prime, index, 0))

        R = jnp.moveaxis(R, 0, index)
    else:
      contract_query = tuple(
          range(len(batch_dims) + len(axis),
                len(batch_dims) + len(axis) + 1))
      contract_z = tuple(range(len(batch_dims), len(batch_dims) + 1))
      # Constructing Z = (K^{'})^{T}V
      # Z (bs, <non-attention dims>, num_heads, channels_m, channels_v)
      Z = lax.dot_general(
          key_prime,
          value,
          ((attention_dims_t, attention_dims_t), (batch_dims_t, batch_dims_t)),
          precision=precision)
      # Constructing W = Q^{'}Z = Q^{'}(K^{'})^{T}V
      # q (bs, <non-attention dims>, num_heads, <attention dims>, channels_m)
      # Z (bs, <non-attention dims>, num_heads, channels_m, channels_v)
      # W (bs,  <non-attention dims>, num_heads, <attention dims>, channels_v)
      W = lax.dot_general(
          query_prime,
          Z, ((contract_query, contract_z), (batch_dims_t, batch_dims_t)),
          precision=precision)
      if not self.renormalize_attention:
        # Bidirectional, not-normalized attention.
        perm_inv = _invert_perm(qk_perm)
        result = W.transpose(perm_inv)
        return result
      else:
        # Bidirectional, normalized attention.
        thick_all_ones = jnp.zeros(key.shape[0:-1]) + jnp.ones(
            key_extra.shape[0:len(axis)])
        contract_key = tuple(
            range(len(batch_dims),
                  len(batch_dims) + len(axis)))
        contract_thick_all_ones = tuple(
            range(thick_all_ones.ndim - len(axis), thick_all_ones.ndim))
        # Construct T = (K^{'})^{T} 1_L
        # k (bs, <non-attention dims>, num_heads, <attention dims>, channels)
        T = lax.dot_general(
            key_prime,
            thick_all_ones, ((contract_key, contract_thick_all_ones),
                             (batch_dims_t, batch_dims_t)),
            precision=precision)

        # Construct partition function: R = Q^{'} T = Q^{'}(K^{'})^{T} 1_L
        # q_p (bs, <non-attention dims>, num_heads, <attention dims>, channs_m)
        # T   (bs, <non-attention dims>, num_heads, channels_m)
        R = lax.dot_general(
            query_prime,
            T, (((query_prime.ndim - 1,), (T.ndim - 1,)),
                (batch_dims_t, range(0,
                                     len(T.shape) - 1))),
            precision=precision)

    R = R + 2 * self.numerical_stabilizer * (
        jnp.abs(R) <= self.numerical_stabilizer)
    R = jnp.reciprocal(R)
    R = jnp.expand_dims(R, len(R.shape))
    # W (bs, <non-attention dims>, num_heads, <attention dims>, channels_v)
    # R (bs, <non-attention dims>, num_heads, <attention dims>, extra_channel)
    result = W * R
    # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
    perm_inv = _invert_perm(qk_perm)
    result = result.transpose(perm_inv)
    return result



def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)

def dot_product_attention_Modified(query,
                          key,
                          value,
                          dtype=jnp.float32,
                          bias=None,
                          axis=None,
                          broadcast_dropout=True,
                          dropout_rng=None,
                          dropout_rate=0.,
                          deterministic=False,
                          precision=None):
  """DEPRECATION WARNING:
 "The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs.


  Args:
    query: queries for calculating attention with shape of `[batch_size, dim1,
      dim2, ..., dimN, num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
      ..., dimN, num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size, dim1,
      dim2,..., dimN, num_heads, value_channels]`.
    dtype: the dtype of the computation (default: float32)
    bias: bias for the attention weights. This can be used for incorporating
      autoregressive mask, padding mask, proximity bias.
    axis: axises over which the attention is applied.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
  """
  assert key.shape[:-1] == value.shape[:-1]
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1])

  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
      raise ValueError('Attention axis must be between the batch '
                       'axis and the last-two axes.')
  depth = query.shape[-1]
  n = key.ndim
  # batch_dims is  <bs, <non-attention dims>, num_heads>
  batch_dims = tuple(np.delete(range(n), axis + (n - 1,)))
  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
  qk_perm = batch_dims + axis + (n - 1,)
  key = key.transpose(qk_perm)
  query = query.transpose(qk_perm)
  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  query = query / jnp.sqrt(depth).astype(dtype)
  batch_dims_t = tuple(range(len(batch_dims)))

  #softMax on key
  key_dims=tuple(range(key.ndim - len(axis), key.ndim))
  key_soft = softmax(key, axis=key_dims)
  key_soft = key_soft.astype(dtype)

  # carry out the dot product between softMax(key)T and value
  part_results = lax.dot_general(
      key_soft,
      value, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
      precision=precision)

  # apply attention bias: masking, droput, proximity bias, ect.
  if bias is not None:
    part_results = part_results + bias



  # apply dropout
  if not deterministic and dropout_rate > 0.:
    if dropout_rng is None:
      dropout_rng = make_rng()
    keep_prob = jax.lax.tie_in(part_results, 1.0 - dropout_rate)
    if broadcast_dropout:
      # dropout is broadcast across the batch+head+non-attention dimension
      dropout_dims = part_results.shape[-(2 * len(axis)):]
      dropout_shape = (tuple([1] * len(batch_dims_t)) + dropout_dims)
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, part_results.shape)
    multiplier = (keep.astype(part_results.dtype) /
                  jnp.asarray(keep_prob, dtype=dtype))
    part_results = part_results * multiplier

  # carry out the dot product between query and part_results
  results = lax.dot_general(
      query,
      part_results, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
      precision=precision)

  # normalize the results
  norm_dims = tuple(range(results.ndim - len(axis), results.ndim))
  results = results/cmath.sqrt(norm_dims)

  # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
  perm_inv = _invert_perm(qk_perm)
  results = results.transpose(perm_inv)
  return results


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)


@struct.dataclass
class _CacheEntry:
  key: np.ndarray
  value: np.ndarray
  i: np.ndarray


def scan_in_dim(*args, **kwargs):
  warnings.warn('scan_in_dim moved to flax.jax_utils',
                DeprecationWarning)
  return jax_utils.scan_in_dim(*args, **kwargs)


class Cache(Collection):
  """The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Collect intermediate activations for efficient autoregressive decoding."""

  def initialize_cache(self, shape, dtype=None):
    """Initialize the cache for the given input shape.

    Args:
      shape: the shape of the batch and attention dimensions.
      dtype: the dtype of the autoregressive cache.
    Returns:
      the initialized cache
    """
    if dtype is None:
      dtype = jnp.float32
    def _init(shape_data):
      ndim = int(shape_data[0])
      tail_shape = tuple(shape_data[1:])
      full_shape = shape + tail_shape
      if len(full_shape) != ndim:
        raise ValueError('Shape should be a tuple with the shape of the batch'
                         'and attention dims.')
      return _CacheEntry(key=jnp.zeros(full_shape, dtype=dtype),
                         value=jnp.zeros(full_shape, dtype=dtype),
                         i=jnp.zeros((), jnp.uint32))
    return Cache(jax.tree_map(_init, self.state))


jax.tree_util.register_pytree_node(
    Cache, iterate_collection, collection_from_iterable)


class MultiHeadDotProductAttention_Modified(Module):
  """The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Multi-head dot-product attention."""

  def apply(self,
            inputs_q,
            inputs_kv,
            num_heads,
            dtype=jnp.float32,
            qkv_features=None,
            out_features=None,
            attention_axis=None,
            causal_mask=False,
            padding_mask=None,
            key_padding_mask=None,
            segmentation=None,
            key_segmentation=None,
            cache=None,
            broadcast_dropout=True,
            dropout_rng=None,
            dropout_rate=0.,
            deterministic=False,
            precision=None,
            kernel_init=default_kernel_init,
            bias_init=zeros,
            bias=True,
            attention_fn=FastAttention.dot_product_attention):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
      inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]`
        or None for self-attention, inn which case key/values will be derived
        from inputs_q.
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      attention_axis: axes over which the attention is applied ( 'None' means
        attention over all axes, but batch, heads, and features).
      causal_mask: boolean specifying whether to apply a causal mask on the
        attention weights. If True, the output at timestep `t` will not depend
        on inputs at timesteps strictly greater than `t`.
      padding_mask: boolean specifying query tokens that are pad token w/ False.
      key_padding_mask: boolean specifying key-value tokens that are pad token
        w/ False.
      segmentation: segment indices for packed inputs_q data.
      key_segmentation: segment indices for packed inputs_kv data.
      cache: an instance of `flax.nn.attention.Cache` used for efficient
        autoregressive decoding.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
      query, key, value, and returns output of shape
      `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``

    Returns:
      output of shape `[bs, dim1, dim2, ..., dimN, features]`.
    """

    assert causal_mask or not cache, (
        'Caching is only support for causal attention.')

    if inputs_kv is None:
      inputs_kv = inputs_q

    is_self_attention = inputs_kv is inputs_q

    if attention_axis is None:
      attention_axis = tuple(range(1, inputs_q.ndim - 1))

    features = out_features or inputs_q.shape[-1]
    qkv_features = qkv_features or inputs_q.shape[-1]

    assert qkv_features % num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // num_heads

    dense = DenseGeneral.partial(
        axis=-1,
        features=(num_heads, head_dim),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        precision=precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, dims..., n_heads, n_features_per_head]
    query, key, value = (dense(inputs_q, dtype=dtype, name='query'),
                         dense(inputs_kv, dtype=dtype, name='key'),
                         dense(inputs_kv, dtype=dtype, name='value'))

    if cache:
      assert isinstance(cache, Cache), 'cache must be an instance of Cache'
      if self.is_initializing():
        cache.store(np.array((key.ndim,) + key.shape[-2:], dtype=np.int32))
      else:
        cache_entry = cache.retrieve(None)
        expected_shape = list(cache_entry.key.shape[:-2])
        for attn_dim in attention_axis:
          expected_shape[attn_dim] = 1
        expected_shape = tuple(expected_shape) + inputs_q.shape[-1:]
        if expected_shape != inputs_q.shape:
          raise ValueError('Invalid shape provided, '
                           'expected shape %s instead got %s.' %
                           (expected_shape, inputs_q.shape))

        if not isinstance(cache_entry, _CacheEntry):
          raise ValueError('Cache is not initialized.')

        cshape = cache_entry.key.shape
        indices = [0] * len(cshape)
        i = cache_entry.i
        attn_size = np.prod(np.take(cshape, attention_axis))
        for attn_dim in attention_axis:
          attn_size //= cshape[attn_dim]
          indices[attn_dim] = i // attn_size
          i = i % attn_size

        key = lax.dynamic_update_slice(cache_entry.key, key, indices)
        value = lax.dynamic_update_slice(cache_entry.value, value, indices)
        one = jnp.array(1, jnp.uint32)
        cache_entry = cache_entry.replace(i=cache_entry.i + one,
                                          key=key,
                                          value=value)
        cache.store(cache_entry)

    # create attention masks
    mask_components = []

    if causal_mask:
      if cache and not self.is_initializing():
        bias_pre_shape = (1,) * (key.ndim - 1)
        attn_shape = tuple(np.take(key.shape, attention_axis))
        attn_size = np.prod(attn_shape)
        ii = jnp.arange(attn_size, dtype=jnp.uint32)
        mask = ii < cache_entry.i
        mask_components.append(mask.reshape(bias_pre_shape + attn_shape))
      else:
        mask_components.append(_make_causal_mask(key, attention_axis))

    if (padding_mask is not None or key_padding_mask is not None) and not cache:
      if key_padding_mask is None:
        if is_self_attention:
          key_padding_mask = padding_mask
        else:
          key_padding_shape = [inputs_kv.shape[dim] for dim in attention_axis]
          key_padding_mask = jnp.full(key_padding_shape, True)
      if padding_mask is None:
        if is_self_attention:
          padding_mask = key_padding_mask
        else:
          padding_shape = [inputs_q.shape[dim] for dim in attention_axis]
          padding_mask = jnp.full(padding_shape, True)

      padding_mask = make_padding_mask(
          padding_mask_query=padding_mask,
          padding_mask_key=key_padding_mask,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis)
      mask_components.append(padding_mask)

    if segmentation is not None:
      if key_segmentation is None:
        assert is_self_attention
        key_segmentation = segmentation
      segmentation_mask = make_padding_mask(
          padding_mask_query=segmentation,
          padding_mask_key=key_segmentation,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis,
          segmentation_mask=True)
      mask_components.append(segmentation_mask)

    if mask_components:
      attention_mask = mask_components[0]
      for component in mask_components[1:]:
        attention_mask = jnp.logical_and(attention_mask, component)

      # attention mask in the form of attention bias
      attention_bias = lax.select(
          attention_mask > 0, jnp.full(attention_mask.shape, 0.).astype(dtype),
          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
      attention_bias = None

    # apply attention
    x = attention_fn(
        query,
        key,
        value,
        dtype=dtype,
        axis=attention_axis,
        bias=attention_bias,
        precision=precision,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic)

    # back to the original inputs dimensions
    out = DenseGeneral(
        x,
        features=features,
        axis=(-2, -1),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        dtype=dtype,
        precision=precision,
        name='out')

    return out


# TODO(flax-dev): Consider refactoring MultiHeadDotProductAttention and moving
# causal_mask and cache support into this class instead.
SelfAttention = MultiHeadDotProductAttention_Modified.partial(inputs_kv=None)


def make_padding_mask(padding_mask_query,
                      padding_mask_key,
                      query_shape,
                      key_shape,
                      attention_axis=None,
                      segmentation_mask=False):
  """The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Makes padding mask for attention weights.

  In case of 1d inputs (i.e., `[bs, len, features]`, the attention weights will
  be `[bs, len, len]` and this function makes a square matrix [len, len].

  Args:
    padding_mask_query: padding mask of query <bs, qdim1,.., qdimn>
    padding_mask_key: padding mask of query <bs, key1,.., keyn>
    query_shape: shape of the query
    key_shape: shape of the key, which is equal to the shape of value.
    attention_axis: axis over which attention is applied.
    segmentation_mask: bool: if true use equality on cartesian product rather
      than outer product for constructing segmentation masks.
  Returns:
    The padding mask for attention weights.
  """
  assert query_shape[0] == key_shape[0]
  assert len(query_shape) == len(key_shape)

  ndim = len(key_shape)
  if attention_axis is None:
    attention_axis = tuple(range(1, ndim - 2))
  assert isinstance(attention_axis, tuple)
  for ax in attention_axis:
    if not (ndim >= 3 and 1 <= ax < ndim - 2):
      raise ValueError(
          'Attention axis must be between the batch axis and the last-two axes.'
      )

  mask_shape_final = (query_shape[0], 1)  #  batch_size, 1 (for all heads)s
  for ax in attention_axis:
    mask_shape_final += (query_shape[ax],)
  for ax in attention_axis:
    mask_shape_final += (key_shape[ax],)

  padding_mask_query = padding_mask_query[..., None]
  padding_mask_key = padding_mask_key[..., None]
  perm = (0,) + tuple(np.flip(np.arange(padding_mask_key.ndim)))[:-1]
  if segmentation_mask:
    mask = jnp.equal(padding_mask_query, padding_mask_key.transpose(perm))
  else:
    mask = jnp.multiply(padding_mask_query, padding_mask_key.transpose(perm))

  mask = mask.reshape(mask_shape_final)
  mask = jax.lax.convert_element_type(mask, jnp.float32)
  return mask


def _make_causal_mask(key, attention_axis=None, self_mask=False):
  """The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Makes a causal mask, to be used for masking out the future for attention.

  In case of 1d inputs (i.e., `[bs, len, features]`, the attention weights will
  be `[bs, len, len]` and this function makes a square matrix [len, len] with
  zeros in upper triangle and ones in lower triangle.

  Args:
    key: shape of the key, which is equal to the shape of value and is
      assumed to be equal to the shape of the query (since this is used in
      self-attention when decoding).
    attention_axis: axis over which attention is applied.
    self_mask: if mask out the diagonal or not.

  Returns:
    A causal mask to be used to mask out future positions.
  """
  if attention_axis is None:
    attention_axis = tuple(range(1, key.ndim - 2))
  assert isinstance(attention_axis, tuple)
  for ax in attention_axis:
    if not (key.ndim >= 3 and 1 <= ax < key.ndim - 2):
      raise ValueError(
          'Attention axis must be between the batch axis and the last-two axes.'
      )

  mask_shape = tuple([1] * (key.ndim - len(attention_axis) - 1))
  mask_shape_final = mask_shape
  for _ in range(2):
    flatten_dim = 1
    for ax in attention_axis:
      mask_shape_final += (key.shape[ax],)
      flatten_dim *= key.shape[ax]
    mask_shape += (flatten_dim,)

  def tri(n, m, k=0):
    # Tie in the key to avoid the mask becoming a constant.
    # This way XLA can construct the mask during computation and fuse it
    # with the attention ops.
    x = lax.tie_in(key, jnp.arange(n, dtype=jnp.int32))
    y = lax.tie_in(key, jnp.arange(m, dtype=jnp.int32))
    mask = lax.ge(
        (lax.broadcast_in_dim(x, shape=(n, m), broadcast_dimensions=(0,))) + k,
        lax.broadcast(y, [n]))
    return mask

  k = -1 if self_mask else 0
  mask = tri(*mask_shape[-2:], k=k).reshape(mask_shape_final)
  return mask