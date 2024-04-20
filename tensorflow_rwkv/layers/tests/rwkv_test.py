import os
import pytest
import tensorflow as tf
from tensorflow_rwkv.layers.rwkv import RWKV, RWKVRNNCell, _wkv

class Identity(tf.keras.layers.Wrapper):
  """Identity wrapper layer"""
  def call(self, inputs, **kwargs):
    return self.layer(inputs, **kwargs)

def _create_model(rnn_mode, input_shape, hidden_dim):
  _, _, embed_dim = input_shape
  inputs = tf.keras.Input(shape=input_shape[1:])
  if rnn_mode:
    layer = tf.keras.layers.RNN(
      cell=RWKVRNNCell(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        reg_lambda=0.0, name="rwkv"),
      return_sequences=True,
      return_state=False,
      name="rnn")
  else:
    layer = Identity(
      RWKV(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        reg_lambda=0.0),
      name="rnn")
  outputs = layer(inputs)
  return tf.keras.Model(inputs=inputs, outputs=outputs, name="RWKVModel")

def _forward_gpt_mode(inputs, hidden_dim, ckpt_path):
  inputs_op = tf.convert_to_tensor(inputs, dtype=tf.float32)
  model = _create_model(rnn_mode=False, input_shape=inputs_op.shape, hidden_dim=hidden_dim)
  load_status = model.load_weights(ckpt_path, by_name=True)
  if load_status: load_status.assert_consumed()
  return model(inputs_op)

def _forward_rnn_mode(inputs, hidden_dim, ckpt_path):
  inputs_op = tf.convert_to_tensor(inputs, dtype=tf.float32)
  model = _create_model(rnn_mode=True, input_shape=inputs_op.shape, hidden_dim=hidden_dim)
  load_status = model.load_weights(ckpt_path, by_name=True)
  if load_status: load_status.assert_consumed()
  return model(inputs_op)

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_forward(dtype):
  """Assert that forward outputs of RWKV RNN mode and GPT mode are near enough"""
  if dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
  else:
    tf.keras.mixed_precision.set_global_policy('float32')
  B = 2
  T = 4
  embed_dim = 8
  hidden_dim = 16
  inputs = tf.random.uniform(shape=(B, T, embed_dim), dtype=tf.float32)
  # Init weights
  ckpt_path = "test"
  model = _create_model(rnn_mode=False, input_shape=inputs.shape, hidden_dim=hidden_dim)
  model.save_weights(filepath=ckpt_path, overwrite=True, save_format="h5")
  # Load for GPT and RNN mode and compute outputs
  outputs_gpt_mode = _forward_gpt_mode(inputs, hidden_dim=hidden_dim, ckpt_path=ckpt_path)
  outputs_rnn_mode = _forward_rnn_mode(inputs, hidden_dim=hidden_dim, ckpt_path=ckpt_path)
  # Compare outputs
  tf.debugging.assert_near(outputs_gpt_mode, outputs_rnn_mode, rtol=1e-3, atol=1e-3)
  # Clean up
  os.remove(ckpt_path)

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_gradients(dtype):
  """Gradient check for WKV op"""
  tol = 1e-3 if dtype == tf.float32 else 0.5
  B = 2
  T = 8
  C = 32
  k = tf.random.uniform(shape=(B, T, C), dtype=dtype)
  v = tf.random.uniform(shape=(B, T, C), dtype=dtype)
  w = -tf.exp(tf.random.uniform(shape=(C,), dtype=dtype))
  u = tf.random.uniform(shape=(C,), dtype=dtype)
  theoretical, numerical = tf.test.compute_gradient(_wkv, [k, v, w, u])
  tf.debugging.assert_near(theoretical[0], numerical[0], rtol=tol, atol=tol)
  tf.debugging.assert_near(theoretical[1], numerical[1], rtol=tol, atol=tol)
  tf.debugging.assert_near(theoretical[2], numerical[2], rtol=tol, atol=tol)
  tf.debugging.assert_near(theoretical[3], numerical[3], rtol=tol, atol=tol)

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_keras_gpt_mode(dtype):
  if dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
  else:
    tf.keras.mixed_precision.set_global_policy('float32')
  B = 2
  T = 4
  embed_dim = 8
  hidden_dim = 16
  inputs = tf.random.uniform(shape=(B, T, embed_dim), dtype=dtype)
  input_layer = tf.keras.Input(shape=inputs.shape[1:])
  rwkv_layer = RWKV(embed_dim=embed_dim, hidden_dim=hidden_dim, reg_lambda=0.0)
  output_layer = rwkv_layer(input_layer)
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="RWKVModel")
  expected_output_shape = rwkv_layer.compute_output_shape(inputs.shape)
  actual_output = model(inputs)
  assert tf.keras.backend.dtype(output_layer) == dtype
  assert actual_output.shape[1:] == expected_output_shape[1:]

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_keras_rnn_mode(dtype):
  if dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
  else:
    tf.keras.mixed_precision.set_global_policy('float32')
  B = 2
  T = 4
  embed_dim = 8
  hidden_dim = 16
  inputs = tf.random.uniform(shape=(B, T, embed_dim), dtype=dtype)
  input_layer = tf.keras.Input(shape=inputs.shape[1:])
  rwkv_layer = tf.keras.layers.RNN(
    cell=RWKVRNNCell(embed_dim=embed_dim, hidden_dim=hidden_dim, reg_lambda=0.0),
    return_sequences=True, return_state=False)
  output_layer = rwkv_layer(input_layer)
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="RWKVModel")
  expected_output_shape = rwkv_layer.compute_output_shape(inputs.shape)
  actual_output = model(inputs)
  assert tf.keras.backend.dtype(output_layer) == dtype
  assert actual_output.shape[1:] == expected_output_shape[1:]
