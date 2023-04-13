import pytest
import tensorflow as tf
from tensorflow_rwkv.layers.rwkv import RWKV, RWKVRNNCell, _wkv

# python -m pytest -v --functions-durations=20 --modules-durations=5 $SKIP_CUSTOM_OP_TESTS_FLAG $EXTRA_ARGS ./tensorflow_rwkv

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
def test_forward():
  """Assert that forward outputs of RWKV RNN mode and GPT mode are near enough"""
  B = 2
  T = 4
  embed_dim = 8
  hidden_dim = 16
  inputs = tf.random.uniform(shape=(B, T, embed_dim), dtype=tf.float32)
  # Init weights
  ckpt_path = "test"
  model = _create_model(rnn_mode=False, input_shape=inputs.shape, hidden_dim=hidden_dim)
  model.save_weights(filepath=ckpt_path, overwrite=True, save_format="h5")
  # GPT mode model
  outputs_gpt_mode = _forward_gpt_mode(inputs, hidden_dim=hidden_dim, ckpt_path=ckpt_path)
  outputs_rnn_mode = _forward_rnn_mode(inputs, hidden_dim=hidden_dim, ckpt_path=ckpt_path)
  tf.debugging.assert_near(outputs_gpt_mode, outputs_rnn_mode)

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.with_device(["cpu", "gpu"])
def test_gradients():
  """Gradient check for WKV op"""
  B = 1
  T = 4
  C = 32
  k = tf.random.uniform(shape=(B, T, C), dtype=tf.float32)
  v = tf.random.uniform(shape=(B, T, C), dtype=tf.float32)
  w = -tf.exp(tf.random.uniform(shape=(C,), dtype=tf.float32))
  u = tf.random.uniform(shape=(C,), dtype=tf.float32)
  theoretical, numerical = tf.test.compute_gradient(_wkv, [k, v, w, u])
  tf.debugging.assert_near(theoretical[0], numerical[0], rtol=1e-4, atol=1e-4)
  tf.debugging.assert_near(theoretical[1], numerical[1], rtol=1e-4, atol=1e-4)
  tf.debugging.assert_near(theoretical[2], numerical[2], rtol=1e-4, atol=1e-4)
  tf.debugging.assert_near(theoretical[3], numerical[3], rtol=1e-4, atol=1e-4)

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.with_device(["cpu", "gpu"])
def test_keras_gpt_mode():
  B = 2
  T = 4
  embed_dim = 8
  hidden_dim = 16
  inputs = tf.random.uniform(shape=(B, T, embed_dim), dtype=tf.float32)
  input_layer = tf.keras.Input(shape=inputs.shape[1:])
  rwkv_layer = RWKV(embed_dim=embed_dim, hidden_dim=hidden_dim, reg_lambda=0.0)
  output_layer = rwkv_layer(input_layer)
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="RWKVModel")
  expected_output_shape = rwkv_layer.compute_output_shape(inputs.shape)
  actual_output = model(inputs)
  expected_output_type = "float32"
  assert tf.keras.backend.dtype(output_layer) == expected_output_type
  assert actual_output.shape[1:] == expected_output_shape[1:]

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.with_device(["cpu", "gpu"])
def test_keras_rnn_mode():
  B = 2
  T = 4
  embed_dim = 8
  hidden_dim = 16
  inputs = tf.random.uniform(shape=(B, T, embed_dim), dtype=tf.float32)
  input_layer = tf.keras.Input(shape=inputs.shape[1:])
  rwkv_layer = tf.keras.layers.RNN(
    cell=RWKVRNNCell(embed_dim=inputs.shape[2], hidden_dim=16, reg_lambda=0.0),
    return_sequences=True, return_state=False)
  output_layer = rwkv_layer(input_layer)
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="RWKVModel")
  expected_output_shape = rwkv_layer.compute_output_shape(inputs.shape)
  actual_output = model(inputs)
  expected_output_type = "float32"
  assert tf.keras.backend.dtype(output_layer) == expected_output_type
  assert actual_output.shape[1:] == expected_output_shape[1:]
