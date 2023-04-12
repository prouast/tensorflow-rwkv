import pytest
import numpy as np
import tensorflow as tf
from tensorflow_rwkv.layers.rwkv import RWKV, RWKVRNNCell

def _forward_gpt_mode(inputs, embed_dim, hidden_dim, reg_lambda):
  inputs_op = tf.convert_to_tensor(inputs, dtype=tf.float32)
  assert embed_dim == inputs_op.shape[2] 
  output = RWKV(
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    reg_lambda=reg_lambda
  )(inputs_op)
  return output

def _forward_rnn_mode(inputs, embed_dim, hidden_dim, reg_lambda):
  inputs_op = tf.convert_to_tensor(inputs, dtype=tf.float32)
  assert embed_dim == inputs_op.shape[2]
  output = tf.keras.layers.RNN(
    cell=RWKVRNNCell(
      embed_dim=embed_dim,
      hidden_dim=hidden_dim,
      reg_lambda=reg_lambda, name="rwkv"),
    return_sequences=True,
    return_state=False,
    name="rnn"
  )(inputs_op)
  return output

def _create_test_data():
  # (B, T, C) = (2, 4, 8)
  inputs_np = np.array(
    [[[1.0, 2.0, 3.0, 0.0, 0.5, -.1, -1.0, 0.0],
      [1.1, 2.1, 4.0, 0.1, 0.4, -.2, -1.1, 0.0],
      [1.5, 2.5, 5.0, 0.0, 0.3, -.4, -1.5, 0.0],
      [0.9, 1.9, 6.0, -.1, 0.2, -.8, -0.9, 0.0]],
     [[1.0, 2.0, 3.0, 0.0, 0.5, -.1, -1.0, 0.0],
      [1.1, 2.1, 4.0, 0.1, 0.4, -.2, -1.1, 0.0],
      [1.5, 2.5, 5.0, 0.0, 0.3, -.4, -1.5, 0.0],
      [0.9, 1.9, 6.0, -.1, 0.2, -.8, -0.9, 0.0]]],
    dtype=np.float32)
  return inputs_np

def test_forward():
  inputs_np = _create_test_data()
  inputs = tf.convert_to_tensor(inputs_np, dtype=tf.float32)

  outputs_gpt_mode = _forward_gpt_mode(inputs, embed_dim=8, hidden_dim=16, reg_lambda=0.)
  print("outputs_gpt_mode: {}".format(outputs_gpt_mode))

  outputs_rnn_mode = _forward_gpt_mode(inputs, embed_dim=8, hidden_dim=16, reg_lambda=0.)
  print("outputs_rnn_mode: {}".format(outputs_rnn_mode))
  assert False
