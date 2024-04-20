import tensorflow as tf
from typeguard import typechecked
from tensorflow_rwkv.utils.resource_loader import LazySO

_wkv_so = LazySO("custom_ops/layers/_wkv_ops.so")

def _wkv(k, v, w, u, name=None):
  """Compute wkv for GPT mode.
  Args:
    k: Keys. Expected shape (B, T, hidden_dim)
    v: Values. Expected shape (B, T, hidden_dim)
    w: -exp(time_decay). Expected shape (hidden_dim,)
    u: aka time_first. Expected shape (hidden_dim,)
  Returns:
    wkv: Shape (B, T, hidden_dim)
  """
  with tf.name_scope(name or "wkv"):
    op_call = _wkv_so.ops.wkv # Naming is derived from REGISTER_OP("WKV")
    return op_call(k, v, w, u)

@tf.RegisterGradient("WKV")
def _wkv_grad(op, grad_output):
  """Compute wkv gradients for GPT mode.
  Args:
    op: The wkv op, containing its inputs
    grad_output: The upstream grads for wkv
  Returns:
    gk: Shape (B, T, hidden_dim)
    gv: Shape (B, T, hidden_dim)
    gw: Shape (hidden_dim,)
    gu: Shape (hidden_dim,)
  """
  k = tf.convert_to_tensor(op.inputs[0], name="k")
  v = tf.convert_to_tensor(op.inputs[1], name="v")
  w = tf.convert_to_tensor(op.inputs[2], name="w")
  u = tf.convert_to_tensor(op.inputs[3], name="u")
  gwkv = tf.convert_to_tensor(grad_output, name="grad_output")
  op_call = _wkv_so.ops.wkv_grad # Naming is derived from REGISTER_OP("WKVGrad")
  grads = op_call(k, v, w, u, gwkv)
  gk = tf.convert_to_tensor(grads[0], name="gk")
  gv = tf.convert_to_tensor(grads[1], name="gv")
  gw = tf.convert_to_tensor(grads[2], name="gw")
  gu = tf.convert_to_tensor(grads[3], name="gu")
  assert k.shape == gk.shape
  assert v.shape == gv.shape
  assert w.shape == gw.shape
  assert u.shape == gu.shape
  return [gk, gv, gw, gu]

@tf.keras.utils.register_keras_serializable(package="RWKV")
class RWKV(tf.keras.layers.Layer):
  @typechecked
  def __init__(self, embed_dim: int, hidden_dim: int, reg_lambda: float, **kwargs):
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.reg_lambda = reg_lambda
    self.ln_1 = tf.keras.layers.LayerNormalization(name="ln_1")
    self.ln_2 = tf.keras.layers.LayerNormalization(name="ln_2")
    self.time_shift = lambda x: tf.keras.layers.ZeroPadding1D(padding=(1, 0))(x[:,:-1])
    super().__init__(**kwargs)
  def build(self, input_shape):
    super().build(input_shape)
    with tf.name_scope(self.name):
      # Time mixing - Mix parameters
      self.tm_mix_k = self.add_weight(
        shape=(self.embed_dim,),
        initializer=tf.keras.initializers.RandomUniform(0.0, 2./self.embed_dim),
        constraint=tf.keras.constraints.NonNeg(),
        name='tm_mix_k')
      self.tm_mix_v = self.add_weight(
        shape=(self.embed_dim,),
        initializer=tf.keras.initializers.RandomUniform(0.0, 2./self.embed_dim),
        constraint=tf.keras.constraints.NonNeg(),
        name='tm_mix_v')
      self.tm_mix_r = self.add_weight(
        shape=(self.embed_dim,),
        initializer=tf.keras.initializers.RandomUniform(0.0, (2./self.embed_dim)**0.5),
        constraint=tf.keras.constraints.NonNeg(),
        name='tm_mix_r')
      # Time mixing - KVR layer weights
      self.tm_key_weights = self.add_weight(
        shape=(self.embed_dim, self.hidden_dim),
        initializer='glorot_uniform',
        regularizer=tf.keras.regularizers.L2(self.reg_lambda),
        name='tm_key_weights')
      self.tm_value_weights = self.add_weight(
        shape=(self.embed_dim, self.hidden_dim),
        initializer='glorot_uniform',
        regularizer=tf.keras.regularizers.L2(self.reg_lambda),
        name='tm_value_weights')
      self.tm_receptance_weights = self.add_weight(
        shape=(self.embed_dim, self.hidden_dim),
        initializer='glorot_uniform',
        regularizer=tf.keras.regularizers.L2(self.reg_lambda),
        name='tm_receptance_weights')
      # Time mixing - time_decay and time_first
      self.time_decay = self.add_weight(
        shape=(self.hidden_dim,),
        initializer='glorot_uniform',
        constraint=tf.keras.constraints.NonNeg(),
        name='time_decay')
      self.time_first = self.add_weight(
        shape=(self.hidden_dim,),
        initializer=tf.keras.initializers.RandomUniform(0.0, 0.05),
        name='time_first')
      # Time mixing - Output layer weights
      self.output_weights = self.add_weight(
        shape=(self.hidden_dim, self.embed_dim),
        initializer='glorot_uniform',
        regularizer=tf.keras.regularizers.L2(self.reg_lambda),
        name='tm_output_weights')
      # Channel mixing - Mix parameters
      self.cm_mix_k = self.add_weight(
        shape=(self.embed_dim,),
        initializer=tf.keras.initializers.RandomUniform(0.0, 2./self.embed_dim),
        constraint=tf.keras.constraints.NonNeg(),
        name='cm_mix_k')
      self.cm_mix_r = self.add_weight(
        shape=(self.embed_dim,),
        initializer=tf.keras.initializers.RandomUniform(0.0, 2./self.embed_dim),
        constraint=tf.keras.constraints.NonNeg(),
        name='cm_mix_r')
      # Channel mixing - KR layer weights
      self.cm_key_weights = self.add_weight(
        shape=(self.embed_dim, 4*self.hidden_dim),
        initializer='glorot_uniform',
        regularizer=tf.keras.regularizers.L2(self.reg_lambda),
        name='cm_key_weights')
      self.cm_value_weights = self.add_weight(
        shape=(4*self.hidden_dim, self.embed_dim),
        initializer='glorot_uniform',
        regularizer=tf.keras.regularizers.L2(self.reg_lambda),
        name='cm_value_weights')
      self.cm_receptance_weights = self.add_weight(
        shape=(self.embed_dim, self.embed_dim),
        initializer='glorot_uniform',
        regularizer=tf.keras.regularizers.L2(self.reg_lambda),
        name='cm_receptance_weights')
    self.built = True
  def get_config(self):
    config = {
      "embed_dim": self.embed_dim,
      "hidden_dim": self.hidden_dim,
      "reg_lambda": self.reg_lambda
    }
    base_config = super().get_config()
    config.update(base_config)
    return config
  def time_mixing(self, inputs):
    """Apply time mixing to inputs in GPT mode.
    Args:
      inputs: Expected shape (B, T, embed_dim)
    Returns:
      out: Shape (B, T, embed_dim)
    """
    # Mix x with the previous timestep to produce xk, xv, xr
    inputs_1 = self.time_shift(inputs) # (B, T, embed_dim)
    xk = inputs * self.tm_mix_k + inputs_1 * (1 - self.tm_mix_k) # (B, T, embed_dim)
    xv = inputs * self.tm_mix_v + inputs_1 * (1 - self.tm_mix_v) # (B, T, embed_dim)
    xr = inputs * self.tm_mix_r + inputs_1 * (1 - self.tm_mix_r) # (B, T, embed_dim)
    # Learn key, value, and receptance from xk, xv, xr 
    k = xk @ self.tm_key_weights # (B, T, hidden_dim)
    v = xv @ self.tm_value_weights # (B, T, hidden_dim)
    r = xr @ self.tm_receptance_weights # (B, T, hidden_dim)
    # Compute wkv
    wkv = _wkv(k=k, v=v, w=-tf.exp(self.time_decay), u=self.time_first) # (B, T, hidden_dim)
    # rwkv
    rwkv = tf.keras.activations.sigmoid(r) * wkv # (B, T, hidden_dim)
    # Compute output
    x = rwkv @ self.output_weights # (B, T, embed_dim)
    return x
  def channel_mixing(self, inputs):
    """Apply channel mixing to inputs in GPT mode.
    Args:
      inputs: Expected shape (B, T, embed_dim)
    Returns:
      x: Shape (B, T, embed_dim)
    """
    # Mix x with the previous timestep to produce xk, xr
    inputs_1 = self.time_shift(inputs) # (B, T, embed_dim)
    xk = inputs * self.cm_mix_k + inputs_1 * (1 - self.cm_mix_k) # (B, T, embed_dim)
    xr = inputs * self.cm_mix_r + inputs_1 * (1 - self.cm_mix_r) # (B, T, embed_dim)
    # Compute k and r
    k = xk @ self.cm_key_weights # (B, T, 4*embed_dim)
    r = xr @ self.cm_receptance_weights # (B, T, embed_dim)
    # Compute kv
    kv = tf.math.square(tf.keras.activations.relu(k)) @ self.cm_value_weights # (B, T, embed_dim)
    # Compute rkv
    rkv = tf.keras.activations.sigmoid(r) * kv # (B, T, embed_dim)
    return rkv
  def call(self, inputs):
    """Apply this layer to inputs in GPT mode.
    Args:
      inputs: Expected shape (B, T, embed_dim)
    Returns:
      x: Shape (B, T, embed_dim)
    """
    x = inputs # (B, T, n_embed)
    x = x + self.time_mixing(self.ln_1(x)) # (B, T, n_embed)
    x = x + self.channel_mixing(self.ln_2(x)) # (B, T, n_embed)
    return x
  def compute_output_shape(self, input_shape):
    return input_shape

class RWKVRNNCell(tf.keras.layers.Layer):
  @typechecked
  def __init__(self, embed_dim: int, hidden_dim: int, reg_lambda: float, **kwargs):
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.reg_lambda = reg_lambda
    self.ln_1 = tf.keras.layers.LayerNormalization(name="ln_1")
    self.ln_2 = tf.keras.layers.LayerNormalization(name="ln_2")
    self.state_size = [self.embed_dim, self.embed_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim]
    self.output_size = self.embed_dim
    super().__init__(**kwargs)
  def build(self, input_shape):
    super().build(input_shape)
    # Time mixing - Mix parameters
    self.tm_mix_k = self.add_weight(
      shape=(self.embed_dim,),
      initializer=tf.keras.initializers.RandomUniform(0.0, 2./self.embed_dim),
      constraint=tf.keras.constraints.NonNeg(),
      name='tm_mix_k')
    self.tm_mix_v = self.add_weight(
      shape=(self.embed_dim,),
      initializer=tf.keras.initializers.RandomUniform(0.0, 2./self.embed_dim),
      constraint=tf.keras.constraints.NonNeg(),
      name='tm_mix_v')
    self.tm_mix_r = self.add_weight(
      shape=(self.embed_dim,),
      initializer=tf.keras.initializers.RandomUniform(0.0, (2./self.embed_dim)**0.5),
      constraint=tf.keras.constraints.NonNeg(),
      name='tm_mix_r')
    # Time mixing - KVR layer weights
    self.tm_key_weights = self.add_weight(
      shape=(self.embed_dim, self.hidden_dim),
      initializer='glorot_uniform',
      regularizer=tf.keras.regularizers.L2(self.reg_lambda),
      name='tm_key_weights')
    self.tm_value_weights = self.add_weight(
      shape=(self.embed_dim, self.hidden_dim),
      initializer='glorot_uniform',
      regularizer=tf.keras.regularizers.L2(self.reg_lambda),
      name='tm_value_weights')
    self.tm_receptance_weights = self.add_weight(
      shape=(self.embed_dim, self.hidden_dim),
      initializer='glorot_uniform',
      regularizer=tf.keras.regularizers.L2(self.reg_lambda),
      name='tm_receptance_weights')
    # Time mixing - time_decay and time_first
    self.time_decay = self.add_weight(
      shape=(self.hidden_dim,),
      initializer='glorot_uniform',
      constraint=tf.keras.constraints.NonNeg(),
      name='time_decay')
    self.time_first = self.add_weight(
      shape=(self.hidden_dim,),
      initializer=tf.keras.initializers.RandomUniform(0.0, 0.05),
      name='time_first')
    # Time mixing - Output layer weights
    self.output_weights = self.add_weight(
      shape=(self.hidden_dim, self.embed_dim),
      initializer='glorot_uniform',
      regularizer=tf.keras.regularizers.L2(self.reg_lambda),
      name='tm_output_weights')
    # Channel mixing - Mix parameters
    self.cm_mix_k = self.add_weight(
      shape=(self.embed_dim,),
      initializer=tf.keras.initializers.RandomUniform(0.0, 2./self.embed_dim),
      constraint=tf.keras.constraints.NonNeg(),
      name='cm_mix_k')
    self.cm_mix_r = self.add_weight(
      shape=(self.embed_dim,),
      initializer=tf.keras.initializers.RandomUniform(0.0, 2./self.embed_dim),
      constraint=tf.keras.constraints.NonNeg(),
      name='cm_mix_r')
    # Channel mixing - KR layer weights
    self.cm_key_weights = self.add_weight(
      shape=(self.embed_dim, 4*self.hidden_dim),
      initializer='glorot_uniform',
      regularizer=tf.keras.regularizers.L2(self.reg_lambda),
      name='cm_key_weights')
    self.cm_value_weights = self.add_weight(
      shape=(4*self.hidden_dim, self.embed_dim),
      initializer='glorot_uniform',
      regularizer=tf.keras.regularizers.L2(self.reg_lambda),
      name='cm_value_weights')
    self.cm_receptance_weights = self.add_weight(
      shape=(self.embed_dim, self.embed_dim),
      initializer='glorot_uniform',
      regularizer=tf.keras.regularizers.L2(self.reg_lambda),
      name='cm_receptance_weights')
    self.built = True
  def get_config(self):
    config = {
      "embed_dim": self.embed_dim,
      "hidden_dim": self.hidden_dim,
      "reg_lambda": self.reg_lambda
    }
    base_config = super().get_config()
    config.update(base_config)
    return config
  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    if inputs is not None:
      batch_size = tf.shape(inputs)[0]
      dtype = inputs.dtype
    if batch_size is None or dtype is None:
      raise ValueError(
        "batch_size and dtype cannot be None while constructing initial "
        f"state. Received: batch_size={batch_size}, dtype={dtype}")
    def create_zeros(unnested_state_size):
      flat_dims = tf.TensorShape(unnested_state_size).as_list()
      init_state_size = [batch_size] + flat_dims
      return tf.zeros(init_state_size, dtype=dtype)
    if tf.nest.is_nested(self.state_size):
        return list(tf.nest.map_structure(create_zeros, self.state_size))
    else:
        return list(create_zeros(self.state_size))
  def time_mixing(self, inputs, prev_state):
    """Apply time mixing to inputs in RNN mode.
    Args:
      inputs: Expected shape (B, embed_dim)
      prev_state: 5-tuple (inputs_channel_mixing, inputs_time_mixing, num, den, q)
    Returns:
      rkv: Shape (B, embed_dim)
      new_state: 4-tuple (inputs_time_mixing, num, den, q)
    """
    # Mix x with the previous timestep to produce xk, xv, xr
    _, prev_state_inputs, prev_state_num, prev_state_den, prev_state_q = prev_state
    xk = inputs * self.tm_mix_k + prev_state_inputs * (1 - self.tm_mix_k) # (B, embed_dim)
    xv = inputs * self.tm_mix_v + prev_state_inputs * (1 - self.tm_mix_v) # (B, embed_dim)
    xr = inputs * self.tm_mix_r + prev_state_inputs * (1 - self.tm_mix_r) # (B, embed_dim)
    # Learn key, value, and receptance from xk, xv, xr 
    k = xk @ self.tm_key_weights # (B, hidden_dim)
    v = xv @ self.tm_value_weights # (B, hidden_dim)
    r = xr @ self.tm_receptance_weights # (B, hidden_dim)
    # Apply activation function to r
    sr = tf.keras.activations.sigmoid(r) # (B, hidden_dim)
    # Compute wkv
    w = self.time_first + k # (B, hidden_dim)
    q = tf.math.maximum(prev_state_q, w) # (B, hidden_dim)
    e1 = tf.exp(prev_state_q - q) # (B, hidden_dim)
    e2 = tf.exp(w - q) # (B, hidden_dim)
    num = e1 * prev_state_num + e2 * v # (B, hidden_dim)
    den = e1 * prev_state_den + e2 # (B, hidden_dim)
    wkv = num / den # (B, hidden_dim)
    # Compute states to pass on
    w = -tf.exp(self.time_decay) + prev_state_q # (B, hidden_dim)
    q = tf.math.maximum(w, k) # (B, hidden_dim)
    e1 = tf.exp(w - q) # (B, hidden_dim)
    e2 = tf.exp(k - q) # (B, hidden_dim)
    new_state_inputs = inputs # (B, embed_dim)
    new_state_num = e1 * prev_state_num + e2 * v # (B, hidden_dim)
    new_state_den = e1 * prev_state_den + e2 # (B, hidden_dim)
    new_state_q = q # (B, hidden_dim)
    # Compute output
    x = (sr * wkv) @ self.output_weights # (B, embed_dim)
    return x, (new_state_inputs, new_state_num, new_state_den, new_state_q)
  def channel_mixing(self, inputs, prev_state):
    """Apply channel mixing to inputs in RNN mode.
    Args:
      inputs: Expected shape (B, embed_dim)
      prev_state: 5-tuple (inputs_channel_mixing, inputs_time_mixing, num, den, q)
    Returns:
      rkv: Shape (B, embed_dim)
      new_state: 1-tuple (inputs_channel_mixing)
    """
    # Mix x with the previous timestep to produce xk, xr
    prev_state_inputs, _, _, _, _ = prev_state
    xk = inputs * self.cm_mix_k + prev_state_inputs * (1 - self.cm_mix_k) # (B, embed_dim)
    xr = inputs * self.cm_mix_r + prev_state_inputs * (1 - self.cm_mix_r) # (B, embed_dim)
    # Compute k and r
    k = xk @ self.cm_key_weights # (B, 4*embed_dim)
    r = xr @ self.cm_receptance_weights # (B, embed_dim)
    # Compute kv
    kv = tf.math.square(tf.keras.activations.relu(k)) @ self.cm_value_weights # (B, embed_dim)
    # Compute rkv
    rkv = tf.keras.activations.sigmoid(r) * kv # (B, embed_dim)
    return rkv, (inputs,)
  def call(self, inputs, prev_state):
    """Apply this layer to inputs in RNN mode.
    Args:
      inputs: Expected shape (B, embed_dim)
      prev_state: 5-tuple (inputs_channel_mixing, inputs_time_mixing, num, den, q)
    Returns:
      x: Shape (B, embed_dim)
      new_state: 5-tuple (inputs_channel_mixing, inputs_time_mixing, num, den, q)
    """
    x = inputs # (B, embed_dim)
    x_tm, new_state_tm = self.time_mixing(self.ln_1(x), prev_state=prev_state)
    x = x + x_tm # (B, embed_dim)
    x_cm, new_state_cm = self.channel_mixing(self.ln_2(x), prev_state=prev_state)
    x = x + x_cm # (B, embed_dim)
    return x, new_state_cm + new_state_tm
  def compute_output_shape(self, input_shape):
    return input_shape
  