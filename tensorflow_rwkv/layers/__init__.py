# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Additional layers that conform to Keras API."""

from tensorflow_rwkv.layers.adaptive_pooling import (
    AdaptiveAveragePooling1D,
    AdaptiveMaxPooling1D,
    AdaptiveAveragePooling2D,
    AdaptiveMaxPooling2D,
    AdaptiveAveragePooling3D,
    AdaptiveMaxPooling3D,
)

from tensorflow_rwkv.layers.embedding_bag import EmbeddingBag
from tensorflow_rwkv.layers.gelu import GELU
from tensorflow_rwkv.layers.max_unpooling_2d import MaxUnpooling2D
from tensorflow_rwkv.layers.max_unpooling_2d_v2 import MaxUnpooling2DV2
from tensorflow_rwkv.layers.maxout import Maxout
from tensorflow_rwkv.layers.multihead_attention import MultiHeadAttention
from tensorflow_rwkv.layers.normalizations import FilterResponseNormalization
from tensorflow_rwkv.layers.normalizations import GroupNormalization
from tensorflow_rwkv.layers.normalizations import InstanceNormalization
from tensorflow_rwkv.layers.optical_flow import CorrelationCost
from tensorflow_rwkv.layers.poincare import PoincareNormalize
from tensorflow_rwkv.layers.polynomial import PolynomialCrossing
from tensorflow_rwkv.layers.snake import Snake
from tensorflow_rwkv.layers.sparsemax import Sparsemax
from tensorflow_rwkv.layers.spectral_normalization import SpectralNormalization
from tensorflow_rwkv.layers.spatial_pyramid_pooling import SpatialPyramidPooling2D
from tensorflow_rwkv.layers.tlu import TLU
from tensorflow_rwkv.layers.wrappers import WeightNormalization
from tensorflow_rwkv.layers.esn import ESN
from tensorflow_rwkv.layers.stochastic_depth import StochasticDepth
from tensorflow_rwkv.layers.noisy_dense import NoisyDense
from tensorflow_rwkv.layers.crf import CRF