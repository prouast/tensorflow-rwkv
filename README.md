# tensorflow-rwkv

**TensorFlow RWKV** contains a TensorFlow implementation of the RWKV layer.
It includes two variants of the RWKV layer

- *GPT mode*: Processes time steps in parallel, intended for training. Relies on custom op implementation (CPU and GPU/CUDA).
- *RNN mode*: Processes time steps sequentially, intended for inference.

This repository is derived from [TensorFlow Addons](https://github.com/tensorflow/addons), since [TensorFlow custom-op](https://github.com/tensorflow/custom-op) is stale and no longer supports recent TensorFlow versions.

## Installation

Needs to be built from source as a pip package and then installed.

#### Requirements

- Python: 3.7, 3.8, 3.9, 3.10
- TensorFlow: 2.11
- Compiler: GCC 9.3.1
- cuDNN: 8.1
- CUDA: 11.2

#### Build from source

##### CPU Custom Ops
```
git clone https://github.com/prouast/tensorflow-rwkv.git
cd tensorflow-rwkv

# This script links project with TensorFlow dependency
python3 ./configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_rwkv-*.whl
```

##### GPU and CPU Custom Ops
```
git clone https://github.com/prouast/tensorflow-rwkv.git
cd tensorflow-rwkv

export TF_NEED_CUDA="1"

# Set these if the below defaults are different on your system
export TF_CUDA_VERSION="11"
export TF_CUDNN_VERSION="8"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"

# This script links project with TensorFlow dependency
python3 ./configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_rwkv-*.whl
```
