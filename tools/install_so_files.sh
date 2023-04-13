set -e -x

if [ "$TF_NEED_CUDA" == "1" ]; then
  CUDA_FLAG="--crosstool_top=@ubuntu20.04-gcc9_manylinux2014-cuda11.8-cudnn8.6-tensorrt8.4_config_cuda//crosstool:toolchain"
fi

bazel build $CUDA_FLAG //tensorflow_rwkv/...
cp ./bazel-bin/tensorflow_rwkv/custom_ops/layers/_*_ops.so ./tensorflow_rwkv/custom_ops/layers/
