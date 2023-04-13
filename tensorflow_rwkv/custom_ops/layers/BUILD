licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_rwkv:tensorflow_rwkv.bzl", "custom_op_library")

custom_op_library(
    name = "_wkv_ops.so",
    srcs = [
        "cc/kernels/wkv_op.cc",
        "cc/kernels/wkv_op.h",
        "cc/ops/wkv_op.cc",
    ],
    cuda_deps = [
        "@cub_archive//:cub",
    ],
    cuda_srcs = [
        "cc/kernels/wkv_op.h",
        "cc/kernels/wkv_op_gpu.cu.cc",
    ],
)
