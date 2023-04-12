package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "layers",
    srcs = glob(["*.py"]),
    data = [
        "//tensorflow_rwkv/custom_ops/layers:_correlation_cost_ops.so",
        "//tensorflow_rwkv/custom_ops/layers:_wkv_ops.so",
    ],
    deps = [
        "//tensorflow_rwkv/testing",
        "//tensorflow_rwkv/utils",
    ],
)

py_test(
    name = "layers_test",
    size = "large",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":layers",
    ],
)
