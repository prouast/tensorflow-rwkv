licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "utils",
    srcs = glob(["*.py"]),
    data = [
        "//tensorflow_rwkv:conftest.py",
        "//tensorflow_rwkv:options.py",
    ],
)

py_test(
    name = "keras_utils_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":utils",
    ],
)
