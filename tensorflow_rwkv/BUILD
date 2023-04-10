licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

py_library(
    name = "tensorflow_rwkv",
    srcs = glob(["*.py"]),
    deps = [
        "//tensorflow_rwkv/layers",
        "//tensorflow_rwkv/utils",
    ],
)

py_test(
    name = "tensorflow_rwkv_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":tensorflow_rwkv",
    ],
)
