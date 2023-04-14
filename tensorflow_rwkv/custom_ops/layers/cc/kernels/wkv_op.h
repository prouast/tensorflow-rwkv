// Copyright (C) Philipp Rouast - All Rights Reserved
// Unauthorized copying of this program, via any medium is strictly prohibited
// Proprietary and confidential
// Written by Philipp Rouast <philipp@rouast.com>, April 2023

#ifndef tensorflow_rwkv_LAYERS_KERNELS_WKV_OP_H_
#define tensorflow_rwkv_LAYERS_KERNELS_WKV_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace rwkv {
namespace functor {

template <typename Device, typename Dtype>
struct WKVFunctor {
  Status operator()(OpKernelContext* context,
                    // Inputs
                    const Tensor& k, const Tensor& v, const Tensor& w, const Tensor& u,
                    // Outputs
                    Tensor* wkv);
};

template <typename Device, typename Dtype>
struct WKVGradFunctor {
  Status operator()(OpKernelContext* context,
                    // Inputs
                    const Tensor& k, const Tensor& v, const Tensor& w, const Tensor& u, const Tensor& gwkv,
                    // Outputs
                    Tensor* gk, Tensor* gv, Tensor* gw, Tensor* gu);
};

}  // namespace functor
}  // namespace rwkv
}  // namespace tensorflow

#endif  // tensorflow_rwkv_LAYERS_KERNELS_WKV_OP_H_
