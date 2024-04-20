// Adapted from BlinkDL RWKV-v4
// https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/cuda/wkv_cuda.cu
// https://johanwind.github.io/2023/03/23/rwkv_details.html

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow_rwkv/custom_ops/layers/cc/kernels/wkv_op.h"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/work_sharder.h"

#define MIN_VALUE (-1e38)

namespace tensorflow {
namespace rwkv {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Dtype>
struct WKVFunctor<CPUDevice, Dtype> {
  Status operator()(OpKernelContext* context,
                    const Tensor& k, const Tensor& v, const Tensor& w, const Tensor& u,
                    Tensor* wkv) {
    
    // Get dims
    const TensorShape &k_shape = k.shape();
    const int32 B = k_shape.dim_size(0);
    const int32 T = k_shape.dim_size(1);
    const int32 C = k_shape.dim_size(2);

    // Get tensors
    const auto _k = k.tensor<Dtype, 3>();
    const auto _v = v.tensor<Dtype, 3>();
    const auto _w = w.tensor<Dtype, 1>();
    const auto _u = u.tensor<Dtype, 1>();
    auto _wkv = wkv->tensor<Dtype, 3>();
    _wkv.setZero();

    // Estimate cost per channel
    // T * (2*max + 4*exp + 10*add + 6*mul + 1*div)
    const int64 cost_per_channel = T * 
      (10 * Eigen::TensorOpCost::AddCost<Dtype>() +
       6 * Eigen::TensorOpCost::MulCost<Dtype>() +
       Eigen::TensorOpCost::DivCost<Dtype>() +
       4 * 10 * (Eigen::TensorOpCost::MulCost<Dtype>() + Eigen::TensorOpCost::MulCost<Dtype>()));

    // Work can be done in parallel for B * C
    // https://github.com/tensorflow/tensorflow/blob/cb619a5b7dae6deb268366e154dd6876ea1801c8/tensorflow/tsl/platform/threadpool.h
    // https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/cuda/wkv_cuda.cu
    const auto work = [&](Eigen::Index start, Eigen::Index end) -> void {
      for (Eigen::Index idx = start; idx < end; ++idx) {
        const int _b = idx / C;
        const int _c = idx % C;
        Dtype p(0), q(0), o(MIN_VALUE);
        // p and q are running sums divided by exp(o) (to avoid overflows)
        for (int t = 0; t < T; t++) {
          Dtype no = std::max(o, _u(_c) + _k(_b, t, _c));
          Dtype A = exp(o - no);
          Dtype B = exp(_u(_c) + _k(_b, t, _c) - no);
          _wkv(_b, t, _c) = (A * p + B * _v(_b, t, _c)) / (A * q + B);
          no = std::max(_w(_c) + o, _k(_b, t, _c));
          A = exp(_w(_c) + o - no);
          B = exp(_k(_b, t, _c) - no);
          p = A * p + B * _v(_b, t, _c);
          q = A * q + B;
          o = no;
        }
      }
    };
    auto thread_pool = context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(B*C, cost_per_channel, work);
    return Status();
  }
};

template <typename Dtype>
struct WKVGradFunctor<CPUDevice, Dtype> {
  Status operator()(OpKernelContext* context,
                    const Tensor& k, const Tensor& v, const Tensor& w, const Tensor& u, const Tensor& gwkv,
                    Tensor* gk, Tensor* gv, Tensor* gw, Tensor* gu) {
    
    // Get dims
    const TensorShape &k_shape = k.shape();
    const int32 B = k_shape.dim_size(0);
    const int32 T = k_shape.dim_size(1);
    const int32 C = k_shape.dim_size(2);

    // Get tensors
    const auto _k = k.tensor<Dtype, 3>();
    const auto _v = v.tensor<Dtype, 3>();
    const auto _w = w.tensor<Dtype, 1>();
    const auto _u = u.tensor<Dtype, 1>();
    const auto _gwkv = gwkv.tensor<Dtype, 3>();
    auto _gk = gk->tensor<Dtype, 3>();
    _gk.setZero();
    auto _gv = gv->tensor<Dtype, 3>();
    _gv.setZero();
    auto _gw = gw->tensor<Dtype, 1>();
    _gw.setZero();
    auto _gu = gu->tensor<Dtype, 1>();
    _gu.setZero();

    // Estimate cost per channel
    // T * (3*max + 7*exp + 35*add + 26*mul + 1*div)
    const int64 cost_per_channel = T * 
      (35 * Eigen::TensorOpCost::AddCost<Dtype>() +
       26 * Eigen::TensorOpCost::MulCost<Dtype>() +
       Eigen::TensorOpCost::DivCost<Dtype>() +
       7 * 10 * (Eigen::TensorOpCost::MulCost<Dtype>() + Eigen::TensorOpCost::MulCost<Dtype>()));

    // Work can be done in parallel for B * C
    // https://github.com/tensorflow/tensorflow/blob/cb619a5b7dae6deb268366e154dd6876ea1801c8/tensorflow/tsl/platform/threadpool.h
    // https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/cuda/wkv_cuda.cu
    const auto work = [&](Eigen::Index start, Eigen::Index end) -> void {
      for (Eigen::Index idx = start; idx < end; ++idx) {
        const int _b = idx / C;
        const int _c = idx % C;

        Dtype y[T], z[T], zexp[T];
        Dtype gw(0), gu(0);
        Dtype p(0), q(0);
        Dtype dpdw(0), dqdw(0);
        Dtype o(MIN_VALUE), one(1);
        for (int t = 0; t < T; t++) {
          Dtype no = std::max(o, _k(_b, t, _c) + _u(_c));
          Dtype A = exp(o - no);
          Dtype B = exp(_k(_b, t, _c) + _u(_c) - no);

          Dtype num = A * p + B * _v(_b, t, _c);
          Dtype iden = one / (A * q + B);

          y[t] = num * iden;
          z[t] = iden;
          zexp[t] = _k(_b, t, _c) + _u(_c) - no;

          gw += _gwkv(_b, t, _c) * (dpdw - dqdw * y[t]) * iden * A;
          gu += _gwkv(_b, t, _c) * (_v(_b, t, _c) - y[t]) * B * iden;

          no = std::max(_w(_c) + o, _k(_b, t, _c));
          A = exp(_w(_c) + o - no);
          B = exp(_k(_b, t, _c) - no);
          dpdw = A * (p + dpdw);
          dqdw = A * (q + dqdw);
          p = A * p + B * _v(_b, t, _c);
          q = A * q + B;
          o = no;
        }
        
        Dtype gp(0), gq(0);
        Dtype _o(MIN_VALUE);
        for (int t = T - 1; t >= 0; t--) {
          Dtype A = _gwkv(_b, t, _c) * z[t] * exp(zexp[t]);
          Dtype B = exp(_k(_b, t, _c) + _o);
          _gk(_b, t, _c) = A * (_v(_b, t, _c) - y[t]) + B * (gp * _v(_b, t, _c) + gq);
          _gv(_b, t, _c) = A + B * gp;

          Dtype no = std::max(_w(_c) + _o, zexp[t] - _k(_b, t, _c) - _u(_c));
          A = exp(_w(_c) + _o - no);
          B = _gwkv(_b, t, _c) * z[t] * exp(zexp[t] - _k(_b, t, _c) - _u(_c) - no);
          gp = A * gp + B;
          gq = A * gq - B * y[t];
          _o = no;
        }
        
        // Accumulate gradients for batch elements
        _gw(_c) += gw;
        _gu(_c) += gu;
      }
    };
    auto thread_pool = context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(B*C, cost_per_channel, work);
    return Status();
  }
};

}  // end namespace functor

template <typename Device, typename Dtype>
class WKVOp : public OpKernel {
 public:
  explicit WKVOp(OpKernelConstruction* context) : OpKernel(context) { }
  void Compute(OpKernelContext* context) override {
    const Tensor& k = context->input(0); // (B, T, C)
    const Tensor& v = context->input(1); // (B, T, C)
    const Tensor& w = context->input(2); // (C,)
    const Tensor& u = context->input(3); // (C,)

    OP_REQUIRES(context, k.shape() == v.shape(),
                errors::InvalidArgument("Input shapes of k and v have to be the same"));
    OP_REQUIRES(context, w.shape() == u.shape(),
                errors::InvalidArgument("Input shapes of w and u have to be the same"));

    // Allocate memory for the output
    Tensor* wkv;
    OP_REQUIRES_OK(context, context->allocate_output(0, k.shape(), &wkv));

    functor::WKVFunctor<Device, Dtype> wkvFunc;
    Status s = wkvFunc(context, k, v, w, u, wkv);

    OP_REQUIRES_OK(context, s);
  }
};

template <typename Device, typename Dtype>
class WKVGradOp : public OpKernel {
 public:
  explicit WKVGradOp(OpKernelConstruction* context) : OpKernel(context) { }
  void Compute(OpKernelContext* context) override {
    const Tensor& k = context->input(0); // (B, T, C)
    const Tensor& v = context->input(1); // (B, T, C)
    const Tensor& w = context->input(2); // (C,)
    const Tensor& u = context->input(3); // (C,)
    const Tensor& gwkv = context->input(4); // (B, T, C)

    const TensorShape &k_shape = k.shape();
    const TensorShape &v_shape = v.shape();
    const TensorShape &w_shape = w.shape();
    const TensorShape &u_shape = u.shape();
    const TensorShape &gwkv_shape = gwkv.shape();
    OP_REQUIRES(context, k_shape == v_shape,
                errors::InvalidArgument("Input shapes of k, v, and gwkv have to be the same"));
    OP_REQUIRES(context, v_shape == gwkv_shape,
                errors::InvalidArgument("Input shapes of k, v, and gwkv have to be the same"));
    OP_REQUIRES(context, w_shape == u_shape,
                errors::InvalidArgument("Input shapes of w and u have to be the same"));

    // Allocate memory for the outputs
    Tensor* gk;
    OP_REQUIRES_OK(context, context->allocate_output(0, k_shape, &gk));
    Tensor* gv;
    OP_REQUIRES_OK(context, context->allocate_output(1, v_shape, &gv));
    Tensor* gw;
    OP_REQUIRES_OK(context, context->allocate_output(2, w_shape, &gw));
    Tensor* gu;
    OP_REQUIRES_OK(context, context->allocate_output(3, u_shape, &gu));

    functor::WKVGradFunctor<Device, Dtype> wkvGrad;
    Status s = wkvGrad(context, k, v, w, u, gwkv, gk, gv, gw, gu);

    OP_REQUIRES_OK(context, s);
  }
};

// Register the CPU kernels.
#define REGISTER_WKV_OP_CPU(T)                               \
  REGISTER_KERNEL_BUILDER(Name("WKV")                        \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<T>("T"),       \
                          WKVOp<CPUDevice, T>)               \
  REGISTER_KERNEL_BUILDER(Name("WKVGrad")                    \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<T>("T"),       \
                          WKVGradOp<CPUDevice, T>)

REGISTER_WKV_OP_CPU(Eigen::half);
REGISTER_WKV_OP_CPU(float);
REGISTER_WKV_OP_CPU(bfloat16);
#undef REGISTER_WKV_OP_CPU

// Register the GPU kernels.
#if GOOGLE_CUDA

#define REGISTER_WKV_OP_GPU(T)                               \
  REGISTER_KERNEL_BUILDER(Name("WKV")                        \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<T>("T"),       \
                          WKVOp<GPUDevice, T>)               \
  REGISTER_KERNEL_BUILDER(Name("WKVGrad")                    \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<T>("T"),       \
                          WKVGradOp<GPUDevice, T>)

REGISTER_WKV_OP_GPU(float);
REGISTER_WKV_OP_GPU(bfloat16);
REGISTER_WKV_OP_GPU(Eigen::half);
#undef REGISTER_WKV_OP_GPU

#endif  // GOOGLE_CUDA

}  // namespace rwkv
}  // namespace tensorflow
