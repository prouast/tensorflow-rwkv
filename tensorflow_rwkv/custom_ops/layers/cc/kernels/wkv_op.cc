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

namespace tensorflow {
namespace rwkv {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Dtype>
struct WKVFunctor<CPUDevice, Dtype> {
  Status operator()(OpKernelContext* context, const Tensor& input_a_t,
                    const Tensor& input_b_t, Tensor* output_t,
                    /* params */
                    int kernel_size, int max_displacement, int stride_1,
                    int stride_2, int pad, TensorFormat data_format) {
    const int32 oN = GetTensorDim(*output_t, FORMAT_NCHW, 'N');
    // const int32 oC = GetTensorDim(*output_t, FORMAT_NCHW, 'C');
    const int32 oH = GetTensorDim(*output_t, FORMAT_NCHW, 'H');
    const int32 oW = GetTensorDim(*output_t, FORMAT_NCHW, 'W');
    const int32 iH = GetTensorDim(input_a_t, data_format, 'H');
    const int32 iW = GetTensorDim(input_a_t, data_format, 'W');
    const int32 iC = GetTensorDim(input_a_t, data_format, 'C');

    const int K = kernel_size * kernel_size * iC;

    const auto input_a = input_a_t.tensor<Dtype, 4>();
    const auto input_b = input_b_t.tensor<Dtype, 4>();
    auto output = output_t->tensor<Dtype, 4>();
    output.setZero();

    const int kernel_rad = (kernel_size - 1) / 2;
    const int displacement_rad = max_displacement / stride_2;
    const int displacement_size = 2 * displacement_rad + 1;

    const bool is_NCHW = (data_format == FORMAT_NCHW);
    // estimate operations per pixel
    const int64 cost_per_pixel =
        iC * ((2 * displacement_rad + 1) * (2 * displacement_rad + 1)) *
        ((2 * kernel_rad + 1) * (2 * kernel_rad + 1)) *
        (Eigen::TensorOpCost::MulCost<Dtype>() +
         Eigen::TensorOpCost::AddCost<Dtype>());

    const auto work = [&](Eigen::Index start, Eigen::Index end) -> void {
      for (Eigen::Index id = start; id < end; ++id) {
        const int n = id / (oH * oW);
        const int h = (id / oW) % oH;
        const int w = id % oW;
        const int h1 = (h - pad) * stride_1 + max_displacement + kernel_rad;
        const int w1 = (w - pad) * stride_1 + max_displacement + kernel_rad;
        for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
          for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
            const int tc = (tj + displacement_rad) * displacement_size +
                           (ti + displacement_rad);

            const int w2 = w1 + ti * stride_2;
            const int h2 = h1 + tj * stride_2;

            for (int j = -kernel_rad; j <= kernel_rad; ++j) {
              // out-of-bound test
              if (!FastBoundsCheck(h1 + j, iH) || !FastBoundsCheck(h2 + j, iH))
                continue;
              for (int i = -kernel_rad; i <= kernel_rad; ++i) {
                if (!FastBoundsCheck(w1 + i, iW) ||
                    !FastBoundsCheck(w2 + i, iW))
                  continue;
                for (int c = 0; c < iC; ++c) {
                  // eq. (1) in FlowNet: Learning Optical Flow with
                  // Convolutional Networks
                  if (is_NCHW) {
                    output(n, tc, h, w) += input_a(n, c, h1 + j, w1 + i) *
                                           input_b(n, c, h2 + j, w2 + i);
                  } else {
                    output(n, tc, h, w) += input_a(n, h1 + j, w1 + i, c) *
                                           input_b(n, h2 + j, w2 + i, c);
                  }
                }
              }
            }
            output(n, tc, h, w) /= K;
          }
        }
      }
    };
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(oN * oH * oW, cost_per_pixel, work);
    return Status();
  }
};

template <typename Dtype>
struct WKVGradFunctor<CPUDevice, Dtype> {
  Status operator()(OpKernelContext* context, const Tensor& input_a_t,
                    const Tensor& input_b_t, const Tensor& topdiff_t,
                    Tensor* output_a_gradient_t, Tensor* output_b_gradient_t,
                    /* params */
                    int kernel_size, int max_displacement, int stride_1,
                    int stride_2, int pad, TensorFormat data_format) {
    const int32 iN = GetTensorDim(input_a_t, data_format, 'N');
    const int32 iC = GetTensorDim(input_a_t, data_format, 'C');
    const int32 iH = GetTensorDim(input_a_t, data_format, 'H');
    const int32 iW = GetTensorDim(input_a_t, data_format, 'W');

    // topdiff is NCHW
    // const int32 oC = GetTensorDim(topdiff_t, FORMAT_NCHW, 'C');
    const int32 oH = GetTensorDim(topdiff_t, FORMAT_NCHW, 'H');
    const int32 oW = GetTensorDim(topdiff_t, FORMAT_NCHW, 'W');

    const auto topdiff = topdiff_t.tensor<Dtype, 4>();
    const auto input_a = input_a_t.tensor<Dtype, 4>();
    const auto input_b = input_b_t.tensor<Dtype, 4>();
    auto output_a_gradient = output_a_gradient_t->tensor<Dtype, 4>();
    auto output_b_gradient = output_b_gradient_t->tensor<Dtype, 4>();
    output_a_gradient.setZero();
    output_b_gradient.setZero();

    const int kernel_rad = (kernel_size - 1) / 2;
    const int displacement_rad = max_displacement / stride_2;
    const int displacement_size = 2 * displacement_rad + 1;
    const int K = kernel_size * kernel_size * iC;

    const bool is_NCHW = (data_format == FORMAT_NCHW);
    // estimate operations per pixel
    const int64 cost_per_pixel =
        2 * iC * ((2 * displacement_rad + 1) * (2 * displacement_rad + 1)) *
        ((2 * kernel_rad + 1) * (2 * kernel_rad + 1)) *
        (Eigen::TensorOpCost::MulCost<Dtype>() +
         Eigen::TensorOpCost::AddCost<Dtype>());

    const auto work = [&](Eigen::Index start, Eigen::Index end) -> void {
      for (Eigen::Index id = start; id < end; ++id) {
        const int n = id / (oH * oW);
        const int h = (id / oW) % oH;
        const int w = id % oW;
        const int h1 = (h - pad) * stride_1 + max_displacement + kernel_rad;
        const int w1 = (w - pad) * stride_1 + max_displacement + kernel_rad;

        for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
          for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
            const int tc = (tj + displacement_rad) * displacement_size +
                           (ti + displacement_rad);

            const int w2 = w1 + ti * stride_2;
            const int h2 = h1 + tj * stride_2;

            for (int j = -kernel_rad; j <= kernel_rad; ++j) {
              // out-of-bound test
              if (!FastBoundsCheck(h1 + j, iH) || !FastBoundsCheck(h2 + j, iH))
                continue;
              for (int i = -kernel_rad; i <= kernel_rad; ++i) {
                if (!FastBoundsCheck(w1 + i, iW) ||
                    !FastBoundsCheck(w2 + i, iW))
                  continue;
                for (int c = 0; c < iC; ++c) {
                  // eq. (1) in FlowNet: Learning Optical Flow with
                  // Convolutional Networks
                  if (is_NCHW) {
                    output_a_gradient(n, c, h1 + j, w1 + i) +=
                        topdiff(n, tc, h, w) * input_b(n, c, h2 + j, w2 + i) /
                        K;
                    output_b_gradient(n, c, h2 + j, w2 + i) +=
                        topdiff(n, tc, h, w) * input_a(n, c, h1 + j, w1 + i) /
                        K;
                  } else {
                    output_a_gradient(n, h1 + j, w1 + i, c) +=
                        topdiff(n, tc, h, w) * input_b(n, h2 + j, w2 + i, c) /
                        K;
                    output_b_gradient(n, h2 + j, w2 + i, c) +=
                        topdiff(n, tc, h, w) * input_a(n, h1 + j, w1 + i, c) /
                        K;
                  }
                }
              }
            }
          }
        }
      }
    };

    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(iN * oH * oW, cost_per_pixel, work);

    return Status();
  }
};

}  // end namespace functor

template <typename Device, typename T>
class WKVOp : public OpKernel {
 public:
  explicit WKVOp(OpKernelConstruction* context) : OpKernel(context) { }
  void Compute(OpKernelContext* context) override {
    const Tensor& k = context->input(0); // (B, N, C)
    const Tensor& v = context->input(1); // (B, N, C)
    const Tensor& w = context->input(2); // (C,)
    const Tensor& u = context->input(3); // (C,)

    OP_REQUIRES(context, k.shape() == v.shape(),
                errors::InvalidArgument("Input shapes of k and v have to be the same"));
    OP_REQUIRES(context, w.shape() == u.shape(),
                errors::InvalidArgument("Input shapes of w and u have to be the same"));

    // Allocate memory for the output
    Tensor* wkv;
    OP_REQUIRES_OK(context, context->allocate_output(0, k.shape, &wkv));

    functor::WKVFunctor<Device, T> wkvFunc;
    Status s = wkvFunc(context, k, v, w, u, wkv);

    OP_REQUIRES_OK(context, s);
  }
};

template <typename Device, typename T>
class WKVGradOp : public OpKernel {
 public:
  explicit WKVGradOp(OpKernelConstruction* context) : OpKernel(context) { }
  void Compute(OpKernelContext* context) override {
    const Tensor& k = context->input(0); // (B, N, C)
    const Tensor& v = context->input(1); // (B, N, C)
    const Tensor& w = context->input(2); // (C,)
    const Tensor& u = context->input(3); // (C,)
    const Tensor& gwkv = context->input(4); // (B, N, C)

    const TensorShape &k_shape = k.shape();
    const TensorShape &v_shape = v.shape();
    const TensorShape &w_shape = w.shape();
    const TensorShape &u_shape = w.shape();
    const TensorShape &gwkv_shape = gwkv.shape();
    OP_REQUIRES(context, k_shape == v_shape == gwkv_shape,
                errors::InvalidArgument("Input shapes of k, v, and gwkv have to be the same"));
    OP_REQUIRES(context, w_shape == u_shape,
                errors::InvalidArgument("Input shapes of w and u have to be the same"));

    // Allocate memory for the outputs
    Tensor* gk;
    OP_REQUIRES_OK(context, context->allocate_output(0, k_shape, &gk));
    Tensor* gv;
    OP_REQUIRES_OK(context, context->allocate_output(1, v_shape, &gv));
    Tensor* gw;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({k_shape.dim_size(0), w_shape.dim_size(0)}), &gw));
    Tensor* gu;
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({k_shape.dim_size(0), u_shape.dim_size(0)}), &gu));

    functor::WKVGradFunctor<Device, T> wkvGrad;
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

TF_CALL_float(REGISTER_WKV_OP_CPU);
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

TF_CALL_float(REGISTER_WKV_OP_GPU);
#undef REGISTER_WKV_OP_GPU

#endif  // GOOGLE_CUDA

}  // namespace rwkv
}  // namespace tensorflow
