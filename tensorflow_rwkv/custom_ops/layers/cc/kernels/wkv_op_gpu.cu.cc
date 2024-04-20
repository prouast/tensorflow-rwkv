// Adapted from BlinkDL RWKV-v4
// https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/cuda/wkv_cuda.cu
// https://johanwind.github.io/2023/03/23/rwkv_details.html

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <assert.h>

#include "gpu/cub/device/device_reduce.cuh"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow_rwkv/custom_ops/layers/cc/kernels/wkv_op.h"

#define MIN_VALUE (-1e38)

namespace tensorflow {
namespace rwkv {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

namespace {

template <typename Dtype>
__global__ void wkv_forward(const int B, const int T, const int C,
                            const Dtype *__restrict__ const _k, const Dtype *__restrict__ const _v, const Dtype *__restrict__ const _w, const Dtype *__restrict__ const _u,
                            Dtype *__restrict__ const _wkv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    Dtype u = _u[_c];
    Dtype w = _w[_c];
    const Dtype *__restrict__ const k = _k + _offset;
    const Dtype *__restrict__ const v = _v + _offset;
    Dtype *__restrict__ const wkv = _wkv + _offset;

    Dtype p = 0, q = 0, o = MIN_VALUE;
    // p and q are running sums divided by exp(o) (to avoid overflows)
    for (int t = 0; t < T; t++) {
        const int i = t * C;

        Dtype no = max(o, u + k[i]);
        Dtype A = exp(o - no);
        Dtype B = exp(u + k[i] - no);
        wkv[ii] = (A * p + B * v[i]) / (A * q + B);

        no = max(w + o, k[i]);
        A = exp(w + o - no);
        B = exp(k[i] - no);
        p = A * p + B * v[i];
        q = A * q + B;
        o = no;
    }
}

template <typename Dtype>
__global__ void wkv_backward(const int B, const int T, const int C,
                             const Dtype *__restrict__ const _k, const Dtype *__restrict__ const _v, const Dtype *__restrict__ const _w, const Dtype *__restrict__ const _u, const Dtype *__restrict__ const _gwkv,
                             Dtype *__restrict__ const _gk, Dtype *__restrict__ const _gv, Dtype *__restrict__ const _gw, Dtype *__restrict__ const _gu) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    Dtype u = _u[_c];
    Dtype w = _w[_c];
    const Dtype *__restrict__ const k = _k + _offset;
    const Dtype *__restrict__ const v = _v + _offset;
    const Dtype *__restrict__ const gwkv = _gwkv + _offset;

    Dtype *__restrict__ const gk = _gk + _offset;
    Dtype *__restrict__ const gv = _gv + _offset;

    Dtype y[T], z[T], zexp[T];

    Dtype gw = 0, gu = 0;
    Dtype p = 0, q = 0;
    Dtype dpdw = 0, dqdw = 0;
    Dtype o = MIN_VALUE;
    for (int t = 0; t < T; t++) {
        const int i = t * C;
        Dtype no = max(o, k[i] + u);
        Dtype A = exp(o - no);
        Dtype B = exp(k[i] + u - no);

        Dtype num = A * p + B * v[i];
        Dtype iden = 1 / (A * q + B);

        y[t] = num * iden;
        z[t] = iden;
        zexp[t] = k[i] + u - no;

        gw += gwkv[i] * (dpdw - dqdw * y[t]) * iden * A;
        gu += gwkv[i] * (v[i] - y[t]) * B * iden;

        no = max(w + o, k[i]);
        A = exp(w + o - no);
        B = exp(k[i] - no);
        dpdw = A * (p + dpdw);
        dqdw = A * (q + dqdw);
        p = A * p + B * v[i];
        q = A * q + B;
        o = no;
    }

    Dtype gp = 0, gq = 0;
    o = MIN_VALUE;
    for (int t = T - 1; t >= 0; t--) {
        const int i = t * C;
        Dtype A = gwkv[i] * z[t] * exp(zexp[t]);
        Dtype B = exp(k[i] + o);
        gk[i] = A * (v[i] - y[t]) + B * (gp * v[i] + gq);
        gv[i] = A + B * gp;

        Dtype no = max(w + o, zexp[t] - k[i] - u);
        A = exp(w + o - no);
        B = gwkv[i] * z[t] * exp(zexp[t] - k[i] - u - no);
        gp = A * gp + B;
        gq = A * gq - B * y[t];
        o = no;
    }

    // Accumulate gradients for batch elements
    _gw[_c] += gw;
    _gu[_c] += gu;
}

};  // namespace

template <typename Dtype>
struct WKVFunctor<GPUDevice, Dtype> {
  Status operator()(OpKernelContext *context,
                    const Tensor& k, const Tensor& v, const Tensor& w, const Tensor& u,
                    Tensor* wkv) {
    
    const int THREADS_PER_BLOCK = 32;

    const int32 B = k.dimension(0);
    const int32 T = k.dimension(1);
    const int32 C = k.dimension(2);

    dim3 threadsPerBlock( min(C, THREADS_PER_BLOCK) );
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);

    const GPUDevice &d = context->eigen_gpu_device();

    wkv_forward<<<numBlocks, threadsPerBlock, 0, d.stream()>>>(
        B, T, C,
        k.flat<Dtype>.data(), v.flat<Dtype>.data(), w.flat<Dtype>.data(), u.flat<Dtype>.data(), 
        wkv->flat<Dtype>.data());

    return Status();
  }
};

template <typename Dtype>
struct WKVGradFunctor<GPUDevice, Dtype> {
  Status operator()(OpKernelContext* context,
                    const Tensor& k, const Tensor& v, const Tensor& w, const Tensor& u, const Tensor& gwkv,
                    Tensor* gk, Tensor* gv, Tensor* gw, Tensor* gu) {

    const int THREADS_PER_BLOCK = 32;

    const int32 B = k.dimension(0);
    const int32 T = k.dimension(1);
    const int32 C = k.dimension(2);

    dim3 threadsPerBlock( min(C, THREADS_PER_BLOCK) );
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);

    wkv_backward<<<numBlocks, threadsPerBlock>>>(
        B, T, C,
        k.flat<Dtype>.data(), v.flat<Dtype>.data(), w.flat<Dtype>.data(), u.flat<Dtype>.data(), gwkv.flat<Dtype>.data(),
        gk->flat<Dtype>.data(), gv->flat<Dtype>.data(), gw->flat<Dtype>.data(), gu->flat<Dtype>.data());

    return Status();
  }
};

}  // namespace functor
}  // namespace rwkv
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
