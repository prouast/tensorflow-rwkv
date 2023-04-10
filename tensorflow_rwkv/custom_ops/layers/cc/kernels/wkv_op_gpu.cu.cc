
// TODO
//#if GOOGLE_CUDA

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

template <typename T>
__global__ void wkv_forward(const int B, const int N, const int C,
                            const T *__restrict__ const _k, const T *__restrict__ const _v, const T *__restrict__ const _w, const T *__restrict__ const _u,
                            T *__restrict__ const _wkv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * N * C + _c;

    T u = _u[_c];
    T w = _w[_c];
    const T *__restrict__ const k = _k + _offset;
    const T *__restrict__ const v = _v + _offset;
    T *__restrict__ const wkv = _wkv + _offset;

    T p = 0, q = 0, o = MIN_VALUE;
    // p and q are running sums divided by exp(o) (to avoid overflows)
    for (int i = 0; i < N; i++) {
        const int ii = i * C;

        T no = max(o, u + k[ii]);
        T A = exp(o - no);
        T B = exp(u + k[ii] - no);
        wkv[ii] = (A * p + B * v[ii]) / (A * q + B);

        no = max(w + o, k[ii]);
        A = exp(w + o - no);
        B = exp(k[ii] - no);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }
}

template <typename T>
__global__ void wkv_backward(const int B, const int N, const int C,
                             const T *__restrict__ const _k, const T *__restrict__ const _v, const T *__restrict__ const _w, const T *__restrict__ const _u, const T *__restrict__ const _gwkv,
                             T *__restrict__ const _gk, T *__restrict__ const _gv, T *__restrict__ const _gw, T *__restrict__ const _gu) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * N * C + _c;
    const int Nmax = N

    T u = _u[_c];
    T w = _w[_c];
    const T *__restrict__ const k = _k + _offset;
    const T *__restrict__ const v = _v + _offset;
    const T *__restrict__ const gwkv = _gwkv + _offset;

    T *__restrict__ const gk = _gk + _offset;
    T *__restrict__ const gv = _gv + _offset;

    T y[Nmax], z[Nmax], zexp[Nmax];

    T gw = 0, gu = 0;
    T p = 0, q = 0;
    T dpdw = 0, dqdw = 0;
    T o = MIN_VALUE;
    for (int i = 0; i < N; i++) {
        const int ii = i * C;
        T no = max(o, k[ii] + u);
        T A = exp(o - no);
        T B = exp(k[ii] + u - no);

        T num = A * p + B * v[ii];
        T iden = 1 / (A * q + B);

        y[i] = num * iden;
        z[i] = iden;
        zexp[i] = k[ii] + u - no;

        gw += gwkv[ii] * (dpdw - dqdw * y[i]) * iden * A;
        gu += gwkv[ii] * (v[ii] - y[i]) * B * iden;

        no = max(w + o, k[ii]);
        A = exp(w + o - no);
        B = exp(k[ii] - no);
        dpdw = A * (p + dpdw);
        dqdw = A * (q + dqdw);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }

    T gp = 0, gq = 0;
    o = MIN_VALUE;
    for (int i = N - 1; i >= 0; i--) {
        const int ii = i * C;
        T A = gwkv[ii] * z[i] * exp(zexp[i]);
        T B = exp(k[ii] + o);
        gk[ii] = A * (v[ii] - y[i]) + B * (gp * v[ii] + gq);
        gv[ii] = A + B * gp;

        T no = max(w + o, zexp[i] - k[ii] - u);
        A = exp(w + o - no);
        B = gwkv[ii] * z[i] * exp(zexp[i] - k[ii] - u - no);
        gp = A * gp + B;
        gq = A * gq - B * y[i];
        o = no;
    }

    // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass, even though it's not in the forward pass
    // TODO
    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] += gw * _w[_c];
    _gu[_offsetBC] += gu;
}

};  // namespace

template <typename Dtype>
struct WKVFunctor<GPUDevice, Dtype> {
  Status operator()(OpKernelContext *context,
                    const Tensor& k, const Tensor& v, const Tensor& w, const Tensor& u,
                    Tensor* wkv) {
    
    const int THREADS_PER_BLOCK = 32;

    const int32 B = k.dimension(0);
    const int32 N = k.dimension(1);
    const int32 C = k.dimension(2);

    dim3 threadsPerBlock( min(C, THREADS_PER_BLOCK) );
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);

    const GPUDevice &d = context->eigen_gpu_device();

    // TODO apply -exp(decay) here?

    wkv_forward<<<numBlocks, threadsPerBlock, 0, d.stream()>>>(
        B, N, C,
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
    const int32 N = k.dimension(1);
    const int32 C = k.dimension(2);

    dim3 threadsPerBlock( min(C, THREADS_PER_BLOCK) );
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);

    wkv_backward<<<numBlocks, threadsPerBlock>>>(
        B, N, C,
        k.flat<Dtype>.data(), v.flat<Dtype>.data(), w.flat<Dtype>.data(), u.flat<Dtype>.data(), gwkv.flat<Dtype>.data(),
        gk->flat<Dtype>.data(), gv->flat<Dtype>.data(), gw->flat<Dtype>.data(), gu->flat<Dtype>.data());

    return Status();
  }
};

template struct WKVFunctor<GPUDevice, float>;
template struct WKVGradFunctor<GPUDevice, float>;

}  // namespace functor
}  // namespace rwkv
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
