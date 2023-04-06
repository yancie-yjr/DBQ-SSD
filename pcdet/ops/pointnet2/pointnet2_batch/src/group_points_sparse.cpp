#include <torch/serialize/tensor.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <THC/THC.h>
#include "group_points_sparse_gpu.h"

extern THCState *state;


int group_points_sparse_grad_wrapper_fast(int b, int c, int n, int m, int nsample, 
    at::Tensor grad_out_tensor, at::Tensor indices_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor) {

    float *grad_points = grad_points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    const float *grad_out = grad_out_tensor.data<float>();
    const long *indices = indices_tensor.data<long>();

    group_points_sparse_grad_kernel_launcher_fast(b, c, n, m, nsample, grad_out, indices, idx, grad_points);
    return 1;
}


int group_points_sparse_wrapper_fast(int b, int c, int n, int m, int nsample, 
    at::Tensor points_tensor, at::Tensor indices_tensor, at::Tensor idx_tensor, at::Tensor out_tensor) {

    const float *points = points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *out = out_tensor.data<float>();
    const long *indices = indices_tensor.data<long>();

    group_points_sparse_kernel_launcher_fast(b, c, n, m, nsample, points, indices, idx, out);
    return 1;
}
