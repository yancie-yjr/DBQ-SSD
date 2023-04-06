#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <THC/THC.h>

#include "sparse_indexing_gpu.h"

extern THCState *state;


void sparse_indexing_get_wrapper_fast(int b, int c, int n, int m, 
    at::Tensor points_tensor, at::Tensor indices_tensor, at::Tensor out_tensor){
    const float *points = points_tensor.data<float>();
    const long *indices = indices_tensor.data<long>();
    float *out = out_tensor.data<float>();

    sparse_indexing_get_kernel_launcher_fast(b, c, n, m, points, indices, out);
}


void sparse_indexing_put_wrapper_fast(int b, int c, int n, int m,
    at::Tensor points_tensor, at::Tensor indices_tensor, at::Tensor out_tensor){
    const float *points = points_tensor.data<float>();
    const long *indices = indices_tensor.data<long>();
    float *out = out_tensor.data<float>();

    sparse_indexing_put_kernel_launcher_fast(b, c, n, m, points, indices, out);
}


void sparse_indexing_replace_wrapper_fast(int b, int c, int n, int m,
    at::Tensor points_tensor, at::Tensor indices_tensor, at::Tensor out_tensor){
    const float *points = points_tensor.data<float>();
    const long *indices = indices_tensor.data<long>();
    float *out = out_tensor.data<float>();

    sparse_indexing_replace_kernel_launcher_fast(b, c, n, m, points, indices, out);
}
