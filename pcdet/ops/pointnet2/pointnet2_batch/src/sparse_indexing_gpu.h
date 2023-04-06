#ifndef _SPARSE_INDEXING_GPU_H
#define _SPARSE_INDEXING_GPU_H

#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

void sparse_indexing_get_wrapper_fast(int b, int c, int n, int m, 
    at::Tensor points_tensor, at::Tensor indices_tensor, at::Tensor out_tensor);

void sparse_indexing_get_kernel_launcher_fast(int b, int c, int n, int m, 
    const float *points, const long *indices, float *out);

void sparse_indexing_put_wrapper_fast(int b, int c, int n, int m, 
    at::Tensor points_tensor, at::Tensor indices_tensor, at::Tensor out_tensor);

void sparse_indexing_put_kernel_launcher_fast(int b, int c, int n, int m, 
    const float *points, const long *indices, float *out);

void sparse_indexing_replace_wrapper_fast(int b, int c, int n, int m, 
    at::Tensor points_tensor, at::Tensor indices_tensor, at::Tensor out_tensor);

void sparse_indexing_replace_kernel_launcher_fast(int b, int c, int n, int m, 
    const float *points, const long *indices, float *out);

#endif
