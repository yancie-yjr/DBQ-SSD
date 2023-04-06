#ifndef _GROUP_POINTS_SPARSE_GPU_H
#define _GROUP_POINTS_SPARSE_GPU_H

#include <torch/serialize/tensor.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>


int group_points_sparse_wrapper_fast(int b, int c, int n, int m, int nsample, 
    at::Tensor points_tensor, at::Tensor indices_tensor, at::Tensor idx_tensor, at::Tensor out_tensor);

void group_points_sparse_kernel_launcher_fast(int b, int c, int n, int m, int nsample, 
    const float *points, const long *indices, const int *idx, float *out);

int group_points_sparse_grad_wrapper_fast(int b, int c, int n, int m, int nsample, 
    at::Tensor grad_out_tensor, at::Tensor indices_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor);

void group_points_sparse_grad_kernel_launcher_fast(int b, int c, int n, int m, int nsample, 
    const float *grad_out, const long *indices, const int *idx, float *grad_points);

#endif
