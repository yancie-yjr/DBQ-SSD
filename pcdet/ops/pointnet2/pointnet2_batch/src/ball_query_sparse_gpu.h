#ifndef _BALL_QUERY_SPARSE_GPU_H
#define _BALL_QUERY_SPARSE_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int ball_query_sparse_wrapper_fast(int b, int n, int m, int k, float radius, int nsample, 
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor indices_tensor,
    at::Tensor idx_tensor);

void ball_query_sparse_kernel_launcher_fast(int b, int n, int m, int k, float radius, int nsample, 
	const float *new_xyz, const float *xyz, const long *indices, int *idx);
#endif
