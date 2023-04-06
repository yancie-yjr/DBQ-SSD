#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "group_points_sparse_gpu.h"

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

__global__ void group_points_sparse_grad_kernel_fast(int b, int c, int n, int m, int nsample, 
    const float *__restrict__ grad_out, const long *__restrict__ indices, 
    const int *__restrict__ idx, float *__restrict__ grad_points) {
    // grad_out: (M, C, nsample)
    // indices: (M, 2)
    // idx: (M, nsample)
    // output:
    //      grad_points: (B, C, N)
    int num_instances = m * c * nsample;
    CUDA_KERNEL_LOOP(index, num_instances){
        int sample_idx = index % nsample;
        int c_idx = (index / nsample) % c;
        int m_idx = index / (c * nsample);
        
        const float grad_out_v = grad_out[index];
        const int bs_idx = indices[m_idx * 2];
        const int pt_idx = idx[m_idx * nsample + sample_idx];

        atomicAdd(grad_points + bs_idx * c * n + c_idx * n + pt_idx, grad_out_v);    
    }
}

void group_points_sparse_grad_kernel_launcher_fast(int b, int c, int n, int m, int nsample, 
    const float *grad_out, const long *indices, const int *idx, float *grad_points) {
    // grad_out: (M, C, nsample)
    // indices: (M, 2)
    // idx: (M, nsample)
    // output:
    //      grad_points: (B, C, N)
    cudaError_t err;
    dim3 blocks(DIVUP(m * c * nsample, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    group_points_sparse_grad_kernel_fast<<<blocks, threads>>>(b, c, n, m, nsample, grad_out, indices, idx, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void group_points_sparse_kernel_fast(int b, int c, int n, int m, int nsample, 
    const float *__restrict__ points, const long *__restrict__ indices, const int *__restrict__ idx,
    float *__restrict__ out) {
    // points: (B, C, N)
    // idx: (M, nsample)
    // indices: (M, 2)
    // output:
    //      out: (M, C, nsample)

    int num_instances = m * c * nsample;
    CUDA_KERNEL_LOOP(index, num_instances){
        int sample_idx = index % nsample;
        int c_idx = (index / nsample) % c;
        int m_idx = index / (c * nsample);
        
        const int bs_idx = indices[m_idx * 2];
        const int pt_idx = idx[m_idx * nsample + sample_idx];
        
        out[index] = points[bs_idx * c * n + c_idx * n + pt_idx];
    }
}

void group_points_sparse_kernel_launcher_fast(int b, int c, int n, int m, int nsample, 
    const float *points, const long* indices, const int *idx, float *out) {
    // points: (B, C, N)
    // idx: (M, nsample)
    // indices: (M, 2)
    // output:
    //      out: (M, C, nsample)
    cudaError_t err;
    dim3 blocks(DIVUP(m * c * nsample, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    group_points_sparse_kernel_fast<<<blocks, threads>>>(b, c, n, m, nsample, points, indices, idx, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
