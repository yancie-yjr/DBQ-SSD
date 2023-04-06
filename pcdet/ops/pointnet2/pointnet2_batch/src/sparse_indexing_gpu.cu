#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "sparse_indexing_gpu.h"

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

__global__ void sparse_indexing_get_gpu_kernel_fast(int b, int c, int n, int m, 
    const float *__restrict__ points, const long *__restrict__ indices, float *__restrict__ out) {
    // points: (B, C, N)
    // indices: (M, 2)
    // output:
    //      out: (M, C)
    CUDA_KERNEL_LOOP(index, m * c){
        int m_idx = index / c;
        int c_idx = index % c;
        int bs_idx = indices[m_idx * 2];
        int pt_idx = indices[m_idx * 2 + 1];

        out[index] = points[bs_idx * c * n + c_idx * n + pt_idx];    
    }
}

void sparse_indexing_get_kernel_launcher_fast(int b, int c, int n, int m, 
    const float *points, const long *indices, float *out) {
    // points: (B, C, N)
    // indices: (M, 2)
    // output:
    //      out: (M, C)

    cudaError_t err;
    dim3 blocks(max(DIVUP(m * c, THREADS_PER_BLOCK), 1));
    dim3 threads(THREADS_PER_BLOCK);

    sparse_indexing_get_gpu_kernel_fast<<<blocks, threads>>>(
        b, c, n, m, points, indices, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void sparse_indexing_put_gpu_kernel_fast(int b, int c, int n, int m, 
    const float *__restrict__ points, const long *__restrict__ indices, float *__restrict__ out) {
    // points: (M, C)
    // indices: (M, 2)
    // output:
    //      out: (B, C, N)

    CUDA_KERNEL_LOOP(index, m * c){
        int m_idx = index / c;
        int c_idx = index % c;
        int bs_idx = indices[m_idx * 2];
        int pt_idx = indices[m_idx * 2 + 1];

        //if (is_replace_add){
            //out[bs_idx * c * n + c_idx * n + pt_idx] += points[index];
        //}
        //else{
            //out[bs_idx * c * n + c_idx * n + pt_idx] = points[index];
        //}
        atomicAdd(out + bs_idx * c * n + c_idx * n + pt_idx, points[index]);
    }
}

void sparse_indexing_put_kernel_launcher_fast(int b, int c, int n, int m, 
    const float *points, const long *indices, float *out) {
    // points: (M, C)
    // indices: (M, 2)
    // output:
    //      out: (B, C, N)

    cudaError_t err;
    dim3 blocks(max(DIVUP(m * c, THREADS_PER_BLOCK), 1));
    dim3 threads(THREADS_PER_BLOCK);

    sparse_indexing_put_gpu_kernel_fast<<<blocks, threads>>>(
        b, c, n, m, points, indices, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void sparse_indexing_replace_gpu_kernel_fast(int b, int c, int n, int m, 
    const float *__restrict__ points, const long *__restrict__ indices, float *__restrict__ out) {
    // points: (M, C)
    // indices: (M, 2)
    // output:
    //      out: (B, C, N)

    CUDA_KERNEL_LOOP(index, m * c){
        int m_idx = index / c;
        int c_idx = index % c;
        int bs_idx = indices[m_idx * 2];
        int pt_idx = indices[m_idx * 2 + 1];

        out[bs_idx * c * n + c_idx * n + pt_idx] = points[index];
    }
}

void sparse_indexing_replace_kernel_launcher_fast(int b, int c, int n, int m, 
    const float *points, const long *indices, float *out) {
    // points: (M, C)
    // indices: (M, 2)
    // output:
    //      out: (B, C, N)

    cudaError_t err;
    dim3 blocks(max(DIVUP(m * c, THREADS_PER_BLOCK), 1));
    dim3 threads(THREADS_PER_BLOCK);

    sparse_indexing_replace_gpu_kernel_fast<<<blocks, threads>>>(
        b, c, n, m, points, indices, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
