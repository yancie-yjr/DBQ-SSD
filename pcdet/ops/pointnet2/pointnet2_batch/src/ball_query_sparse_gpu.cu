#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_sparse_gpu.h"
#include "cuda_utils.h"

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

__global__ void ball_query_sparse_kernel_fast(
    int b, int n, int m, int k, float radius, int nsample, 
    const float *__restrict__ new_xyz, const float *__restrict__ xyz,
    const long *__restrict__ indices, int *__restrict__ idx) {
    // new_xyz: (B, K, 3)
    // xyz: (B, N, 3),
    // indices: (M, 2)
    // output:
    // idx: (M, nsample)
    CUDA_KERNEL_LOOP(t_idx, m){
        int bs_idx = indices[t_idx * 2];
        int pt_idx = indices[t_idx * 2 + 1];

        new_xyz += (bs_idx * k + pt_idx) * 3; 
        xyz += bs_idx * n * 3;
        idx += t_idx * nsample;

        float radius2 = radius * radius;
        float new_x = new_xyz[0];
        float new_y = new_xyz[1];
        float new_z = new_xyz[2];

        int cnt = 0;
        for (int j = 0; j < n; ++j) {
            float x = xyz[j * 3 + 0];
            float y = xyz[j * 3 + 1];
            float z = xyz[j * 3 + 2];
            float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
            if (d2 < radius2){
                if (cnt == 0){
                    for (int l = 0; l < nsample; ++l) {
                        idx[l] = j;
                    }
                }
                idx[cnt] = j;
                ++cnt;
                if (cnt >= nsample) break;
            }
        }
    }
}


void ball_query_sparse_kernel_launcher_fast(
    int b, int n, int m, int k, float radius, int nsample,
    const float *new_xyz, const float *xyz,
    const long *indices, int *idx) {
    // new_xyz: (B, K, 3)
    // xyz: (B, N, 3)
    // indices: (M, 2)
    // output:
    //      idx: (M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_sparse_kernel_fast<<<blocks, threads>>>(b, n, m, k, radius, nsample, new_xyz, xyz, indices, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
