#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CheckError(call)                                           \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess) {                                \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr,                                        \
                "code: %d, reason: %s\n",                          \
                error,                                             \
                cudaGetErrorString(error));                        \
        }                                                          \
    }

__host__ __device__ double absDet(const double mat[9]);

__global__ void niggliReduceKernel(const double cell_src[][9], double cell_dst[][9], double tol, int num_iterations);

std::vector<double> niggliReduce(const std::vector<double>& cell, double tol, int num_iterations);
