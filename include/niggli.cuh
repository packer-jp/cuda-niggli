#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

__host__ __device__ float absDet(const float mat[9]);

__global__ void niggliReduceKernel(const float cell_src[][9], float cell_dst[][9], float tol, int num_iterations);

std::vector<float> niggliReduce(const std::vector<float>& cell, float tol, int num_iterations);
