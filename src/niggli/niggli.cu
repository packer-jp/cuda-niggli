#include "niggli.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

__host__ __device__ double absDet(const double mat[9])
{
    double det = mat[0 * 3 + 0] * (mat[1 * 3 + 1] * mat[2 * 3 + 2] - mat[1 * 3 + 2] * mat[2 * 3 + 1]) - mat[0 * 3 + 1] * (mat[1 * 3 + 0] * mat[2 * 3 + 2] - mat[1 * 3 + 2] * mat[2 * 3 + 0]) + mat[0 * 3 + 2] * (mat[1 * 3 + 0] * mat[2 * 3 + 1] - mat[1 * 3 + 1] * mat[2 * 3 + 0]);
    return std::abs(det);
}

__device__ void multiplyAndUpdate(const double src[9], double dst[9])
{
    double temp[9];
    temp[0 * 3 + 0] = src[0 * 3 + 0] * dst[0 * 3 + 0] + src[0 * 3 + 1] * dst[1 * 3 + 0] + src[0 * 3 + 2] * dst[2 * 3 + 0];
    temp[0 * 3 + 1] = src[0 * 3 + 0] * dst[0 * 3 + 1] + src[0 * 3 + 1] * dst[1 * 3 + 1] + src[0 * 3 + 2] * dst[2 * 3 + 1];
    temp[0 * 3 + 2] = src[0 * 3 + 0] * dst[0 * 3 + 2] + src[0 * 3 + 1] * dst[1 * 3 + 2] + src[0 * 3 + 2] * dst[2 * 3 + 2];
    temp[1 * 3 + 0] = src[1 * 3 + 0] * dst[0 * 3 + 0] + src[1 * 3 + 1] * dst[1 * 3 + 0] + src[1 * 3 + 2] * dst[2 * 3 + 0];
    temp[1 * 3 + 1] = src[1 * 3 + 0] * dst[0 * 3 + 1] + src[1 * 3 + 1] * dst[1 * 3 + 1] + src[1 * 3 + 2] * dst[2 * 3 + 1];
    temp[1 * 3 + 2] = src[1 * 3 + 0] * dst[0 * 3 + 2] + src[1 * 3 + 1] * dst[1 * 3 + 2] + src[1 * 3 + 2] * dst[2 * 3 + 2];
    temp[2 * 3 + 0] = src[2 * 3 + 0] * dst[0 * 3 + 0] + src[2 * 3 + 1] * dst[1 * 3 + 0] + src[2 * 3 + 2] * dst[2 * 3 + 0];
    temp[2 * 3 + 1] = src[2 * 3 + 0] * dst[0 * 3 + 1] + src[2 * 3 + 1] * dst[1 * 3 + 1] + src[2 * 3 + 2] * dst[2 * 3 + 1];
    temp[2 * 3 + 2] = src[2 * 3 + 0] * dst[0 * 3 + 2] + src[2 * 3 + 1] * dst[1 * 3 + 2] + src[2 * 3 + 2] * dst[2 * 3 + 2];
    for (int i = 0; i < 9; ++i) {
        dst[i] = temp[i];
    }
}

__global__ void niggliReduceKernel(
    const double cell_src[][9],
    double cell_dst[][9],
    double tol,
    int num_iterations)
{
    long start = clock64();
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double eps = tol * std::pow(absDet(cell_src[idx]), 1.0 / 3.0);
    double cell[9];
    for (int i = 0; i < 9; ++i) {
        cell[i] = cell_src[idx][i];
    }
    double M[9];
    double A, B, C, E, N, Y;
    int i;
    for (i = 0; i < num_iterations; ++i) {
        A = cell[0 * 3 + 0] * cell[0 * 3 + 0] + cell[0 * 3 + 1] * cell[0 * 3 + 1] + cell[0 * 3 + 2] * cell[0 * 3 + 2];
        B = cell[1 * 3 + 0] * cell[1 * 3 + 0] + cell[1 * 3 + 1] * cell[1 * 3 + 1] + cell[1 * 3 + 2] * cell[1 * 3 + 2];
        C = cell[2 * 3 + 0] * cell[2 * 3 + 0] + cell[2 * 3 + 1] * cell[2 * 3 + 1] + cell[2 * 3 + 2] * cell[2 * 3 + 2];
        E = 2 * (cell[1 * 3 + 0] * cell[2 * 3 + 0] + cell[1 * 3 + 1] * cell[2 * 3 + 1] + cell[1 * 3 + 2] * cell[2 * 3 + 2]);
        N = 2 * (cell[2 * 3 + 0] * cell[0 * 3 + 0] + cell[2 * 3 + 1] * cell[0 * 3 + 1] + cell[2 * 3 + 2] * cell[0 * 3 + 2]);
        Y = 2 * (cell[0 * 3 + 0] * cell[1 * 3 + 0] + cell[0 * 3 + 1] * cell[1 * 3 + 1] + cell[0 * 3 + 2] * cell[1 * 3 + 2]);

        // A1
        if (B + eps < A || std::abs(A - B) < eps && std::abs(E) > std::abs(N) + eps) {
            M[0 * 3 + 0] = 0, M[0 * 3 + 1] = -1, M[0 * 3 + 2] = 0;
            M[1 * 3 + 0] = -1, M[1 * 3 + 1] = 0, M[1 * 3 + 2] = 0;
            M[2 * 3 + 0] = 0, M[2 * 3 + 1] = 0, M[2 * 3 + 2] = -1;
            multiplyAndUpdate(M, cell);
        }

        A = cell[0 * 3 + 0] * cell[0 * 3 + 0] + cell[0 * 3 + 1] * cell[0 * 3 + 1] + cell[0 * 3 + 2] * cell[0 * 3 + 2];
        B = cell[1 * 3 + 0] * cell[1 * 3 + 0] + cell[1 * 3 + 1] * cell[1 * 3 + 1] + cell[1 * 3 + 2] * cell[1 * 3 + 2];
        C = cell[2 * 3 + 0] * cell[2 * 3 + 0] + cell[2 * 3 + 1] * cell[2 * 3 + 1] + cell[2 * 3 + 2] * cell[2 * 3 + 2];
        E = 2 * (cell[1 * 3 + 0] * cell[2 * 3 + 0] + cell[1 * 3 + 1] * cell[2 * 3 + 1] + cell[1 * 3 + 2] * cell[2 * 3 + 2]);
        N = 2 * (cell[2 * 3 + 0] * cell[0 * 3 + 0] + cell[2 * 3 + 1] * cell[0 * 3 + 1] + cell[2 * 3 + 2] * cell[0 * 3 + 2]);
        Y = 2 * (cell[0 * 3 + 0] * cell[1 * 3 + 0] + cell[0 * 3 + 1] * cell[1 * 3 + 1] + cell[0 * 3 + 2] * cell[1 * 3 + 2]);

        // A2
        if (C + eps < B || std::abs(B - C) < eps && std::abs(N) > std::abs(Y) + eps) {
            M[0 * 3 + 0] = -1, M[0 * 3 + 1] = 0, M[0 * 3 + 2] = 0;
            M[1 * 3 + 0] = 0, M[1 * 3 + 1] = 0, M[1 * 3 + 2] = -1;
            M[2 * 3 + 0] = 0, M[2 * 3 + 1] = -1, M[2 * 3 + 2] = 0;
            multiplyAndUpdate(M, cell);
            continue;
        }

        double ll = (std::abs(E) < eps) ? 0 : (E / std::abs(E));
        double m = (std::abs(N) < eps) ? 0 : (N / std::abs(N));
        double n = (std::abs(Y) < eps) ? 0 : (Y / std::abs(Y));

        if (ll * m * n == 1) {
            // A3
            double i = (ll == -1) ? -1 : 1;
            double j = (m == -1) ? -1 : 1;
            double k = (n == -1) ? -1 : 1;
            double M[9] = {i, 0.0, 0.0, 0.0, j, 0.0, 0.0, 0.0, k};
            M[0 * 3 + 0] = i, M[0 * 3 + 1] = 0, M[0 * 3 + 2] = 0;
            M[1 * 3 + 0] = 0, M[1 * 3 + 1] = j, M[1 * 3 + 2] = 0;
            M[2 * 3 + 0] = 0, M[2 * 3 + 1] = 0, M[2 * 3 + 2] = k;
            multiplyAndUpdate(M, cell);
        } else if (ll * m * n == 0 || ll * m * n == -1) {
            // A4
            double i = (ll == 1) ? -1 : 1;
            double j = (m == 1) ? -1 : 1;
            double k = (n == 1) ? -1 : 1;

            if (i * j * k == -1) {
                if (n == 0) {
                    k = -1;
                } else if (m == 0) {
                    j = -1;
                } else if (ll == 0) {
                    i = -1;
                }
            }
            M[0 * 3 + 0] = i, M[0 * 3 + 1] = 0, M[0 * 3 + 2] = 0;
            M[1 * 3 + 0] = 0, M[1 * 3 + 1] = j, M[1 * 3 + 2] = 0;
            M[2 * 3 + 0] = 0, M[2 * 3 + 1] = 0, M[2 * 3 + 2] = k;
            multiplyAndUpdate(M, cell);
        }

        A = cell[0 * 3 + 0] * cell[0 * 3 + 0] + cell[0 * 3 + 1] * cell[0 * 3 + 1] + cell[0 * 3 + 2] * cell[0 * 3 + 2];
        B = cell[1 * 3 + 0] * cell[1 * 3 + 0] + cell[1 * 3 + 1] * cell[1 * 3 + 1] + cell[1 * 3 + 2] * cell[1 * 3 + 2];
        C = cell[2 * 3 + 0] * cell[2 * 3 + 0] + cell[2 * 3 + 1] * cell[2 * 3 + 1] + cell[2 * 3 + 2] * cell[2 * 3 + 2];
        E = 2 * (cell[1 * 3 + 0] * cell[2 * 3 + 0] + cell[1 * 3 + 1] * cell[2 * 3 + 1] + cell[1 * 3 + 2] * cell[2 * 3 + 2]);
        N = 2 * (cell[2 * 3 + 0] * cell[0 * 3 + 0] + cell[2 * 3 + 1] * cell[0 * 3 + 1] + cell[2 * 3 + 2] * cell[0 * 3 + 2]);
        Y = 2 * (cell[0 * 3 + 0] * cell[1 * 3 + 0] + cell[0 * 3 + 1] * cell[1 * 3 + 1] + cell[0 * 3 + 2] * cell[1 * 3 + 2]);

        // A5
        if (std::abs(E) > B + eps || std::abs(E - B) < eps && 2 * N < Y - eps || std::abs(E + B) < eps && -eps > Y) {
            double s = E / std::abs(E);
            M[0 * 3 + 0] = 1, M[0 * 3 + 1] = 0, M[0 * 3 + 2] = 0;
            M[1 * 3 + 0] = 0, M[1 * 3 + 1] = 1, M[1 * 3 + 2] = 0;
            M[2 * 3 + 0] = 0, M[2 * 3 + 1] = -s, M[2 * 3 + 2] = 1;
            multiplyAndUpdate(M, cell);
            continue;
        }

        // A6
        if (std::abs(N) > A + eps || std::abs(N - A) < eps && 2 * E < Y - eps || std::abs(N + A) < eps && -eps > Y) {
            double s = N / std::abs(N);
            M[0 * 3 + 0] = 1, M[0 * 3 + 1] = 0, M[0 * 3 + 2] = 0;
            M[1 * 3 + 0] = 0, M[1 * 3 + 1] = 1, M[1 * 3 + 2] = 0;
            M[2 * 3 + 0] = -s, M[2 * 3 + 1] = 0, M[2 * 3 + 2] = 1;
            multiplyAndUpdate(M, cell);
            continue;
        }

        // A7
        if (std::abs(Y) > A + eps || std::abs(Y - A) < eps && 2 * E < N - eps || std::abs(Y + A) < eps && -eps > N) {
            double s = Y / std::abs(Y);
            M[0 * 3 + 0] = 1, M[0 * 3 + 1] = 0, M[0 * 3 + 2] = 0;
            M[1 * 3 + 0] = -s, M[1 * 3 + 1] = 1, M[1 * 3 + 2] = 0;
            M[2 * 3 + 0] = 0, M[2 * 3 + 1] = 0, M[2 * 3 + 2] = 1;
            multiplyAndUpdate(M, cell);
            continue;
        }

        // A8
        if (-eps > E + N + Y + A + B || std::abs(E + N + Y + A + B) < eps && eps < Y + 2 * (A + N)) {
            M[0 * 3 + 0] = 1, M[0 * 3 + 1] = 0, M[0 * 3 + 2] = 0;
            M[1 * 3 + 0] = 0, M[1 * 3 + 1] = 1, M[1 * 3 + 2] = 0;
            M[2 * 3 + 0] = 1, M[2 * 3 + 1] = 1, M[2 * 3 + 2] = 1;
            multiplyAndUpdate(M, cell);
            continue;
        }
        break;
    }
    for (int i = 0; i < 9; ++i) {
        cell_dst[idx][i] = cell[i];
    }
}

std::vector<double> niggliReduce(const std::vector<double>& cell, double tol, int num_iterations)
{
    int num_elements = cell.size();
    int num_cells = num_elements / 9;

    double* cell_src = nullptr;
    double* cell_dst = nullptr;
    cudaMalloc((void**)&cell_src, num_elements * sizeof(double));
    cudaMalloc((void**)&cell_dst, num_elements * sizeof(double));

    cudaMemcpy(cell_src, cell.data(), num_elements * sizeof(double), cudaMemcpyHostToDevice);

    niggliReduceKernel<<<num_cells, 1>>>(reinterpret_cast<double(*)[9]>(cell_src), reinterpret_cast<double(*)[9]>(cell_dst), tol, num_iterations);
    cudaDeviceSynchronize();

    std::vector<double> result(num_elements);
    cudaMemcpy(result.data(), cell_dst, num_elements * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(cell_src);
    cudaFree(cell_dst);

    return result;
}
