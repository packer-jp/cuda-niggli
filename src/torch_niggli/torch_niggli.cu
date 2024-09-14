#include "niggli.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <chrono>
#include <pybind11/pybind11.h>
#include <torch/extension.h>


torch::Tensor niggliReduceTorch(torch::Tensor cell, float tol = 1e-5, int num_iterations = 1000)
{
    cell = cell.contiguous();
    TORCH_CHECK(cell.device().is_cuda(), "Cell tensor must be on CUDA device");

    int num_cells = cell.size(0);

    torch::Tensor result = torch::empty({num_cells, 3, 3}, torch::dtype(torch::kFloat).device(torch::kCUDA));

    cudaDeviceSynchronize();

    niggliReduceKernel<<<1, num_cells>>>(reinterpret_cast<const float(*)[9]>(cell.data_ptr<float>()), reinterpret_cast<float(*)[9]>(result.data_ptr<float>()), tol, num_iterations);

    cudaDeviceSynchronize();

    return result;
}

PYBIND11_MODULE(torch_niggli, m)
{
    m.def("niggli_reduce", &niggliReduceTorch, "A function that reduces cell using the Niggli reduction algorithm on the GPU.",
        pybind11::arg("cell"), pybind11::arg("tol") = 1e-5, pybind11::arg("num_iterations") = 1000);
}
