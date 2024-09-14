#include "niggli.cuh"
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

void generateRandomCells(std::vector<float>& cells, int batch_size, std::mt19937& rng)
{
    std::normal_distribution<float> dist(0.0, 1.0);
    for (int i = 0; i < batch_size * 9; ++i) {
        cells[i] = dist(rng);
    }
}

void checkNiggliReduced(const std::vector<float>& cell, float originalVolume, float tol = 1e-5)
{
    float reducedVolume = absDet(cell.data());

    assert(std::abs(reducedVolume - originalVolume) < tol * originalVolume);

    float eps = tol * std::pow(reducedVolume, 1.0 / 3.0);

    float A = cell[0 * 3 + 0] * cell[0 * 3 + 0] + cell[0 * 3 + 1] * cell[0 * 3 + 1] + cell[0 * 3 + 2] * cell[0 * 3 + 2];
    float B = cell[1 * 3 + 0] * cell[1 * 3 + 0] + cell[1 * 3 + 1] * cell[1 * 3 + 1] + cell[1 * 3 + 2] * cell[1 * 3 + 2];
    float C = cell[2 * 3 + 0] * cell[2 * 3 + 0] + cell[2 * 3 + 1] * cell[2 * 3 + 1] + cell[2 * 3 + 2] * cell[2 * 3 + 2];

    float E = 2 * (cell[1 * 3 + 0] * cell[2 * 3 + 0] + cell[1 * 3 + 1] * cell[2 * 3 + 1] + cell[1 * 3 + 2] * cell[2 * 3 + 2]);
    float N = 2 * (cell[2 * 3 + 0] * cell[0 * 3 + 0] + cell[2 * 3 + 1] * cell[0 * 3 + 1] + cell[2 * 3 + 2] * cell[0 * 3 + 2]);
    float Y = 2 * (cell[0 * 3 + 0] * cell[1 * 3 + 0] + cell[0 * 3 + 1] * cell[1 * 3 + 1] + cell[0 * 3 + 2] * cell[1 * 3 + 2]);

    assert(A < B + eps);
    assert(B < C + eps);
    assert(std::abs(E) < B + eps);
    assert(std::abs(N) < A + eps);
    assert(std::abs(Y) < A + eps);
    assert(E + N + Y + A + B > -eps);
}

int main()
{
    std::random_device rd;
    std::mt19937 rng(rd());
    const float tol = 1e-5;
    const int num_iterations = 1000;
    const int batch_size = 64;

    std::vector<float> cells(batch_size * 9);
    generateRandomCells(cells, batch_size, rng);

    std::vector<float> originalVolumes(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        originalVolumes[i] = absDet(&cells[i * 9]);
    }

    std::vector<float> reduced_cells = niggliReduce(cells, tol, num_iterations);

    for (int i = 0; i < batch_size; ++i) {
        std::vector<float> cell(reduced_cells.begin() + i * 9, reduced_cells.begin() + (i + 1) * 9);
        checkNiggliReduced(cell, originalVolumes[i], tol);
    }
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
