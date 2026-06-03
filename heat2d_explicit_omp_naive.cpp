// heat2d_explicit_omp_naive.cpp
// Equação do calor 2D — método totalmente explícito FTCS.
// OpenMP naive: omp parallel for com barreira global implícita a cada passo.
// Inclui comparação com solução exata:
// u(x,y,t) = sin(pi x) sin(pi y) exp(-2 pi^2 alpha t).

#include "heat2d_explicit_common.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    omp_set_dynamic(0);

    #pragma omp parallel
    {
        #pragma omp single
        std::printf("Threads: %d\n", omp_get_num_threads());
    }

    heat2d::Params p;
    if (!heat2d::load_params_strict("param.txt", p)) return 1;

    const int N = p.N;
    const int T = p.T;
    const int TILE = p.TILE;
    const std::size_t ld = static_cast<std::size_t>(N);

    const double h = heat2d::compute_h(p);
    const double dt = heat2d::compute_dt(p);
    const double lam = heat2d::compute_lambda(p);

    std::vector<double> U0(static_cast<std::size_t>(N) * ld, 0.0);
    std::vector<double> U1(static_cast<std::size_t>(N) * ld, 0.0);
    heat2d::initialize_exact_plain(U0.data(), U1.data(), N, ld, p);

    double* src = U0.data();
    double* dst = U1.data();

    const auto t0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < T; ++step) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ii = 1; ii <= N - 2; ii += TILE) {
            for (int jj = 1; jj <= N - 2; jj += TILE) {
                const int i_end = std::min(N - 2, ii + TILE - 1);
                const int j_end = std::min(N - 2, jj + TILE - 1);
                for (int i = ii; i <= i_end; ++i) {
                    for (int j = jj; j <= j_end; ++j) {
                        dst[heat2d::idx2(i, j, ld)] =
                            src[heat2d::idx2(i, j, ld)] +
                            lam * (src[heat2d::idx2(i + 1, j, ld)] +
                                   src[heat2d::idx2(i - 1, j, ld)] +
                                   src[heat2d::idx2(i, j + 1, ld)] +
                                   src[heat2d::idx2(i, j - 1, ld)] -
                                   4.0 * src[heat2d::idx2(i, j, ld)]);
                    }
                }
            }
        }

        std::swap(src, dst);
        // Fronteiras permanecem zero nos dois buffers.
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    const double final_time = static_cast<double>(T) * dt;
    const heat2d::ErrorStats err = heat2d::compute_errors(src, N, ld, p, final_time);
    heat2d::print_summary("omp_naive", p, dt, lam, secs, err);
    heat2d::write_output("output.txt", src, N, ld, h);

    return 0;
}
