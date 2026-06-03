// heat2d_explicit_omp_naive_nofs.cpp
// Equação do calor 2D — método totalmente explícito FTCS.
// OpenMP naive com mitigação de false sharing:
// campos alinhados, leading dimension padded e first-touch paralelo.
// Inclui comparação com solução exata.

#include "heat2d_explicit_common.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
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

    constexpr std::size_t doubles_per_cacheline = heat2d::CACHELINE_BYTES / sizeof(double);
    const std::size_t ld = heat2d::round_up(static_cast<std::size_t>(N), doubles_per_cacheline);
    const std::size_t NN = static_cast<std::size_t>(N) * ld;

    const double h = heat2d::compute_h(p);
    const double dt = heat2d::compute_dt(p);
    const double lam = heat2d::compute_lambda(p);

    heat2d::AlignedBuffer<double> U0, U1;
    if (!U0.allocate(NN) || !U1.allocate(NN)) {
        std::cerr << "Erro: falha na alocação alinhada dos campos.\n";
        return 1;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        std::fill_n(U0.data() + heat2d::idx2(i, 0, ld), ld, 0.0);
        std::fill_n(U1.data() + heat2d::idx2(i, 0, ld), ld, 0.0);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 1; i <= N - 2; ++i) {
        const double x = static_cast<double>(i) * h;
        for (int j = 1; j <= N - 2; ++j) {
            const double y = static_cast<double>(j) * h;
            U0[heat2d::idx2(i, j, ld)] = heat2d::exact_solution(x, y, 0.0, p);
        }
    }

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
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    const double final_time = static_cast<double>(T) * dt;
    const heat2d::ErrorStats err = heat2d::compute_errors(src, N, ld, p, final_time);
    heat2d::print_summary("omp_naive_nofs", p, dt, lam, secs, err);
    heat2d::write_output("output.txt", src, N, ld, h);

    return 0;
}
