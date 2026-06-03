// heat2d_explicit_omp_busywait_nobarrier.cpp
// Equação do calor 2D — método totalmente explícito FTCS.
// OpenMP com divisão 1D em faixas de linhas e espera ocupada local.
// Sem barreira global por passo: cada thread espera apenas as vizinhas.
// Inclui comparação com solução exata.

#include "heat2d_explicit_common.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <omp.h>

#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
  static inline void spin_pause() noexcept { _mm_pause(); }
#else
  static inline void spin_pause() noexcept { std::this_thread::yield(); }
#endif

static inline void wait_until_at_least(const std::atomic<int>& slot, int expected) noexcept {
    unsigned spins = 0;
    while (slot.load(std::memory_order_acquire) < expected) {
        spin_pause();
        if ((++spins & 0x3FFu) == 0u) std::this_thread::yield();
    }
}

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

    const int max_threads = omp_get_max_threads();
    std::unique_ptr<std::atomic<int>[]> progress(new std::atomic<int>[static_cast<std::size_t>(max_threads)]);
    for (int t = 0; t < max_threads; ++t) progress[t].store(0, std::memory_order_relaxed);

    const auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(none) shared(N,T,TILE,ld,lam,U0,U1,progress)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const heat2d::Range rows = heat2d::split_closed_interval(1, N - 2, tid, nt);

        auto wait_neighbors = [&](int expected) {
            if (tid > 0)      wait_until_at_least(progress[tid - 1], expected);
            if (tid + 1 < nt) wait_until_at_least(progress[tid + 1], expected);
        };

        for (int step = 0; step < T; ++step) {
            wait_neighbors(step);

            const double* src = (step & 1) ? U1.data() : U0.data();
            double* dst = (step & 1) ? U0.data() : U1.data();

            if (!rows.empty()) {
                for (int ii = rows.first; ii <= rows.last; ii += TILE) {
                    const int i_end = std::min(rows.last, ii + TILE - 1);
                    for (int jj = 1; jj <= N - 2; jj += TILE) {
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
            }

            progress[tid].store(step + 1, std::memory_order_release);
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    const double* result = (T & 1) ? U1.data() : U0.data();
    const double final_time = static_cast<double>(T) * dt;
    const heat2d::ErrorStats err = heat2d::compute_errors(result, N, ld, p, final_time);
    heat2d::print_summary("omp_busywait", p, dt, lam, secs, err);
    heat2d::write_output("output.txt", result, N, ld, h);

    return 0;
}
