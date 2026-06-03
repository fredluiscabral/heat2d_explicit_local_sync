// heat2d_explicit_mpi_naive_1d.cpp
// Equação do calor 2D — método totalmente explícito FTCS, MPI puro.
// Decomposição 1D por linhas interiores.
// Inclui comparação com solução exata:
// u(x,y,t) = sin(pi x) sin(pi y) exp(-2 pi^2 alpha t).

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

struct Params {
    int N = 0;
    double alpha = 0.0;
    int T = 0;
    int TILE = 0;
};

static inline std::string ltrim(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
        [](unsigned char ch){ return !std::isspace(ch); }));
    return s;
}

static inline std::string rtrim(std::string s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
        [](unsigned char ch){ return !std::isspace(ch); }).base(), s.end());
    return s;
}

static inline std::string trim(std::string s) { return rtrim(ltrim(s)); }

static inline std::string tolower_str(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

static inline std::size_t idx2(int i, int j, int ld) {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(ld) +
           static_cast<std::size_t>(j);
}

static bool load_params_strict_rank0(const std::string& fname, Params& p) {
    std::ifstream fin(fname);
    if (!fin) {
        std::cerr << "Erro: não foi possível abrir " << fname << " (obrigatório).\n";
        return false;
    }

    std::unordered_map<std::string, std::string> kv;
    std::string line;

    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;

        kv[tolower_str(trim(line.substr(0, pos)))] = trim(line.substr(pos + 1));
    }

    try {
        if (!kv.count("n"))     { std::cerr << "Erro: falta 'N'.\n"; return false; }
        if (!kv.count("alpha")) { std::cerr << "Erro: falta 'alpha'.\n"; return false; }
        if (!kv.count("t"))     { std::cerr << "Erro: falta 'T'.\n"; return false; }
        if (!kv.count("tile"))  { std::cerr << "Erro: falta 'TILE'.\n"; return false; }

        p.N     = std::stoi(kv["n"]);
        p.alpha = std::stod(kv["alpha"]);
        p.T     = std::stoi(kv["t"]);
        p.TILE  = std::stoi(kv["tile"]);
    } catch (const std::exception& e) {
        std::cerr << "Erro: conversão de parâmetros: " << e.what() << "\n";
        return false;
    }

    if (p.N < 3) {
        std::cerr << "Erro: N>=3.\n";
        return false;
    }
    if (p.alpha <= 0.0) {
        std::cerr << "Erro: alpha>0.\n";
        return false;
    }
    if (p.T < 0) {
        std::cerr << "Erro: T>=0.\n";
        return false;
    }
    if (p.TILE < 1 || p.TILE > p.N - 2) {
        std::cerr << "Erro: 1<=TILE<=N-2.\n";
        return false;
    }

    return true;
}

static inline void split_range(int total, int parts, int coord, int& start, int& count) {
    const int base = total / parts;
    const int rem = total % parts;
    count = base + (coord < rem ? 1 : 0);
    start = coord * base + std::min(coord, rem);
}

static inline double pi() {
    return std::acos(-1.0);
}

static inline double compute_h(const Params& p) {
    return 1.0 / static_cast<double>(p.N - 1);
}

static inline double compute_dt(const Params& p) {
    const double h = compute_h(p);
    return 0.90 * (h * h) / (4.0 * p.alpha);
}

static inline double compute_lambda(const Params& p) {
    const double h = compute_h(p);
    const double dt = compute_dt(p);
    return p.alpha * dt / (h * h);
}

static inline double exact_solution(double x, double y, double t, const Params& p) {
    const double k = pi();
    return std::sin(k * x) * std::sin(k * y) *
           std::exp(-2.0 * k * k * p.alpha * t);
}

static void exchange_halo_rows(std::vector<double>& A,
                               int ni,
                               int ld,
                               int nbr_north,
                               int nbr_south,
                               MPI_Comm comm) {
    MPI_Status status;

    if (nbr_north != MPI_PROC_NULL) {
        MPI_Sendrecv(&A[idx2(1, 0, ld)], ld, MPI_DOUBLE, nbr_north, 10,
                     &A[idx2(0, 0, ld)], ld, MPI_DOUBLE, nbr_north, 11,
                     comm, &status);
    } else {
        std::fill_n(&A[idx2(0, 0, ld)], ld, 0.0);
    }

    if (nbr_south != MPI_PROC_NULL) {
        MPI_Sendrecv(&A[idx2(ni, 0, ld)], ld, MPI_DOUBLE, nbr_south, 11,
                     &A[idx2(ni + 1, 0, ld)], ld, MPI_DOUBLE, nbr_south, 10,
                     comm, &status);
    } else {
        std::fill_n(&A[idx2(ni + 1, 0, ld)], ld, 0.0);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Params p;

    if (rank == 0) {
        if (!load_params_strict_rank0("param.txt", p)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::printf("MPI processes: %d\n", size);
        std::fflush(stdout);
    }

    MPI_Bcast(&p.N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p.alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p.T, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p.TILE, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (size > p.N - 2) {
        if (rank == 0) {
            std::cerr << "Erro: número de ranks não pode exceder N-2 na decomposição 1D.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    const int N = p.N;
    const int T = p.T;
    const int TILE = p.TILE;
    const int ld = N;

    const double h = compute_h(p);
    const double dt = compute_dt(p);
    const double lam = compute_lambda(p);

    const int interior_rows = N - 2;
    int istart_local = 0;
    int icount_local = 0;
    split_range(interior_rows, size, rank, istart_local, icount_local);

    const int ni = icount_local;
    const int ig0 = 1 + istart_local;

    const std::size_t local_size =
        static_cast<std::size_t>(ni + 2) * static_cast<std::size_t>(ld);

    std::vector<double> U0(local_size, 0.0);
    std::vector<double> U1(local_size, 0.0);

    const int nbr_north = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int nbr_south = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;

    for (int li = 1; li <= ni; ++li) {
        const int gi = ig0 + (li - 1);
        const double x = static_cast<double>(gi) * h;

        for (int j = 1; j <= N - 2; ++j) {
            const double y = static_cast<double>(j) * h;
            U0[idx2(li, j, ld)] = exact_solution(x, y, 0.0, p);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int step = 0; step < T; ++step) {
        std::vector<double>& src_vec = (step & 1) ? U1 : U0;
        std::vector<double>& dst_vec = (step & 1) ? U0 : U1;

        exchange_halo_rows(src_vec, ni, ld, nbr_north, nbr_south, MPI_COMM_WORLD);

        for (int ii = 1; ii <= ni; ii += TILE) {
            const int i_end = std::min(ni, ii + TILE - 1);

            for (int jj = 1; jj <= N - 2; jj += TILE) {
                const int j_end = std::min(N - 2, jj + TILE - 1);

                for (int li = ii; li <= i_end; ++li) {
                    for (int j = jj; j <= j_end; ++j) {
                        dst_vec[idx2(li, j, ld)] =
                            src_vec[idx2(li, j, ld)] +
                            lam * (src_vec[idx2(li + 1, j, ld)] +
                                   src_vec[idx2(li - 1, j, ld)] +
                                   src_vec[idx2(li, j + 1, ld)] +
                                   src_vec[idx2(li, j - 1, ld)] -
                                   4.0 * src_vec[idx2(li, j, ld)]);
                    }
                }
            }
        }
    }

    const double t1 = MPI_Wtime();
    const double elapsed = t1 - t0;

    double elapsed_max = 0.0;
    MPI_Reduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const std::vector<double>& result = (T & 1) ? U1 : U0;
    const double final_time = static_cast<double>(T) * dt;

    double l1_local = 0.0;
    double l2_local = 0.0;
    double linf_local = 0.0;

    for (int li = 1; li <= ni; ++li) {
        const int gi = ig0 + (li - 1);
        const double x = static_cast<double>(gi) * h;

        for (int j = 1; j <= N - 2; ++j) {
            const double y = static_cast<double>(j) * h;
            const double diff = result[idx2(li, j, ld)] - exact_solution(x, y, final_time, p);
            const double ad = std::abs(diff);

            l1_local += ad;
            l2_local += diff * diff;
            linf_local = std::max(linf_local, ad);
        }
    }

    double l1_sum = 0.0;
    double l2_sum = 0.0;
    double linf_max = 0.0;

    MPI_Reduce(&l1_local, &l1_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&l2_local, &l2_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&linf_local, &linf_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Amostra para output.txt: coleta pontos em stride 16.
    std::vector<int> ig_local;
    std::vector<int> jg_local;
    std::vector<double> val_local;

    for (int li = 1; li <= ni; ++li) {
        const int gi = ig0 + (li - 1);
        if (gi % 16 != 0) continue;

        for (int j = 16; j <= N - 2; j += 16) {
            ig_local.push_back(gi);
            jg_local.push_back(j);
            val_local.push_back(result[idx2(li, j, ld)]);
        }
    }

    const int mloc = static_cast<int>(ig_local.size());

    std::vector<int> counts;
    std::vector<int> displs;
    int gtot = 0;

    if (rank == 0) counts.resize(static_cast<std::size_t>(size));

    MPI_Gather(&mloc, 1, MPI_INT,
               rank == 0 ? counts.data() : nullptr,
               1,
               MPI_INT,
               0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        displs.resize(static_cast<std::size_t>(size), 0);

        for (int r = 0; r < size; ++r) {
            displs[static_cast<std::size_t>(r)] = gtot;
            gtot += counts[static_cast<std::size_t>(r)];
        }
    }

    std::vector<int> ig_all;
    std::vector<int> jg_all;
    std::vector<double> val_all;

    if (rank == 0) {
        ig_all.resize(static_cast<std::size_t>(gtot));
        jg_all.resize(static_cast<std::size_t>(gtot));
        val_all.resize(static_cast<std::size_t>(gtot));
    }

    MPI_Gatherv(ig_local.data(), mloc, MPI_INT,
                rank == 0 ? ig_all.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_INT,
                0,
                MPI_COMM_WORLD);

    MPI_Gatherv(jg_local.data(), mloc, MPI_INT,
                rank == 0 ? jg_all.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_INT,
                0,
                MPI_COMM_WORLD);

    MPI_Gatherv(val_local.data(), mloc, MPI_DOUBLE,
                rank == 0 ? val_all.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout.precision(17);
        std::cout << "Variant : mpi_puro\n";
        std::cout << "N : " << p.N << "\n";
        std::cout << "alpha : " << p.alpha << "\n";
        std::cout << "T : " << p.T << "\n";
        std::cout << "TILE : " << p.TILE << "\n";
        std::cout << "dt : " << dt << "\n";
        std::cout << "lambda : " << lam << "\n";
        std::cout << "final_time : " << final_time << "\n";

        const double denom = static_cast<double>(N) * static_cast<double>(N);
        std::cout << "L1_error : " << l1_sum / denom << "\n";
        std::cout << "L2_error : " << std::sqrt(l2_sum / denom) << "\n";
        std::cout << "Linf_error : " << linf_max << "\n";
        std::cout << "Tempo : " << elapsed_max << " s\n";

        std::unordered_map<std::uint64_t, double> mapv;
        mapv.reserve(static_cast<std::size_t>(gtot) * 13 / 10 + 1);

        auto key = [](int i, int j) -> std::uint64_t {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(i)) << 32) |
                   static_cast<std::uint32_t>(j);
        };

        for (int k = 0; k < gtot; ++k) {
            mapv[key(ig_all[static_cast<std::size_t>(k)], jg_all[static_cast<std::size_t>(k)])] =
                val_all[static_cast<std::size_t>(k)];
        }

        std::ofstream fout("output.txt");
        if (!fout) {
            std::cerr << "Erro: não foi possível abrir output.txt para escrita.\n";
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        fout.setf(std::ios::fixed);
        fout.precision(8);

        for (int i = 0; i < N; i += 16) {
            const double x = static_cast<double>(i) * h;

            for (int j = 0; j < N; j += 16) {
                const double y = static_cast<double>(j) * h;
                double v = 0.0;

                if (i >= 1 && i <= N - 2 && j >= 1 && j <= N - 2) {
                    const auto it = mapv.find(key(i, j));
                    if (it != mapv.end()) v = it->second;
                }

                fout << x << " " << y << " " << v << "\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
