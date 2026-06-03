#ifndef HEAT2D_EXPLICIT_COMMON_HPP
#define HEAT2D_EXPLICIT_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace heat2d {

constexpr std::size_t CACHELINE_BYTES = 64;

struct Params {
    int N = 0;
    double alpha = 0.0;
    int T = 0;
    int TILE = 0;
};

struct Range {
    int first = 0;
    int last = -1;
    bool empty() const { return first > last; }
};

struct ErrorStats {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
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

static inline std::string trim(std::string s) {
    return rtrim(ltrim(s));
}

static inline std::string tolower_str(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

static bool load_params_strict(const std::string& fname, Params& p) {
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

static inline std::size_t idx2(int i, int j, std::size_t ld) {
    return static_cast<std::size_t>(i) * ld + static_cast<std::size_t>(j);
}

static inline std::size_t round_up(std::size_t x, std::size_t m) {
    return ((x + m - 1) / m) * m;
}

static inline Range split_closed_interval(int first, int last, int part, int parts) {
    const int n = last - first + 1;
    if (n <= 0 || parts <= 0 || part < 0 || part >= parts) return Range{1, 0};

    const int base = n / parts;
    const int rem = n % parts;
    const int count = base + (part < rem ? 1 : 0);
    const int offset = part * base + std::min(part, rem);

    if (count <= 0) return Range{1, 0};
    return Range{first + offset, first + offset + count - 1};
}

template <typename T>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept {
        ptr_ = other.ptr_;
        n_ = other.n_;
        other.ptr_ = nullptr;
        other.n_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            n_ = other.n_;
            other.ptr_ = nullptr;
            other.n_ = 0;
        }
        return *this;
    }

    ~AlignedBuffer() { release(); }

    bool allocate(std::size_t n, std::size_t alignment = CACHELINE_BYTES) {
        release();
        n_ = n;
        if (n == 0) {
            ptr_ = nullptr;
            return true;
        }

        void* raw = nullptr;
        const int rc = posix_memalign(&raw, alignment, n * sizeof(T));
        if (rc != 0) {
            ptr_ = nullptr;
            n_ = 0;
            return false;
        }

        ptr_ = static_cast<T*>(raw);
        return true;
    }

    void release() {
        if (ptr_) std::free(ptr_);
        ptr_ = nullptr;
        n_ = 0;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

    std::size_t size() const { return n_; }

    T& operator[](std::size_t i) { return ptr_[i]; }
    const T& operator[](std::size_t i) const { return ptr_[i]; }

private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};

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

// Solução exata compatível com contorno de Dirichlet nulo:
// u(x,y,t) = sin(pi x) sin(pi y) exp(-2 pi^2 alpha t).
static inline double exact_solution(double x, double y, double t, const Params& p) {
    const double k = pi();
    return std::sin(k * x) * std::sin(k * y) *
           std::exp(-2.0 * k * k * p.alpha * t);
}

static inline void zero_two_fields(double* U0, double* U1, int N, std::size_t ld) {
    for (int i = 0; i < N; ++i) {
        std::fill_n(U0 + idx2(i, 0, ld), ld, 0.0);
        std::fill_n(U1 + idx2(i, 0, ld), ld, 0.0);
    }
}

static inline void initialize_exact_plain(double* U0,
                                          double* U1,
                                          int N,
                                          std::size_t ld,
                                          const Params& p) {
    zero_two_fields(U0, U1, N, ld);

    const double h = compute_h(p);
    for (int i = 1; i <= N - 2; ++i) {
        const double x = static_cast<double>(i) * h;
        for (int j = 1; j <= N - 2; ++j) {
            const double y = static_cast<double>(j) * h;
            U0[idx2(i, j, ld)] = exact_solution(x, y, 0.0, p);
        }
    }
}

static inline ErrorStats compute_errors(const double* U,
                                        int N,
                                        std::size_t ld,
                                        const Params& p,
                                        double final_time) {
    const double h = compute_h(p);

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;

    for (int i = 0; i < N; ++i) {
        const double x = static_cast<double>(i) * h;
        for (int j = 0; j < N; ++j) {
            const double y = static_cast<double>(j) * h;
            const double diff = U[idx2(i, j, ld)] - exact_solution(x, y, final_time, p);
            const double ad = std::abs(diff);

            sum_abs += ad;
            sum_sq += diff * diff;
            max_abs = std::max(max_abs, ad);
        }
    }

    const double denom = static_cast<double>(N) * static_cast<double>(N);

    ErrorStats e;
    e.l1 = sum_abs / denom;
    e.l2 = std::sqrt(sum_sq / denom);
    e.linf = max_abs;
    return e;
}

static inline void write_output(const std::string& fname,
                                const double* U,
                                int N,
                                std::size_t ld,
                                double h) {
    std::ofstream fout(fname);
    if (!fout) {
        std::cerr << "Erro: não foi possível abrir " << fname << " para escrita.\n";
        return;
    }

    fout.setf(std::ios::fixed);
    fout.precision(8);

    for (int i = 0; i < N; i += 16) {
        const double x = i * h;
        for (int j = 0; j < N; j += 16) {
            const double y = j * h;
            fout << x << " " << y << " " << U[idx2(i, j, ld)] << "\n";
        }
    }
}

static inline void print_summary(const char* variant,
                                 const Params& p,
                                 double dt,
                                 double lambda,
                                 double elapsed,
                                 const ErrorStats& err) {
    const double final_time = static_cast<double>(p.T) * dt;

    std::cout.precision(17);
    std::cout << "Variant : " << variant << "\n";
    std::cout << "N : " << p.N << "\n";
    std::cout << "alpha : " << p.alpha << "\n";
    std::cout << "T : " << p.T << "\n";
    std::cout << "TILE : " << p.TILE << "\n";
    std::cout << "dt : " << dt << "\n";
    std::cout << "lambda : " << lambda << "\n";
    std::cout << "final_time : " << final_time << "\n";
    std::cout << "L1_error : " << err.l1 << "\n";
    std::cout << "L2_error : " << err.l2 << "\n";
    std::cout << "Linf_error : " << err.linf << "\n";
    std::cout << "Tempo : " << elapsed << " s\n";
}

} // namespace heat2d

#endif // HEAT2D_EXPLICIT_COMMON_HPP
