// Authored by Lukas Riemer
//
// COMPILATION
// Compile with clang, use appropriate optimization flags,
// especially to enable simd, e. g. '-mavx2 -mfma', if available.
// This is tested to be compilable with clang-6.0.0.
// Example compilation command: 'clang++ task_lu.cpp -fopenmp -O3 -mavx2 -mfma -o task_lu'.
// Example compilation command with openblas: 'clang++ task_lu.cpp <path>/libopenblas.a -fopenmp -Ofast -mavx2 -mfma -o task_lu'
// Then execute, without any arguments.
//
// ORGANIZATION
// 0. Prelude
// 1. LU building blocks
// 2. LU versions
// 3. Helper, correctness and benching functions
// 4. Main
//
// BENCH OPTIONS
// If set the three LU building blocks are benchmarked, single threaded.
#define BENCH_BUILDING_BLOCKS true
// Control the benched dimensions for the building blocks,
// only powers of two valid.
#define BENCH_BB_MIN_LEN 4
#define BENCH_BB_MAX_LEN 512
// Control the benched dimensions for the factorization,
// only powers of two valid.
// All combinations on this 'grid' are benchmarked.
#define BENCH_GRID_MIN_LEN 4
#define BENCH_GRID_MAX_LEN 128
#define BENCH_BLOCK_MIN_LEN 32
#define BENCH_BLOCK_MAX_LEN 256
// 1 => Loop-parallelized, 2 => Task-parallelized
#define BENCH_FACTORIZE 2
//
// MISC
// If set performs a full correctness check.
#define CHECK_CORRECTNESS true
// If set uses cblas routines, tested with openblas.
#define USE_BLAS true
// If set seeds randomness to 42.
#define SEED_RAND_42 true
//
// SIZE PARAMETERS (Task-parallelized)
// The most important parameters for the performance of the decomposition
// are the size of the of the grid and the size of each block. This is a
// (surely non-exhaustive) description of the situation:
// If the grid size is ...
// - too small, too little inter-thread parallelism is available,
//   resulting in longer idle times.
// - too large, unnecessary overhead is introduced,
//   e. g. each block needs it's own task that needs to be spawned and managed.
// If the block size is ...
// - too small, not enough intra-thread parallelism is available,
//   e. g. for simd on unrolled loops.
// - too large, blocks don't (entirely) fit into (L1) cache.
// Also, typically, a too large block size also implies a too small grid size
// and vice versa.

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <omp.h>

#if USE_BLAS
#include <mkl_cblas.h>
#endif

// 1. LU building blocks

// An in-place LU decomposition, is applied to the diagonal blocks.
void diag_fact(std::vector<float>& block)
{
    std::size_t n = std::round(std::sqrt(block.size()));
    uint32_t ops = 0;
    for (auto i = 0; i < n; ++i) {
        for (auto j = i; j < n; j++) {
            float sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += block[i * n + k] * block[k * n + j];
            }
            block[i * n + j] = block[i * n + j] - sum;
        }
        for (int j = i + 1; j < n; j++) {
            float sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += block[j * n + k] * block[k * n + i];
            }
            block[j * n + i] = (1 / block[i * n + i]) * (block[j * n + i] - sum);
        }
    }
}

// Used to update the block-row to the right of a just computed diagonal block.
void row_update(std::vector<float>& block, std::vector<float>& diagonal_block)
{
#if USE_BLAS
    int n = std::round(std::sqrt(block.size()));
    cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, n, n, 1.0, &diagonal_block[0], n, &block[0], n);
#else    
    std::size_t n = std::round(std::sqrt(block.size()));
    for (auto j = 0; j < n; ++j) {
        for (auto k = 0; k < j; ++k) {
            const auto lfactor = diagonal_block[j * n + k];
            #pragma omp simd
            for (auto i = 0; i < n; ++i) {
                block[j * n + i] -= lfactor * block[k * n + i];
            }
        }
    }
#endif
}

// Used to update the block-col below a just computed diagonal block.
void col_update(std::vector<float>& block, std::vector<float>& diagonal_block)
{
#if USE_BLAS
    int n = std::round(std::sqrt(block.size()));
    cblas_strsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, &diagonal_block[0], n, &block[0], n);
#else
    std::size_t n = std::round(std::sqrt(block.size()));
    for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < n; ++j) {
            float lsum = 0.0;
            #pragma omp simd
            for (auto k = 0; k < j; ++k) {
                lsum += diagonal_block[k * n + j] * block[i * n + k];
            }
            block[i * n + j] = (block[i * n + j] - lsum) / diagonal_block[j * n + j];
        }
    }
#endif
}

// Used to update the remaining inner block-rectangle.
void trail_update(std::vector<float>& block, std::vector<float>& col_block,
                  std::vector<float>& row_block)
{
#if USE_BLAS
    int n = std::round(std::sqrt(col_block.size()));
    float alpha = -1.0;
    float beta = 1.0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, &col_block[0], n, &row_block[0], n, beta, &block[0], n);
#else
    std::size_t n = std::round(std::sqrt(block.size()));
    for (auto j = 0; j < n; ++j) {
        for (auto i = 0; i < n; ++i) {
            #pragma omp simd
            for (auto k = 0; k < n; ++k) {
                block[j * n + k] -= col_block[i * n + k] * row_block[j * n + i];
            }
        }
    }
#endif
}

// 2. LU versions

void factorize_loop_parallelized(
    std::vector<std::vector<std::vector<float>>>& blocks)
{
    auto num_blocks = blocks.size();

    // With this version, all parallelism is within each iteration.

    #pragma omp parallel
    for (auto i = 0; i < num_blocks; ++i) {
        #pragma omp single
        diag_fact(blocks[i][i]);

        #pragma omp for nowait
        for (auto j = i + 1; j < num_blocks; ++j) {
            row_update(blocks[i][j], blocks[i][i]);
        }

        #pragma omp for
        for (auto j = i + 1; j < num_blocks; ++j) {
            col_update(blocks[j][i], blocks[i][i]);
        }

        // Without collapse that would be similar to schedule(static, n).
        #pragma omp for collapse(2)
        for (auto j = i + 1; j < num_blocks; ++j) {
            for (auto k = i + 1; k < num_blocks; ++k) {
                trail_update(blocks[j][k], blocks[i][k], blocks[j][i]);
            }
        }
    }
}

void factorize_task_parallelized(
    std::vector<std::vector<std::vector<float>>>& blocks)
{
    auto nb = blocks.size();

    // With this version, all available parallelism can be exploited.

    // A placeholder to convince some compilers.
    // A nullpointer for whatever reason is proven to be silently incorrect.
    float task_p [nb * nb];

#pragma omp parallel
#pragma omp single
    for (auto i = 0; i < nb; ++i) {

        // Use priority(1) to suggest priorized execution to the runtime.
        #pragma omp task depend(inout : task_p[i * nb + i]) priority(1)
        diag_fact(blocks[i][i]);

        for (auto j = i + 1; j < nb; ++j) {
            #pragma omp task depend(inout : task_p[j * nb + i])   \
                             depend(in : task_p[i * nb + i]) 
            col_update(blocks[j][i], blocks[i][i]);

            #pragma omp task depend(inout : task_p[i * nb + j])   \
                             depend(in : task_p[i * nb + i]) 
            row_update(blocks[i][j], blocks[i][i]);
        }

        for (auto j = i + 1; j < nb; ++j) {
            for (auto k = i + 1; k < nb; ++k) {
                #pragma omp task depend(inout :                     \
                                            task_p[j * nb + k])     \
                                 depend(in :                        \
                                            task_p[i * nb + k],     \
                                            task_p[j * nb + i])
                trail_update(blocks[j][k], blocks[i][k], blocks[j][i]);
            }
        }
    }
}

// 3. Helper, correctness and benching functions

std::vector<std::vector<std::vector<float>>> random_blocked_mat(int block_len,
                                                                int blocks_len)
{
#if SEED_RAND_42
    std::mt19937 mersenne_engine{ 42 };
#else
    std::random_device dev;
    std::mt19937 mersenne_engine{ dev() };
#endif
    std::uniform_real_distribution<float> dist{ -(float)std::pow(10, 2),
                                                (float)std::pow(10, 2) };
    const auto gen_rand = [&]() { return dist(mersenne_engine); };
    std::vector<std::vector<std::vector<float>>> mat(blocks_len);
    for (auto j = 0; j < blocks_len; ++j) {
        mat[j] = std::vector<std::vector<float>>(blocks_len);
        for (auto k = 0; k < blocks_len; ++k) {
            std::vector<float> vec(block_len * block_len);
            std::generate(std::begin(vec), std::end(vec), gen_rand);
            if(j == k){
                // Dominate it diagonally for reasonable configurations.
                for(auto jj = 0; jj < block_len; ++jj){
                    for(auto kk = 0; kk < block_len; ++kk){
                        vec[jj * block_len + kk] = 100000000;
                    }
                }
            }
            mat[j][k] = vec;
        }
    }
    return mat;
}

// Transforms a blocked matrix into a coherent one.
std::vector<float>
unblock_mat(std::vector<std::vector<std::vector<float>>> blocked_mat)
{
    auto blocks_len = blocked_mat.size();
    std::size_t block_len = std::round(std::sqrt(blocked_mat[0][0].size()));
    auto block_size = block_len * block_len;
    auto blocks_size = blocks_len * blocks_len;
    auto line_len = blocks_len * block_len;
    std::vector<float> unblocked_mat(blocks_size * block_size);
    for (auto j = 0; j < line_len; ++j) {
        for (auto k = 0; k < line_len; ++k) {
            auto blocks_j = j / block_len;
            auto block_j = j % block_len;
            auto blocks_k = k / block_len;
            auto block_k = k % block_len;

            unblocked_mat[line_len * j + k]
                = blocked_mat[blocks_j][blocks_k]
                             [block_j * block_len + block_k];
        }
    }
    return unblocked_mat;
}

int main_correctness()
{
    const float EPSILON = 0.0001;

    // Defining the number of elemnts in a block (block_len * block_len),
    // and the number of blocks in the matrix (blocks_len * blocks_len).
    auto block_len = 128;
    auto blocks_len = 8;

    auto mat_blocked_orig = random_blocked_mat(block_len, blocks_len);
    auto mat_unblocked_orig = unblock_mat(mat_blocked_orig);

    // Do the correct thing
    auto mat_unblocked = mat_unblocked_orig;
    diag_fact(mat_unblocked);

    // Do the thing to test
    auto mat_blocked = mat_blocked_orig;
    factorize_task_parallelized(mat_blocked);
    auto result_unblocked = unblock_mat(mat_blocked);

    for (auto i = 0; i < mat_unblocked.size(); ++i) {
        const auto x = result_unblocked[i];
        const auto y = mat_unblocked[i];
        if (std::abs(x - y) >= EPSILON) {
            std::cout << "ERROR: Computed LU entries not correct" << i << std::endl;
        }
    }

    // Do the correct thing
    mat_unblocked = mat_unblocked_orig;
    diag_fact(mat_unblocked);

    // Do the thing to test
    mat_blocked = mat_blocked_orig;
    factorize_loop_parallelized(mat_blocked);
    result_unblocked = unblock_mat(mat_blocked);

    for (auto i = 0; i < mat_unblocked.size(); ++i) {
        const auto x = result_unblocked[i];
        const auto y = mat_unblocked[i];
        if (std::abs(x - y) >= EPSILON) {
            std::cout << "ERROR: Computed LU entries not correct" << i << std::endl;
        }
    }

    std::cout << "OK: Correctness asserted for all LU versions"
              << std::endl;

    return 0;
}

class bench_samples {

public:
    bench_samples()
        : samples(0)
        , start_point()
        , end_point(){};
    inline void sample_start()
    {
        atomic_signal_fence(std::memory_order::memory_order_seq_cst);
        start_point = std::chrono::system_clock::now();
        atomic_signal_fence(std::memory_order::memory_order_seq_cst);
    };
    inline void sample_end()
    {
        atomic_signal_fence(std::memory_order::memory_order_seq_cst);
        end_point = std::chrono::system_clock::now();
        atomic_signal_fence(std::memory_order::memory_order_seq_cst);
        add_sample(end_point - start_point);
    };
    void add_sample(std::chrono::nanoseconds sample)
    {
        samples.push_back(sample);
    };
    std::chrono::nanoseconds mean() const
    {
        return std::accumulate(samples.begin(), samples.end(),
                               std::chrono::nanoseconds(0))
            / samples.size();
    };
    std::chrono::nanoseconds min() const
    {
        return *std::min_element(samples.begin(), samples.end());
    };
    std::chrono::nanoseconds max() const
    {
        return *std::max_element(samples.begin(), samples.end());
    };
    std::chrono::nanoseconds std_dev() const
    {
        auto avg = mean();
        auto n = samples.size();
        auto skipped_count = 0;
        auto skipped_count_ref = &skipped_count;
        std::vector<double> errors(n);
        std::transform(samples.begin(), samples.end(), errors.begin(),
                       [&](std::chrono::nanoseconds sample) {
                           // Skip heavy outliers.
                           if (std::abs((sample - avg).count()) > 10 * avg.count()) {
                               *skipped_count_ref += 1;
                               return 0.0;
                           }
                           return (sample - avg).count()
                               * ((sample - avg).count() / ((double)n - 1));
                       });
        auto variance = std::accumulate(errors.begin(), errors.end(), 0.0)
            * (((double)n - 1) / (n - 1 - skipped_count));
        return std::chrono::nanoseconds(
            (int64_t)std::round(std::sqrt(variance)));
    };

private:
    std::vector<std::chrono::nanoseconds> samples;
    std::chrono::_V2::system_clock::time_point start_point;
    std::chrono::_V2::system_clock::time_point end_point;
};

// 4. Main

void bench_factorize()
{
    // Benching variables
    auto warmup_iters = 3;
    signed long min_bench_iters = 10;
    auto max_ns_per_bench = std::chrono::nanoseconds(2000000000);
    auto min_block_size = BENCH_BLOCK_MIN_LEN;
    auto max_block_size = BENCH_BLOCK_MAX_LEN;
    auto min_grid_size = BENCH_GRID_MIN_LEN;
    auto max_grid_size = BENCH_GRID_MAX_LEN;

    // Benching
    volatile float write_to = 0.0;
    std::vector<bench_samples> samples(0);
    for (auto block_size = min_block_size; block_size <= max_block_size;
         block_size *= 2) {
        for (auto grid_size = min_grid_size; grid_size <= max_grid_size;
             grid_size *= 2) {

            // Setup
            auto sample = bench_samples();
            auto mat = random_blocked_mat(block_size, grid_size);

            // Warmup
            auto start = std::chrono::system_clock::now();
            for (auto i = 0; i < warmup_iters; ++i) {
#if BENCH_FACTORIZE == 1
                factorize_loop_parallelized(mat);
#endif
#if BENCH_FACTORIZE == 2
                factorize_task_parallelized(mat);
#endif
                write_to = mat.back().back().back();
            }
            auto end = std::chrono::system_clock::now();
            auto calculated_bench_iters = max_ns_per_bench
                / std::chrono::nanoseconds((end - start) / warmup_iters);
            auto bench_iters
                = std::max(calculated_bench_iters, min_bench_iters);

            // Bench
            for (auto i = 0; i < bench_iters; ++i) {
                sample.sample_start();
#if BENCH_FACTORIZE == 1
                factorize_loop_parallelized(mat);
#endif
#if BENCH_FACTORIZE == 2
                factorize_task_parallelized(mat);
#endif
                sample.sample_end();

                // To 100% ensure LU must actually be computed.
                write_to = mat.back().back().back();
            }

            // Write out results, specific to lu factorization.
            auto n = block_size * grid_size;
            auto ops = 2.0 / 3.0 * n * n * n + n * n / 2.0 - n / 6.0;
            auto mean_throughput = ops / sample.mean().count();
            auto min_throughput = ops / sample.max().count();
            auto max_throughput = ops / sample.min().count();
            auto stddev = mean_throughput
                * ((float)sample.std_dev().count() / sample.mean().count());

            std::cout << "(" << n * n << ")"
                      << " " << grid_size << "x" << grid_size << " grid, "
                      << block_size << "x" << block_size
                      << " each, throughput: " << mean_throughput << "+/-"
                      << stddev << " (" << min_throughput << ","
                      << max_throughput << ")" << std::endl;
        }
    }
}

void bench_building_blocks()
{
    // Benching variables
    auto warmup_iters = 3;
    signed long min_bench_iters = 50;
    auto max_ns_per_bench = std::chrono::nanoseconds(1000000000);
    auto min_block_len = BENCH_BB_MIN_LEN;
    auto max_block_len = BENCH_BB_MAX_LEN;

    std::cout << "Benching LU building blocks single threaded mean gflops: "
                 "col|row|trail"
              << std::endl;

    // Benching
    volatile float write_to = 0.0;
    for (auto block_len = min_block_len; block_len <= max_block_len;
         block_len *= 2) {

        // Setup
        auto block = unblock_mat(random_blocked_mat(block_len, 1));
        auto col = unblock_mat(random_blocked_mat(block_len, 1));
        auto row = unblock_mat(random_blocked_mat(block_len, 1));

        float gflops[3];
        for (auto bb = 0; bb < 3; ++bb) {

            // Warmup
            auto start = std::chrono::system_clock::now();
            for (auto i = 0; i < warmup_iters; ++i) {
                if (bb == 0) {
                    col_update(col, block);
                }
                if (bb == 1) {
                    row_update(row, block);
                }
                if (bb == 2) {
                    trail_update(block, col, row);
                }
                write_to = block[0];
            }
            auto end = std::chrono::system_clock::now();
            auto calculated_bench_iters = max_ns_per_bench
                / std::chrono::nanoseconds((end - start) / warmup_iters);
            auto bench_iters
                = std::max(calculated_bench_iters, min_bench_iters);

            // Bench
            auto sample = bench_samples();
            for (auto i = 0; i < bench_iters; ++i) {
                sample.sample_start();
                if (bb == 0) {
                    col_update(col, block);
                }
                if (bb == 1) {
                    row_update(row, block);
                }
                if (bb == 2) {
                    trail_update(block, col, row);
                }
                write_to = block[0];
                sample.sample_end();
            }

            // Compute the bench results.
            auto n = (float)block_len;
            float mean_gflops;
            if (bb == 0) {
                mean_gflops = (n * n * n + n * n) / sample.mean().count();
            }
            if (bb == 1) {
                mean_gflops = (n * n * n) / sample.mean().count();
            }
            if (bb == 2) {
                mean_gflops = 2 * n * n * n / sample.mean().count();
            }
            gflops[bb] = mean_gflops;
        }

        std::cout << "LU building blocks mean gflops at block len " << block_len
                  << " : " << std::fixed << std::setprecision(2) << gflops[0]
                  << "|" << gflops[1] << "|" << gflops[2] << std::endl;
    }
}

int main()
{
#if CHECK_CORRECTNESS
    main_correctness();
#endif
#if BENCH_BUILDING_BLOCKS
    bench_building_blocks();
#endif
    bench_factorize();
}
