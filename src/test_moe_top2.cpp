#include "moe_cpu_top2.h"
#include "moe_cuda_top2.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>

// Simple testing macro
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Test failed: " << message << std::endl; \
            exit(1); \
        } \
    } while (0)

// Generate random float vector
std::vector<float> generate_random_vector(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    std::vector<float> vec(size);
    for (auto& x : vec) x = dis(gen);
    return vec;
}

// Compare two integer vectors (for assignments)
bool compare_int_vectors(const std::vector<std::vector<std::pair<int, float>>>& a, 
                        const std::vector<std::vector<std::pair<int, float>>>& b, 
                        const std::string& test_name) {
    if (a.size() != b.size()) {
        std::cerr << test_name << ": Size mismatch (" << a.size() << " vs " << b.size() << ")" << std::endl;
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) {
            std::cerr << test_name << ": Assignment size mismatch at index " << i << std::endl;
            return false;
        }
        for (size_t j = 0; j < a[i].size(); ++j) {
            if (a[i][j].first != b[i][j].first) {
                std::cerr << test_name << ": Expert index mismatch at token " << i << ", expert " << j 
                          << " (" << a[i][j].first << " vs " << b[i][j].first << ")" << std::endl;
                return false;
            }
            if (std::abs(a[i][j].second - b[i][j].second) > 1e-5) {
                std::cerr << test_name << ": Weight mismatch at token " << i << ", expert " << j 
                          << " (" << a[i][j].second << " vs " << b[i][j].second << ")" << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Compare two float vectors with tolerance
bool compare_float_vectors(const std::vector<float>& a, const std::vector<float>& b, 
                          float tolerance, const std::string& test_name) {
    if (a.size() != b.size()) {
        std::cerr << test_name << ": Size mismatch (" << a.size() << " vs " << b.size() << ")" << std::endl;
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cerr << test_name << ": Mismatch at index " << i << " (" << a[i] << " vs " << b[i] << ")" << std::endl;
            return false;
        }
    }
    return true;
}

// Test MoE router with timing
double test_moe_router(int batch_size, int seq_len, int embed_dim, int num_experts, bool is_cuda) {
    std::string test_name = "MoE Router Top-2 (batch=" + std::to_string(batch_size) + 
                           ", seq=" + std::to_string(seq_len) + 
                           ", embed=" + std::to_string(embed_dim) + 
                           ", experts=" + std::to_string(num_experts) + ")";
    
    // Generate input data
    std::vector<float> token_embeddings = generate_random_vector(batch_size * seq_len * embed_dim);
    std::vector<float> router_weights = generate_random_vector(embed_dim * num_experts);

    double duration = 0.0;
    std::vector<std::vector<std::pair<int, float>>> assignments;

    if (is_cuda) {
        // GPU timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        assignments = moe_router_cuda(token_embeddings.data(), router_weights.data(), 
                                     batch_size, seq_len, embed_dim, num_experts);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float duration_ms;
        cudaEventElapsedTime(&duration_ms, start, stop);
        duration = duration_ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        // CPU timing
        auto start = std::chrono::high_resolution_clock::now();
        assignments = moe_router_cpu(token_embeddings, router_weights, 
                                    batch_size, seq_len, embed_dim, num_experts);
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Compare results
    std::vector<std::vector<std::pair<int, float>>> cpu_assignments = moe_router_cpu(
        token_embeddings, router_weights, batch_size, seq_len, embed_dim, num_experts);
    TEST_ASSERT(compare_int_vectors(cpu_assignments, assignments, test_name), 
                test_name + ": CPU and GPU assignments differ");
    std::cout << test_name << ": Passed (" << (is_cuda ? "CUDA" : "CPU") << ": " << duration << " ms)" << std::endl;

    return duration;
}

// Test MoE expert MLP with timing
double test_moe_expert_mlp(int batch_size, int seq_len, int embed_dim, int hidden_dim_moe, 
                           int num_experts, bool is_cuda) {
    std::string test_name = "MoE Expert MLP Top-2 (batch=" + std::to_string(batch_size) + 
                           ", seq=" + std::to_string(seq_len) + 
                           ", embed=" + std::to_string(embed_dim) + 
                           ", hidden_moe=" + std::to_string(hidden_dim_moe) + 
                           ", experts=" + std::to_string(num_experts) + ")";
    
    // Generate input data
    std::vector<float> token_embeddings = generate_random_vector(batch_size * seq_len * embed_dim);
    std::vector<std::vector<std::pair<int, float>>> assignments(batch_size * seq_len);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_experts - 1);
    std::uniform_real_distribution<float> weight_dis(0.0f, 1.0f);
    for (auto& token_assignments : assignments) {
        token_assignments.resize(2);
        token_assignments[0] = {dis(gen), weight_dis(gen)};
        token_assignments[1] = {dis(gen), weight_dis(gen)};
        while (token_assignments[1].first == token_assignments[0].first) {
            token_assignments[1].first = dis(gen);
        }
        float sum = token_assignments[0].second + token_assignments[1].second;
        token_assignments[0].second /= sum;
        token_assignments[1].second /= sum;
    }
    std::vector<std::vector<float>> expert_weights(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        expert_weights[e] = generate_random_vector(embed_dim * hidden_dim_moe);
    }

    double duration = 0.0;
    std::vector<float> outputs;

    if (is_cuda) {
        // GPU timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        outputs = moe_expert_mlp_cuda(token_embeddings.data(), expert_weights, assignments, 
                                     batch_size, seq_len, embed_dim, hidden_dim_moe, num_experts);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float duration_ms;
        cudaEventElapsedTime(&duration_ms, start, stop);
        duration = duration_ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        // CPU timing
        auto start = std::chrono::high_resolution_clock::now();
        outputs = moe_expert_mlp_cpu(token_embeddings, expert_weights, assignments, 
                                    batch_size, seq_len, embed_dim, hidden_dim_moe);
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Compare results
    std::vector<float> cpu_outputs = moe_expert_mlp_cpu(token_embeddings, expert_weights, assignments, 
                                                       batch_size, seq_len, embed_dim, hidden_dim_moe);
    TEST_ASSERT(compare_float_vectors(cpu_outputs, outputs, 1e-5, test_name), 
                test_name + ": CPU and GPU outputs differ");
    std::cout << test_name << ": Passed (" << (is_cuda ? "CUDA" : "CPU") << ": " << duration << " ms)" << std::endl;

    return duration;
}

// Test Dense MLP with timing
double test_dense_mlp(int batch_size, int seq_len, int embed_dim, int hidden_dim_dense, bool is_cuda) {
    std::string test_name = "Dense MLP (batch=" + std::to_string(batch_size) + 
                           ", seq=" + std::to_string(seq_len) + 
                           ", embed=" + std::to_string(embed_dim) + 
                           ", hidden_dense=" + std::to_string(hidden_dim_dense) + ")";
    
    // Generate input data
    std::vector<float> token_embeddings = generate_random_vector(batch_size * seq_len * embed_dim);
    std::vector<float> mlp_weights = generate_random_vector(embed_dim * hidden_dim_dense);

    double duration = 0.0;
    std::vector<float> outputs;

    if (is_cuda) {
        // GPU timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        outputs = dense_mlp_cuda(token_embeddings.data(), mlp_weights.data(), 
                                batch_size, seq_len, embed_dim, hidden_dim_dense);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float duration_ms;
        cudaEventElapsedTime(&duration_ms, start, stop);
        duration = duration_ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        // CPU timing
        auto start = std::chrono::high_resolution_clock::now();
        outputs = dense_mlp_cpu(token_embeddings, mlp_weights, 
                               batch_size, seq_len, embed_dim, hidden_dim_dense);
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Compare results
    std::vector<float> cpu_outputs = dense_mlp_cpu(token_embeddings, mlp_weights, 
                                                  batch_size, seq_len, embed_dim, hidden_dim_dense);
    TEST_ASSERT(compare_float_vectors(cpu_outputs, outputs, 1e-5, test_name), 
                test_name + ": CPU and GPU outputs differ");
    std::cout << test_name << ": Passed (" << (is_cuda ? "CUDA" : "CPU") << ": " << duration << " ms)" << std::endl;

    return duration;
}

// Run a single test case with timing
void run_test_case(int test_case_num, int batch_size, int seq_len, int embed_dim, 
                   int hidden_dim_dense, int num_experts) {
    int hidden_dim_moe = hidden_dim_dense / num_experts; // MoE hidden dim = dense hidden dim / num_experts
    std::cout << "\nTest Case " << test_case_num << " (batch=" << batch_size << ", seq=" << seq_len 
              << ", embed=" << embed_dim << ", hidden_dense=" << hidden_dim_dense 
              << ", hidden_moe=" << hidden_dim_moe << ", experts=" << num_experts << "):\n";

    // MoE timings
    double cpu_router_time = test_moe_router(batch_size, seq_len, embed_dim, num_experts, false);
    double cuda_router_time = test_moe_router(batch_size, seq_len, embed_dim, num_experts, true);
    double cpu_expert_time = test_moe_expert_mlp(batch_size, seq_len, embed_dim, hidden_dim_moe, 
                                                num_experts, false);
    double cuda_expert_time = test_moe_expert_mlp(batch_size, seq_len, embed_dim, hidden_dim_moe, 
                                                 num_experts, true);
    double cpu_moe_time = cpu_router_time + cpu_expert_time;
    double cuda_moe_time = cuda_router_time + cuda_expert_time;

    // Dense MLP timings
    double cpu_mlp_time = test_dense_mlp(batch_size, seq_len, embed_dim, hidden_dim_dense, false);
    double cuda_mlp_time = test_dense_mlp(batch_size, seq_len, embed_dim, hidden_dim_dense, true);

    // Print timings
    std::cout << "Execution Times:\n";
    std::cout << "CPU MoE (Router + Expert MLP): " << cpu_moe_time << " ms\n";
    std::cout << "CUDA MoE (Router + Expert MLP): " << cuda_moe_time << " ms\n";
    std::cout << "CPU Dense MLP: " << cpu_mlp_time << " ms\n";
    std::cout << "CUDA Dense MLP: " << cuda_mlp_time << " ms\n";
}

int main() {
    // Test case 1: Standard dimensions
    run_test_case(1, 2, 64, 128, 256, 8);

    // Test case 2: Single token
    run_test_case(2, 1, 1, 64, 128, 4);

    // Test case 3: Small dimensions
    run_test_case(3, 1, 2, 32, 64, 2);

    // Test case 4: Large batch and experts
    run_test_case(4, 16, 128, 256, 512, 16);

    // Test case 5: Large batch and large number of experts
    run_test_case(5, 16, 128, 256, 512, 64);

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}