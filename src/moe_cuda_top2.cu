#include "moe_cuda_top2.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

#define TILE_WIDTH 16

// Check CUDA errors
#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)

// Tiled matrix multiplication kernel
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < M && t * TILE_WIDTH + threadIdx.x < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (col < N && t * TILE_WIDTH + threadIdx.y < K) {
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void max_reduction_top2_kernel(const float* logits, int* assignments, float* weights, int M, int N) {
    __shared__ float s_logits[TILE_WIDTH][2];
    __shared__ int s_indices[TILE_WIDTH][2];

    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Initialize local max values and indices, each thread must hold TOP 2 MAX values that it has seen
    float max_val[2] = {-1e9f, -1e9f};
    int max_idx[2] = {-1, -1};

    // Each thread processes a subset of logits, each block process one token
    for (int i = tid; i < N; i += blockDim.x) {
        float val = logits[token_idx * N + i];
        if (val > max_val[0]) {
            max_val[1] = max_val[0];
            max_idx[1] = max_idx[0];
            max_val[0] = val;
            max_idx[0] = i;
        } else if (val > max_val[1]) {
            max_val[1] = val;
            max_idx[1] = i;
        }
    }

    // Store in shared memory
    s_logits[tid][0] = max_val[0]; //here we know max_val[0] >= max_val[0]
    s_logits[tid][1] = max_val[1];
    s_indices[tid][0] = max_idx[0];
    s_indices[tid][1] = max_idx[1];
    __syncthreads();

    // Perform top 2 reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Compare second max at tid vs first max at tid+s: if smaller, do move the tid+s first max to tid second max, else do nothing
            if (s_logits[tid][1] < s_logits[tid + s][0]) {
                s_logits[tid][1] = s_logits[tid + s][0];
                s_indices[tid][1] = s_indices[tid + s][0];
            }
        }
        __syncthreads();
    }

    // Thread 0 computes softmax and stores results
    if (tid == 0 && token_idx < M) {
        float max_logit = s_logits[0][0]; // this is the top 1 max value in the logits
        float exp_sum = 0.0f;
        float exp_vals[2];
        exp_vals[0] = expf(s_logits[0][0] - max_logit);
        exp_vals[1] = expf(s_logits[0][1] - max_logit);
        exp_sum = exp_vals[0] + exp_vals[1];
        assignments[token_idx * 2] = s_indices[0][0];
        assignments[token_idx * 2 + 1] = s_indices[0][1];
        weights[token_idx * 2] = exp_vals[0] / exp_sum;
        weights[token_idx * 2 + 1] = exp_vals[1] / exp_sum;
    }
}


std::vector<std::vector<std::pair<int, float>>> moe_router_cuda(
    const float* token_embeddings, const float* router_weights,
    int batch_size, int seq_len, int embed_dim, int num_experts) {
    int M = batch_size * seq_len;
    float *d_token_embeddings, *d_router_weights, *d_logits, *d_weights;
    int *d_assignments;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_token_embeddings, M * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_router_weights, embed_dim * num_experts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, M * num_experts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_assignments, M * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, M * 2 * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_token_embeddings, token_embeddings, M * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_router_weights, router_weights, embed_dim * num_experts * sizeof(float), cudaMemcpyHostToDevice));

    // Matrix multiplication: token_embeddings (M x embed_dim) x router_weights (embed_dim x num_experts)
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((num_experts + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_token_embeddings, d_router_weights, d_logits, M, embed_dim, num_experts);
    CUDA_CHECK(cudaGetLastError());

    // Top-2 expert selection
    threadsPerBlock = dim3(TILE_WIDTH);
    numBlocks = dim3(M);
    max_reduction_top2_kernel<<<numBlocks, threadsPerBlock>>>(d_logits, d_assignments, d_weights, M, num_experts);
    CUDA_CHECK(cudaGetLastError());

    // Copy assignments and weights back
    std::vector<int> assignments(M * 2);
    std::vector<float> weights(M * 2);
    CUDA_CHECK(cudaMemcpy(assignments.data(), d_assignments, M * 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weights.data(), d_weights, M * 2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Convert to vector of pairs
    std::vector<std::vector<std::pair<int, float>>> result(M, std::vector<std::pair<int, float>>(2));
    for (int i = 0; i < M; ++i) {
        result[i][0] = {assignments[i * 2], weights[i * 2]};
        result[i][1] = {assignments[i * 2 + 1], weights[i * 2 + 1]};
    }

    // Free memory
    CUDA_CHECK(cudaFree(d_token_embeddings));
    CUDA_CHECK(cudaFree(d_router_weights));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_weights));

    return result;
}

__global__ void scatter_weighted_outputs_kernel(
    const float* d_expert_outputs, const int* d_token_indices, const float* d_token_weights,
    float* d_outputs, int num_tokens, int hidden_dim_moe) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Token index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Hidden dimension index

    if (i < num_tokens && j < hidden_dim_moe) {
        int token_idx = d_token_indices[i];
        float weight = d_token_weights[i];
        float val = d_expert_outputs[i * hidden_dim_moe + j] * weight;
        atomicAdd(&d_outputs[token_idx * hidden_dim_moe + j], val);
    }
}
std::vector<float> moe_expert_mlp_cuda(
    const float* token_embeddings,
    const std::vector<std::vector<float>>& expert_weights,
    const std::vector<std::vector<std::pair<int, float>>>& assignments,
    int batch_size, int seq_len, int embed_dim, int hidden_dim_moe, int num_experts) {
    int M = batch_size * seq_len;

    // Group tokens by expert
    std::vector<std::vector<std::pair<int, float>>> tokens_per_expert(num_experts);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < 2; ++k) {
            int expert_idx = assignments[i][k].first;
            tokens_per_expert[expert_idx].push_back({i, assignments[i][k].second});
        }
    }

    // Allocate device memory
    float *d_token_embeddings, *d_expert_weights, *d_outputs;
    CUDA_CHECK(cudaMalloc(&d_token_embeddings, M * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_expert_weights, num_experts * embed_dim * hidden_dim_moe * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outputs, M * hidden_dim_moe * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_outputs, 0, M * hidden_dim_moe * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_token_embeddings, token_embeddings, M * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> all_expert_weights;
    for (int e = 0; e < num_experts; ++e) {
        all_expert_weights.insert(all_expert_weights.end(), expert_weights[e].begin(), expert_weights[e].end());
    }
    CUDA_CHECK(cudaMemcpy(d_expert_weights, all_expert_weights.data(), num_experts * embed_dim * hidden_dim_moe * sizeof(float), cudaMemcpyHostToDevice));

    // Process each expert
    dim3 threadsPerBlockMatmul(TILE_WIDTH, TILE_WIDTH);
    dim3 threadsPerBlockScatter(16, 16); // Adjust as needed for your GPU
    for (int e = 0; e < num_experts; ++e) {
        if (tokens_per_expert[e].empty()) continue;
        int num_tokens = tokens_per_expert[e].size();

        // Allocate and copy token indices and weights
        int *d_token_indices;
        float *d_token_weights;
        CUDA_CHECK(cudaMalloc(&d_token_indices, num_tokens * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_token_weights, num_tokens * sizeof(float)));
        std::vector<int> token_indices(num_tokens);
        std::vector<float> token_weights(num_tokens);
        for (int i = 0; i < num_tokens; ++i) {
            token_indices[i] = tokens_per_expert[e][i].first;
            token_weights[i] = tokens_per_expert[e][i].second;
        }
        CUDA_CHECK(cudaMemcpy(d_token_indices, token_indices.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_token_weights, token_weights.data(), num_tokens * sizeof(float), cudaMemcpyHostToDevice));

        // Create temporary input and output arrays
        float *d_expert_inputs, *d_expert_outputs;
        CUDA_CHECK(cudaMalloc(&d_expert_inputs, num_tokens * embed_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_expert_outputs, num_tokens * hidden_dim_moe * sizeof(float)));

        // Gather inputs for this expert
        for (int i = 0; i < num_tokens; ++i) {
            int token_idx = token_indices[i];
            CUDA_CHECK(cudaMemcpy(d_expert_inputs + i * embed_dim,
                                 d_token_embeddings + token_idx * embed_dim,
                                 embed_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        // Expert computation: matmul_tiled_kernel
        dim3 numBlocksMatmul((hidden_dim_moe + TILE_WIDTH - 1) / TILE_WIDTH, (num_tokens + TILE_WIDTH - 1) / TILE_WIDTH);
        matmul_tiled_kernel<<<numBlocksMatmul, threadsPerBlockMatmul>>>(d_expert_inputs,
                                                                       d_expert_weights + e * embed_dim * hidden_dim_moe,
                                                                       d_expert_outputs,
                                                                       num_tokens, embed_dim, hidden_dim_moe);
        CUDA_CHECK(cudaGetLastError());

        // Scatter outputs with weights
        dim3 numBlocksScatter((num_tokens + threadsPerBlockScatter.x - 1) / threadsPerBlockScatter.x,
                              (hidden_dim_moe + threadsPerBlockScatter.y - 1) / threadsPerBlockScatter.y);
        scatter_weighted_outputs_kernel<<<numBlocksScatter, threadsPerBlockScatter>>>(d_expert_outputs,
                                                                                    d_token_indices,
                                                                                    d_token_weights,
                                                                                    d_outputs,
                                                                                    num_tokens, hidden_dim_moe);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_expert_inputs));
        CUDA_CHECK(cudaFree(d_expert_outputs));
        CUDA_CHECK(cudaFree(d_token_indices));
        CUDA_CHECK(cudaFree(d_token_weights));
    }

    // Copy outputs back
    std::vector<float> outputs(M * hidden_dim_moe);
    CUDA_CHECK(cudaMemcpy(outputs.data(), d_outputs, M * hidden_dim_moe * sizeof(float), cudaMemcpyDeviceToHost));

    // Free memory
    CUDA_CHECK(cudaFree(d_token_embeddings));
    CUDA_CHECK(cudaFree(d_expert_weights));
    CUDA_CHECK(cudaFree(d_outputs));

    return outputs;
}

std::vector<float> dense_mlp_cuda(
    const float* token_embeddings, const float* mlp_weights,
    int batch_size, int seq_len, int embed_dim, int hidden_dim) {
    int M = batch_size * seq_len;
    float *d_token_embeddings, *d_mlp_weights, *d_outputs;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_token_embeddings, M * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_weights, embed_dim * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outputs, M * hidden_dim * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_token_embeddings, token_embeddings, M * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mlp_weights, mlp_weights, embed_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));

    // Matrix multiplication
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((hidden_dim + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_token_embeddings, d_mlp_weights, d_outputs,
                                                       M, embed_dim, hidden_dim);
    CUDA_CHECK(cudaGetLastError());

    // Copy outputs back
    std::vector<float> outputs(M * hidden_dim);
    CUDA_CHECK(cudaMemcpy(outputs.data(), d_outputs, M * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));

    // Free memory
    CUDA_CHECK(cudaFree(d_token_embeddings));
    CUDA_CHECK(cudaFree(d_mlp_weights));
    CUDA_CHECK(cudaFree(d_outputs));

    return outputs;
}
