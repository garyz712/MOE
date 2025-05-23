
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

// Top-2 expert selection kernel
__global__ void max_reduction_top2_kernel(const float* logits, int* assignments, float* weights, int M, int N) {
    __shared__ float s_logits[TILE_WIDTH][2];
    __shared__ int s_indices[TILE_WIDTH][2];

    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Initialize local max values and indices
    float max_val[2] = {-1e9f, -1e9f};
    int max_idx[2] = {-1, -1};

    // Each thread processes a subset of logits
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
    s_logits[tid][0] = max_val[0];
    s_logits[tid][1] = max_val[1];
    s_indices[tid][0] = max_idx[0];
    s_indices[tid][1] = max_idx[1];
    __syncthreads();

    // Perform reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Compare first max
            if (s_logits[tid][0] < s_logits[tid + s][0]) {
                s_logits[tid][1] = s_logits[tid][0];
                s_indices[tid][1] = s_indices[tid][0];
                s_logits[tid][0] = s_logits[tid + s][0];
                s_indices[tid][0] = s_indices[tid + s][0];
            }
            // Compare second max
            if (s_logits[tid][1] < s_logits[tid + s][0] && s_indices[tid + s][0] != s_indices[tid][0]) {
                s_logits[tid][1] = s_logits[tid + s][0];
                s_indices[tid][1] = s_indices[tid + s][0];
            }
            if (s_logits[tid][1] < s_logits[tid + s][1] && s_indices[tid + s][1] != s_indices[tid][0]) {
                s_logits[tid][1] = s_logits[tid + s][1];
                s_indices[tid][1] = s_indices[tid + s][1];
            }
        }
        __syncthreads();
    }

    // Thread 0 computes softmax and stores results
    if (tid == 0 && token_idx < M) {
        float max_logit = s_logits[0][0];
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

// New kernel to combine expert outputs without atomicAdd
__global__ void combine_expert_outputs_kernel(
    const float* d_expert_outputs, const int* d_assignments, const float* d_weights,
    float* d_outputs, int M, int hidden_dim_moe, int num_experts) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x; // Token index
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;   // Hidden dimension index

    if (token_idx < M && dim_idx < hidden_dim_moe) {
        // Get the two expert indices and their weights for this token
        int expert_idx1 = d_assignments[token_idx * 2];
        int expert_idx2 = d_assignments[token_idx * 2 + 1];
        float weight1 = d_weights[token_idx * 2];
        float weight2 = d_weights[token_idx * 2 + 1];

        // Compute weighted sum of expert outputs for this token and dimension
        float output = 0.0f;
        if (expert_idx1 >= 0 && expert_idx1 < num_experts) {
            output += weight1 * d_expert_outputs[expert_idx1 * M * hidden_dim_moe + token_idx * hidden_dim_moe + dim_idx];
        }
        if (expert_idx2 >= 0 && expert_idx2 < num_experts) {
            output += weight2 * d_expert_outputs[expert_idx2 * M * hidden_dim_moe + token_idx * hidden_dim_moe + dim_idx];
        }

        // Write directly to output (no atomic operation needed)
        d_outputs[token_idx * hidden_dim_moe + dim_idx] = output;
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

std::vector<float> moe_expert_mlp_cuda(
    const float* token_embeddings,
    const std::vector<std::vector<float>>& expert_weights,
    const std::vector<std::vector<std::pair<int, float>>>& assignments,
    int batch_size, int seq_len, int embed_dim, int hidden_dim_moe, int num_experts) {
    int M = batch_size * seq_len;

    // Allocate device memory
    float *d_token_embeddings, *d_expert_weights, *d_expert_outputs, *d_outputs;
    int *d_assignments;
    float *d_weights;

    CUDA_CHECK(cudaMalloc(&d_token_embeddings, M * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_expert_weights, num_experts * embed_dim * hidden_dim_moe * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_expert_outputs, num_experts * M * hidden_dim_moe * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outputs, M * hidden_dim_moe * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_outputs, 0, M * hidden_dim_moe * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_assignments, M * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, M * 2 * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_token_embeddings, token_embeddings, M * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> all_expert_weights;
    for (int e = 0; e < num_experts; ++e) {
        all_expert_weights.insert(all_expert_weights.end(), expert_weights[e].begin(), expert_weights[e].end());
    }
    CUDA_CHECK(cudaMemcpy(d_expert_weights, all_expert_weights.data(), num_experts * embed_dim * hidden_dim_moe * sizeof(float), cudaMemcpyHostToDevice));

    // Copy assignments and weights to device
    std::vector<int> flat_assignments(M * 2);
    std::vector<float> flat_weights(M * 2);
    for (int i = 0; i < M; ++i) {
        flat_assignments[i * 2] = assignments[i][0].first;
        flat_assignments[i * 2 + 1] = assignments[i][1].first;
        flat_weights[i * 2] = assignments[i][0].second;
        flat_weights[i * 2 + 1] = assignments[i][1].second;
    }
    CUDA_CHECK(cudaMemcpy(d_assignments, flat_assignments.data(), M * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, flat_weights.data(), M * 2 * sizeof(float), cudaMemcpyHostToDevice));

    // Process each expert
    dim3 threadsPerBlockMatmul(TILE_WIDTH, TILE_WIDTH);
    for (int e = 0; e < num_experts; ++e) {
        // Compute expert outputs for all tokens
        dim3 numBlocksMatmul((hidden_dim_moe + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
        matmul_tiled_kernel<<<numBlocksMatmul, threadsPerBlockMatmul>>>(
            d_token_embeddings,
            d_expert_weights + e * embed_dim * hidden_dim_moe,
            d_expert_outputs + e * M * hidden_dim_moe,
            M, embed_dim, hidden_dim_moe);
        CUDA_CHECK(cudaGetLastError());
    }

    // Combine expert outputs using weights
    dim3 threadsPerBlockCombine(16, 16);
    dim3 numBlocksCombine((M + threadsPerBlockCombine.x - 1) / threadsPerBlockCombine.x,
                          (hidden_dim_moe + threadsPerBlockCombine.y - 1) / threadsPerBlockCombine.y);
    combine_expert_outputs_kernel<<<numBlocksCombine, threadsPerBlockCombine>>>(
        d_expert_outputs, d_assignments, d_weights, d_outputs, M, hidden_dim_moe, num_experts);
    CUDA_CHECK(cudaGetLastError());

    // Copy outputs back
    std::vector<float> outputs(M * hidden_dim_moe);
    CUDA_CHECK(cudaMemcpy(outputs.data(), d_outputs, M * hidden_dim_moe * sizeof(float), cudaMemcpyDeviceToHost));

    // Free memory
    CUDA_CHECK(cudaFree(d_token_embeddings));
    CUDA_CHECK(cudaFree(d_expert_weights));
    CUDA_CHECK(cudaFree(d_expert_outputs));
    CUDA_CHECK(cudaFree(d_outputs));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_weights));

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
