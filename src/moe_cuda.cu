#include "moe_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define TILE_WIDTH 16
#define MAX_THREADS_PER_BLOCK 512

// Optimized matrix multiplication with tiling
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH]; //only store a square tile into the shared memory
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; //each block compute one output tile, and there are num of tiles * num of tiles blocks
    int by = blockIdx.y;
    int tx = threadIdx.x; //each thread compute an output element in a tile
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty; //C, sA, sB, are row major but threads are column major
    int col = bx * TILE_WIDTH + tx; //Enable coalesced thread memory access tx for reading A and writing C
    float sum = 0.0f;

    for (int t=0; t<(K+TILE_WIDTH-1)/TILE_WIDTH; ++t){ //iterate through all tiles, each iteration is just a normal gemm mul on a GPU block of size (tile_size, tile_size)
        if (row<M && t*TILE_WIDTH + tx<K){ //Ensures the row is within A’s bounds and the column (tile offset t * TILE_WIDTH plus thread’s tx) is within A’s columns.
            sA[ty][tx] = A[row * K + t * TILE_WIDTH + tx];//each continuous tx in a column in a block process a continuous row in A, because A is stored in row major
        }else{
            sA[ty][tx] = 0.0f;
        }
        if (col<N && t*TILE_WIDTH + ty<K){ //Ensures the column is within B’s bounds. and the row (tile offset t * TILE_WIDTH plus thread’s ty) is within B’s rows.
            sB[ty][tx] = B[(t*TILE_WIDTH + ty)*N + col];//each continuous tx in a column in a block process a continuous row in B, so each ty process a different row
        }else{
            sB[ty][tx] = 0.0f;
        }
        __syncthreads(); // wait until all A, B tiles are loaded

        for (int k = 0; k<TILE_WIDTH; ++k){
            sum += sA[ty][k] * sB[k][tx]; //compute a partial sum at one entry of each output tile, repeat this for num of tiles times in the outer for loop
        }
        __syncthreads(); //wait until all partial sum in the output tile are ready
    }

    if (row < M && col<N){ //at this point all output entries are ready to be written in C
        C[row * N + col] = sum; // each kernel calculate only one output element
    }

}

// MoE expert MLP kernel with per-token expert dispatching
__global__ void moe_expert_mlp_kernel(const float* token_embeddings, const float* expert_weights,
    const int* assignments, float* outputs,
    int M, int K, int N, int num_experts) {
    int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx < M && hidden_idx < N) {
    // Get the assigned expert for this token
        int expert_idx = assignments[token_idx];
        float sum = 0.0f;
        // Compute output: token_embeddings[token_idx] @ expert_weights[expert_idx]
        for (int k = 0; k < K; ++k) {
            sum += token_embeddings[token_idx * K + k] *
            expert_weights[expert_idx * K * N + k * N + hidden_idx];
        }
        outputs[token_idx * N + hidden_idx] = sum;
    }
}


// Parallel max reduction kernel for top-1 selection, one block should cover all moe selection for each token
__global__ void max_reduction_top1_kernel(const float* logits, int* assignments, int M, int N) {
    extern __shared__ float sdata[]; // create a shared space for each token to do reduction within the block
    float* s_logits = sdata; //create a pointer to the shared memory to store logits
    int* s_indices = (int*)(sdata + blockDim.x); //store the indices right after the MOE logits for each token
 
    int tid = threadIdx.x; //thread id
    int token_idx = blockIdx.x; //each block only do reduction for one token
    int idx = token_idx * N + tid; //global logits address for each logit in each token

    // Load logits and indices into shared memory
    float val = (tid < N && token_idx < M) ? logits[idx] : -1e30f;
    int expert_idx = (tid < N && token_idx < M) ? tid : 0;
    s_logits[tid] = val;
    s_indices[tid] = expert_idx;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s>0; s>>=1){
        if (tid < s && tid+s < N){
            if (s_logits[tid] < s_logits[tid+s]){
                s_logits[tid] = s_logits[tid+s];
                s_indices[tid] = s_indices[tid+s];
            }
        }
        __syncthreads();
    }

    // Write result for this token
    if (tid == 0 && token_idx < M){
        assignments[token_idx] = s_indices[0];
    }
}

// MoE Router CUDA Implementation
std::vector<int> moe_router_cuda(const float* token_embeddings,
                                 const float* router_weights,
                                 int batch_size, int seq_len, int embed_dim, int num_experts) {
    int M = batch_size * seq_len;
    int K = embed_dim;
    int N = num_experts;

    float *d_token_embeddings, *d_router_weights, *d_logits;
    int* d_assignments;

    cudaMalloc(&d_token_embeddings, M * K * sizeof(float));
    cudaMalloc(&d_router_weights, K * N * sizeof(float));
    cudaMalloc(&d_logits, M * N * sizeof(float));
    cudaMalloc(&d_assignments, M * sizeof(int));

    cudaMemcpy(d_token_embeddings, token_embeddings, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_router_weights, router_weights, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Matrix multiplication
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_token_embeddings, d_router_weights, d_logits, M, K, N);

    // Parallel max reduction for top-1 selection
    int threads = min(N, MAX_THREADS_PER_BLOCK);
    int blocks = M; // One block per token
    size_t shared_mem_size = threads * (sizeof(float) + sizeof(int));
    max_reduction_top1_kernel<<<blocks, threads, shared_mem_size>>>(d_logits, d_assignments, M, N);

    // Copy assignments back
    std::vector<int> assignments(M);
    cudaMemcpy(assignments.data(), d_assignments, M * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_token_embeddings);
    cudaFree(d_router_weights);
    cudaFree(d_logits);
    cudaFree(d_assignments);

    return assignments;
}

// Expert MLP CUDA Implementation
std::vector<float> moe_expert_mlp_cuda(const float* token_embeddings,
    const std::vector<std::vector<float>>& expert_weights,
    const std::vector<int>& assignments,
    int batch_size, int seq_len, int embed_dim, int hidden_dim) {
    int M = batch_size * seq_len;
    int K = embed_dim;
    int N = hidden_dim;
    int num_experts = expert_weights.size();

    float *d_token_embeddings, *d_expert_weights, *d_outputs;
    int* d_assignments;

    cudaMalloc(&d_token_embeddings, M * K * sizeof(float));
    cudaMalloc(&d_expert_weights, num_experts * K * N * sizeof(float));
    cudaMalloc(&d_outputs, M * N * sizeof(float));
    cudaMalloc(&d_assignments, M * sizeof(int));

    // Copy data
    cudaMemcpy(d_token_embeddings, token_embeddings, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_assignments, assignments.data(), M * sizeof(int), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < num_experts; ++i) {
        cudaMemcpy(d_expert_weights + i * K * N, expert_weights[i].data(), K * N * sizeof(float),
        cudaMemcpyHostToDevice);
    }

    // Launch kernel
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    moe_expert_mlp_kernel<<<numBlocks, threadsPerBlock>>>(d_token_embeddings, d_expert_weights,
                        d_assignments, d_outputs,
                        M, K, N, num_experts);

    // Copy results back
    std::vector<float> outputs(M * N);
    cudaMemcpy(outputs.data(), d_outputs, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_token_embeddings);
    cudaFree(d_expert_weights);
    cudaFree(d_outputs);
    cudaFree(d_assignments);

    return outputs;
}


// Dense MLP CUDA Implementation
std::vector<float> dense_mlp_cuda(const float* token_embeddings,
                                  const float* mlp_weights,
                                  int batch_size, int seq_len, int embed_dim, int hidden_dim) {
    int M = batch_size * seq_len;
    int K = embed_dim;
    int N = hidden_dim;

    float *d_token_embeddings, *d_mlp_weights, *d_outputs;

    cudaMalloc(&d_token_embeddings, M * K * sizeof(float));
    cudaMalloc(&d_mlp_weights, K * N * sizeof(float));
    cudaMalloc(&d_outputs, M * N * sizeof(float));

    cudaMemcpy(d_token_embeddings, token_embeddings, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp_weights, mlp_weights, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_token_embeddings, d_mlp_weights, d_outputs, M, K, N);

    std::vector<float> outputs(M * N);
    cudaMemcpy(outputs.data(), d_outputs, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_token_embeddings);
    cudaFree(d_mlp_weights);
    cudaFree(d_outputs);

    return outputs;
}