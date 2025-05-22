#ifndef MOE_CUDA_TOP2_H
#define MOE_CUDA_TOP2_H

#include <vector>
#include <utility>

std::vector<std::vector<std::pair<int, float>>> moe_router_cuda(
    const float* token_embeddings, const float* router_weights,
    int batch_size, int seq_len, int embed_dim, int num_experts);

std::vector<float> moe_expert_mlp_cuda(
    const float* token_embeddings,
    const std::vector<std::vector<float>>& expert_weights,
    const std::vector<std::vector<std::pair<int, float>>>& assignments,
    int batch_size, int seq_len, int embed_dim, int hidden_dim_moe, int num_experts);

std::vector<float> dense_mlp_cuda(
    const float* token_embeddings, const float* mlp_weights,
    int batch_size, int seq_len, int embed_dim, int hidden_dim);

#endif