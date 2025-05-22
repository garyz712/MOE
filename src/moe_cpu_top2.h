
#ifndef MOE_CPU_TOP2_H
#define MOE_CPU_TOP2_H

#include <vector>
#include <utility>

std::vector<std::vector<std::pair<int, float>>> moe_router_cpu(
    const std::vector<float>& token_embeddings,
    const std::vector<float>& router_weights,
    int batch_size, int seq_len, int embed_dim, int num_experts);

std::vector<float> moe_expert_mlp_cpu(
    const std::vector<float>& token_embeddings,
    const std::vector<std::vector<float>>& expert_weights,
    const std::vector<std::vector<std::pair<int, float>>>& assignments,
    int batch_size, int seq_len, int embed_dim, int hidden_dim_moe);

std::vector<float> dense_mlp_cpu(
    const std::vector<float>& token_embeddings,
    const std::vector<float>& mlp_weights,
    int batch_size, int seq_len, int embed_dim, int hidden_dim);

#endif