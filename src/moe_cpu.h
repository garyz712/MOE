#ifndef MOE_CPU_H
#define MOE_CPU_H

#include <vector>

std::vector<int> moe_router_cpu(const std::vector<float>& token_embeddings,
                                const std::vector<float>& router_weights,
                                int batch_size, int seq_len, int embed_dim, int num_experts);

std::vector<float> moe_expert_mlp_cpu(const std::vector<float>& token_embeddings,
                                      const std::vector<std::vector<float>>& expert_weights,
                                      const std::vector<int>& assignments,
                                      int batch_size, int seq_len, int embed_dim, int hidden_dim);

std::vector<float> dense_mlp_cpu(const std::vector<float>& token_embeddings,
                                 const std::vector<float>& mlp_weights,
                                 int batch_size, int seq_len, int embed_dim, int hidden_dim);

#endif // MOE_CPU_H