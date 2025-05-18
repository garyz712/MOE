#ifndef MOE_CUDA_H
#define MOE_CUDA_H

#include <vector>

std::vector<int> moe_router_cuda(const float* token_embeddings,
                                 const float* router_weights,
                                 int batch_size, int seq_len, int embed_dim, int num_experts);

std::vector<float> moe_expert_mlp_cuda(const float* token_embeddings,
                                       const std::vector<std::vector<float>>& expert_weights,
                                       const std::vector<int>& assignments,
                                       int batch_size, int seq_len, int embed_dim, int hidden_dim);

std::vector<float> dense_mlp_cuda(const float* token_embeddings,
                                  const float* mlp_weights,
                                  int batch_size, int seq_len, int embed_dim, int hidden_dim);

#endif // MOE_CUDA_H