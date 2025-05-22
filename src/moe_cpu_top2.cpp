#include "moe_cpu_top2.h"
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <functional>

std::vector<std::vector<std::pair<int, float>>> moe_router_cpu(
    const std::vector<float>& token_embeddings,
    const std::vector<float>& router_weights,
    int batch_size, int seq_len, int embed_dim, int num_experts) {
    int M = batch_size * seq_len;
    std::vector<float> logits(M * num_experts);
    // Matrix multiplication: token_embeddings (M x embed_dim) x router_weights (embed_dim x num_experts)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < num_experts; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < embed_dim; ++k) {
                sum += token_embeddings[i * embed_dim + k] * router_weights[k * num_experts + j];
            }
            logits[i * num_experts + j] = sum;
        }
    }
    // Top-2 expert selection with softmax weights
    std::vector<std::vector<std::pair<int, float>>> assignments(M, std::vector<std::pair<int, float>>(2));
    for (int i = 0; i < M; ++i) {
        std::vector<std::pair<float, int>> logit_pairs(num_experts);
        for (int j = 0; j < num_experts; ++j) {
            logit_pairs[j] = {logits[i * num_experts + j], j};
        }
        std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + 2, logit_pairs.end(), 
                          std::greater<std::pair<float, int>>());
        // Compute softmax over top-2 logits
        float max_logit = logit_pairs[0].first;
        float sum_exp = 0.0f;
        for (int j = 0; j < 2; ++j) {
            logit_pairs[j].first = std::exp(logit_pairs[j].first - max_logit);
            sum_exp += logit_pairs[j].first;
        }
        for (int j = 0; j < 2; ++j) {
            assignments[i][j] = {logit_pairs[j].second, logit_pairs[j].first / sum_exp};
        }
    }
    return assignments;
}

std::vector<float> moe_expert_mlp_cpu(
    const std::vector<float>& token_embeddings,
    const std::vector<std::vector<float>>& expert_weights,
    const std::vector<std::vector<std::pair<int, float>>>& assignments,
    int batch_size, int seq_len, int embed_dim, int hidden_dim_moe) {
    int M = batch_size * seq_len;
    std::vector<float> outputs(M * hidden_dim_moe, 0.0f);

    // Process each token with its top-2 experts
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < 2; ++k) {
            int expert_idx = assignments[i][k].first;
            float weight = assignments[i][k].second;
            for (int j = 0; j < hidden_dim_moe; ++j) {
                float sum = 0.0f;
                for (int d = 0; d < embed_dim; ++d) {
                    sum += token_embeddings[i * embed_dim + d] *
                           expert_weights[expert_idx][d * hidden_dim_moe + j];
                }
                outputs[i * hidden_dim_moe + j] += weight * sum;
            }
        }
    }

    return outputs;
}

std::vector<float> dense_mlp_cpu(
    const std::vector<float>& token_embeddings,
    const std::vector<float>& mlp_weights,
    int batch_size, int seq_len, int embed_dim, int hidden_dim) {
    int M = batch_size * seq_len;
    std::vector<float> outputs(M * hidden_dim, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < hidden_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < embed_dim; ++k) {
                sum += token_embeddings[i * embed_dim + k] * mlp_weights[k * hidden_dim + j];
            }
            outputs[i * hidden_dim + j] = sum;
        }
    }
    return outputs;
}