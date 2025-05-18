#include "moe_cpu.h"
#include <algorithm>

std::vector<int> moe_router_cpu(const std::vector<float>& token_embeddings,
                                const std::vector<float>& router_weights,
                                int batch_size, int seq_len, int embed_dim, int num_experts) {
    int num_tokens = batch_size * seq_len;
    std::vector<float> logits(num_tokens * num_experts);

    // Compute router logits: token_embeddings @ router_weights
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < num_experts; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < embed_dim; ++k) {
                sum += token_embeddings[i * embed_dim + k] * router_weights[k * num_experts + j];
            }
            logits[i * num_experts + j] = sum;
        }
    }

    // Top-k (k=1) expert selection
    std::vector<int> assignments(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        int best_expert = 0;
        float max_val = logits[i * num_experts];
        for (int j = 1; j < num_experts; ++j) {
            if (logits[i * num_experts + j] > max_val) {
                max_val = logits[i * num_experts + j];
                best_expert = j;
            }
        }
        assignments[i] = best_expert;
    }

    return assignments;
}

std::vector<float> moe_expert_mlp_cpu(const std::vector<float>& token_embeddings,
                                      const std::vector<std::vector<float>>& expert_weights,
                                      const std::vector<int>& assignments,
                                      int batch_size, int seq_len, int embed_dim, int hidden_dim) {
    int num_tokens = batch_size * seq_len;
    std::vector<float> outputs(num_tokens * hidden_dim, 0.0f);

    // Compute MLP for each token based on assigned expert
    for (int i = 0; i < num_tokens; ++i) {
        int expert_id = assignments[i];
        const std::vector<float>& weights = expert_weights[expert_id];
        for (int j = 0; j < hidden_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < embed_dim; ++k) {
                sum += token_embeddings[i * embed_dim + k] * weights[k * hidden_dim + j];
            }
            outputs[i * hidden_dim + j] = sum; // No activation for simplicity
        }
    }

    return outputs;
}

std::vector<float> dense_mlp_cpu(const std::vector<float>& token_embeddings,
                                 const std::vector<float>& mlp_weights,
                                 int batch_size, int seq_len, int embed_dim, int hidden_dim) {
    int num_tokens = batch_size * seq_len;
    std::vector<float> outputs(num_tokens * hidden_dim, 0.0f);

    // Compute dense MLP: token_embeddings @ mlp_weights
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < hidden_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < embed_dim; ++k) {
                sum += token_embeddings[i * embed_dim + k] * mlp_weights[k * hidden_dim + j];
            }
            outputs[i * hidden_dim + j] = sum; // No activation for simplicity
        }
    }

    return outputs;
}