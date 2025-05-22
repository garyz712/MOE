Project Proposal: GPU-Accelerated Mixture-of-Experts (MoE) Router with Variable-k Expert Routing
Summary
This project aims to implement a GPU-accelerated inference kernel for the router component of a Mixture-of-Experts (MoE) transformer model. The focus is on supporting variable-k routing, where each token can dynamically route to either one or two experts (i.e., k=1 or k=2) based on router confidence. The project will compare the speed and accuracy of the MoE model against a standard dense MLP layer under this routing regime.

Background Information
Mixture-of-Experts (MoE) models selectively activate only a subset of model parameters for each input token, enabling substantial computational savings and improved model capacity. Traditional MoE designs use fixed-k routing (e.g., always top-1 or top-2 experts). This project extends that by allowing per-token flexibility—each token can route to either 1 or 2 experts, depending on the router's output distribution.

This dynamic routing aims to improve model efficiency without significantly compromising accuracy. The project will compare this approach against standard dense MLPs to evaluate trade-offs in inference time and predictive performance.

Computation Details
Inputs: Token embeddings and router logits.

Routing Strategy: Dynamic k ∈ {1, 2} based on confidence thresholding.

Output: Expert assignments per token and aggregated MLP outputs from selected experts.

Project Explanation
The project involves the following core components:

MoE Router Kernel: Implements GPU routing logic that supports variable expert selection (k=1 or k=2 per token).

Expert MLP Execution: Parallel expert computations with efficient token-to-expert dispatch.

Baseline Comparison: A dense MLP implementation for comparison in terms of speed and accuracy.

Questions to Address
What inference speedup does dynamic k=1/2 routing provide over dense MLPs?

How does allowing k=1 for confident tokens affect overall accuracy and load balance?

Can a variable-k MoE retain accuracy while reducing computational cost?

Previous GPU Implementations
Prior models like Switch Transformer (k=1) and DeepSpeed-MoE (k=2) have demonstrated efficient GPU MoE implementations. However, none explore adaptive k-selection per token, which this project introduces.

Technical Challenges
Efficient GPU implementation of dynamic routing decisions (k=1 or 2).

Load balancing across experts given per-token k.

Memory management for dynamically varying expert outputs and token scatter/gather operations.

Problems to Solve
Design routing logic that adaptively selects 1 or 2 experts based on token confidence.

Optimize expert dispatch and aggregation in CUDA with minimal overhead.

Compare MoE and MLP models with controlled benchmarks.

Deliverables and Goals
CPU baseline implementations for MoE and dense MLP.

CUDA kernels for MoE routing and expert computation with dynamic k.

Benchmark results comparing MoE vs MLP for inference speed and accuracy.

Documentation detailing implementation, challenges, and analysis.

