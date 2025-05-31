

# GPU-Accelerated Mixture-of-Experts (MoE) with Sparse Top-k Routing and MLP Comparison

## Summary

This project implements a GPU-accelerated inference kernel for the **router component** of a Mixture-of-Experts (MoE) transformer model. It supports **fixed top-`k` expert routing**, where `k` is predefined (either 1 or 2), allowing each token to be routed to either one or two experts. Additionally, the project includes a comparison of inference speed and accuracy between different sprase MoE architectures and a standard dense MLP layer.

## Background

Mixture-of-Experts models selectively activate a small subset of experts per input token, significantly reducing computation compared to dense layers. This project explores the performance characteristics of fixed-top-`k` MoE routing (with `k=1` and `k=2`) versus traditional dense MLPs, using CUDA to accelerate inference on GPUs.

## Design Overview

* **Routing Modes**: Supports both `k=1` and `k=2` routing, where each token is assigned to the top-1 or top-2 experts based on router logits. The value of `k` is fixed at runtime.
* **Expert Execution**: Tokens assigned to the same expert are batched and processed together using shared expert MLPs.
* **Comparative Baseline**: A standard dense MLP layer is implemented as a baseline for performance and accuracy comparison.

## Computation Details

* **Inputs**: Token embeddings and router weight matrices.
* **Routing Output**: Expert assignments per token and corresponding MLP/MOE outputs.
* **k Values**: `k=1` and `k=2` are both supported with separate implementations.

## Project Components

1. **MoE Router**: GPU-accelerated router to assign each token to its top-`k` experts.
2. **Expert Execution**: Efficient CUDA kernel for dispatching and computing expert MLPs.
3. **Comparison**: Evaluate and benchmark the MoE model against a dense MLP layer in terms of speed and accuracy.

## Key Questions

* What are the performance benefits of top-`k` MoE routing (with `k=1/2`) over a dense MLP?
* How does increasing `k` from 1 to 2 affect GPU utilization and model accuracy?

## Technical Challenges

* Efficiently batching tokens by expert index for parallel GPU execution.
* Designing fast CUDA kernels for MLP inference under sparse token-to-expert routing.
* Managing memory access patterns to ensure high throughput.

## Deliverables

* CPU baseline for MoE and dense MLP inference.
* CUDA kernels for MoE routing and expert MLP computation.
* Benchmarks comparing speed and accuracy across `k=1`, `k=2`, and dense MLP.
* Documentation and analysis of routing behavior and performance.

## Usage

* cd into src folder
* Compile the files like: nvcc -o test_moe_top2 test_moe_top2.cpp moe_cpu_top2.cpp moe_cuda_top2.cu -std=c++11
* Run the testbench: ./test_moe_top2
* You will see somthing like below:

    
**Result (K=2):**

    Test Case 4 (batch=16, seq=128, embed=256, hidden_dense=512, hidden_moe=32, experts=16):
    MoE Router Top-2 (batch=16, seq=128, embed=256, experts=16): Passed (CPU: 57.2919 ms)
    MoE Router Top-2 (batch=16, seq=128, embed=256, experts=16): Passed (CUDA: 1.08826 ms)
    MoE Expert MLP Top-2 (batch=16, seq=128, embed=256, hidden_moe=32, experts=16): Passed (CPU: 295.797 ms)
    MoE Expert MLP Top-2 (batch=16, seq=128, embed=256, hidden_moe=32, experts=16): Passed (CUDA: 1.82934 ms)
    Dense MLP (batch=16, seq=128, embed=256, hidden_dense=512): Passed (CPU: 2375.61 ms)
    Dense MLP (batch=16, seq=128, embed=256, hidden_dense=512): Passed (CUDA: 5.44253 ms)
    Execution Times:
    CPU MoE (Router + Expert MLP): 353.089 ms
    CUDA MoE (Router + Expert MLP): 2.9176 ms
    CPU Dense MLP: 2375.61 ms
    CUDA Dense MLP: 5.44253 ms

All tests passed!

* Finally, download Nvidia Nsight compute and run the profiler using: ncu -o rope_profile_top2 --set full ./test_moe_top2

## Performance Analysis: K=1 MOE vs K=2 MOE, Dense MLP vs Sparse MOE, CPU MOE/MLP vs GPU MOE/MLP

The experiment results highlight significant performance differences between dense MLP, K=1 MOE, and K=2 MOE implementations across CPU and CUDA platforms. For dense MLP, CPU execution times range from 0.028425 ms to 2363.94 ms, while CUDA drastically reduces this to 0.1984 ms to 5.85469 ms, showcasing GPU's superior parallel processing for large-scale matrix operations. K=1 MOE, combining router and expert MLP, performs better on CPU (0.01699 ms to 262.114 ms) than dense MLP due to selective expert computation, with CUDA further accelerating it to 0.470208 ms to 2.43638 ms, though its advantage diminishes with smaller inputs (e.g., Test Case 2). K=2 MOE, selecting two experts with weighted combination, increases CPU times (0.042753 ms to 353.089 ms) due to added complexity, while CUDA maintains efficiency (0.483232 ms to 3.90237 ms). Although it lags slightly behind K=1 MOE due to the top-2 reduction and combination overhead, K=2 MOE providing better GPU MOE vs CPU MOE speed up than K=1 MOE due to better utilization of GPU's parallelism.

Across all test cases, CUDA consistently outperforms CPU by orders of magnitude, with dense MLP benefiting most from GPU acceleration due to its uniform computation, while MOE variants leverage sparse expert selection to reduce workload, especially effective with larger expert pools (e.g., 64 experts in Test Case 5). For smaller inputs (Test Cases 2 and 3), CPU MoE implementations are competitive or faster than CUDA due to minimal data transfer overhead, but scalability favors CUDA as batch and sequence sizes increase (Test Cases 4 and 5), where dense MLP CPU times soar (up to 2394.01 ms) compared to CUDA's 5.85469 ms. K=2 MOE introduces additional latency over K=1 MOE due to dual expert processing, but its flexibility in expert weighting could enhance model accuracy so it is widely used (e.g. Deepseek), making CUDA the preferred choice for large-scale deployments despite the slight performance trade-off. 



