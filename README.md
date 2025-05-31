

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

## Performance Comparisons between K=1 MOE vs K=2 MOE, Dense MLP vs Sparse MOE, CPU MOE/MLP vs GPU MOE/MLP:

The experiment results highlight significant performance differences between dense MLP, K=1 MOE, and K=2 MOE implementations across CPU and CUDA platforms. For dense MLP, CPU execution times range from 0.028425 ms to 2363.94 ms, while CUDA drastically reduces this to 0.1984 ms to 5.85469 ms, showcasing GPU's superior parallel processing for large-scale matrix operations. K=1 MOE, combining router and expert MLP, performs better on CPU (0.01699 ms to 262.114 ms) than dense MLP due to selective expert computation, with CUDA further accelerating it to 0.470208 ms to 2.43638 ms, though its advantage diminishes with smaller inputs (e.g., Test Case 2). K=2 MOE, selecting two experts with weighted combination, increases CPU times (0.042753 ms to 353.089 ms) due to added complexity, while CUDA maintains efficiency (0.483232 ms to 3.90237 ms). Although it lags slightly behind K=1 MOE due to the top-2 reduction and combination overhead, K=2 MOE providing better GPU MOE vs CPU MOE speed up than K=1 MOE due to better utilization of GPU's parallelism.

Across all test cases, CUDA consistently outperforms CPU by orders of magnitude, with dense MLP benefiting most from GPU acceleration due to its uniform computation, while MOE variants leverage sparse expert selection to reduce workload, especially effective with larger expert pools (e.g., 64 experts in Test Case 5). For smaller inputs (Test Cases 2 and 3), CPU MoE implementations are competitive or faster than CUDA due to minimal data transfer overhead, but scalability favors CUDA as batch and sequence sizes increase (Test Cases 4 and 5), where dense MLP CPU times soar (up to 2394.01 ms) compared to CUDA's 5.85469 ms. K=2 MOE introduces additional latency over K=1 MOE due to dual expert processing, but its flexibility in expert weighting could enhance model accuracy so it is widely used (e.g. Deepseek), making CUDA the preferred choice for large-scale deployments despite the slight performance trade-off. 

## Profiling and Potential Improvements

* Dense MLP Kernel:
The profiling of the MLP kernel on an NVIDIA RTX A5000 GPU, executed in 357.76 µs over 417,318 cycles at 1.17 GHz, reveals 11.53 M kernel instances (1.08 M global), with memory traffic showing 125.22 MB in L1/TEX cache (6.72% hit rate), 4.19 MB in L2 cache (97.60% hit rate), 3.26 MB in device memory, and 10.49 M shared memory requests, indicating heavy global memory reliance and poor L1 data locality despite the matmul_tiled_kernel using a 16x16 tile size to optimize memory access; this suggests potential improvements like adjusting tile size, enhancing coalesced access, or tuning thread blocks to reduce the 1.08 M global instances and improve the L1 hit rate.

* MOE Expert Kernel:
The profiling of the MoE expert MLP kernel (moe_expert_mlp_kernel) on an NVIDIA RTX A5000 GPU, executed in 34.08 µs over 39,377 cycles at 1.15 GHz, shows 526.34 K kernel instances entirely from global memory (525.31 K requests), with memory traffic of 15.32 MB in L1/TEX cache (54.2% hit rate), 65.54 KB in L2 cache (83.09% hit rate), and 2.63 MB in device memory, alongside minimal system memory usage (36.74 KB); the kernel's sequential matrix multiplication loop for per-token expert dispatching, processing 1.58 M elements with a 16x16 thread block, indicates better L1 cache utilization than the dense MLP kernel 

* MOE Top 1 Router Kernel:
The profiling of the max_reduction_top1_kernel on an NVIDIA RTX A5000 GPU, executed in 7.36 µs over 8,371 cycles at 1.13 GHz, shows 6.14 K global kernel instances (4.10 K requests) and 66.60 K shared memory instances (37.22 K requests), with memory traffic of 524.29 KB in L1/TEX cache (4.71% hit rate), 61.79 KB in L2 cache (23.50% hit rate), and 640.00 B in device memory, alongside no compression or peer memory usage; the kernel's efficient use of shared memory for parallel reduction within blocks (up to 512 threads) to select the top-1 expert per token is evident, but the low L1/TEX and L2 cache hit rates suggest limited data reuse, indicating potential optimization opportunities through improved data locality or adjusting block sizes to better align with the 64-thread configuration and reduce global memory dependency.

* MOE Top 2 Router Kernel:
The profiling of the max_reduction_top2_kernel on an NVIDIA RTX A5000 GPU, executed in 12.06 µs over 13,985 cycles at 1.16 GHz with a grid size of (2048, 1, 1) and block size of (16, 1, 1), shows 16.38 K global kernel instances (8.19 K requests) and 120.81 K shared memory instances (72.20 K requests), with memory traffic of 524.29 KB in L1/TEX cache (21.97% hit rate), 202.02 KB in L2 cache (37.54% hit rate), and 128.00 B in device memory; the kernel's implementation, which uses a 16-thread block to perform a top-2 expert selection with softmax weighting, leverages shared memory (s_logits and s_indices) for parallel reduction across 2048 tokens
