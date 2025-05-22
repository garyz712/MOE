

# Project Proposal: GPU-Accelerated Mixture-of-Experts (MoE) Router with Fixed Top-k Routing and MLP Comparison

## Summary

This project implements a GPU-accelerated inference kernel for the **router component** of a Mixture-of-Experts (MoE) transformer model. It supports **fixed top-`k` expert routing**, where `k` is predefined (either 1 or 2), allowing each token to be routed to either one or two experts. Additionally, the project includes a comparison of inference speed and accuracy between the MoE architecture and a standard dense MLP layer.

## Background

Mixture-of-Experts models selectively activate a small subset of experts per input token, significantly reducing computation compared to dense layers. This project explores the performance characteristics of fixed-top-`k` MoE routing (with `k=1` and `k=2`) versus traditional MLPs, using CUDA to accelerate inference on GPUs.

## Design Overview

* **Routing Modes**: Supports both `k=1` and `k=2` routing, where each token is assigned to the top-1 or top-2 experts based on router logits. The value of `k` is fixed at runtime (not adaptive).
* **Expert Execution**: Tokens assigned to the same expert are batched and processed together using shared expert MLPs.
* **Comparative Baseline**: A standard dense MLP layer is implemented as a baseline for performance and accuracy comparison.

## Computation Details

* **Inputs**: Token embeddings and router weight matrices.
* **Routing Output**: Expert assignments per token and corresponding MLP outputs.
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



