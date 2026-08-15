---
layout: default
title: Research & Engineering Projects
permalink: /projects/
---

# Research & Engineering Projects

This page serves as a living log of my engineering implementations, focusing on low-level deep learning architectures, autograd engines, and compiler-optimized training pipelines.

---

### 1. Foundation Model Pre-training on Constrained Accelerators (JAX / XLA) --- August 2026
*Developed a custom JAX/XLA pipeline to pre-train a 354M parameter autoregressive transformer on a strict 16GB VRAM ceiling.*
* **Compiler Optimization:** Mitigated host RAM exhaustion and computational graph bloat by leveraging `jax.lax.scan`, forcing the High-Level Optimizer (HLO) to compile a static, unrolled hardware loop across 24 transformer blocks.
* **Memory Architecture:** Resolved VRAM fragmentation by overriding JAX's default BFC allocation in favor of the `platform` allocator (`cudaMalloc`). Designed a constant-memory gradient accumulation loop with an $O(1)$ state size utilizing functional carry states.
* **Memory-Compute Trade-offs:** Implemented activation rematerialization (gradient checkpointing) to dynamically exchange a 33% increase in FLOPs for a 60% reduction in peak memory, ensuring the $O(T^2)$ attention mechanism remains strictly within memory bounds.
* **Fault Tolerance:** Engineered an OS-level checkpointing system integrated with direct Linux kernel (`sysfs`) polling to detect AC power loss, enabling graceful GPU shutdown and state preservation during grid failures.
* **[Link to GitHub Repository]()**

### 2. Custom Autograd Engine & Multivariable Calculus Implementation (NumPy) --- April 2026
*Developed a from-scratch, object-oriented autograd engine in pure NumPy to rigorously model the multivariable calculus underlying the backward pass.*
* **Automatic Differentiation:** Engineered a custom DAG-based `Value` class to track gradients through complex scalar and matrix operations without reliance on high-level autodiff primitives.
* **Model Architecture:** Built and trained Multi-Layer Perceptrons (MLPs) for non-linear optimization, scaling the framework to support and train a miniature GPT model via hand-crafted forward and backward passes.
* **[Link to GitHub Repository]()**
