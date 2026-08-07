---
layout: default
---

# [Monsur Abdullah]
**Potential Systems Researcher & Engineer | MLSys, JAX/XLA, & High-Performance Deep Learning**

[GitHub]() | [LinkedIn](linkedin.com/in/moonsur161) | [arXiv/Draft]() | [Email](monsur.ab.161@gmail.com)

---

## About Me
I am a curious and independent researcher and perhaps an upcoming systems engineer with a background in Electrical and Electronic Engineering (EEE), trying to build my specialization in Machine Learning Systems (MLSys) and compiler optimization. 

My research lies at the intersection of deep learning and hardware architecture. I design training pipelines that optimize data flows, manage custom memory allocations, and manipulate intermediate representations (IR) to maximize throughput on constrained accelerators. A core philosophy of my work is that models must be strictly optimized at the single-node memory level before they can effectively scale across compute clusters. 

Currently, I am actively seeking fully funded PhD or Pre-Doctoral opportunities for Fall 2027 in labs focused on High-Performance Computing (HPC) and ML Compilers.

---

## Research & Engineering Projects

### 1. Foundation Model Pre-training on Constrained Accelerators (JAX / XLA)
*Developed a custom JAX/XLA pipeline to pre-train a 354M parameter autoregressive transformer on a strict 16GB VRAM ceiling.*
* **Compiler Optimization:** Mitigated host RAM exhaustion and computational graph bloat by leveraging `jax.lax.scan`, forcing the High-Level Optimizer (HLO) to compile a static, unrolled hardware loop across 24 transformer blocks.
* **Memory Architecture:** Resolved VRAM fragmentation by overriding JAX's default BFC allocation in favor of the `platform` allocator (`cudaMalloc`). Designed a constant-memory gradient accumulation loop with an $O(1)$ state size utilizing functional carry states.
* **Memory-Compute Trade-offs:** Implemented activation rematerialization (gradient checkpointing) to dynamically exchange a 33% increase in FLOPs for a 60% reduction in peak memory, ensuring the $O(T^2)$ attention mechanism remains strictly within memory bounds.
* **Fault Tolerance:** Engineered an OS-level checkpointing system integrated with direct Linux kernel (`sysfs`) polling to detect AC power loss, enabling graceful GPU shutdown and state preservation during grid failures.
* **[Link to GitHub Repository]()**

### 2. Custom Autograd Engine & Multivariable Calculus Implementation (NumPy)
*Developed a from-scratch, object-oriented autograd engine in pure NumPy to rigorously model the multivariable calculus underlying the backward pass.*
* **Automatic Differentiation:** Engineered a custom DAG-based `Value` class to track gradients through complex scalar and matrix operations without reliance on high-level autodiff primitives.
* **Model Architecture:** Built and trained Multi-Layer Perceptrons (MLPs) for non-linear optimization, scaling the framework to support and train a miniature GPT model via hand-crafted forward and backward passes.
* **[Link to GitHub Repository]()**

---

## Technical Expertise
* **Frameworks & Compilers:** JAX, XLA, PyTorch, NumPy, custom autodiff mechanics.
* **Hardware & Systems:** High-Performance Computing (HPC), mixed-precision optimization (`bfloat16`/`float32`), Tensor Core utilization, memory fragmentation defense.
* **Low-Level Engineering:** Advanced Python internals (generators, closures, custom decorators), Linux kernel polling, fault-tolerant I/O, atomic file operations, OS-level page caching (`np.memmap`).
