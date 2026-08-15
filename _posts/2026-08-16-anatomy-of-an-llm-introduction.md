---
layout: post
title: "Anatomy of an LLM: From First Principles to Edge Silicon"
date: 2026-08-16
comments: true
---

> *"You are not the only one cursed with knowledge."*  
> — **Thanos**, *Avengers: Infinity War*

When it comes to deep learning, most modern frameworks abstract away the curse of knowledge.
You call `.backward()` or `jax.grad()`, and the math simply happens. But an architecture that cannot be understood
at its absolute lowest level cannot be effectively optimized at scale. 

To truly understand the multivariable calculus and hardware constraints underlying large language models, I decided to build one from the ground up.

### The Origin: From Micrograd to Multidimensional Tensors
The initial spark came from Andrej Karpathy's excellent `micrograd` repository. It was a brilliant demonstration of
reverse-mode automatic differentiation, but it was limited to scalar values. In reality, neural networks push massive multidimensional matrices. 

Inspired by his foundational work, I decided to write my own object-oriented Autograd engine in pure NumPy.
I expanded the scope to handle dynamic tensor broadcasting, matrix multiplications, and custom non-linear activations.
I swapped out standard recursive backward passes for iterative Depth-First Search (DFS) to handle deep computational graphs
without hitting Python's recursion limits. Using this pure NumPy engine, I successfully trained a character-level model on the
`tiny-shakespeare` dataset, replicating Karpathy's results entirely from scratch.

### The Leap: GPT-2 Medium on Edge Silicon
Understanding the math was only the first half of the equation; understanding the hardware was the second.

I decided to leave NumPy behind and transition to JAX/XLA to build and pre-train a GPT-2 Medium model (354M parameters) using
the OpenWebText dataset. To make this an actual engineering challenge, I constrained the environment strictly to consumer edge hardware:
an NVIDIA RTX 5080 Laptop (16GB VRAM) and 32GB of DDR5 RAM. 

Training a model of this size on a 16GB VRAM budget required aggressive systems engineering.
I utilized mixed precision (`bfloat16` and `float32`), bypassed standard memory allocators to defeat VRAM fragmentation,
implemented activation rematerialization, and engineered some interesting stuffs currently out of the scope of my discussion.

### The Roadmap: Anatomy of an LLM
Over the upcoming series of logs, I will dissect every single design choice, mathematical derivation, and hardware optimization that went into this pipeline, function by function.

*   **Part 1:** `make_embedding`
*   **Part 2:** `make_linear`
*   **Part 3:** `make_dropout`
*   **Part 4:** `make_layernorm`
*   **Part 5:** `make_rmsnorm`
*   **Part 6:** `make_rope_cache`
*   **Part 7:** `make_learned_pos_emb`
*   **Part 8:** `make_causal_self_attention`
*   **Part 9:** `make_feed_forward`
*   **Part 10:** `make_feed_forward_swiglu`
*   **Part 11:** `make_transformer_block`
*   **Part 12:** `make_lm_head`
*   **Part 13:** `make_cross_entropy_loss`
*   **Part 14:** `make_adam`
*   **Part 15:** `make_step_lr`
*   **Part 16:** `make_cosine_annealing_lr`
*   **Part 17:** `make_cosine_with_warmup`

The abstractions provided by modern frameworks are powerful, but they hide the true cost of computation.
When we invoke `jax.value_and_grad`, the compiler handles the multivariable calculus invisibly, turning the backward pass into a black box. 

To truly lift the hood, we won't just look at the forward pass in JAX. For every architectural component in this series,
I will also try to pull back the curtain and present the equivalent handwritten `_backward` method from my custom NumPy autograd engine,
breaking down the exact calculus that the compiler is executing behind the scenes.
