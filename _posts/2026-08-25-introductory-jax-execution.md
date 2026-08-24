---
layout: post
title: "Introductory JAX - Part 04: LLVM, PTX, and the Hardware Handoff"
short_title: "Introductory JAX - Execution"
date: 2026-08-24 00:00:00
comments: true
categories: [jax]
---

> *"Everything is a copy of a copy of a copy."*
> — **Narrator**, *Fight Club*

Referring to the previous blog, we now have the XLA-compiled executable, or **Optimized HLO**, in our hands. What happens next? This is passed into LLVM. What's that?

> *Note: The upcoming sections are more system-oriented than JAX-oriented. To understand the basic idea of how a JIT-compiled function is executed in JAX, the intermediate sections can be skipped; reading only the final summary of this article is enough.*

LLVM doesn't stand for an acronym. It's just a compiler framework. But why do we actually *need* another compiler framework here, and why don't we just convert the HLO directly into assembly or machine language?

Say we have N front-end frameworks (JAX, PyTorch, TensorFlow) and M hardware targets (Intel CPUs, NVIDIA GPUs, Google TPUs). If we try a one-to-one mapping between the HLO generated from each framework and each piece of hardware, it creates an N × M maintenance nightmare. For every new framework or hardware released, dozens of new translators must be written.

Another problem is that if XLA experts wrote the direct machine language, it wouldn't be as extremely optimized as it is now. Achieving deep expertise in both XLA math and hardware-optimized low-level code is an unbearable hassle for the small lifetime we have.

But, as always, there is an exception: the JAX-TPU combination. Because Google built both, they know how both work. The XLA compiler translates the HLO directly into TPU machine code (LIR, or Low-level Intermediate Representation), completely bypassing LLVM.

For everything else, we hand the HLO over to LLVM. Two things happen here:

1. **Blind Assembly-Type Translation:** LLVM cannot read HLO natively. Initially, it translates the HLO into its own internal language called **LLVM IR (Intermediate Representation)**. This IR is hardware-independent and identical for all CPUs and GPUs.
2. **Hardware-Oriented Optimization:** LLVM then optimizes this IR for the specific hardware architecture being used (Turing, Hopper, Blackwell, etc.). The elimination of dead code also happens here.

So, what happens next? Again, the path splits:

1. **CPU as Hardware:** If we're on a machine with only a CPU, the LLVM IR is translated directly into `x86_64` machine code. Simple.
2. **GPU as Hardware:** If we're on an NVIDIA GPU (like the free Colab T4), NVIDIA provides an LLVM backend translator called **NVPTX**. It translates the optimized program into **PTX (Parallel Thread Execution)**.

    What is this PTX? It's a pseudo, fake assembly language created by NVIDIA. Why fake? Because despite all the hardware coming from NVIDIA, architectures vary wildly. JAX code translated to Turing machine code will not work on Blackwell. PTX is the solution: a stable, universal language that all NVIDIA GPUs understand.

    Problem solved? No.

    Because PTX is a fake assembly language, the GPU silicon still can't execute it. The NVIDIA Driver (Display Driver for consumer GPUs, or User-Mode Driver for server GPUs) contains its own tiny but lightning-fast compiler. The driver looks at the PTX, looks at the exact physical GPU slotted into the motherboard, and does one final translation. It translates PTX into **SASS (Streaming Assembler)**. This SASS is the true machine language.

    Problem solved? Yes.

Now we have `x86_64` machine code for the CPU and SASS for the NVIDIA GPU. Both are binary 1s and 0s. But it is not wise to make Python manually push this binary to the hardware. The solution is **PJRT (Portable JAX Runtime)**. PJRT manages the machine code and takes action based on the hardware:

1. **For CPUs:** PJRT writes the `x86_64` machine code directly into the system RAM and flags the block as Executable. When it's time to run, the CPU's instruction fetcher pulls this block into its cache (L3 → L2 → L1) and executes it.
2. **For GPUs:** The Kernel-Mode Driver locks the SASS into pinned system RAM. The CPU drops a signal to the GPU's Ring Buffer. The GPU uses its DMA (Direct Memory Access) engine to reach across the motherboard's PCIe bus, pull the SASS into VRAM, and execute it using its silicon cores.
3. **For TPUs:** PJRT sees the LIR (Low-level Intermediate Representation) and executes it.

    The problem with GPUs is that the PCIe bus is incredibly narrow compared to the colossal bandwidth inside the GPU itself. This bottleneck is solved in flagship server architectures like Grace Hopper, where NVIDIA puts their ARM-based CPU (Grace) and GPU (Hopper) onto the exact same board. They are wired together using **NVLink-C2C**, which provides unified memory. Because the CPU and GPU share the same RAM, data transfer latency vanishes and DMA over PCIe is no longer needed.

There is one more interesting thing. The time between the CPU signaling the Ring Buffer and the GPU executing the math is not spent lazily. The CPU immediately moves on to the next line of Python code. This is called **Asynchronous Dispatch**.

When the CPU evaluates a JAX function, it immediately returns a `jax.Array`. This is just a hollow pointer in system RAM waiting for the GPU to finish its calculation. This data safely lives in the VRAM across your training loops without ever crossing the PCIe bus. The only time this movement halts is if we explicitly ask Python to look at the output by calling `print(output)` or converting it via `np.asarray()`. In this case, the CPU is forced to wait until the calculation ends, making it a **synchronous** block.

To summarize how a JIT-compiled function is executed in JAX:

* **For CPU:** Tracing with Jaxpr → HLO Formation → XLA Compilation → LLVM IR Generation → LLVM Optimization → `x86_64` Code Generation → PJRT Execution
* **For GPU:** Tracing with Jaxpr → HLO Formation → XLA Compilation → LLVM IR Generation → LLVM Optimization → PTX Code Generation → SASS Code Generation → PJRT Execution
* **For TPU:** Tracing with jaxpr → HLO formation → XLA compilation → LIR (TPU machine code) generation → PJRT execution

And with this, the JIT-compilation of a JAX function is officially wrapped up.
