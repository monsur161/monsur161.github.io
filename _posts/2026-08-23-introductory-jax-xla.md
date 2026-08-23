---
layout: post
title: "Introductory JAX - Part 03: XLA, Kernel Fusion, and the Memory Wall"
short_title: "Introductory JAX - XLA"
date: 2026-08-23 00:00:00
comments: true
categories: [jax]
---

> *"Words are just... complicated airflow."*
> — **Kendall Roy,**, *Succession*

Now, we'll move to the next phase of compilation, where the HLO is handed over to XLA. What is XLA? It stands for Accelerated Linear Algebra, which is a machine learning compiler that speeds up models from different frameworks like JAX, PyTorch, and TensorFlow on various hardware (CPU, GPU, TPU, etc.). But, XLA's primary goal is not to make the math faster. Its primary goal is to **fight the memory wall**.

Before we proceed further, let's learn some jargon that we'll face in a while:

**1. Kernel:** A small, specialized C++/CUDA function that runs on the GPU. It reads data from slow VRAM, does some math in the fast SRAM, and writes the answer back to slow VRAM.

**2. HBM / VRAM (Global Memory):** Tensors live here. It's huge in size, but relatively slow to access.

**3. SRAM / Registers (Local Memory):** Located directly on the processing cores. It's very fast, but tiny.

**4. Eager execution:** JAX's default behavior when a function is not JIT-compiled and called upon.

> There's a very brief section discussing data movement in the GPU. As we need to keep that essence in mind to understand what XLA does, please [take a look](https://monsur161.github.io/precision/2026/08/19/the-mechanics-of-fp16-and-bf16.html).

## Eager Execution

In eager mode, XLA has no idea about the future. While executing an operation, it has no idea what the next line of Python code is going to ask it to do. Therefore, the code is executed line-by-line, operation-by-operation. The Python interpreter sends individual commands to the GPU one at a time.

Remember our function? Let's write that here again.

```python
@jax.jit            # Comment out for eager execution
def f(x):
    return jnp.sin(x**10 + jnp.log(2.0))

x_in = jnp.array([1.0])
```

For simplicity, we've made the input tensor a 1D vector of shape `(1,)`. So, if we comment out the decorator, it's executed eagerly like below:

| Operation | Action | Memory Transfer |
| --- | --- | --- |
| **Kernel 1 ($x^{10}$)** | Read $x$ from VRAM. Compute $x^{10}$ via the standard power kernel. | 1 Read |
|  | Write the intermediate `temp_pow` tensor to VRAM. | 1 Write |
| **Kernel 2 (Logarithm)** | Push scalar 2.0 from CPU to device. Compute $\ln(2.0)$. | 0 Reads (Host transfer) |
|  | Write the intermediate `temp_log` tensor to VRAM. | 1 Write |
| **Kernel 3 (Addition)** | Read `temp_pow` from VRAM. Read `temp_log` from VRAM. | 2 Reads |
|  | Compute `temp_pow + temp_log`. Write intermediate `temp_add` to VRAM. | 1 Write |
| **Kernel 4 (Sine)** | Read `temp_add` from VRAM. Compute sine wave. | 1 Read |
|  | Write final output to VRAM. | 1 Write |
| **Total Cost** |  | **4 Reads, 4 Writes** |

So, it takes 4 reads and 4 writes, a total of 8 memory transfers, to execute `f(x)` in eager mode. Now, if we had used our original input `x_in = jnp.array([1.0, 3.0])` here, would it take double the transfers, half for the first element and half for the other? **No**. Why? Because GPUs do not execute arrays element-by-element in a sequence, like:

```python
x_in_flat = x_in.flatten()
out = []
for x in x_in_flat:
    out.append(jnp.sin(x**10 + jnp.log(2.0)))
out = jnp.asarray(out)
out = out.reshape(x_in.shape)
```

Instead, they use an architecture called SIMT (Single Instruction, Multiple Threads). When Python dispatches the $x^{10}$ kernel in eager mode, it doesn't launch one kernel for the 1.0 and a second kernel for the 3.0. Rather, it launches a single vectorized kernel. The GPU reads the entire 2-element array from VRAM in one single memory transaction, assigns the 1.0 to Thread 0 and the 3.0 to Thread 1, executes them simultaneously, and writes the entire 2-element result array back to VRAM in one single transaction. This keeps the number of memory transfers the same.

Then, where's the problem? The problem is the increasing overhead with the increasing shape of the tensors. For a single element of float32, we needed to move 4 bytes of data for each of the 8 transfers. And this byte size grows proportionally with the size of the tensor, so we need to move 8 bytes later for each transfer when we have the original `x_in`. The movement of large sizes of data creates a latency penalty.

## The Solution: XLA Kernel Fusion

We know that while executing a JIT-compiled function, a trace named `jaxpr` is formed first, and then it's converted to HLO and forwarded to XLA. XLA performs a static analysis of the entire network at once. It looks for adjacent operations that can be mathematically combined or "fused" into a single, custom GPU kernel. And then, instead of launching 4 separate kernels, XLA writes a brand new, bespoke kernel on the fly that does all the operations simultaneously.

How? Let's verify it for our function `f(x)`. To visualize the fused kernel, we need to see how XLA rewrites what it receives from the HLO. To compare them, we'll generate both the HLO and XLA executable version. Below our JIT-compiled function, we'll use the `.compile()` and `.as_text()` methods. Let's add the following block:

```python
# Creating the HLO
lowered = f.lower(x_in)

# Creating the XLA executable
compiled = lowered.compile()

# Printing both
print("HLO representation:\n")
print(lowered.compiler_ir())
print("\nCompiled version:\n")
print(compiled.as_text())

```

This gives us the following:

```text
HLO representation:

module @jit_f attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2xf32>) -> (tensor<2xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<2xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<2xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<2xf32>
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %4 = stablehlo.log %cst : tensor<f32>
    %5 = stablehlo.convert %4 : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %7 = stablehlo.add %3, %6 : tensor<2xf32>
    %8 = stablehlo.sine %7 : tensor<2xf32>
    return %8 : tensor<2xf32>
  }
}


Compiled version:

HloModule jit_f, is_scheduled=true, entry_computation_layout={(f32[2]{0})->f32[2]{0}}, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}

%fused_computation (param_0.3: f32[2]) -> f32[2] {
  %param_0.3 = f32[2]{0} parameter(0)
  %mul.3 = f32[2]{0} multiply(%param_0.3, %param_0.3), metadata={op_name="jit(f)/mul" source_file="/tmp/ipykernel_1819/2254706847.py" source_line=3 source_end_line=3 source_column=19 source_end_column=39}
  %mul.2 = f32[2]{0} multiply(%mul.3, %mul.3), metadata={op_name="jit(f)/mul" source_file="/tmp/ipykernel_1819/2254706847.py" source_line=3 source_end_line=3 source_column=19 source_end_column=39}
  %mul.1 = f32[2]{0} multiply(%mul.2, %mul.2), metadata={op_name="jit(f)/mul" source_file="/tmp/ipykernel_1819/2254706847.py" source_line=3 source_end_line=3 source_column=19 source_end_column=39}
  %mul.0 = f32[2]{0} multiply(%mul.3, %mul.1), metadata={op_name="jit(f)/mul" source_file="/tmp/ipykernel_1819/2254706847.py" source_line=3 source_end_line=3 source_column=19 source_end_column=39}
  %constant.0 = f32[] constant(0.693147182)
  %broadcast.0 = f32[2]{0} broadcast(%constant.0), dimensions={}
  %add.0 = f32[2]{0} add(%mul.0, %broadcast.0), metadata={op_name="jit(f)/add" source_file="/tmp/ipykernel_1819/2254706847.py" source_line=3 source_end_line=3 source_column=19 source_end_column=39}
  ROOT %sin.0 = f32[2]{0} sine(%add.0), metadata={op_name="jit(f)/sin" source_file="/tmp/ipykernel_1819/2254706847.py" source_line=3 source_end_line=3 source_column=11 source_end_column=40}
}

ENTRY %main.1 (x.1: f32[2]) -> f32[2] {
  %x.1 = f32[2]{0} parameter(0), metadata={op_name="x"}
  ROOT %add_sine_fusion = f32[2]{0} fusion(%x.1), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(f)/sin" source_file="/tmp/ipykernel_1819/2254706847.py" source_line=3 source_end_line=3 source_column=11 source_end_column=40}
}

```

The HLO representation is explained in detail in the [previous blog](https://monsur161.github.io/jax/2026/08/22/introductory-jax-hlo.html). We'll take care of the XLA kernel fusion here. Before starting to read that, there are a few things to observe.

**1. Layout Specifier:** Each tensor now has a `{0}` attached to it, like `f32[2]{0}`. What's that? There's a brief section at the end of this blog on layout specifiers. It's recommended to go through that section if the concept seems new.

**2. Metadata:** XLA has attached something like:
`metadata={op_name="jit(f)/mul" source_file="/tmp/ipykernel_1819/2254706847.py" source_line=3...}`
to the end of every single line. Why?

Once the code is converted into HLO and sent to the GPU, the GPU executes pure machine code. There's no source code anymore. But what if the input data was somehow corrupted, causing one of these operations to generate a NaN (Not a Number) or trigger a memory overflow? The GPU would crash. And at that point, the hardware would just throw a peculiar memory fault without this metadata, and we would have no trace to it. To prevent this, XLA is leaving this metadata tag to trace the error back to the source code. It just refers to the origin of any error by directly pointing to it.

Now, let's look directly at the calculation steps:

* `HloModule jit_f, is_scheduled=true, entry_computation_layout={(f32[2]{0})->f32[2]{0}}, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}` $\rightarrow$ This is the global configuration. Here:

    * `HloModule jit_f` $\rightarrow$ Tells the hardware to look here for a self-contained module compiled from a function named f.
    
    * `is_scheduled=true` $\rightarrow$ Making it True enables the hardware to have a chronological timeline of what to do after what. Before this, the math graph was just a web of dependencies (like add after exponentiation, sine after adding, etc.). But hardware doesn't understand it. The `is_scheduled=true` here means that XLA has already run a complex memory-planning algorithm to determine the exact, nanosecond-by-nanosecond chronological order these operations must fire in to minimize VRAM usage.
    
    * `entry_computation_layout={(f32[2]{0})->f32[2]{0}}` $\rightarrow$ This is the strict memory formatting of what comes in and what goes out.
    
    * `allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}` $\rightarrow$ SPMD (Single Program, Multiple Data) takes place when a cluster of GPUs is used instead of a single one. These flags allow XLA to shard the input array and output array into pieces and distribute them across multiple GPUs while using a cluster.


* `%fused_computation (param_0.3: f32[2]) -> f32[2]` $\rightarrow$ Identifies param_0.3 as a float32 input of shape (2,) to the GPU kernel named `%fused_computation`, and the output shape of the function is also in float32 and of shape (2,).


* ```text
  %param_0.3 = f32[2]{0} parameter(0)
  %mul.3 = f32[2]{0} multiply(%param_0.3, %param_0.3), metadata={...}
  %mul.2 = f32[2]{0} multiply(%mul.3, %mul.3), metadata={...}
  %mul.1 = f32[2]{0} multiply(%mul.2, %mul.2), metadata={...}
  %mul.0 = f32[2]{0} multiply(%mul.3, %mul.1), metadata={...}
  ```

  Here, `%param_0.3 = f32[2]{0} parameter(0)` tells the GPU to grab the raw input data coming across the motherboard and safely store it in a variable named `%param_0.3`. The later part calculates $x^{10}$. This calculation was already optimized, and there's nothing left to do here.


* `%constant.0 = f32[] constant(0.693147182)` $\rightarrow$ In the stablehlo output, the compiler loaded 2.0 (with `%cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>` ) and explicitly called a log primitive (with `%4 = stablehlo.log %cst : tensor<f32>` ). But here, the log operation is completely gone. XLA's optimizer understood that `jnp.log(2.0)` is made of constants only. Instead of forcing the GPU to calculate the logarithm during **runtime**, the compiler calculated it mathematically on the CPU during **compile time** ($\ln(2) \approx 0.693147$). Then the answer was hardcoded directly into the graph, which saves execution cycles.


* `%broadcast.0 = f32[2]{0} broadcast(%constant.0), dimensions={}` $\rightarrow$ Broadcasts the constant to match the dimension.


* `%add.0 = f32[2]{0} add(%mul.0, %broadcast.0), metadata={...}` $\rightarrow$ Performs the addition of the exponent and the logarithm.


* `ROOT %sin.0 = f32[2]{0} sine(%add.0), metadata={...}` $\rightarrow$ Performs the sine operation on the result from the addition, but why with `ROOT`? `ROOT` identifies something similar to `return` in Python, but in terms of XLA. When an operation is tagged as `ROOT`, it produces the final output tensor that gets handed back to Python. There is always exactly one `ROOT` per computation block, and it's attached directly to the final mathematical operation instead of a separate command. Why?

  If XLA used a separate `return` command, the GPU might allocate a temporary register to hold the result of the sine wave, and then execute a second instruction to move that register to the output buffer. But tagging the operation itself as the `ROOT` makes XLA fuse the math and the return pipeline into a single hardware step. The Tensor Cores or ALUs calculate the sine wave and write the result directly into the output memory address. It eliminates an unnecessary memory move inside the silicon.


* `ENTRY %main.1 (x.1: f32[2]) -> f32[2]` $\rightarrow$ This `ENTRY` computation is the absolute top-level function in XLA. The CPU will hand it over to the GPU (or TPU) to execute the program.

    * `ENTRY %main.1` $\rightarrow$ Tells the hardware to start the execution from here. The `%main.1` is just the internal identifier that XLA assigned to it.
    
    * `(x.1: f32[2])` $\rightarrow$ This is the input argument. XLA knows exactly what is coming from the source code across the PCIe bus. It's a 1D array containing two 32-bit floats.

    * `-> f32[2]` $\rightarrow$ Identifies that the hardware must allocate a new memory block in VRAM for a 2-element array of 32-bit floats before it finishes. This allocation happens before any calculation.


* `%x.1 = f32[2]{0} parameter(0)` $\rightarrow$ A few things happen here.
    
    * `parameter(0)` $\rightarrow$ Tells the compiler to grab the first raw input provided from the VRAM to the `ENTRY` function.

    * `f32[2]{0}` $\rightarrow$ Performs strict typing of this data and confirms its physical layout in memory. The `{0}` indicates a standard, flat 1D layout (stride mapping).

    * `%x.1 =` $\rightarrow$ We identified `x.1: f32[2]` as the single input argument in the `ENTRY` block. Here, the physical memory gets bound to the local variable `%x.1` so that the rest of the block can interact with it.


* `ROOT %add_sine_fusion = f32[2]{0} fusion(%x.1), kind=kLoop, calls=%fused_computation, metadata={...}` $\rightarrow$ It indicates how the GPU cores will operate.

    * `ROOT` $\rightarrow$ Identifies the end and the returning value of this function.

    * `%add_sine_fusion =` $\rightarrow$ The internal name of the result.

    * `fusion(%x.1)` $\rightarrow$ It's an operation, just like `multiply` or `sine`. But it's a fusion operation. Simply, it's the input data passed into the fusion engine.

    * `kind=kLoop` $\rightarrow$ When XLA analyzes a graph, it has to decide how to map the math to the parallel threads of the GPU. As every operation inside this particular `%fused_computation` (exponentiation, addition, sine) is strictly **element-wise**, XLA decides to assign it the `kLoop` fusion strategy. Now, what is `kLoop`?
    
      `kLoop` means that XLA has generated a massively parallel `for` loop. The GPU will spawn one thread for each element in the tensor, where:
      
      * Thread 0 will grab index 0 of `%x.1` (the `1.0`), pull it into a register, run the math ($1.0^{10} + \ln(2)$), apply the sine, and write the answer to the output array.
      
      * Thread 1 will simultaneously do the exact same thing for index 1 (the `3.0`).


    * `calls=%fused_computation` $\rightarrow$ This is the pointer. It tells the hardware to look up the `%fused_computation` block (identified by `%fused_computation (param_0.3: f32[2]) -> f32[2]`) and execute that entire sequence of mathematics as a single unit.


That's enough for now. Let's wrap it up here.


## Extension: Layout Specifier

In Python, we visualize 2D arrays as grids. But the GPU does not. Physical memory (VRAM) is not a 2D grid. It's a single, long, 1D line of storage slots. To store a 2D matrix on this 1D line, the hardware has to flatten it. And, how we choose to flatten it is the **"layout."**

### The Two Ways to Flatten: Row-Major vs. Column-Major

Say we have a simple $2 \times 3$ matrix:

|  | Col 0 | Col 1 | Col 2 |
| --- | --- | --- | --- |
| **Row 0** | A | B | C |
| **Row 1** | D | E | F |

We only have two logical ways to pick up these letters and place them onto our 1D memory street:

1. **Row-Major (The C/Python/JAX Standard):** Read across the first row, then the second row, and so on. Row-Major means that rows are the dimension that changes slow and column dimensions change fast. It's specified by `{1, 0}`, and this is a layout specifier.

    * *Memory slots:* `[0]:A, [1]:B, [2]:C, [3]:D, [4]:E, [5]:F`

2. **Column-Major (The Fortran/MATLAB Standard):** Read down the first column, then the second column, and so on. Column-Major means that columns are the dimension that changes slow and row dimensions change fast. It's specified by `{0, 1}`, and this is also a layout specifier.

    * *Memory slots:* `[0]:A, [1]:D, [2]:B, [3]:E, [4]:C, [5]:F`

> **Key insight:** The same matrix looks completely different in physical silicon depending on which layout is used. The hardware needs a mathematical syntax to track this.

Now, for the XLA output, the compiler saw `f32[2]{0}`. As a 1D vector (`[1.0, 3.0]`) was passed in there, it only has a single axis (Dimension 0). There is nothing to decide about whether to read it by rows or columns. A 1D tensor is already a flat line. Hence, XLA simply assigns it the layout `{0}`.

But, if it all ends up in memory anyway, why do we care? The answer is **Cache Lines**. What's that? When the GPU pulls data from VRAM, it doesn't pull a single float. It pulls a massive chunk of continuous memory all at once, which we call a Cache Line.

If the original input tensor is stored in a **Row-Major** layout, and we write a loop that adds up columns (stepping `A -> D`), the GPU pulls a massive chunk of memory to get `A`, but the rest (`B` and `C`) is of no use because it needs to jump straight to `D`. This forces the GPU to constantly buffer between VRAM and SRAM and lowers the FLOPS. So, the layout specifier is needed to read the correct orientation.
