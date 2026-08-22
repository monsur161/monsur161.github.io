---
layout: post
title: "Introductory JAX - Part 02: Lowering, StableHLO, and the Universal IR"
short_title: "Introductory JAX - HLO"
date: 2026-08-22 12:00:00
comments: true
categories: [llm]
---

> *"Speak as you might to a young child, or a golden retriever. It wasn't brains that got me here, I can assure you that."*
> — **John Tuld**, *Margin Call*

Now that we know some basic ideas of JAX, let's continue to explore it further. We stopped at building the *jaxpr* graph of a JIT-compiled pure function. That was the scope of the first blog. But, what happens next?

Once the Tracer finishes its run, we are left with a JAXPR. But there is a massive problem: the *XLA* compiler is a completely separate piece of *C++* software built by Google. It has absolutely no idea what *Python*, *JAX*, or a *jaxpr* is. This *XLA* engine understands something called **HLO (High-Level Optimizer)**.

So, our phase 2 is **Lowering**. Lowering what? The *jaxpr*. To what? To *HLO*. So, lowering is the act of translating the Python-specific *jaxpr* into the universal mathematical language that XLA actually understands.

### JAXPR vs. HLO

* **JAXPR** is essentially a Python list of mathematical intents. It knows the shapes and types, but it is still tied to JAX's internal logic.
* **HLO** is something strict. It defines the exact multidimensional tensor operations, the strict memory boundaries, and the hardware-agnostic algebra that needs to occur.

## The Universal ML Language: StableHLO

*XLA* was not just built for JAX. TensorFlow uses it. PyTorch uses it (via PyTorch/XLA). Julia uses it. Because all these different frameworks need to utilize the exact same XLA compiler, they all agreed to lower their internal graphs into a shared intermediate representation (IR) called **StableHLO** (which is a dialect of MLIR, the Multi-Level Intermediate Representation architecture).

> Previously, XLA used its own internal MLIR dialect (`mhlo`, which stands for **MLIR HLO**). But as PyTorch, JAX, and other frameworks all started relying on XLA, Google and the open-source community created **StableHLO**. This is a locked, version-controlled standard that guarantees backward compatibility. Modern JAX has fully migrated to this new standard.

During lowering, JAX takes every single primitive operation in the JAXPR (like `add`, `mul`, `dot_general`) and meticulously translates them into their exact HLO equivalents.

## Seeing the HLO

How do we see the HLO? We can use the `.lower()` and `.compiler_ir` methods in JAX. Let's get to the first function that we wrote back in our first blog.

```python
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    return jnp.sin(x**10 + jnp.log(2.0))

x_in = jnp.array([1.0, 3.0])
```

We have forced it to be a JIT-compiled function with the `@jax.jit` decorator. We could trace the graph even if it's not JIT-compiled. But the HLO translation occurs only when it's a JIT-compiled one. So, to visualize the HLO representation, we'll write the following:

```python
lowered = f.lower(x_in)
print(lowered.compiler_ir())
```

So, the `.lower` method explicitly lowers the function `f` for the argument `x_in`, which is stored in `lowered`. Then the `compiler_ir` method gives the object representation of `lowered`. And what do we get? We get the following:

```text
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
```

There are a few things to observe here before reading it.

**1. Framework Independence:** We had `jaxpr` mentioned in the trace of the function `abs_val` in our first blog. But in the HLO representation above, there is absolutely no mention of JAX or Python, and we can also check it for that `abs_val` function as well. This was the purpose of HLO mentioned a few paragraphs before. Also, if we write this exact same mathematical function in TensorFlow and lower it to HLO, the resulting text will be 100% identical.

**2. Broadcasting is made explicit:** In HLO, the mathematical operations must strictly match tensor dimensions. The lowering phase has the dimensions written and matched in every phase of its IR.

Let's start reading the HLO translation line-by-line:

* `module @jit_f attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32}` $\rightarrow$
This is modern JAX preparing the graph for **SPMD (Single Program, Multiple Data)** execution. At present, our script is running on a single device. Still, the HLO is explicitly stamped with partition data. When we want to use a cluster of GPUs, we'll use `jax.sharding` for splitting and the partition numbers will change according to the number of GPUs in the cluster.
One thing to notice here is the `mhlo`. While the operations have been successfully migrated to the universal `stablehlo` standard, some of the **module-level metadata**—specifically the attributes that tell the compiler how to physically shard and distribute the memory across multiple GPUs or TPUs (`num_partitions`, `num_replicas`)—still rely on legacy MHLO definitions.


* `func.func public @main(%arg0: tensor<2xf32>) -> (tensor<2xf32> {jax.result_info = "result"})` $\rightarrow$ Here, `%arg0` is the input variable of shape `(2,)` and data type `float32`. Now, a 1D tensor containing exactly two 32-bit floats is expected by this JIT-compiled function.


* ```text
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
  %1 = stablehlo.multiply %0, %0 : tensor<2xf32>
  %2 = stablehlo.multiply %1, %1 : tensor<2xf32>
  %3 = stablehlo.multiply %0, %2 : tensor<2xf32>
  ```

  They calculate $x^{10}$. During lowering, the compiler frontend looked at $x^{10}$ and realized that computing a generalized power function in silicon is extremely expensive, which usually requires calculating $e^{10 \ln(x)}$. However, simply multiplying a number by itself takes a single, lightning-fast clock cycle on the GPU. Even then, it didn't decide to execute it in 9 consecutive cycles by multiplying $x$ by itself for 9 consecutive times. Instead, it uses a hardware-level optimization called **Exponentiation by Squaring**:


  * `%0 = stablehlo.multiply %arg0, %arg0` $\rightarrow$ Calculates $x \cdot x = x^2$.
  * `%1 = stablehlo.multiply %0, %0` $\rightarrow$ Squares the previous result: $x^2 \cdot x^2 = x^4$.
  * `%2 = stablehlo.multiply %1, %1` $\rightarrow$ Squares it again: $x^4 \cdot x^4 = x^8$.
  * `%3 = stablehlo.multiply %0, %2` $\rightarrow$ Multiplies the first result by the third: $x^2 \cdot x^8 = \mathbf{x^{10}}$.


  Because it rewrote the math to use pure multiplications, it completely bypassed a naive `pow(x, 10.0)` call. This eliminated the need to allocate a `10.0` constant and erased the need for a `broadcast_in_dim` operation entirely. The compiler optimized the code before XLA even touched it. And this is the **O** in **HLO**.


* ```text
  %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  %4 = stablehlo.log %cst : tensor<f32>
  %5 = stablehlo.convert %4 : tensor<f32>
  %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<2xf32>
  ```

  Now, the compiler executes `jnp.log(2.0)`.
  * `%cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>` $\rightarrow$ Loads the literal constant value $2.0$.
  * `%4 = stablehlo.log %cst : tensor<f32>` $\rightarrow$ Computes the natural logarithm of the constant.
  * `%5 = stablehlo.convert %4 : tensor<f32>` $\rightarrow$ A safety check ensuring the type exactly matches the precision pipeline.
  * `%6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<2xf32>` $\rightarrow$ Hardware math units (ALUs) can't generally add a scalar to an array directly. They require the data blocks to match in size. The `broadcast_in_dim` command physically duplicates the scalar $\ln(2)$ in memory to create a `2xf32` tensor, matching the shape of your $x^{10}$ tensor so they can be fed into the ALUs side-by-side.


* ```text
  %7 = stablehlo.add %3, %6 : tensor<2xf32>
  %8 = stablehlo.sine %7 : tensor<2xf32>
  return %8 : tensor<2xf32>
  ```

  Now that we have both the exponent and the logarithm on the right side of the equation formatted as `tensor<2xf32>`, XLA executes the final primitives.
  * `%7 = stablehlo.add %3, %6 : tensor<2xf32>` $\rightarrow$ Adds the arrays together: $x^{10} + \ln(2)$.
  * `%8 = stablehlo.sine %7 : tensor<2xf32>` $\rightarrow$ Applies the sine wave function to the result.
  * `return %8 : tensor<2xf32>` $\rightarrow$ Passes the memory pointer of the final tensor back to the system.


## But, But, But... Why Do We Need To Know This?

When building neural networks from scratch, inspecting this output allows us to catch memory leaks or redundant calculations. Assume we wrote a function in mixed-precision and then found XLA converting types back and forth (`f32` to `f16` to `f32`) inside the loop. What we need to do then is stop and design a better pipeline, as the FLOPS will underperform during the original training with the lagging one.

In the next blog, we'll move on to understanding the next phase of JIT-compilation. Let's call it a day.
