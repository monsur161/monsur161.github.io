---
layout: post
title: "The Mechanics of 16 Bit Elements: Speed, Precision, and Trade-Offs"
short_title: "The Mechanics of FP16 & BF16"
date: 2026-08-19
comments: true
categories: [precision]
---

> *"There are three ways to make a living in this business: be first, be smarter, or cheat."*
> — **John Tuld**, *Margin Call*

Now it's time to learn float16 (Half Precision). It's denoted by E5M10, which gives us 5 Exponent bits and 10 Mantissa bits. And at this point, we should be capable of proving that the maximum positive value accessible by float16 is

$$\text{Max Value} = \mathbf{65,504}$$

Compared to the $10^{38}$ limit of float32, 65,504 is extremely small. Using this data type can lead us to face a problem named "**gradient overflow**". As expressed by the name itself, gradient overflow indicates the explosion of gradients due to accumulation in consecutive steps. If a sequence of mathematical operations causes a number to spike above 65,504, the hardware returns `NaN` and crashes the training process.

### Step Size Comparison (FP16 and FP32)

Using half the bits of float32, we trade memory with precision. Let's look at how FP16 handles our favorite number: 1024.

In FP16, 1024 looks like this: `0 11001 0000000000`.
Technically, what will be the immediate next step taken by this number? It will be the rightmost mantissa bit flipped: `0 11001 0000000001`.

Using our alternate calculation method: the next integer power step after 1024 is 2048. The distance between them is 1024. In FP16, our 10 mantissa bits give us $2^{10}$ (or 1024) unique combinations. When we divide the distance by the number of combinations, we get:

$$\text{Step Size} = \frac{2048 - 1024}{2^{10}} = \frac{1024}{1024} = \mathbf{1.0}$$

This means the absolute smallest step FP16 can take after 1024 is exactly 1. There's nothing in between, no fraction. The immediate next representable number is 1025. Fractional precision is entirely gone at this magnitude, though it's present in small-scale values.

### Perks of Float16

Before going to the perks, we'll name two terms:

* **GEMM:** Stands for **General Matrix Multiplication**, like we did in high-school mathematics.
* **FLOPS:** Stands for **Floating Point Operations per Second**. It's used to measure the raw computing power and speed of a processor, especially for tasks involving complex decimal numbers.

In plain language, when performing GEMM, what does the hardware (CPU or GPU) perform? It does 2 things repetitively:

* Loading the data
* Performing arithmetic calculation

> We have a small section on this at the end of this blog. That's completely from a hardware engineering perspective and optional to read.

So, if we have a $1 \times 2$ matrix $X$ and a $2 \times 1$ matrix $Y$ in float16 and perform matrix multiplication, we need to move those 4 elements, then perform 2 multiplications and 1 addition. If the elements are stored as float32 (the matrices being denoted as $X_{32}$ and $Y_{32}$), then each being 4 bytes, we have 16 bytes of data transfer for 3 arithmetic operations. And it becomes 8 bytes of data transfer if they were float16 (the matrices being denoted as $X_{16}$ and $Y_{16}$) for the same number of operations.

Now, our hardware is blinkingly fast at arithmetic operations. But the real bottleneck is caused by the data transfer. Certainly, transferring half the data for the same number of operations enables float16 to have higher FLOPS in GEMM compared to float32. This speed is the advantage of using FP16, and now we know the *why*.

But, is the relationship linear, or, do we get double FLOPS with half the size of data? Two things to remember here:

* We work with large matrices.
* Accumulation in float16 loses precision.

What our hardware does silently is accumulating the results in float32 to keep the precision, while the multiplication is done in float16 if we use $X_{16}$ and $Y_{16}$. The data loading stays fast as each element is just 2 bytes, but the intermediate conversion wastes some time. As a result, the final FLOPS is higher but not twice of FP32.

---

# The Google Brain Alternative: BF16

As float16 has no way other than sacrificing its precision for speed, and float32 has to appear in the intermediate arithmetic accumulation, why not get a larger multiplier to get access to a higher range of numbers while having those 16 bits? The answer became bfloat16 (Brain Floating Point), introduced by Google. It's denoted by E8M7, giving us 8 Exponent bits and 7 Mantissa bits.

Notice something familiar? It has the exact same 8-bit exponent structure as FP32. Let's see what happens to the math. And again, we should be capable of proving that the maximum positive value accessible by bfloat16 is:

$$\text{Max Value} \approx \mathbf{3.39 \times 10^{38}}$$

It's something we desire. Even though it only takes up 16 bits of memory, bfloat16 can hold essentially the exact same maximum physical value as the massive 32-bit float.

### Step Size Comparison (BF16, FP16 and FP32)

In BF16, 1024 looks like this: `0 10001001 0000000`.
What is the immediate next step it can take? If we flip the rightmost mantissa bit to `0 10001001 0000001`, what number do we actually get?

Let's use our distance formula. The distance to the next integer power (2048) is still 1024. But this time, our 7 mantissa bits only give us $2^7$ (or 128) unique combinations to slice up that distance.

$$\text{Step Size} = \frac{2048 - 1024}{2^{7}} = \frac{1024}{128} = \mathbf{8.0}$$

So, we'll have to take 8 steps each time we move forward from 1024. The next representable number after 1024 is neither 1024.000122 (like FP32) nor 1025 (like FP16). The next number is exactly **1032**. Any mathematical operation resulting in a number between 1024 and 1032 will be ruthlessly rounded to the nearest available step. If we compare the step size of FP32, FP16, and BF16 at this stage, it will be crawling vs. stepping vs. long-jumping respectively. This is the hidden cost of BF16.

### What are the perks (and the trade-offs)?

The main perk of BF16 is its dynamic range. We can drop BF16 directly into a deep learning model without having to write complex "gradient scaling" code, because it is virtually impossible to overflow $10^{38}$ during regular training. Also, it has the same 2 bytes per element, which enables us to perform faster data movement during training, whereas intermediate upcasting will still be required to hold precision.

A good choice of mixed precision can be bfloat16 and float32. We'll get both speed and precision. But the combination should be taken care of during the design period, otherwise any flaw in the data flow can contaminate the data stream or cause a bloat in the memory. We'll discuss it later in another blog.

---

# Optional: The Physical Reality of Data Movement

When we talk about the bottleneck of data movement during GEMM, it isn't just moving data from the computer's system RAM to the GPU. The true bottleneck is the movement of data entirely **inside the GPU itself**. Before any math happens, our matrices ($X$ and $Y$) are pushed across the motherboard's PCIe bus into the GPU's memory (VRAM). But what happens next?

**The Internal GPU Hierarchy**
Once the data is inside VRAM, the GEMM operation starts. However, the hardware math units that do the actual multiplication (ALUs or Tensor Cores) cannot read directly from VRAM. The data must travel through a strict, physical pipeline:

* **VRAM (HBM):** The main storage (e.g., 80GB on an NVIDIA A100). Huge, but relatively slow.
* **L2 Cache:** A smaller pool of much faster memory shared across the whole GPU.
* **L1 Cache / Shared Memory (SRAM):** Tiny, blisteringly fast memory located right next to the processing cores.
* **Registers:** The absolute fastest memory slots, bolted directly to the math units.

**The GEMM Data Cycle**
When we earlier referred to "moving 4 elements," here is the exact journey each of those elements take:

1. The GPU reads the elements from **VRAM** and pulls them into the ultra-fast **Shared Memory**.
2. The elements are loaded directly into the **Registers**.
3. The math units execute the multiply-and-accumulate operation.
4. The final result is written back out to VRAM.

**Why 16-bit Floats Dominate This Process**
This internal pipeline connecting VRAM to the math units has a fixed physical width, known as Memory Bandwidth. When we use FP32, every single number acts like a 4-byte block. If you use FP16 or BF16, every number is a 2-byte block. By switching to 16-bit data types, we instantly push twice as many numbers through the exact same physical wires in the same amount of time. We can also fit twice as many numbers into the tiny SRAM caches. This keeps the math units constantly fed with data, preventing them from sitting idle and drastically pushing up our FLOPS.
