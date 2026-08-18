---
layout: post
title: "ABC of Floating-Point Numbers"
short_title: "ABC of Floating-Point"
date: 2026-08-19
comments: true
categories: [precision]
---

> *"I have measured out my life with coffee spoons."*
> — **T.S. Eliot**, *The Love Song of J. Alfred Prufrock*

### The Need for Limits

How precisely do we need to represent any number in neural networks? Or, how large of a range of numbers do we need to represent the parameters
of a neural network? This is not something as arbitrary as picking a ball from a black box of multiple colors and guessing the color. But this
can be easily calculated on paper. To do this, we need to understand a few data types, precisely float32, float16, and bfloat16. And we'll also
learn some basic explanation with 8-bit floats (FP8). Let's start.

**Float32 (FP32 - Single Precision):** Denoted as E8M23. It uses 32 bits: 1 for the sign, 8 for the exponent, and 23 for the mantissa (precision).
It provides an enormous range of values and pinpoint accuracy, but it takes up 4 bytes of memory per number. When training a model with billions of
parameters, passing FP32 tensors back and forth starves your GPU's memory bandwidth.

**Float16 (FP16 - Half Precision):** Denoted as E5M10. FP16 uses 16 bits (2 bytes): 1 for the sign, 5 for the exponent, and 10 for the mantissa.
It uses half the memory when compared to Float32, but reducing from E8 to E5 drastically lowers the maximum number it can represent.

**Bfloat16 (BF16 - Brain Floating Point):** Denoted by E8M7 and developed by Google Brain. This is a smart alternative to FP16. It also uses 16 bits,
but it redistributes them: 1 for the sign, 8 for the exponent (same as FP32), and only 7 for the mantissa. It has lower precision (fewer decimal places)
but has the massive dynamic range of FP32.

How do we identify the range of numbers represented by any data type? Let's explore.

### The Standard IEEE 754 Formula

For any sort of floating-point data type, the hardware interprets the bits using this formula:

$$
\text{Value} = (-1)^{\text{sign}} \times 2^{(\text{exponent} - \text{bias})} \times (1 + \text{mantissa}) \tag{1}
$$

This looks like a cryptographic nightmare at first glance. Let's get through it step-by-step.

We know the basic idea of computer science: a computer never understands anything except binary. So, we have to represent everything in
**0**s and **1**s. Almost everyone studied basic concepts of binary and decimal numbers covered in our high-school curriculum, which certainly includes:

* Converting from decimal to binary and vice-versa
* Signed and unsigned representation of binary numbers
* Calculating the 1's complement of a binary number
* Calculating the 2's complement of a binary number

Assuming we have this much knowledge, we can proceed further.

#### The Sign
The first thing in Equation 1 is the **sign**. This is simple. For signed representation of binary numbers, the number is considered to be positive
if the leftmost bit (the MSB, or Most Significant Bit) is 0, and negative if it's 1. It's that simple.

#### The Exponent

Then comes the **exponent**. What is it? Let's consider FP8, specifically the format known as E4M3. Here, we have:

* **1 bit** for the Sign
* **4 bits** for the Exponent
* **3 bits** for the Mantissa

With 4 bits for the exponent, we can represent $2^4 = 16$ different integers, from `0000` (0) to `1111` (15).
However, IEEE 754 reserves the maximum exponent of all 1s (`1111`) strictly to represent special cases like `Infinity` or `NaN` (Not a Number).
As soon as our exponent hits that ceiling, we're doomed. Therefore, the maximum *usable* exponent for a normal number is `1110`, which is **14**
in decimal. That's the role of the exponent.

But, why do we need the term **bias** here?

#### The Need For Bias

Normally, computers represent negative integers using a system called **Two's Complement**. However, Two's Complement has a problem: a negative number
like -1 is stored as all ones (`1111`), while a small positive number like 1 is stored as `0001`.

If we used Two's Complement for our floating-point exponents, the hardware would look at the bits `1111` (-1) and `0001` (+1), and the circuitry would
assume `1111` is larger because the raw binary value is higher. To compare two floating-point numbers, the CPU would have to unpack the float, extract
the exponent, convert the Two's Complement, and *then* compare them. That is far too much work for a simple hardware operation.

**The Solution: Biased Representation**
Instead of using a sign bit for the exponent, the exponent bits are simply treated as a standard, unsigned positive integer (where `0000` is the
absolute smallest and `1111` is the absolute largest). This introduces the following logic:

* Small unsigned binary integers (below the bias) become negative exponents (representing very small fractions).
* Large unsigned binary integers (above the bias) become positive exponents (representing large numbers).

To get our negative exponents back in reality, we shift the center point by subtracting a constant **bias**. This creates perfect lexicographical
ordering, where the hardware can just compare the raw bits to know which number is bigger.

The IEEE 754 standard dictates that the bias is calculated based on the number of exponent bits ($E$), using the formula:

$$\text{Bias} = 2^{E-1} - 1$$

For our 4-bit exponent in FP8 ($E = 4$):

$$\text{Bias} = 2^{4-1} - 1 = 2^3 - 1 = \mathbf{7}$$

Our 4 bits can store raw integer values from `0000` (0) to `1111` (15). By applying the formula $(\text{Stored Value} - \text{Bias})$, we shift
this entire range down by 7:

| Stored Binary | Raw Integer | Formula (Raw - Bias) | Actual Exponent | Multiplier |
| --- | --- | --- | --- | --- |
| `0001` | 1 | $1 - 7$ | **-6** | $2^{-6}$ |
| `0110` | 6 | $6 - 7$ | **-1** | $2^{-1}$ |
| `0111` | 7 | $7 - 7$ | **0** | $2^0 = 1$ |
| `1000` | 8 | $8 - 7$ | **+1** | $2^1$ |
| `1110` | 14 | $14 - 7$ | **+7** | $2^7$ |

*(Note: `0000` and `1111` are reserved for special cases like zero, subnormals, and NaNs).*

#### The Mantissa (Precision)

We have only one thing left from Equation 1: the **Mantissa**. If the exponent decides the rough magnitude of our number, the mantissa provides
the precision (making it exactly $1.125$ or $1.375$ etc.). In standard base-10 decimals, the digits after the decimal point represent inverse powers of 10 (tenths, hundredths, thousandths). In binary
floating-point math, the digits after the "radix point" represent a sum of inverse powers of 2.

In FP8 (E4M3), we have exactly 3 Mantissa bits. Each bit acts as a binary switch for a specific fraction:

* **Bit 1 (Leftmost):** $\frac{1}{2^1} = \frac{1}{2} = 0.5$
* **Bit 2 (Middle):** $\frac{1}{2^2} = \frac{1}{4} = 0.25$
* **Bit 3 (Rightmost):** $\frac{1}{2^3} = \frac{1}{8} = 0.125$

Let's look at exactly what happens when the hardware reads specific bit patterns.

If the mantissa is `001`, the first two switches are off ($0$), and only the eighths switch is on ($1$):

$$\text{Fraction}_{001} = \left(0 \times \frac{1}{2}\right) + \left(0 \times \frac{1}{4}\right) + \left(1 \times \frac{1}{8}\right) = 0.125$$

If the mantissa is `011`, the halves switch is off, but the quarters and eighths switches are both on. We simply add their values together:

$$\text{Fraction}_{011} = \left(0 \times \frac{1}{2}\right) + \left(1 \times \frac{1}{4}\right) + \left(1 \times \frac{1}{8}\right) = 0.25 + 0.125 = 0.375$$

Remember Equation 1? The IEEE 754 standard dictates that every normal number has a **hidden leading 1**. Because the first bit of the core
scientific notation would always be a 1, there's no need to waste any memory storing it. This means a raw bit pattern of `001` evaluates to
$(1 + 0.125) = \mathbf{1.125}$, and a pattern of `011` evaluates to $(1 + 0.375) = \mathbf{1.375}$. Therefore, the true multiplier is
$(1 + \text{Mantissa})$.

#### The Maximum Mantissa

To find the absolute highest fractional value the mantissa can provide, we flip all 3 bits to `1` (`111`).

$$\text{Fraction}_{111} = \frac{1}{2^1} + \frac{1}{2^2} + \frac{1}{2^3}$$

This is a finite geometric series. The sum of this specific series approaches, but never quite reaches, 1. We can solve it using the standard
geometric sum formula:

$$S = \frac{a(1-r^n)}{1-r}$$

Where our starting term $a = \frac{1}{2}$, our ratio $r = \frac{1}{2}$, and our number of terms $n = 3$:

$$\text{Fraction}_{111} = \frac{\frac{1}{2} (1 - (\frac{1}{2})^3)}{1 - \frac{1}{2}} = 1 - \frac{1}{8} = 1 - 2^{-3}$$

Adding back our hidden leading 1:

$$(1 + \text{Mantissa}) = 1 + (1 - 2^{-3}) = 2 - 2^{-3}$$

### Putting It All Together

Now that we have dismantled the architecture of the float, we can calculate the absolute maximum positive value this 8-bit format can hold.
We will take the maximum normal exponent (`1110`), the maximum mantissa (`111`), and a positive sign bit (`0`), and plug them back into Equation 1:

* **Sign:** Bit `0` means $(-1)^0 = 1$
* **Exponent:** `1110` is 14 in decimal. $14 - 7 = 7$.
* **Mantissa:** `111` gives us $(2 - 2^{-3})$.

$$\text{Value} = 1 \times 2^{7} \times (2 - 2^{-3})$$

Let's distribute the $2^{7}$ inside the parentheses:

$$\text{Value} = (2^{7} \times 2^1) - (2^{7} \times 2^{-3})$$

$$\text{Value} = 2^{8} - 2^4$$

$$\text{Value} = 256 - 16 = \mathbf{240}$$

The highest number an FP8 (E4M3) standard float can represent is exactly 240.

What about the lowest (most negative) number? Because floating-point numbers are perfectly symmetrical, finding the most negative number requires
zero extra math. We simply flip the sign bit from `0` to `1`. The exponent and mantissa remain entirely maxed out.

$$\text{Lowest Value} = (-1)^1 \times 2^{7} \times (2 - 2^{-3}) = \mathbf{-240}$$

So, with just 8 bits, we can mathematically cover a dynamic range from -240 to +240.

***

### An Exception: OCP FP8

But, but, but...

Why are we learning this? To understand deep learning at some point, obviously. And since deep learning requires a more dynamic range, the AI
industry (NVIDIA, ARM, Intel) created **OCP FP8**, bypassing the IEEE 754 standard. In OCP FP8, stead of reserving the entire `1111` exponent block
for NaNs and Infinities, they only reserve the absolute final bit pattern (`1111` exponent + `111` mantissa) for NaN, and completely threw away the
concept of Infinity. This allows the exponent to reach 15, enabling **448** to be the maximum number representable by FP8. And why is it 448?
You should prove it on your pen-and-paper at this point.
