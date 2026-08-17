---
layout: post
title: "The Mechanics of Shape and Spread: Understanding Distributions"
short_title: "Understanding Distributions"
date: 2026-08-17
comments: true
categories: [math]
---

> *"To understand God's thoughts we must study statistics, for these are the measure of His purpose."*
> — **Florence Nightingale**

We have two types of distributions in our primary concern. One is the uniform distribution, and the other is the normal distribution.
Let's discuss them briefly. This discussion is mostly limited to some high-school level mathematics, basic integration, and the basic
idea of systems (linearity and continuity).

A **uniform distribution** of data is like a perfectly flat, rectangular block. It represents a scenario where every single outcome
within a specific range is equally likely to happen. There are no peaks, no favorites, and no rare extremes. It's just a flat, even
playing field from a starting point $a$ to an ending point $b$.

<img src="/assets/images/uniform-light.png" class="img-light" alt="Uniform distribution plot">
<img src="/assets/images/uniform-dark.png" class="img-dark" alt="Uniform distribution plot">

On the other hand, the **normal distribution** is the famous "bell curve." Unlike the flat box, it has a clear favorite: the data
heavily clusters around a central peak. As we move further away from that center in either direction, the chances of an outcome occurring
drop off symmetrically, creating sweeping tails. It’s the shape that we see mostly everywhere in nature.

<img src="/assets/images/normal-light.png" class="img-light" alt="Normal distribution plot">
<img src="/assets/images/normal-dark.png" class="img-dark" alt="Normal distribution plot">

We have three parameters to count on for each type of distribution:

1. **Mean ($\mu$)**: This is simply the average, or the "center of mass" of our data. In statistics, we often see this written as $E[X]$,
which stands for the **Expected Value**. It's called the expected value because if we were to randomly draw numbers from our distribution
infinitely many times, the average of all those draws would converge exactly to this point.
2. **Variance ($\sigma^2$)**: If the mean tells us where the center is, the variance tells us how aggressively the data is scattered away from
that center. A low variance means the data is tightly packed around the mean (a sharp, narrow spike). A high variance means the data is
widely spread out (a broad, flat shape).
3. **Standard Deviation ($\sigma$)**: This is just the square root of the variance. Why do we need both? Because calculating variance requires
squaring the distances of our data points (as we'll see shortly). That means the unit of variance is squared. Standard deviation takes the
square root to bring our measurement back into the *original units* of our data, making it much easier to interpret in the real world.

Let's derive each of them for a uniform distribution of numbers, say in the range of $[a,b]$.

To derive the mean, we need to know the Probability Density Function (PDF, or $P(x)$) of a uniform distribution. But, what is a PDF?

In simple terms, a PDF is a mathematical function that tells us how probability is distributed across different values. To understand it,
let's start with a discrete example. Think of a standard 6-sided die. Every time we roll it, there is a $\frac{1}{6}$ chance of getting a 1,
a $\frac{1}{6}$ chance of a 2, and so on. If this die had $n$ sides, the probability for any specific side would just be $\frac{1}{n}$.
Because we can cleanly separate and count these outcomes, we call this a *discrete* probability.

So, the mean, or the expected value from a 6-sided die will be:

$$E[X] = \left(1 \times \frac{1}{6}\right) + \left(2 \times \frac{1}{6}\right) + \dots + \left(6 \times \frac{1}{6}\right) = 3.5$$

Which is generally written like:

$$E[X] = \sum x \cdot P(x)$$

But in a uniform distribution, we can't have a fixed number of sample data points. The number of data points between any two numbers lying
between $[a,b]$ is infinite, and hence it's called a **continuous space**. The discrete summation ($\sum$) is replaced by a continuous integral
($\int$) here, with the lower limit being $a$ and the upper limit being $b$. Then, the mean or the expected value of $x$ should be:

$$
E[X] = \int x \cdot f(x) dx \tag{1}
$$

For a Uniform box from $a$ to $b$, denoted as $X \sim U(a, b)$ (expressed that $X$ follows a uniform distribution on the interval $a$ to $b$),
the probability is evenly spread across the whole width of the box. The width is $(b - a)$, so the height of the probability box (the PDF) is
strictly $f(x) = \frac{1}{b-a}$.

Substituting that $f(x)$ into the integral, we get:

$$E[X] = \int_{a}^{b} x \left( \frac{1}{b-a} \right) dx$$

Let's move to calculating the variance. Variance literally means: *"On average, how far away is each data point from the center (the mean)?"*

Let's define our terms:

* Let $X$ be our data point.
* Let $\mu$ be our mean (which is just $E[X]$).

To find the distance, we do $(X - \mu)$. But distances can be negative. If we just averaged them, the negative and positive distances would
cancel out to zero. So, statisticians square the distance to force everything to be positive: $(X - \mu)^2$.

Therefore, the literal definition of variance is the **Expected Value of the squared distances**:

$$
\sigma^2 = E[(X - \mu)^2] \tag{2}
$$

**The Algebraic Derivation (The Shortcut)**

Now, let's expand that polynomial inside the brackets using standard algebra $(a - b)^2 = a^2 - 2ab + b^2$:

$$\sigma^2 = E[X^2 - 2X\mu + \mu^2]$$

The Expected Value operator $E[\dots]$ is linear, which means we can distribute it to each piece inside the brackets separately:

$$\sigma^2 = E[X^2] - E[2X\mu] + E[\mu^2]$$

Now we apply two fundamental rules of Expected Values:

1. **Constants pass through:** The mean ($\mu$) is just a static number (like $0.0$). The number $2$ is also a constant. We can pull constants
2. outside the $E[\dots]$. So, $E[2X\mu]$ becomes $2\mu E[X]$.
3. **The Expected Value of a constant is just the constant:** What is the average of the number 5? It's just 5. So, $E[\mu^2]$ is literally just $\mu^2$.

Let's apply those rules to our distributed formula:

$$\sigma^2 = E[X^2] - 2\mu E[X] + \mu^2$$

**The Final Collapse**

As we can remember, $\mu$ is literally just $E[X]$. Let's replace the $\mu$'s with $E[X]$ to see what happens:

$$\sigma^2 = E[X^2] - 2E[X]E[X] + (E[X])^2$$

Combining those middle terms, we get:

$$\sigma^2 = E[X^2] - 2(E[X])^2 + (E[X])^2$$

$$
\sigma^2 = E[X^2] - (E[X])^2 \tag{3}
$$

As we have the formula now, we can move to calculate the original value of the variance of a uniform distribution. To get this, we need to find
those two pieces: the mean $E[X]$, and the expected square $E[X^2]$. Remember, we had this?

$$E[X] = \int_{a}^{b} x \left( \frac{1}{b-a} \right) dx$$

When we integrate $x$, it becomes $\frac{x^2}{2}$:

$$E[X] = \frac{1}{b-a} \left[ \frac{x^2}{2} \right]_{a}^{b}$$

$$E[X] = \frac{b^2 - a^2}{2(b-a)}$$

Using the difference of squares rule ($b^2 - a^2 = (b-a)(b+a)$), we cancel out the $(b-a)$:

$$
E[X] = \frac{a+b}{2} \tag{4}
$$

*(This makes perfect intuitive sense, that the mean of a uniform box is just the exact middle point between $a$ and $b$.)*

Now we do the exact same thing to get the latter part, but we integrate $x^2$:

$$E[X^2] = \int_{a}^{b} x^2 \left( \frac{1}{b-a} \right) dx$$

The integral of $x^2$ is $\frac{x^3}{3}$:

$$E[X^2] = \frac{1}{b-a} \left[ \frac{x^3}{3} \right]_{a}^{b}$$

$$E[X^2] = \frac{b^3 - a^3}{3(b-a)}$$

Using the difference of cubes rule ($b^3 - a^3 = (b-a)(b^2 + ab + a^2)$), we cancel out the $(b-a)$ again:

$$E[X^2] = \frac{b^2 + ab + a^2}{3}$$

Now we plug our two pieces back into the master Variance formula from earlier:

$$\sigma^2 = E[X^2] - (E[X])^2$$

$$\sigma^2 = \frac{b^2 + ab + a^2}{3} - \left( \frac{a+b}{2} \right)^2$$

Square the mean term on the right:

$$\sigma^2 = \frac{b^2 + ab + a^2}{3} - \frac{a^2 + 2ab + b^2}{4}$$

Here is exactly where the magic happens. To subtract these two fractions, we need a common denominator for **3** and **4**. The lowest common
denominator is **12**.

Multiply the left fraction by $\frac{4}{4}$ and the right fraction by $\frac{3}{3}$:

$$\sigma^2 = \frac{4(b^2 + ab + a^2)}{12} - \frac{3(a^2 + 2ab + b^2)}{12}$$

$$\sigma^2 = \frac{4b^2 + 4ab + 4a^2 - 3a^2 - 6ab - 3b^2}{12}$$

$$\sigma^2 = \frac{b^2 - 2ab + a^2}{12}$$

The numerator here is a perfect square!

$$b^2 - 2ab + a^2 = (b-a)^2$$

And substituting the perfect square back in, we get our clean and general formula:

$$
\sigma^2 = \frac{(b-a)^2}{12} \tag{5}
$$

The standard deviation for a uniform distribution of data points will be simply the square root of its variance, which is:

$$\sigma = \sqrt{\frac{(b-a)^2}{12}} = \frac{b-a}{\sqrt{12}}$$

As we got all three parameters for a uniform distribution of data, what are those for a normal distribution? To make a fair comparison, let's
look at the **Standard Normal Distribution**, as this is the specific, baseline version of the bell curve we use in statistics. Simply, its
parameters will be the following:

* **Mean** = 0
* **Variance** = 1
* **Standard Deviation** = 1

They can also be derived mathematically. But the discussion will run out of the scope of the current blog, and also require some college-level
understanding of mathematics from my perspective. So, we can discuss them anytime later. For now, below is the side-by-side comparison between
what we learned so far:

| Parameter | Uniform Distribution $U(a, b)$ | Standard Normal Distribution $\mathcal{N}(0, 1)$ |
| --- | :---: | :---: |
| **Mean ($\mu$)** | $\frac{a+b}{2}$ | $0$ |
| **Variance ($\sigma^2$)** | $\frac{(b-a)^2}{12}$ | $1$ |
| **Standard Deviation ($\sigma$)** | $\frac{b-a}{\sqrt{12}}$ | $1$ |

### What's Next?

So far, we have only looked at these distributions sitting perfectly still in isolation. We've mapped their shapes and found their centers.
But in the real world, and especially when building systems or training algorithms, data doesn't sit still. Variables scale, shift, and collide
with one another.

So, the next question is, what happens to the mean and variance if we multiply our perfectly flat uniform distribution by a scalar? What if we
add two completely independent distributions together? In the next part of this blog, we will step out of isolation and explore the mathematics of
shifting, scaling, and colliding distributions.
