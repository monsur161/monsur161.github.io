---
layout: post
title: "The Mechanics of Transformation: Scaling, Shifting, and Colliding Distributions"
date: 2026-08-17
comments: true
categories: [math]
---

> *"If you expect disappointment, then you can never really be disappointed."*
> - **MJ, Spider-Man: No Way Home**

### Scaling a Distribution (Multiplication by a Scalar)

Let's head to some of the properties we should know about each of them. We'll consider the uniform distribution first. For a uniform distribution
of data between $[a,b]$, we have mean = $\frac{a+b}{2}$ and variance = $\frac{(b-a)^2}{12}$. What happens when we multiply this distribution by
a scalar, say $c$? And what happens if it's not a scalar, rather another uniform distribution, say $[x,y]$? How do the mean and variance of the
product vary each time?

Let the initial random variable be $X \sim U(a, b)$ (interpreted as X follows a uniform distribution on the interval a to b). If we multiply
this distribution by a constant scalar $c$, we create a new random variable $Z = cX$.

The expectation operator is linear, meaning constants can be pulled out directly:

$$
\mathbb{E}[cX] = c\mathbb{E}[X]
$$

$$
\mu_{Z} = c \left( \frac{a+b}{2} \right)
$$

Variance measures squared deviation from the mean, so the scaling factor must be squared:

$$
\text{Var}(cX) = c^2\text{Var}(X)
$$

$$
\sigma^2_{Z} = c^2 \left( \frac{(b-a)^2}{12} \right)
$$

The new distribution $Z$ remains perfectly uniform. If $c > 0$, it is simply scaled and shifted to a new domain of $[ca, cb]$.

In short, multiplying a random variable by a constant scales the expected value linearly, but it scales the variance quadratically.

<img src="/assets/images/fig1-uniform-scale-light.png" class="img-light" alt="Scalar Multiplication of a Uniform Distribution">
<img src="/assets/images/fig1-uniform-scale-dark.png" class="img-dark" alt="Scalar Multiplication of a Uniform Distribution">

### The Collision (Product of Independent Variables)

Now comes the second question. Let $X \sim U(a, b)$ and let the second distribution be $Y \sim U(x, y)$. To determine the properties of the
product $Z = XY$, we must assume that $X$ and $Y$ are **independent** random variables. Because this is a product of two random variables,
the transformations are no longer strictly linear, and the properties require expanding the expectation operators.

For independent random variables, the expected value of their product is simply the product of their individual expected values:

$$
\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]
$$

$$
\mu_{Z} = \left( \frac{a+b}{2} \right) \left( \frac{x+y}{2} \right)
$$

The variance of the product of two independent random variables is governed by a specific identity. It is not simply the product of their variances.

The definition of variance for $Z$ is derived directly from our expanded variance formula (Equation 3 from Part 1):

$$
\text{Var}(XY) = \mathbb{E}[(XY)^2] - (\mathbb{E}[XY])^2
$$

Because $X$ and $Y$ are independent, we can split the first term:

$$
\text{Var}(XY) = \mathbb{E}[X^2]\mathbb{E}[Y^2] - (\mathbb{E}[X]\mathbb{E}[Y])^2
$$

Next, we take the standard variance identity derived in Equation 3, $\text{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$, and rearrange it to
isolate the second moment ($\mathbb{E}[X^2]$). We do this for both $X$ and $Y$:

$$
\mathbb{E}[X^2] = \text{Var}(X) + \mathbb{E}[X]^2
$$

$$
\mathbb{E}[Y^2] = \text{Var}(Y) + \mathbb{E}[Y]^2
$$

Then we substitute these isolated second moments back into the first term of our original equation ($\mathbb{E}[X^2]\mathbb{E}[Y^2]$):

$$
\mathbb{E}[X^2]\mathbb{E}[Y^2] = \left( \text{Var}(X) + \mathbb{E}[X]^2 \right) \left( \text{Var}(Y) + \mathbb{E}[Y]^2 \right)
$$

After that, we expand this expression by multiplying the binomials:

$$
\mathbb{E}[X^2]\mathbb{E}[Y^2] = \text{Var}(X)\text{Var}(Y) + \text{Var}(X)\mathbb{E}[Y]^2 + \mathbb{E}[X]^2\text{Var}(Y) + \mathbb{E}[X]^2\mathbb{E}[Y]^2
$$

Finally, we substitute this fully expanded form back into the original variance equation:

$$
\text{Var}(XY) = \left[ \text{Var}(X)\text{Var}(Y) + \text{Var}(X)\mathbb{E}[Y]^2 + \text{Var}(Y)\mathbb{E}[X]^2 + \mathbb{E}[X]^2\mathbb{E}[Y]^2 \right] - \mathbb{E}[X]^2\mathbb{E}[Y]^2 \tag{6}
$$

$$
\text{Var}(XY) = \text{Var}(X)\text{Var}(Y) + \text{Var}(X)\mathbb{E}[Y]^2 + \text{Var}(Y)\mathbb{E}[X]^2
$$

Substituting the uniform mean (Equation 4) and uniform variance (Equation 5) derived in Part 1, the final variance of the product becomes:

$$
\sigma^2_{Z} = \left[ \frac{(b-a)^2}{12} \cdot \frac{(x-y)^2}{12} \right] + \left[ \frac{(b-a)^2}{12} \cdot \left(\frac{x+y}{2}\right)^2 \right] + \left[ \frac{(x-y)^2}{12} \cdot \left(\frac{a+b}{2}\right)^2 \right]
$$

Unlike multiplication by a scalar, the product of two uniform distributions **does not result in a uniform distribution**. We'll investigate it
later in another blog.

<img src="/assets/images/fig2-uniform-product-light.png" class="img-light" alt="Product of Two Uniform Distributions">
<img src="/assets/images/fig2-uniform-product-dark.png" class="img-dark" alt="Product of Two Uniform Distributions">

If we had asked the same question for a normal distribution of data instead of a uniform one, what would we have? Let's examine that as well.

When we start with a standard normal distribution (mean $\mu = 0$, variance $\sigma^2 = 1$), transformations behave somewhat elegantly due to
the zero-mean property. Let's define this as $X \sim \mathcal{N}(0, 1)$ (stands for $X$ follows a standard normal distribution with mean=0 and
variance=1)

When we multiply this distribution by a constant scalar $c$, we create a new random variable $Z = cX$.

Because expectation is a linear operator, the mean scales directly by the constant. Since the original mean is zero, the new mean remains
identically zero.

$$
\mathbb{E}[cX] = c\mathbb{E}[X]
$$

$$
\mu_Z = c \cdot 0 = 0
$$

Variance measures squared deviation, so the scalar must be squared.

$$
\text{Var}(cX) = c^2\text{Var}(X)
$$

$$
\sigma^2_Z = c^2 \cdot 1 = c^2
$$

A fundamental property of the normal distribution is that any linear transformation of it remains perfectly normal. The new distribution
is $Z \sim \mathcal{N}(0, c^2)$.

Let $Y \sim \mathcal{N}(\mu, \sigma^2)$ be a second, independent normal distribution with its own arbitrary mean and variance. The new random
variable is $Z = XY$.

Because $X$ and $Y$ are independent, the expectation of their product is the product of their expectations. Since $\mathbb{E}[X]$ is zero,
the entire mean collapses to zero, regardless of the mean of $Y$.

$$
\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]
$$

$$
\mu_Z = 0 \cdot \mu = 0
$$

We can put the values directly into the general variance identity derived previously:

$$
\text{Var}(XY) = \text{Var}(X)\text{Var}(Y) + \text{Var}(X)\mathbb{E}[Y]^2 + \text{Var}(Y)\mathbb{E}[X]^2
$$

Substituting $\text{Var}(X) = 1$ and $\mathbb{E}[X] = 0$, we get:

$$
\text{Var}(XY) = (1)(\sigma^2) + (1)(\mu)^2 + (\sigma^2)(0)^2
$$

$$
\sigma^2_Z = \sigma^2 + \mu^2
$$

Unlike multiplying by a scalar, the product of two normal distributions is **not normal**.

When multiplying two normal variables, the resulting probability density function (PDF) has a very sharp peak at zero and heavily elongated
"fat" tails. If both $X$ and $Y$ are standard normal, the resulting distribution of $Z = XY$ is known as a **Normal Product Distribution**.

<img src="/assets/images/fig4-normal-product-light.png" class="img-light" alt="Product of Two Normal Distributions">
<img src="/assets/images/fig4-normal-product-dark.png" class="img-dark" alt="Product of Two Normal Distributions">

### Shifting and Combining (Addition)

We have one more property to observe before we pause our investigation about uniform and normal distribution for now. What happens to the
mean and variance in case of addition between a uniform distribution and a scalar, and two uniform distribution, instead of multiplication?
And what happens when we do the same but replace the uniform distribution with normal ones?

Shifting from multiplication to addition makes the mathematical operations significantly more straightforward because the expectation operator
is perfectly linear, and the variance of independent variables is strictly additive.

Let the initial random variable be $X \sim U(a, b)$ as before. When we add a constant scalar $c$ to this distribution, we create a new random
variable $Z = X + c$.

Adding a scalar simply shifts the entire distribution along the number line without stretching or squishing it. Because expectation is linear,
adding a constant to a random variable adds that exact constant to the mean:

$$
\mathbb{E}[X + c] = \mathbb{E}[X] + c
$$

$$
\mu_Z = \frac{a+b}{2} + c
$$

Variance measures the spread of the data *around the mean*. Because adding a scalar shifts every single data point by exactly the same amount,
the spread of the data does not change at all. The variance of a constant is zero.

$$
\text{Var}(X + c) = \text{Var}(X) + \text{Var}(c) = \text{Var}(X) + 0
$$

$$
\sigma^2_Z = \frac{(b-a)^2}{12}
$$

The new distribution $Z$ remains perfectly uniform. It is simply shifted to a new domain of $[a+c, b+c]$.

Let the second independent distribution be $Y \sim U(x, y)$. The new random variable is $Z = X + Y$.

The expectation of the sum of two random variables is always the sum of their individual expectations. This holds true regardless of whether
the variables are independent.

$$
\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]
$$

$$
\mu_Z = \frac{a+b}{2} + \frac{x+y}{2} = \frac{a+b+x+y}{2}
$$

For *independent* random variables, the variance of their sum is simply the sum of their individual variances. (Unlike multiplication, there are
no cross-terms to compute).

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)
$$

$$
\sigma^2_Z = \frac{(b-a)^2}{12} + \frac{(y-x)^2}{12} = \frac{(b-a)^2 + (y-x)^2}{12}
$$

**The Distribution Shape:**
The sum of two independent uniform distributions is **never uniform**. Mathematically, the Probability Density Function (PDF) of the sum of two
independent variables is the convolution of their individual PDFs. Convolving two rectangular pulses (uniform distributions) creates a piecewise
linear function. This is something I learnt being a sophomore student during my undergrad, and surely out of the current scope of discussion of
this blog.

Let's head back to the same question we asked before, but this time for normal distribution of data. Keeping with the standard normal distribution
we used previously, $X \sim \mathcal{N}(0, 1)$, the behavior of the normal distribution under addition is also one of its most elegant mathematical
properties.

When we add a constant scalar $c$ to $X$, we create a new random variable $Z = X + c$.

Just like with the uniform distribution, adding a scalar simply shifts the entire distribution along the x-axis without altering its spread.

<img src="/assets/images/fig3-normal-shift-light.png" class="img-light" alt="Shifting a Normal Distribution">
<img src="/assets/images/fig3-normal-shift-dark.png" class="img-dark" alt="Shifting a Normal Distribution">

The expectation operator is linear, so the constant is simply added to the original mean.

$$
\mathbb{E}[X + c] = \mathbb{E}[X] + c
$$

$$
\mu_Z = 0 + c = c
$$

Because a constant has no variance and shifting the data does not change how spread out it is around its center, the variance is entirely unaffected.

$$
\text{Var}(X + c) = \text{Var}(X) + \text{Var}(c) = 1 + 0
$$

$$
\sigma^2_Z = 1
$$

Any linear transformation of a normal distribution results in a normal distribution. The new distribution is $Z \sim \mathcal{N}(c, 1)$. It looks
exactly like the standard normal curve, just shifted so its peak sits at $c$.

Now let $Y \sim \mathcal{N}(\mu, \sigma^2)$ be a second, independent normal distribution with its own arbitrary mean and variance. The new random
variable is $Z = X + Y$.

The expected value of the sum of two random variables is always the sum of their expected values.

$$
\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]
$$

$$
\mu_Z = 0 + \mu = \mu
$$

Because $X$ and $Y$ are independent, the variance of their sum is the sum of their variances.

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)
$$

$$
\sigma^2_Z = 1 + \sigma^2 \tag{7}
$$

The normal distribution is entirely unique in terms of its new shape. Unlike the uniform distribution (which transforms the original rectangular
shape after addition), **the sum of two independent normal distributions is always exactly another normal distribution.**

Mathematically, the probability density function of $Z$ is the convolution of the two original probability density functions. The convolution
of two Gaussian functions yields another perfect Gaussian function. Therefore, $Z \sim \mathcal{N}(\mu, 1 + \sigma^2)$.

This "stability" under addition is one of the foundational reasons the normal distribution appears everywhere in nature and statistics.

### But, Why Do We Even Need To Know This?

It is easy to look at these derivations as abstract statistics, but they form the invisible structural foundation of data transformations.
It might seem like a mathematical detour right now, but we'll return here once we reach the relevant design decisions in our later writings.
Only then will all of this theory truly click into place and make perfect sense.
