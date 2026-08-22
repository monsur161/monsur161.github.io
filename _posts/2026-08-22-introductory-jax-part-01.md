---
layout: post
title: "Introductory JAX - Part 01: Tracers, Pure Functions, and JIT"
short_title: "Introductory JAX - Part 01"
date: 2026-08-22
comments: true
categories: [llm]
---

> *"Where ignorance is bliss, 'tis folly to be wise."*
> — **Thomas Gray**, *Ode on a Distant Prospect of Eton College*

The scripts written from here onward will use Python. And since we'll be using raw JAX throughout our bare-level understanding of the LLM, it's important to understand the basics of its operation. Here's the [dev's guide to installing JAX](https://docs.jax.dev/en/latest/installation.html).

After installation, we can simply incorporate JAX in our script like:

```python
import jax
import jax.numpy as jnp

```

One thing to keep in mind before we jump in: JAX is purely functional. In JAX, we pass in *arguments* as the *parameters* of a function, and catch the *result* as the *output* given by that function. This is how JAX was designed to be. Now that we know it, the first thing we need to learn is the *Just In Time* (JIT) compilation of a JAX Python function.

What is JIT? Before answering that, I think we should recall the basic idea of an *interpreter* and a *compiler*, where:

* An interpreter executes a source code by reading it line-by-line, and,
* A compiler builds an **executable** binary file from the source code by translating it all at once.

So, while using an interpreter, we have to perform the `read-convert-execute-continue` step every time we run the script. Using a compiler saves us from this repetition by compiling the whole script at once in the binary file, unless the script gets changed somewhere later. JIT compilation, often described as sitting somewhere between an interpreter and a compiler, tries to get the best of both. How? When we make a function JIT-compiled, it performs the following steps (simply stated):

* An *interpreter* executes the script line-by-line.
* A *profiler* monitors the actions from the background.
* Any repetitive action, also referred to as a **hot spot**, is identified by the profiler.
* This action is then **optimized** and **compiled** to **machine code** using a *compiler*.
* This compiled code is then **cached**.
* Every time this repetitive execution arrives, the **optimized**, **compiled** and **cached** code is called upon instead of interpreting the **raw** and **uncompiled source code**.

One thing to remember here: the optimization done here is based on the current state of the program, and the cached code is utilized as long as this state holds **True**. As soon as the current state changes at any instance in the future, the execution at that stage returns to the slowed interpretation, which is referred to as **deoptimization**.

JAX provides us with this JIT-compilation facility. That's the main reason we should prefer JAX. But how does JAX work when a function is JIT-compiled versus when it is not?

Let's define a function:

```python
def f(x):
    return jnp.sin(x**10 + jnp.log(2.0))
```

The first thing JAX does when the function is JIT-compiled is to transform `f(x)` into an internal intermediate representation (IR) called **jaxpr**. What is it? It's simply a sequence of primitive operations, with only one unit of computation at a time. There's more to know about jaxpr from the [dev's guide to understanding jaxpr](https://docs.jax.dev/en/latest/jaxpr.html#jax-internals-jaxpr). But knowing when to stop is also wisdom; hence, only what we need is discussed in this log.

Even if `f(x)` wasn't JIT-compiled, we could still visualize what happens when we transform the JIT-compiled version into *jaxpr* using the `make_jaxpr` method. Let's define some input to `f` and get the *jaxpr*:

```python
x_in = jnp.array([1.0, 3.0])
print(jax.make_jaxpr(f)(x_in))
```

Every jaxpr follows a strict functional structure:
`{ lambda [consts] ; [inputs]. let [equations] in [outputs] }`

And ours builds the following:

```text
{ lambda ; a:f32[2]. let
    b:f32[2] = integer_pow[y=10] a
    c:f32[2] = log a
    d:f32[2] = add b c
    e:f32[2] = sin d
  in (e,) }
```

It interprets the sequence of operations to be performed to execute `f(x)` for the input `x_in`. It:

* states that there's no constant;
* identifies that `x_in` is the single input vector `a` of *shape = (2,)* with each element being *float32*;
* raises the exponent of `a` to 10 and calls it *float32* `b` of *shape = (2,)*;
* performs $e$-based logarithm of `a` and calls it *float32* `c` of *shape = (2,)*;
* adds `b` to `c` and calls it *float32* `d` of *shape = (2,)*;
* takes the sine of `d` and calls it *float32* `e` of *shape = (2,)*;
* and then returns `(e,)`, which is always a `tuple` in *jaxpr*.

This whole thing doesn't involve a concrete **value**, and this process is called **Tracing**. So, what is tracing? It's just going through the source code with a **Tracer**, having *no value* but only *shape* and *data type*, to transform it to *jaxpr*. Though we did it above for a function that's not JIT-compiled, it is the absolute first step when a function *is* JIT-compiled.

If it's that easy, then why not try another function? Let's say we want to write a simple one-liner to find the absolute value of the input, and then observe how the *jaxpr* translates this to its IR form.

```python
def abs_val(x):
    return x if x > 0 else -x
print(jax.make_jaxpr(abs_val)(-3.0))
```

Executing this will give us a list of errors. Now that we've confirmed we have an error, let's modify it to catch the error cleanly:

```python
from jax.errors import TracerBoolConversionError, ConcretizationTypeError

def abs_val(x):
    return x if x > 0 else -x

try:
    print(jax.make_jaxpr(abs_val)(3.))
except TracerBoolConversionError as e:
    print(e)
```

It outputs the following:

```text
Attempted boolean conversion of traced array with shape bool[].
The error occurred while tracing the function abs_val at /tmp/ipykernel_1277/1401350888.py:3 for jit. This concrete value was not available in Python because it depends on the value of the argument x.
See [https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError](https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError)
```

Using `help(TracerBoolConversionError)` reveals that *This error occurs when a traced value in JAX is used in a context where a boolean value is expected*. And `help(ConcretizationTypeError)` shows us that *This error occurs when a JAX Tracer object is used in a context where a concrete value is required*.

Essentially, the `TracerBoolConversionError` is a subclass of `ConcretizationTypeError`. What happens during the transformation to *jaxpr* is that a *tracer* (with only a *data type* and *shape*) tries to trace how it travels through `abs_val`. It doesn't understand what to do when it meets the `if` statement. Why? Because it has no concrete value, so it can't confirm which branch of the `if` statement to execute, and it throws an error.

This highlights the primary requirement when we JIT-compile any function: it must be a **pure function**. The [sharp bits of JAX](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions) provides an excellent section on pure functions.

Before even JIT-compiling a function, we need to ensure that the function is *pure* and *JIT-friendly*. How? Let's return to `abs_val`. We could have written it like below:

```python
def abs_val(x):
    return jnp.where(x > 0, x, -x)
print(jax.make_jaxpr(abs_val)(3.0))

```

And it gives the following:

```text
{ lambda ; a:f32[]. let
    b:bool[] = gt a 0.0:f32[]
    c:f32[] = neg a
    d:f32[] = jit[
      name=_where
      jaxpr={ lambda ; b:bool[] a:f32[] c:f32[]. let
          d:f32[] = select_n b c a
        in (d,) }
    ] b a c
  in (d,) }

```

This worked well, and it worked without raising an error while tracing. What is it tracing? It:

* identifies the input as a scalar (denoted by `[]`) `a` of *float32*;
* checks if `a` is greater than (by `gt`) *float32* 0.0 and denotes the result as *bool* `b`;
* negates `a` and denotes it as *float32* `c`;
* initiates `d` as a *float32* with `b`, `a`, and `c` as its arguments, holding a sub-graph with a condition saying:
    * if `b` is `False`, select `c`;
    * if `b` is `True`, select `a`;
    * denote it as `d` and return;
* and returns a tuple `(d,)`.

The `jit` call has appeared inside the `jaxpr` here. Why? Because in JAX, we only form a `jaxpr` if we make a function *JIT-compiled*. When it was written in plain Python using `if-else`, the execution was lazy. When the first condition is satisfied, its internals are interpreted; when it's not, the alternate block is interpreted. But JAX needs to trace *both* branches to generate a complete *jaxpr* graph. That's why we had to rewrite it using the *JIT-friendly* `jnp.where`, which safely evaluates both sides.

What if this *JIT-friendly* version didn't exist? We have another choice:

```python
def abs_val(x):
    return x if x > 0 else -x
print(jax.make_jaxpr(abs_val, static_argnums=(0,))(3.0))
```

And it gives us:

```text
{ lambda ; . let  in (3.0:f32[],) }
```

How do we read this *jaxpr*?

* `lambda ; .` $\rightarrow$ The dot after the semicolon means there are 0 dynamic inputs to this compute graph.
* `let  in` $\rightarrow$ The `let` block is completely empty. No mathematical operations, no `select_n` or `gt`.
* `(3.0:f32[],)` $\rightarrow$ The graph is hardcoded to return the *float32* static scalar 3.0.

So what did we do? We added `static_argnums=(0,)` as the argument to `jax.make_jaxpr`. This tells the tracer *not* to treat the *0-th* argument (`x`) as dynamic during tracing, but rather as a *float32* static constant of value 3.0. Inside the function, the `if` statement successfully evaluates as `True` and the value is returned.

Does this solve the problem of ignorance? **NO**. As long as we call it with the same 3.0, it will be fine. But calling it with some other value, say -3.0, will cause JAX to **recompile a brand new graph** from scratch, returning a *float32* static constant of value 3.0. If we pass 3.0 again later, yet another graph must be compiled, abandoning the previous one. This destroys the entire purpose of caching.

Additionally, using the `static_argnums` argument inside `jax.make_jaxpr` only helps us visualize the trace. It has nothing to do with the actual purity of the main function. The underlying function is still fundamentally impure.

So, to properly JIT-compile a function, we must make it *pure* and *JIT-friendly*. There are three common ways to apply it in Python.

### 1. Using `jax.jit` as a Decorator

We can use `jax.jit` directly as a Python decorator.

```python
# JIT-friendly function
@jax.jit                
def abs_val(x):
    return jnp.where(x > 0, x, -x)
print(jax.make_jaxpr(abs_val)(3.0))

# Lazy function (recompiles on new inputs)
@jax.jit(static_argnames=['x',])
def abs_val(x):
    return x if x > 0 else -x
```

### 2. Using `jax.jit` as a Wrapper

Alternatively, we can write the function normally and pass it through a Python wrapper.

```python
# JIT-compiled function
def abs_val(x):
    return jnp.where(x > 0, x, -x)
abs_val_jit = jax.jit(abs_val)              
print(jax.make_jaxpr(abs_val_jit)(3.0))

# Lazy function
def abs_val(x):
    return x if x > 0 else -x
abs_val_jit = jax.jit(abs_val, static_argnames=['x',])
```

*(Note: Both the decorator and the wrapper accept `static_argnums` or `static_argnames` when dealing with static arguments, which can be passed as a tuple or list).*

### 3. Importing `partial` from `functools`

We can import `partial` from Python's standard `functools` library and use it as a decorator with `jax.jit`.

```python
from functools import partial

# JIT-friendly function
@partial(jax.jit)
def abs_val(x):
    return jnp.where(x > 0, x, -x)
print(jax.make_jaxpr(abs_val)(3.0))

# Lazy function
@partial(jax.jit, static_argnames=['x',])
def abs_val(x):
    return x if x > 0 else -x
```

We intentionally skipped the `jax.make_jaxpr` print statement for all the lazy functions above. Why? Because if we had tried it, it would throw an error. Again, why? Because when we call `jax.make_jaxpr`, a *tracer* is passed in to build the graph. But the lazy function is decorated to identify `x` as a static argument, meaning it expects a concrete, static Python **constant**. Handing it a tracer instead results in an immediate crash.

With this foundation, we now understand:

* The pure functionality of JIT
* JIT compilation basics
* JIT compilation in JAX

*Tracing* is merely the first step before the execution of a JIT-compiled function in JAX. What happens next will be discovered in the next log.
