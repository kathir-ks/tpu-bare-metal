"""Shared helpers for the per-family oracle case modules (cases_*.py).

Determinism is keyed by NAME, not call order, so multiple authors (or parallel
agents) can add cases to different family files without perturbing each other's
draws. randf(key, idx, *shape) is byte-identical across regenerations as long as
the (key, idx, shape) triple is unchanged.
"""
import hashlib
import numpy as np
import jax.numpy as jnp


def _seed(key, idx, shape):
    h = hashlib.sha256(f"{key}|{idx}|{tuple(int(d) for d in shape)}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def randf(key, idx, *shape):
    """Deterministic standard-normal f32 array, identified by (key, idx, shape)."""
    g = np.random.default_rng(_seed(key, idx, shape))
    return g.standard_normal(shape).astype(np.float32)


def randpos(key, idx, *shape, lo=0.5):
    """Deterministic positive f32 array (for log/sqrt/rsqrt domains)."""
    return (np.abs(randf(key, idx, *shape)) + lo).astype(np.float32)


def randint(key, idx, shape, high):
    """Deterministic int32 array in [0, high) (for gather indices etc.)."""
    g = np.random.default_rng(_seed(key, idx, shape) ^ 0x9E3779B9)
    return g.integers(0, high, size=shape, dtype=np.int32)


def f(arr):
    """Tag a float32 input."""
    return ("f32", np.asarray(arr, dtype=np.float32))


def i(arr):
    """Tag an int32 input."""
    return ("i32", np.asarray(arr, dtype=np.int32))


def grad_of_sum(fn, inputs, float_idx):
    """d sum(fn(*inputs)) / d inputs[k] for each k in float_idx (oracle gradient)."""
    import jax

    def scalar(*fargs):
        full = list(inputs)
        for j, k in enumerate(float_idx):
            full[k] = fargs[j]
        return jnp.sum(fn(*full))

    fl = [inputs[k] for k in float_idx]
    grads = jax.grad(scalar, argnums=tuple(range(len(fl))))(*fl)
    if not isinstance(grads, tuple):
        grads = (grads,)
    return grads
