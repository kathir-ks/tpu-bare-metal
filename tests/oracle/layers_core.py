"""Oracle layer cases — core paramful layers (nn.hpp): gelu, linear, rmsnorm.

General layer-case format (shared by all layers_*.py modules):
    key: dict(
        inputs = [ dict(name=..., shape=[...], int=False, high=None), ... ],
                   # ordered forward inputs; int=True => int32 (e.g. ids/targets),
                   # high = exclusive upper bound for int draws
        params = { "name": [shape], ... },   # names MUST match the c.param(...) names
                                             #   the C++ builder requests, in nn.hpp
        fn     = lambda ins, p: out,         # ins: name->jnp array ; p: name->jnp array
        scalar_out = False,                  # True if fn returns a scalar (loss)
    )
The fn here must mirror the nn.hpp formula EXACTLY (same eps, same constants, same axes).
Fixtures store each input + param by name, out, grad_<name> for float inputs, grad_<param>.
"""
import jax
import jax.numpy as jnp

SQRT_2_OVER_PI = 0.7978845608


def gelu(x):
    x3 = x * x * x
    inner = (x + 0.044715 * x3) * SQRT_2_OVER_PI
    return 0.5 * x * (1.0 + jnp.tanh(inner))


def rmsnorm(x, scale, eps=1e-5):
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(ms + eps)) * scale


LAYERS = {
    "gelu__4x8": dict(
        inputs=[dict(name="x", shape=[4, 8])],
        params={},
        fn=lambda ins, p: gelu(ins["x"])),

    "linear__4x6_8": dict(
        inputs=[dict(name="x", shape=[4, 6])],
        params={"fc.w": [6, 8], "fc.b": [8]},
        fn=lambda ins, p: ins["x"] @ p["fc.w"] + p["fc.b"]),

    "rmsnorm__4x8": dict(
        inputs=[dict(name="x", shape=[4, 8])],
        params={"n": [8]},
        fn=lambda ins, p: rmsnorm(ins["x"], p["n"])),
}
