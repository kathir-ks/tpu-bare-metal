"""Oracle layer cases — nn::softmax (max-subtract, paramless). Mirrors layer_table_softmax.hpp.

nn::softmax(g, x, axis) computes:
    m = stop_gradient(reduce_max(x, axis, keepdims=True))
    e = exp(x - m)
    s = reduce_sum(e, axis, keepdims=True)
    return e / s

This is numerically identical to jax.nn.softmax(x, axis=axis).
No learned parameters.
"""
import jax.numpy as jnp

LAYERS = {
    "softmax__4x5_ax1": dict(
        inputs=[dict(name="x", shape=[4, 5])],
        params={},
        fn=lambda ins, p: jnp.exp(ins["x"] - jnp.max(ins["x"], axis=1, keepdims=True))
                          / jnp.sum(jnp.exp(ins["x"] - jnp.max(ins["x"], axis=1, keepdims=True)), axis=1, keepdims=True),
    ),

    "softmax__3x4x5_ax2": dict(
        inputs=[dict(name="x", shape=[3, 4, 5])],
        params={},
        fn=lambda ins, p: jnp.exp(ins["x"] - jnp.max(ins["x"], axis=2, keepdims=True))
                          / jnp.sum(jnp.exp(ins["x"] - jnp.max(ins["x"], axis=2, keepdims=True)), axis=2, keepdims=True),
    ),
}
