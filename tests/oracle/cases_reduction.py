"""Oracle cases — reductions (graph Ops: ReduceSum, ReduceMax; reduce_mean is a
composite). Cover each axis and full reduction.
"""
import jax.numpy as jnp
from oracle_util import randf, f

CASES = {
    "reduce_sum_ax0__2x3":  dict(category="reduction",
                                 inputs=[f(randf("rsum", 0, 2, 3))], fn=lambda a: jnp.sum(a, axis=0)),
    "reduce_sum_ax1__2x3":  dict(category="reduction",
                                 inputs=[f(randf("rsum", 1, 2, 3))], fn=lambda a: jnp.sum(a, axis=1)),
    "reduce_sum_all__2x3":  dict(category="reduction",
                                 inputs=[f(randf("rsum", 2, 2, 3))], fn=lambda a: jnp.sum(a)),
    "reduce_max_ax1__2x3":  dict(category="reduction",
                                 inputs=[f(randf("rmax", 0, 2, 3))], fn=lambda a: jnp.max(a, axis=1)),
    "reduce_mean_ax1__2x3": dict(category="reduction",
                                 inputs=[f(randf("rmean", 0, 2, 3))], fn=lambda a: jnp.mean(a, axis=1)),

    # keepdims
    "reduce_sum_keepdims_ax1__2x3": dict(category="reduction",
                                         inputs=[f(randf("rsum_kd", 0, 2, 3))],
                                         fn=lambda a: jnp.sum(a, axis=1, keepdims=True)),

    # rank-3, per-axis
    "reduce_sum_r3_ax0__2x3x4": dict(category="reduction",
                                     inputs=[f(randf("rsum3", 0, 2, 3, 4))],
                                     fn=lambda a: jnp.sum(a, axis=0)),
    "reduce_sum_r3_ax1__2x3x4": dict(category="reduction",
                                     inputs=[f(randf("rsum3", 1, 2, 3, 4))],
                                     fn=lambda a: jnp.sum(a, axis=1)),
    "reduce_sum_r3_ax2__2x3x4": dict(category="reduction",
                                     inputs=[f(randf("rsum3", 2, 2, 3, 4))],
                                     fn=lambda a: jnp.sum(a, axis=2)),

    # multi-axis
    "reduce_sum_r3_ax02__2x3x4": dict(category="reduction",
                                      inputs=[f(randf("rsum3", 3, 2, 3, 4))],
                                      fn=lambda a: jnp.sum(a, axis=(0, 2))),

    # reduce_max over axis 0 of [3,5]
    "reduce_max_ax0__3x5": dict(category="reduction",
                                inputs=[f(randf("rmax3", 0, 3, 5))],
                                fn=lambda a: jnp.max(a, axis=0)),

    # reduce_mean rank-3 axis 2
    "reduce_mean_r3_ax2__2x3x4": dict(category="reduction",
                                      inputs=[f(randf("rmean3", 0, 2, 3, 4))],
                                      fn=lambda a: jnp.mean(a, axis=2)),
}
