"""Oracle cases — shape ops (graph Ops: Transpose, Reshape, Broadcast). Pure data
movement: tolerance is 0 (must be exact). Seeded with one example per op; extend
with more ranks/perms.
"""
import jax.numpy as jnp
from oracle_util import randf, f

CASES = {
    # --- existing cases ---
    "transpose__2x3":     dict(category="shape",
                               inputs=[f(randf("tr", 0, 2, 3))], fn=lambda a: jnp.transpose(a, (1, 0))),
    "reshape__2x3_to_6":  dict(category="shape",
                               inputs=[f(randf("rs", 0, 2, 3))], fn=lambda a: jnp.reshape(a, (6,))),
    "broadcast__3_to_2x3": dict(category="shape",
                               inputs=[f(randf("bc", 0, 3))],
                               fn=lambda a: jnp.broadcast_to(a, (2, 3))),

    # --- new cases ---

    # rank-3 transpose [2,3,4] with perm (2,0,1) -> [4,2,3]
    "transpose__2x3x4_perm201": dict(category="shape",
                               inputs=[f(randf("tr3", 0, 2, 3, 4))],
                               fn=lambda a: jnp.transpose(a, (2, 0, 1))),

    # transpose_last2 of [2,3,4] (swap last two dims -> [2,4,3])
    "transpose_last2__2x3x4": dict(category="shape",
                               inputs=[f(randf("trl2", 0, 2, 3, 4))],
                               fn=lambda a: jnp.transpose(a, (0, 2, 1))),

    # reshape [2,3,4] -> [6,4]
    "reshape__2x3x4_to_6x4": dict(category="shape",
                               inputs=[f(randf("rs3a", 0, 2, 3, 4))],
                               fn=lambda a: jnp.reshape(a, (6, 4))),

    # reshape [2,3,4] -> [24]
    "reshape__2x3x4_to_24": dict(category="shape",
                               inputs=[f(randf("rs3b", 0, 2, 3, 4))],
                               fn=lambda a: jnp.reshape(a, (24,))),

    # broadcast [1,4] -> [3,4]
    "broadcast__1x4_to_3x4": dict(category="shape",
                               inputs=[f(randf("bc1x4", 0, 1, 4))],
                               fn=lambda a: jnp.broadcast_to(a, (3, 4))),

    # broadcast scalar-row [3] -> [2,3]
    "broadcast__3_to_2x3_row": dict(category="shape",
                               inputs=[f(randf("bcrow", 0, 3))],
                               fn=lambda a: jnp.broadcast_to(a, (2, 3))),

    # non-128-aligned: reshape [2,130] -> [260]
    "reshape__2x130_to_260": dict(category="shape",
                               inputs=[f(randf("rs130", 0, 2, 130))],
                               fn=lambda a: jnp.reshape(a, (260,))),

    # slice [4,4] -> [2,3] : rows 1:3, cols 0:3 (stride 1)
    "slice__4x4_to_2x3": dict(category="shape",
                               inputs=[f(randf("sl", 0, 4, 4))],
                               fn=lambda a: a[1:3, 0:3]),

    # pad [2,3] -> [4,5] : low=(1,0) high=(1,2), zeros
    "pad__2x3_to_4x5": dict(category="shape",
                               inputs=[f(randf("pd", 0, 2, 3))],
                               fn=lambda a: jnp.pad(a, ((1, 1), (0, 2)))),

    # concat along axis 0: [2,3] + [2,3] -> [4,3]
    "concat_ax0__2x3_2x3": dict(category="shape",
                               inputs=[f(randf("cc0", 0, 2, 3)), f(randf("cc0", 1, 2, 3))],
                               fn=lambda a, b: jnp.concatenate([a, b], axis=0)),
    # concat along axis 1 with unequal sizes: [2,3] + [2,2] -> [2,5]
    "concat_ax1__2x3_2x2": dict(category="shape",
                               inputs=[f(randf("cc1", 0, 2, 3)), f(randf("cc1", 1, 2, 2))],
                               fn=lambda a, b: jnp.concatenate([a, b], axis=1)),
}
