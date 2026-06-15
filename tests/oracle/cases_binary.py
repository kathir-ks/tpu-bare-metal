"""Oracle cases — binary elementwise ops with numpy broadcasting (graph Ops: Add,
Sub, Mul, Div, Max, Min). Compare/Select live in cases_misc.py.
"""
import jax.numpy as jnp
from oracle_util import randf, randpos, f

CASES = {
    "add__2x3":         dict(category="elementwise",
                             inputs=[f(randf("add", 0, 2, 3)), f(randf("add", 1, 2, 3))],
                             fn=lambda a, b: a + b),
    "mul__2x3":         dict(category="elementwise",
                             inputs=[f(randf("mul", 0, 2, 3)), f(randf("mul", 1, 2, 3))],
                             fn=lambda a, b: a * b),
    "sub__2x3":         dict(category="elementwise",
                             inputs=[f(randf("sub", 0, 2, 3)), f(randf("sub", 1, 2, 3))],
                             fn=lambda a, b: a - b),
    "div__2x3":         dict(category="elementwise",
                             inputs=[f(randf("div", 0, 2, 3)), f(randpos("div", 1, 2, 3))],
                             fn=lambda a, b: a / b),
    "max__2x3":         dict(category="elementwise",
                             inputs=[f(randf("max", 0, 2, 3)), f(randf("max", 1, 2, 3))],
                             fn=lambda a, b: jnp.maximum(a, b)),
    "min__2x3":         dict(category="elementwise",
                             inputs=[f(randf("min", 0, 2, 3)), f(randf("min", 1, 2, 3))],
                             fn=lambda a, b: jnp.minimum(a, b)),
    "add_bcast__2x3_3": dict(category="elementwise",
                             inputs=[f(randf("addb", 0, 2, 3)), f(randf("addb", 1, 3))],
                             fn=lambda a, b: a + b),
    # scalar-ish broadcast: [2,3] + [1]
    "add_bcast__2x3_1": dict(category="elementwise",
                             inputs=[f(randf("addb1", 0, 2, 3)), f(randf("addb1", 1, 1))],
                             fn=lambda a, b: a + b),
    # broadcast along row: [2,3] * [3]
    "mul_bcast__2x3_3": dict(category="elementwise",
                             inputs=[f(randf("mulb3", 0, 2, 3)), f(randf("mulb3", 1, 3))],
                             fn=lambda a, b: a * b),
    # mutual broadcast: [1,3] + [2,1] -> [2,3]
    "add_mutual__1x3_2x1": dict(category="elementwise",
                                inputs=[f(randf("addmut", 0, 1, 3)), f(randf("addmut", 1, 2, 1))],
                                fn=lambda a, b: a + b),
    # rank-3: [2,2,3] * [2,2,3]
    "mul_rank3__2x2x3": dict(category="elementwise",
                             inputs=[f(randf("mulr3", 0, 2, 2, 3)), f(randf("mulr3", 1, 2, 2, 3))],
                             fn=lambda a, b: a * b),
    # non-128-aligned minor dim: [2,130] + [2,130]
    "add_nonaligned__2x130": dict(category="elementwise",
                                  inputs=[f(randf("addna", 0, 2, 130)), f(randf("addna", 1, 2, 130))],
                                  fn=lambda a, b: a + b),
    # div with randpos denominator (avoids divide-by-zero)
    "div_pos__2x3": dict(category="elementwise",
                         inputs=[f(randf("divp", 0, 2, 3)), f(randpos("divp", 1, 2, 3))],
                         fn=lambda a, b: a / b),
}
