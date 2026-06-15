"""Oracle cases — matmul (graph Op: Dot, batched). Last two axes are the matrix;
leading axes are batch. DEFAULT tolerance is dominated by the bf16 contraction.
"""
from oracle_util import randf, f

CASES = {
    "dot__2x3_3x4":       dict(category="matmul",
                               inputs=[f(randf("dot", 0, 2, 3)), f(randf("dot", 1, 3, 4))],
                               fn=lambda a, b: a @ b),
    "dot_batched__2x2x3_2x3x4": dict(category="matmul",
                               inputs=[f(randf("dotb", 0, 2, 2, 3)), f(randf("dotb", 1, 2, 3, 4))],
                               fn=lambda a, b: a @ b),
    "dot_nonsq__4x7_7x5":       dict(category="matmul",
                               inputs=[f(randf("dot_nonsq", 0, 4, 7)), f(randf("dot_nonsq", 1, 7, 5))],
                               fn=lambda a, b: a @ b),
    "dot_na130__2x130_130x4":   dict(category="matmul",
                               inputs=[f(randf("dot_na130", 0, 2, 130)), f(randf("dot_na130", 1, 130, 4))],
                               fn=lambda a, b: a @ b),
    "dot_b4__2x2x3x4_2x2x4x5": dict(category="matmul",
                               inputs=[f(randf("dot_b4", 0, 2, 2, 3, 4)), f(randf("dot_b4", 1, 2, 2, 4, 5))],
                               fn=lambda a, b: a @ b),
    "dot_vec__1x8_8x1":         dict(category="matmul",
                               inputs=[f(randf("dot_vec", 0, 1, 8)), f(randf("dot_vec", 1, 8, 1))],
                               fn=lambda a, b: a @ b),
}
