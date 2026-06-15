"""Oracle cases — unary elementwise ops (graph Ops: Neg, Exp, Log, Sqrt, Rsqrt,
Tanh, Abs, Logistic). One CASE per (op, shape). category drives tolerance.

Each value: dict(category=..., inputs=[f(...)/i(...)], fn=lambda *ins: out, diff=True).
Keys must match a builder in tests/cpp/op_table_unary.hpp exactly.
"""
import jax
import jax.numpy as jnp
from oracle_util import randf, randpos, f

CASES = {
    "tanh__2x3":     dict(category="unary_transcendental",
                          inputs=[f(randf("tanh", 0, 2, 3))], fn=lambda a: jnp.tanh(a)),
    "exp__2x3":      dict(category="unary_transcendental",
                          inputs=[f(randf("exp", 0, 2, 3) * 0.5)], fn=lambda a: jnp.exp(a)),
    "logistic__2x3": dict(category="unary_transcendental",
                          inputs=[f(randf("logistic", 0, 2, 3))], fn=lambda a: jax.nn.sigmoid(a)),
    "rsqrt__2x3":    dict(category="unary_transcendental",
                          inputs=[f(randpos("rsqrt", 0, 2, 3))], fn=lambda a: jax.lax.rsqrt(a)),
    "neg__2x3":      dict(category="elementwise",
                          inputs=[f(randf("neg", 0, 2, 3))], fn=lambda a: -a),
    "abs__2x3":      dict(category="elementwise",
                          inputs=[f(randf("abs", 0, 2, 3))], fn=lambda a: jnp.abs(a)),
    "log__2x3":      dict(category="unary_transcendental",
                          inputs=[f(randpos("log", 0, 2, 3))], fn=lambda a: jnp.log(a)),
    "sqrt__2x3":     dict(category="unary_transcendental",
                          inputs=[f(randpos("sqrt", 0, 2, 3))], fn=lambda a: jnp.sqrt(a)),
    # rank-1
    "tanh__5":       dict(category="unary_transcendental",
                          inputs=[f(randf("tanh__5", 0, 5))], fn=lambda a: jnp.tanh(a)),
    # rank-3
    "exp__2x2x3":    dict(category="unary_transcendental",
                          inputs=[f(randf("exp__2x2x3", 0, 2, 2, 3) * 0.5)], fn=lambda a: jnp.exp(a)),
    # rank-4
    "tanh__2x2x2x3": dict(category="unary_transcendental",
                          inputs=[f(randf("tanh__2x2x2x3", 0, 2, 2, 2, 3))], fn=lambda a: jnp.tanh(a)),
    # size-1 dim
    "logistic__1x4": dict(category="unary_transcendental",
                          inputs=[f(randf("logistic__1x4", 0, 1, 4))], fn=lambda a: jax.nn.sigmoid(a)),
    "neg__4x1":      dict(category="elementwise",
                          inputs=[f(randf("neg__4x1", 0, 4, 1))], fn=lambda a: -a),
    # non-128-aligned minor dim
    "abs__2x130":    dict(category="elementwise",
                          inputs=[f(randf("abs__2x130", 0, 2, 130))], fn=lambda a: jnp.abs(a)),
    "tanh__3x65":    dict(category="unary_transcendental",
                          inputs=[f(randf("tanh__3x65", 0, 3, 65))], fn=lambda a: jnp.tanh(a)),
}
