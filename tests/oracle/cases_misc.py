"""Oracle cases — misc ops (graph Ops: StopGradient, Select, Compare, Convert, Iota).

Notes for extending:
  * StopGradient: forward is identity; gradient w.r.t. the input must be exactly 0.
  * Compare: output is i1 (boolean). The fixture `out` is f32, so a Compare case must
    cast the boolean result to f32 in `fn` AND the C++ builder must convert(Pred->F32);
    mark diff=False (no gradient through a predicate).
  * Select: select(pred, a, b); gradient flows to a and b per the predicate.
  * Convert / Iota: integer paths; mark diff=False where output is non-float.
"""
import jax
import jax.numpy as jnp
from oracle_util import randf, f

CASES = {
    "stop_gradient__2x3": dict(category="elementwise",
                               inputs=[f(randf("sg", 0, 2, 3))],
                               fn=lambda a: jax.lax.stop_gradient(a)),

    # Compare: a > b -> bool, cast to f32. Non-differentiable.
    "compare_gt__2x3": dict(category="elementwise",
                            inputs=[f(randf("cmp", 0, 2, 3)), f(randf("cmp", 1, 2, 3))],
                            fn=lambda a, b: (a > b).astype(jnp.float32),
                            diff=False),

    # Select: where(a > 0, a, b); gradient flows to both a and b.
    "select_pos__2x3": dict(category="elementwise",
                            inputs=[f(randf("sel", 0, 2, 3)), f(randf("sel", 1, 2, 3))],
                            fn=lambda a, b: jnp.where(a > 0, a, b)),

    # Convert: float identity through convert-to-F32. Differentiable.
    "convert_id__2x3": dict(category="elementwise",
                            inputs=[f(randf("cvt", 0, 2, 3))],
                            fn=lambda a: a),

    # Iota: no-input case; iota({6}, dim=0) as f32. Non-differentiable.
    "iota__6": dict(category="shape",
                    inputs=[],
                    fn=lambda: jnp.arange(6, dtype=jnp.float32),
                    diff=False),
}
