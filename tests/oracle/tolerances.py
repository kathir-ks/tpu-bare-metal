"""Single source of truth for per-op numeric tolerances (capability: precision-policy).

Each entry maps an op *category* to (highest, default) max-abs-error thresholds:
  - highest : f32 dot_precision="HIGHEST" — tight, matches the existing gradcheck
              envelope (max err ~3e-4 on a small matmul chain).
  - default : dot_precision="DEFAULT" (bf16 matmul passes) — looser. The bf16 margin
              is largest for matmul-heavy ops because every contraction term is
              rounded to bf16 (~3 decimal digits) before accumulation.

Derivation: HIGHEST values are the f32 round-off envelope observed on this libtpu
build. DEFAULT values add a measured bf16 margin: ~2e-2 relative for a single
matmul, scaled by the op's contraction depth. These are thresholds, not promises —
the conformance suite prints measured error next to them so slack stays visible.
"""

# category -> (highest_tol, default_tol)
TOL = {
    "elementwise": (1e-4, 1e-4),   # no matmul: bf16 path identical to f32 here
    "unary_transcendental": (1e-4, 1e-4),
    "reduction": (5e-4, 5e-4),     # accumulation order can differ slightly
    "matmul": (1e-3, 3e-2),        # DEFAULT here is the small-K floor; see matmul_default_tol
    "shape": (0.0, 0.0),           # pure data movement: must be exact
    "gather": (0.0, 0.0),          # index/copy: exact
}


def tol_for(category):
    if category not in TOL:
        raise KeyError(f"no tolerance for category {category!r}; add it to tolerances.py")
    return TOL[category]


def matmul_default_tol(k):
    """bf16 (DEFAULT) matmul tolerance as a function of contraction depth K.

    A bf16 dot of depth K sums K products, each rounded to ~3 decimal digits before
    accumulation; the error accumulates roughly as sqrt(K) (random-walk of rounding
    noise). Measured on this libtpu build: K=3 -> ~1.0e-2, K=130 -> ~5.8e-2, which
    fits err ≈ 7e-3 * sqrt(K). We set the threshold a margin above that fit, with a
    2e-2 floor so shallow matmuls keep a meaningful (not hidden-error) bound — bf16
    small-K error is ~1e-2, so a 2e-2 bound still catches a 2x regression. The floor
    also covers batched matmul, whose per-output error runs slightly above the plain
    sqrt(K) fit (measured K=4 batched -> 1.52e-2):

        tol(K) = max(2e-2, 7e-3 * sqrt(K))

    e.g. K=4 -> 2.0e-2, K=8 -> 2.0e-2, K=130 -> 8.0e-2. HIGHEST stays 1e-3 (exact-ish).
    """
    import math
    return max(2e-2, 7e-3 * math.sqrt(float(k)))
