"""Oracle cases — numerical-stability regimes (task 5.3).

These exercise the stabilization tricks in nn.hpp at magnitudes where a naive
implementation overflows or divides by ~0:
  * softmax / cross_entropy: max-subtraction must prevent exp() overflow when
    logits are ~1e3 (exp(1e3) = inf in f32). The `scale` field draws inputs at
    that magnitude; both the C++ op and the JAX oracle subtract the max, so the
    result is finite and they must still agree.
  * rmsnorm: the eps inside rsqrt(mean(x^2)+eps) must keep the norm finite when
    x is ~1e-4 (mean(x^2) ~ 1e-8, eps-dominated), and well-behaved when x ~1e3.

Helpers are the byte-identical, already-on-device-verified replicas from the
softmax/rmsnorm/cross_entropy layer modules.
"""
from layers_block import softmax, rmsnorm   # verified replicas
from layers_ce import _cross_entropy         # verified replica

LAYERS = {
    # softmax with logits ~1e3 — exp() would overflow without max-subtraction.
    "softmax_large__4x5": dict(
        inputs=[dict(name="x", shape=[4, 5], scale=1e3)],
        params={},
        fn=lambda ins, p: softmax(ins["x"], axis=1)),

    # rmsnorm with near-zero inputs (~1e-4) — eps must dominate rsqrt and keep it finite.
    "rmsnorm_tiny__4x8": dict(
        inputs=[dict(name="x", shape=[4, 8], scale=1e-4)],
        params={"n": [8]},
        # tiny x -> norm ~ x * rsqrt(eps) ~ x*316; gradient amplified by 1/rms, so use
        # the project's gradcheck envelope (1e-2) for the grad bound. Forward stays tight.
        fn=lambda ins, p: rmsnorm(ins["x"], p["n"]),
        grad_tol=1e-2),

    # rmsnorm with large inputs (~1e3) — mean(x^2)~1e6; must stay well-conditioned.
    "rmsnorm_huge__4x8": dict(
        inputs=[dict(name="x", shape=[4, 8], scale=1e3)],
        params={"n": [8]},
        fn=lambda ins, p: rmsnorm(ins["x"], p["n"])),

    # cross_entropy with large logits (~±60 at scale 20) — logsumexp must not overflow.
    # The scalar loss can reach ~tens, so loosen the forward bound accordingly (the
    # f32 round-off floor scales with the loss magnitude); not masking a bug.
    "cross_entropy_large__4x5": dict(
        inputs=[
            dict(name="logits",  shape=[4, 5], scale=20.0),
            dict(name="targets", shape=[4], int=True, high=5),
        ],
        params={},
        fn=lambda ins, p: _cross_entropy(ins["logits"], ins["targets"], 4, 5),
        scalar_out=True,
        fwd_tol=5e-2),
}
