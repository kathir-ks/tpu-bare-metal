"""Oracle layer case — pre-norm transformer block (nn.hpp::block).

Mirrors nn::block(c, x, "blk", B, T, D, H, ff):
    a = attention(rmsnorm(x, "blk.n1"), "blk.attn"); h = x + a
    m = linear(gelu(linear(rmsnorm(h, "blk.n2"), "blk.fc1", D, ff)), "blk.fc2", ff, D)
    return h + m

Param names match the c.param(...) names nn.hpp requests:
    blk.n1[D], blk.attn.{q,k,v,o}.w[D,D] (no bias), blk.n2[D],
    blk.fc1.w[D,ff], blk.fc1.b[ff], blk.fc2.w[ff,D], blk.fc2.b[D].

The fn replicates the nn.hpp math EXACTLY: gelu (tanh approx), rmsnorm (eps=1e-5),
causal multi-head attention with 1/sqrt(hd) scaling and a -1e30 additive mask on
the strictly-upper triangle (col>row), softmax over the last axis.
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


def softmax(x, axis):
    m = jax.lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))
    e = jnp.exp(x - m)
    s = jnp.sum(e, axis=axis, keepdims=True)
    return e / s


def attention(x, qw, kw, vw, ow, B, T, D, H):
    hd = D // H
    # linear (no bias): [B,T,D] @ [D,D]
    q = x @ qw
    k = x @ kw
    v = x @ vw

    def split_heads(t):
        r = t.reshape(B, T, H, hd)
        return jnp.transpose(r, (0, 2, 1, 3))  # [B,H,T,hd]

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)
    # scores [B,H,T,T] = q @ k^T / sqrt(hd)
    scores = q @ jnp.swapaxes(k, -1, -2)
    scores = scores * (1.0 / jnp.sqrt(jnp.asarray(hd, jnp.float32)))
    # causal mask: col>row -> -1e30
    row = jnp.arange(T).reshape(T, 1)
    col = jnp.arange(T).reshape(1, T)
    masked = col > row
    maskv = jnp.where(masked, jnp.float32(-1e30), jnp.float32(0.0))  # [T,T]
    scores = scores + maskv  # broadcast over B,H
    attn = softmax(scores, axis=3)  # [B,H,T,T]
    out = attn @ v  # [B,H,T,hd]
    out = jnp.transpose(out, (0, 2, 1, 3))  # [B,T,H,hd]
    out = out.reshape(B, T, D)
    return out @ ow  # linear (no bias) [B,T,D] @ [D,D]


def block(x, p, B, T, D, H, ff):
    n1 = rmsnorm(x, p["blk.n1"])
    a = attention(n1, p["blk.attn.q.w"], p["blk.attn.k.w"],
                  p["blk.attn.v.w"], p["blk.attn.o.w"], B, T, D, H)
    h = x + a
    n2 = rmsnorm(h, p["blk.n2"])
    fc1 = n2 @ p["blk.fc1.w"] + p["blk.fc1.b"]
    g = gelu(fc1)
    fc2 = g @ p["blk.fc2.w"] + p["blk.fc2.b"]
    return h + fc2


_B, _T, _D, _H, _FF = 1, 4, 8, 2, 16

LAYERS = {
    "block__1x4x8_h2_ff16": dict(
        inputs=[dict(name="x", shape=[_B, _T, _D])],
        params={
            "blk.n1": [_D],
            "blk.attn.q.w": [_D, _D],
            "blk.attn.k.w": [_D, _D],
            "blk.attn.v.w": [_D, _D],
            "blk.attn.o.w": [_D, _D],
            "blk.n2": [_D],
            "blk.fc1.w": [_D, _FF],
            "blk.fc1.b": [_FF],
            "blk.fc2.w": [_FF, _D],
            "blk.fc2.b": [_D],
        },
        fn=lambda ins, p: block(ins["x"], p, _B, _T, _D, _H, _FF),
        scalar_out=False,
        # Composite layer: deep backward accumulates TPU-HIGHEST-vs-f32 rounding
        # (~1e-6 relative per matmul), amplified by rmsnorm 1/rms + softmax. Forward
        # stays tight (~2e-4); gradients use the project's gradcheck envelope (1e-2).
        grad_tol=1e-2),
}
