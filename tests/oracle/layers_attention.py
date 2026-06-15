"""Oracle layer case — causal multi-head self-attention (nn.hpp::attention).

Mirrors nn::attention(c, x, "attn", B, T, D, H) EXACTLY:
  q,k,v = x @ W   (linear, bias=false; W is [D,D] row-major)
  split heads: reshape [B,T,H,hd] then transpose (0,2,1,3) -> [B,H,T,hd]
  scores = q @ k^T * (1/sqrt(hd))              -> [B,H,T,T]
  causal mask: where col>row add -1e30          (col/row are T x T iotas)
  softmax over last axis (max-subtracted, stop-grad max)
  out = attn @ v                                -> [B,H,T,hd]
  merge heads: transpose (0,2,1,3) -> [B,T,H,hd], reshape [B,T,D]
  out = out @ Wo                                -> final projection
"""
import jax
import jax.numpy as jnp

B, T, D, H = 1, 4, 8, 2
hd = D // H


def softmax(x, axis):
    m = jax.lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))
    e = jnp.exp(x - m)
    s = jnp.sum(e, axis=axis, keepdims=True)
    return e / s


def attention(x, wq, wk, wv, wo):
    q = x @ wq
    k = x @ wk
    v = x @ wv

    def split_heads(t):
        r = t.reshape(B, T, H, hd)
        return jnp.transpose(r, (0, 2, 1, 3))  # [B,H,T,hd]

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    scores = q @ jnp.swapaxes(k, -1, -2)            # [B,H,T,T]
    scores = scores * (1.0 / jnp.sqrt(float(hd)))
    row = jnp.arange(T)[:, None]
    col = jnp.arange(T)[None, :]
    maskv = jnp.where(col > row, -1e30, 0.0).astype(scores.dtype)  # [T,T]
    scores = scores + maskv
    attn = softmax(scores, axis=3)                  # [B,H,T,T]
    out = attn @ v                                  # [B,H,T,hd]
    out = jnp.transpose(out, (0, 2, 1, 3))          # [B,T,H,hd]
    out = out.reshape(B, T, D)
    return out @ wo


LAYERS = {
    "attention__1x4x8_h2": dict(
        inputs=[dict(name="x", shape=[B, T, D])],
        params={
            "attn.q.w": [D, D],
            "attn.k.w": [D, D],
            "attn.v.w": [D, D],
            "attn.o.w": [D, D],
        },
        fn=lambda ins, p: attention(
            ins["x"], p["attn.q.w"], p["attn.k.w"], p["attn.v.w"], p["attn.o.w"]),
    ),
}
