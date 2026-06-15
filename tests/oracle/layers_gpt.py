"""Oracle case — composite GPT loss (cpp/gpt.hpp gpt_loss), the end-to-end capstone.

Mirrors gpt_logits + gpt_loss exactly:
    tok = wte[ids]                         # embedding (gather)      [B,T,D]
    h   = tok + wpe                        # + positional, broadcast over B
    for l: h = block(h, "h{l}", ...)       # n_layer pre-norm blocks
    h   = rmsnorm(h, "lnf")
    logits = h @ head.w                    # untied head, no bias    [B,T,V]
    loss = cross_entropy(reshape(logits,[B*T,V]), reshape(targets,[B*T]))  # scalar mean

Param NAMES + ORDER match the c.param(...) calls nn.hpp/gpt.hpp make (resolver feeds
each from the fixture):
    wte[V,D], wpe[T,D], then per block l "h{l}.": n1[D], attn.{q,k,v,o}.w[D,D], n2[D],
    fc1.w[D,ff], fc1.b[ff], fc2.w[ff,D], fc2.b[D]; then lnf[D], head.w[D,V].

The layer helpers (gelu/rmsnorm/softmax/attention/cross_entropy) are byte-for-byte the
ones already verified on-device in layers_block.py / layers_ce.py — composed here.
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
    return e / jnp.sum(e, axis=axis, keepdims=True)


def attention(x, qw, kw, vw, ow, B, T, D, H):
    hd = D // H
    q, k, v = x @ qw, x @ kw, x @ vw

    def split_heads(t):
        return jnp.transpose(t.reshape(B, T, H, hd), (0, 2, 1, 3))  # [B,H,T,hd]

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    scores = (q @ jnp.swapaxes(k, -1, -2)) * (1.0 / jnp.sqrt(jnp.asarray(hd, jnp.float32)))
    row = jnp.arange(T).reshape(T, 1)
    col = jnp.arange(T).reshape(1, T)
    maskv = jnp.where(col > row, jnp.float32(-1e30), jnp.float32(0.0))  # [T,T]
    scores = scores + maskv
    attn = softmax(scores, axis=3)
    out = attn @ v                              # [B,H,T,hd]
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, T, D)
    return out @ ow


def block(x, p, prefix, B, T, D, H, ff):
    n1 = rmsnorm(x, p[prefix + ".n1"])
    a = attention(n1, p[prefix + ".attn.q.w"], p[prefix + ".attn.k.w"],
                  p[prefix + ".attn.v.w"], p[prefix + ".attn.o.w"], B, T, D, H)
    h = x + a
    n2 = rmsnorm(h, p[prefix + ".n2"])
    fc1 = gelu(n2 @ p[prefix + ".fc1.w"] + p[prefix + ".fc1.b"])
    fc2 = fc1 @ p[prefix + ".fc2.w"] + p[prefix + ".fc2.b"]
    return h + fc2


def cross_entropy(logits, targets, N, V):
    m = jax.lax.stop_gradient(jnp.max(logits, axis=1, keepdims=True))
    sh = logits - m
    lse = jnp.log(jnp.sum(jnp.exp(sh), axis=1, keepdims=True)) + m
    logp = logits - lse
    cols = jnp.arange(V, dtype=jnp.int32)
    oh = (cols[None, :] == targets[:, None]).astype(jnp.float32)
    return jnp.mean(-jnp.sum(logp * oh, axis=1))


# Tiny config (fast, still exercises every component).
_V, _NL, _H, _D, _FF, _T, _B = 10, 1, 2, 8, 16, 3, 1


def gpt_loss(ins, p):
    ids, targets = ins["ids"], ins["targets"]               # [B,T] int
    tok = p["wte"][ids]                                      # [B,T,D]
    h = tok + p["wpe"]                                       # + [T,D] broadcast over B
    for l in range(_NL):
        h = block(h, p, f"h{l}", _B, _T, _D, _H, _FF)
    h = rmsnorm(h, p["lnf"])
    logits = h @ p["head.w"]                                 # [B,T,V]
    flat = logits.reshape(_B * _T, _V)
    tgt = targets.reshape(_B * _T)
    return cross_entropy(flat, tgt, _B * _T, _V)


def _params():
    p = {"wte": [_V, _D], "wpe": [_T, _D]}
    for l in range(_NL):
        pre = f"h{l}"
        p[f"{pre}.n1"] = [_D]
        p[f"{pre}.attn.q.w"] = [_D, _D]
        p[f"{pre}.attn.k.w"] = [_D, _D]
        p[f"{pre}.attn.v.w"] = [_D, _D]
        p[f"{pre}.attn.o.w"] = [_D, _D]
        p[f"{pre}.n2"] = [_D]
        p[f"{pre}.fc1.w"] = [_D, _FF]
        p[f"{pre}.fc1.b"] = [_FF]
        p[f"{pre}.fc2.w"] = [_FF, _D]
        p[f"{pre}.fc2.b"] = [_D]
    p["lnf"] = [_D]
    p["head.w"] = [_D, _V]
    return p


LAYERS = {
    "gpt__1L_v10_d8_t3": dict(
        inputs=[
            dict(name="ids",     shape=[_B, _T], int=True, high=_V),
            dict(name="targets", shape=[_B, _T], int=True, high=_V),
        ],
        params=_params(),
        fn=gpt_loss,
        scalar_out=True,
        # Composite model (embedding -> block -> lnf -> head -> CE): gradients use the
        # same documented composite envelope as `block` (1e-2, the project's gradcheck
        # norm). Measured here is ~8e-5 — far under it — so this is a meaningful bound,
        # not a masked bug; every component is independently exact on-device.
        grad_tol=1e-2),
}
