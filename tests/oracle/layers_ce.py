"""Oracle layer cases — cross-entropy (nn::cross_entropy in nn.hpp).

nn::cross_entropy(g, logits[N,V], targets[N], N, V) -> scalar mean NLL.

Formula (mirrors nn.hpp exactly):
  m   = stop_gradient(max(logits, axis=1, keepdims=True))
  sh  = logits - m
  lse = log(sum(exp(sh), axis=1, keepdims=True)) + m   # logsumexp [N,1]
  logp = logits - lse                                  # [N,V]
  oh   = one_hot(targets, V)                           # [N,V] float
  picked = sum(logp * oh, axis=1)                      # [N]
  nll  = -picked                                       # [N]
  loss = mean(nll)                                     # scalar

No learnable parameters.  scalar_out=True.
"""
import jax
import jax.numpy as jnp


def _cross_entropy(logits, targets, N, V):
    # mirror nn.hpp: max subtraction for logsumexp stability; max is stop-gradient'd.
    m = jax.lax.stop_gradient(jnp.max(logits, axis=1, keepdims=True))  # [N,1]
    sh = logits - m
    lse = jnp.log(jnp.sum(jnp.exp(sh), axis=1, keepdims=True)) + m    # [N,1]
    logp = logits - lse                                                 # [N,V]
    # one-hot gather: iota over columns == targets
    cols = jnp.arange(V, dtype=jnp.int32)                              # [V]
    tgt = targets[:, None]                                              # [N,1]
    oh = (cols[None, :] == tgt).astype(jnp.float32)                    # [N,V]
    picked = jnp.sum(logp * oh, axis=1)                                 # [N]
    nll = -picked                                                        # [N]
    return jnp.mean(nll)                                                 # scalar


LAYERS = {
    "cross_entropy__4x5": dict(
        inputs=[
            dict(name="logits",  shape=[4, 5]),
            dict(name="targets", shape=[4], int=True, high=5),
        ],
        params={},
        fn=lambda ins, p: _cross_entropy(ins["logits"], ins["targets"], 4, 5),
        scalar_out=True,
    ),
}
