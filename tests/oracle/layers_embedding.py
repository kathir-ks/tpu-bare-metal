"""Oracle layer cases — embedding layer (nn::embedding, gather-based row lookup).

The single param is named exactly "emb" with shape [vocab, dim].
ids is an int32 input with high=vocab (exclusive upper bound).
fn: p["emb"][ins["ids"]]  → out shape [B, T, dim]
"""
import jax.numpy as jnp

LAYERS = {
    "embedding__2x3_v10_d4": dict(
        inputs=[dict(name="ids", shape=[2, 3], int=True, high=10)],
        params={"emb": [10, 4]},
        fn=lambda ins, p: p["emb"][ins["ids"]],
        scalar_out=False,
    ),
}
