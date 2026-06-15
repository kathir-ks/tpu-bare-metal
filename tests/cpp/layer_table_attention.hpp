// layer_table_attention.hpp — builders for the attention layer. Mirrors layers_attention.py.
// Causal multi-head self-attention: nn::attention(c, x, "attn", B=1, T=4, D=8, H=2).
#pragma once
#include "layer_table_base.hpp"

namespace tpu {

inline void register_attention(LayerMap& T) {
    T["attention__1x4x8_h2"] = LayerSpec{
        {{"x", {1, 4, 8}, false}}, false,
        [](TrainCtx& c, const std::vector<Value>& v){
            return nn::attention(c, v[0], "attn", /*B=*/1, /*T=*/4, /*D=*/8, /*H=*/2);
        }};
}

}  // namespace tpu
