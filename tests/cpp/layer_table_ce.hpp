// layer_table_ce.hpp — builders for the cross-entropy layer. Mirrors layers_ce.py.
// nn::cross_entropy(g, logits[N,V], targets[N], N, V) -> scalar mean NLL.
// No learnable parameters; takes Graph& g directly (not TrainCtx&).
#pragma once
#include "layer_table_base.hpp"

namespace tpu {

inline void register_ce(LayerMap& T) {
    // logits: float [4,5], targets: int32 [4], scalar output (mean NLL)
    T["cross_entropy__4x5"] = LayerSpec{
        {{"logits", {4, 5}, false}, {"targets", {4}, true}},
        /*scalar_out=*/true,
        [](TrainCtx& c, const std::vector<Value>& v) {
            return nn::cross_entropy(c.g, v[0], v[1], 4, 5);
        }};
}

}  // namespace tpu
