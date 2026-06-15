// layer_table_embedding.hpp — builders for the embedding layer. Mirrors layers_embedding.py.
#pragma once
#include "layer_table_base.hpp"

namespace tpu {

inline void register_embedding(LayerMap& T) {
    T["embedding__2x3_v10_d4"] = LayerSpec{
        {{"ids", {2, 3}, true}}, false,
        [](TrainCtx& c, const std::vector<Value>& v){
            return nn::embedding(c, v[0], "emb", 10, 4);
        }};
}

}  // namespace tpu
