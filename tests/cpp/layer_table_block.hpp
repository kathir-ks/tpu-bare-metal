// layer_table_block.hpp — builders for the block layer. Mirrors layers_block.py.
// Pre-norm transformer block: nn::block(c, x, "blk", B=1, T=4, D=8, H=2, ff=16).
#pragma once
#include "layer_table_base.hpp"

namespace tpu {

inline void register_block(LayerMap& T) {
    T["block__1x4x8_h2_ff16"] = LayerSpec{
        {{"x", {1, 4, 8}, false}}, false,
        [](TrainCtx& c, const std::vector<Value>& v){
            return nn::block(c, v[0], "blk",
                             /*B=*/1, /*T=*/4, /*D=*/8, /*H=*/2, /*ff=*/16);
        }};
}

}  // namespace tpu
