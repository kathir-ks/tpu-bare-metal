// layer_table_softmax.hpp — builders for nn::softmax (paramless). Mirrors layers_softmax.py.
#pragma once
#include "layer_table_base.hpp"

namespace tpu {

inline void register_softmax(LayerMap& T) {
    // softmax over axis=1 on x[4,5]
    T["softmax__4x5_ax1"] = LayerSpec{
        {{"x", {4, 5}, false}}, false,
        [](TrainCtx& c, const std::vector<Value>& v) {
            return nn::softmax(c.g, v[0], 1);
        }};

    // softmax over axis=2 on x[3,4,5]
    T["softmax__3x4x5_ax2"] = LayerSpec{
        {{"x", {3, 4, 5}, false}}, false,
        [](TrainCtx& c, const std::vector<Value>& v) {
            return nn::softmax(c.g, v[0], 2);
        }};
}

}  // namespace tpu
