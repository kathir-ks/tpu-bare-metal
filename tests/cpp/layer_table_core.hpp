// layer_table_core.hpp — builders for core paramful layers. Mirrors layers_core.py.
#pragma once
#include "layer_table_base.hpp"

namespace tpu {

inline void register_core(LayerMap& T) {
    T["gelu__4x8"] = LayerSpec{
        {{"x", {4, 8}, false}}, false,
        [](TrainCtx& c, const std::vector<Value>& v){ return nn::gelu(c.g, v[0]); }};

    T["linear__4x6_8"] = LayerSpec{
        {{"x", {4, 6}, false}}, false,
        [](TrainCtx& c, const std::vector<Value>& v){ return nn::linear(c, v[0], "fc", 6, 8, true); }};

    T["rmsnorm__4x8"] = LayerSpec{
        {{"x", {4, 8}, false}}, false,
        [](TrainCtx& c, const std::vector<Value>& v){ return nn::rmsnorm(c, v[0], "n", 8); }};
}

}  // namespace tpu
