// layer_table_stability.hpp — builders for numerical-stability regime cases.
// Mirrors tests/oracle/layers_stability.py. Same nn.hpp ops as softmax/rmsnorm/ce,
// just fed extreme-magnitude inputs (from the fixture) to exercise overflow/eps paths.
#pragma once
#include "layer_table_base.hpp"

namespace tpu {

inline void register_stability(LayerMap& T) {
    T["softmax_large__4x5"] = LayerSpec{
        {{"x", {4, 5}, false}}, false,
        [](TrainCtx& c, const std::vector<Value>& v){ return nn::softmax(c.g, v[0], 1); }};

    T["rmsnorm_tiny__4x8"] = LayerSpec{
        {{"x", {4, 8}, false}}, false,
        [](TrainCtx& c, const std::vector<Value>& v){ return nn::rmsnorm(c, v[0], "n", 8); }};

    T["rmsnorm_huge__4x8"] = LayerSpec{
        {{"x", {4, 8}, false}}, false,
        [](TrainCtx& c, const std::vector<Value>& v){ return nn::rmsnorm(c, v[0], "n", 8); }};

    T["cross_entropy_large__4x5"] = LayerSpec{
        {{"logits", {4, 5}, false}, {"targets", {4}, true}}, /*scalar_out=*/true,
        [](TrainCtx& c, const std::vector<Value>& v){ return nn::cross_entropy(c.g, v[0], v[1], 4, 5); }};
}

}  // namespace tpu
