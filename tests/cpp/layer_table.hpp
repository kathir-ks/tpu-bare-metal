// layer_table.hpp — master layer-conformance table: merges every per-layer builder
// header. The conformance suite (test_oracle_layers.cpp) parses a fixture key, looks
// up the LayerSpec, builds the layer via nn.hpp with a resolver feeding the fixture's
// params, and compares forward + gradients to the oracle.
//
// Add a layer by creating layer_table_<name>.hpp (+ tests/oracle/layers_<name>.py)
// and registering it below.
#pragma once

#include "layer_table_base.hpp"
#include "layer_table_core.hpp"
#include "layer_table_embedding.hpp"
#include "layer_table_softmax.hpp"
#include "layer_table_attention.hpp"
#include "layer_table_block.hpp"
#include "layer_table_ce.hpp"
#include "layer_table_gpt.hpp"
#include "layer_table_stability.hpp"

namespace tpu {

inline const LayerMap& layer_table() {
    static const LayerMap T = [] {
        LayerMap m;
        register_core(m);
        register_embedding(m);
        register_softmax(m);
        register_attention(m);
        register_block(m);
        register_ce(m);
        register_gpt(m);
        register_stability(m);
        return m;
    }();
    return T;
}

inline bool layer_table_has(const std::string& key) { return layer_table().count(key) > 0; }

}  // namespace tpu
