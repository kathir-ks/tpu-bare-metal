// layer_table_base.hpp — shared types for per-layer conformance builders.
//
// Each layer header (layer_table_<name>.hpp) includes this and defines
//   inline void register_<name>(LayerMap& T) { T["key"] = LayerSpec{...}; }
// The master layer_table.hpp merges them. The builder receives a TrainCtx (whose
// resolver feeds each c.param(name,...) from the fixture's same-named param) and the
// ordered input Values, and returns the layer output. Mirrors tests/oracle/layers_*.py.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "nn.hpp"

namespace tpu {

struct LayerInput {
    std::string name;
    Shape       shape;
    bool        is_int = false;   // int32 input (e.g. embedding ids / CE targets)
};

struct LayerSpec {
    std::vector<LayerInput> inputs;     // ordered forward inputs
    bool scalar_out = false;            // builder returns a scalar (loss) already
    std::function<Value(TrainCtx&, const std::vector<Value>&)> build;
};

using LayerMap = std::map<std::string, LayerSpec>;

}  // namespace tpu
