// op_table_base.hpp — shared types for the per-family op-table headers.
//
// Each family header (op_table_<family>.hpp) includes this and defines
//   inline void register_<family>(OpBuilderMap& T) { T["key"] = [](Graph&, ...){...}; }
// The master op_table.hpp includes all families and merges them into one map.
// Splitting by family lets parallel authors extend disjoint files without colliding.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "graph.hpp"

namespace tpu {

// A builder takes the graph and the already-created input Values (in0, in1, ...)
// and returns the single output Value for that op case. It must mirror the JAX
// `fn` of the same key in tests/oracle/cases_<family>.py exactly.
using OpBuilder    = std::function<Value(Graph&, const std::vector<Value>&)>;
using OpBuilderMap = std::map<std::string, OpBuilder>;

}  // namespace tpu
