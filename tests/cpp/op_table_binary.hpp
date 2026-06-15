// op_table_binary.hpp — builders for binary elementwise ops. Mirrors cases_binary.py.
#pragma once
#include "op_table_base.hpp"

namespace tpu {

inline void register_binary(OpBuilderMap& T) {
    T["add__2x3"]         = [](Graph& g, const std::vector<Value>& v){ return g.add(v[0], v[1]); };
    T["mul__2x3"]         = [](Graph& g, const std::vector<Value>& v){ return g.mul(v[0], v[1]); };
    T["sub__2x3"]         = [](Graph& g, const std::vector<Value>& v){ return g.sub(v[0], v[1]); };
    T["div__2x3"]         = [](Graph& g, const std::vector<Value>& v){ return g.div(v[0], v[1]); };
    T["max__2x3"]         = [](Graph& g, const std::vector<Value>& v){ return g.max(v[0], v[1]); };
    T["min__2x3"]         = [](Graph& g, const std::vector<Value>& v){ return g.min(v[0], v[1]); };
    T["add_bcast__2x3_3"]      = [](Graph& g, const std::vector<Value>& v){ return g.add(v[0], v[1]); };
    // scalar-ish broadcast: [2,3] + [1]
    T["add_bcast__2x3_1"]      = [](Graph& g, const std::vector<Value>& v){ return g.add(v[0], v[1]); };
    // broadcast along row: [2,3] * [3]
    T["mul_bcast__2x3_3"]      = [](Graph& g, const std::vector<Value>& v){ return g.mul(v[0], v[1]); };
    // mutual broadcast: [1,3] + [2,1] -> [2,3]
    T["add_mutual__1x3_2x1"]   = [](Graph& g, const std::vector<Value>& v){ return g.add(v[0], v[1]); };
    // rank-3: [2,2,3] * [2,2,3]
    T["mul_rank3__2x2x3"]      = [](Graph& g, const std::vector<Value>& v){ return g.mul(v[0], v[1]); };
    // non-128-aligned minor dim: [2,130] + [2,130]
    T["add_nonaligned__2x130"] = [](Graph& g, const std::vector<Value>& v){ return g.add(v[0], v[1]); };
    // div with positive denominator (avoids divide-by-zero)
    T["div_pos__2x3"]          = [](Graph& g, const std::vector<Value>& v){ return g.div(v[0], v[1]); };
}

}  // namespace tpu
