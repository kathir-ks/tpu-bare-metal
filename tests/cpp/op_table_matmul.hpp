// op_table_matmul.hpp — builders for matmul/dot. Mirrors cases_matmul.py.
#pragma once
#include "op_table_base.hpp"

namespace tpu {

inline void register_matmul(OpBuilderMap& T) {
    T["dot__2x3_3x4"]               = [](Graph& g, const std::vector<Value>& v){ return g.dot(v[0], v[1]); };
    T["dot_batched__2x2x3_2x3x4"]   = [](Graph& g, const std::vector<Value>& v){ return g.dot(v[0], v[1]); };
    T["dot_nonsq__4x7_7x5"]         = [](Graph& g, const std::vector<Value>& v){ return g.dot(v[0], v[1]); };
    T["dot_na130__2x130_130x4"]     = [](Graph& g, const std::vector<Value>& v){ return g.dot(v[0], v[1]); };
    T["dot_b4__2x2x3x4_2x2x4x5"]   = [](Graph& g, const std::vector<Value>& v){ return g.dot(v[0], v[1]); };
    T["dot_vec__1x8_8x1"]           = [](Graph& g, const std::vector<Value>& v){ return g.dot(v[0], v[1]); };
}

}  // namespace tpu
