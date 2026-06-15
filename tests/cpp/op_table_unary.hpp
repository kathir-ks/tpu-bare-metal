// op_table_unary.hpp — builders for unary elementwise ops. Mirrors cases_unary.py.
#pragma once
#include "op_table_base.hpp"

namespace tpu {

inline void register_unary(OpBuilderMap& T) {
    T["tanh__2x3"]     = [](Graph& g, const std::vector<Value>& v){ return g.tanh(v[0]); };
    T["exp__2x3"]      = [](Graph& g, const std::vector<Value>& v){ return g.exp(v[0]); };
    T["logistic__2x3"] = [](Graph& g, const std::vector<Value>& v){ return g.logistic(v[0]); };
    T["rsqrt__2x3"]    = [](Graph& g, const std::vector<Value>& v){ return g.rsqrt(v[0]); };
    T["neg__2x3"]      = [](Graph& g, const std::vector<Value>& v){ return g.neg(v[0]); };
    T["abs__2x3"]      = [](Graph& g, const std::vector<Value>& v){ return g.abs(v[0]); };
    T["log__2x3"]      = [](Graph& g, const std::vector<Value>& v){ return g.log(v[0]); };
    T["sqrt__2x3"]     = [](Graph& g, const std::vector<Value>& v){ return g.sqrt(v[0]); };
    // rank-1
    T["tanh__5"]       = [](Graph& g, const std::vector<Value>& v){ return g.tanh(v[0]); };
    // rank-3
    T["exp__2x2x3"]    = [](Graph& g, const std::vector<Value>& v){ return g.exp(v[0]); };
    // rank-4
    T["tanh__2x2x2x3"] = [](Graph& g, const std::vector<Value>& v){ return g.tanh(v[0]); };
    // size-1 dim
    T["logistic__1x4"] = [](Graph& g, const std::vector<Value>& v){ return g.logistic(v[0]); };
    T["neg__4x1"]      = [](Graph& g, const std::vector<Value>& v){ return g.neg(v[0]); };
    // non-128-aligned minor dim
    T["abs__2x130"]    = [](Graph& g, const std::vector<Value>& v){ return g.abs(v[0]); };
    T["tanh__3x65"]    = [](Graph& g, const std::vector<Value>& v){ return g.tanh(v[0]); };
}

}  // namespace tpu
