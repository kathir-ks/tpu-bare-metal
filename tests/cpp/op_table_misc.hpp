// op_table_misc.hpp — builders for misc ops (StopGradient, Select, Compare,
// Convert, Iota). Mirrors cases_misc.py. For Compare cases, convert the i1 result
// to f32 in the builder so it matches the oracle's float `out`.
#pragma once
#include "op_table_base.hpp"

namespace tpu {

inline void register_misc(OpBuilderMap& T) {
    T["stop_gradient__2x3"] = [](Graph& g, const std::vector<Value>& v){ return g.stop_gradient(v[0]); };

    T["compare_gt__2x3"] = [](Graph& g, const std::vector<Value>& v){
        return g.convert(g.compare(v[0], v[1], Cmp::GT), DType::F32);
    };

    T["select_pos__2x3"] = [](Graph& g, const std::vector<Value>& v){
        Value pred = g.compare(v[0], g.scalar(0.0), Cmp::GT);
        return g.select(pred, v[0], v[1]);
    };

    T["convert_id__2x3"] = [](Graph& g, const std::vector<Value>& v){
        return g.convert(v[0], DType::F32);
    };

    T["iota__6"] = [](Graph& g, const std::vector<Value>& /*v*/){
        return g.iota({6}, 0, DType::F32);
    };
}

}  // namespace tpu
