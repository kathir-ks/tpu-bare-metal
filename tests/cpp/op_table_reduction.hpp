// op_table_reduction.hpp — builders for reductions. Mirrors cases_reduction.py.
#pragma once
#include "op_table_base.hpp"

namespace tpu {

inline void register_reduction(OpBuilderMap& T) {
    T["reduce_sum_ax0__2x3"]  = [](Graph& g, const std::vector<Value>& v){ return g.reduce_sum(v[0], {0}); };
    T["reduce_sum_ax1__2x3"]  = [](Graph& g, const std::vector<Value>& v){ return g.reduce_sum(v[0], {1}); };
    T["reduce_sum_all__2x3"]  = [](Graph& g, const std::vector<Value>& v){ return g.reduce_sum(v[0], {0, 1}); };
    T["reduce_max_ax1__2x3"]  = [](Graph& g, const std::vector<Value>& v){ return g.reduce_max(v[0], {1}); };
    T["reduce_mean_ax1__2x3"] = [](Graph& g, const std::vector<Value>& v){ return g.reduce_mean(v[0], {1}); };

    // keepdims
    T["reduce_sum_keepdims_ax1__2x3"]  = [](Graph& g, const std::vector<Value>& v){ return g.reduce_sum(v[0], {1}, true); };

    // rank-3, per-axis
    T["reduce_sum_r3_ax0__2x3x4"]  = [](Graph& g, const std::vector<Value>& v){ return g.reduce_sum(v[0], {0}); };
    T["reduce_sum_r3_ax1__2x3x4"]  = [](Graph& g, const std::vector<Value>& v){ return g.reduce_sum(v[0], {1}); };
    T["reduce_sum_r3_ax2__2x3x4"]  = [](Graph& g, const std::vector<Value>& v){ return g.reduce_sum(v[0], {2}); };

    // multi-axis
    T["reduce_sum_r3_ax02__2x3x4"] = [](Graph& g, const std::vector<Value>& v){ return g.reduce_sum(v[0], {0, 2}); };

    // reduce_max over axis 0 of [3,5]
    T["reduce_max_ax0__3x5"]        = [](Graph& g, const std::vector<Value>& v){ return g.reduce_max(v[0], {0}); };

    // reduce_mean rank-3 axis 2
    T["reduce_mean_r3_ax2__2x3x4"]  = [](Graph& g, const std::vector<Value>& v){ return g.reduce_mean(v[0], {2}); };
}

}  // namespace tpu
