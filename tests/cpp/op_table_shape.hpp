// op_table_shape.hpp — builders for shape ops. Mirrors cases_shape.py.
#pragma once
#include "op_table_base.hpp"

namespace tpu {

inline void register_shape(OpBuilderMap& T) {
    // existing cases
    T["transpose__2x3"]      = [](Graph& g, const std::vector<Value>& v){ return g.transpose(v[0], {1, 0}); };
    T["reshape__2x3_to_6"]   = [](Graph& g, const std::vector<Value>& v){ return g.reshape(v[0], {6}); };
    T["broadcast__3_to_2x3"] = [](Graph& g, const std::vector<Value>& v){ return g.broadcast_to(v[0], {2, 3}); };

    // new cases
    // rank-3 transpose [2,3,4] with perm (2,0,1) -> [4,2,3]
    T["transpose__2x3x4_perm201"]  = [](Graph& g, const std::vector<Value>& v){ return g.transpose(v[0], {2, 0, 1}); };

    // transpose_last2 of [2,3,4] (swap last two dims -> [2,4,3])
    T["transpose_last2__2x3x4"]    = [](Graph& g, const std::vector<Value>& v){ return g.transpose_last2(v[0]); };

    // reshape [2,3,4] -> [6,4]
    T["reshape__2x3x4_to_6x4"]     = [](Graph& g, const std::vector<Value>& v){ return g.reshape(v[0], {6, 4}); };

    // reshape [2,3,4] -> [24]
    T["reshape__2x3x4_to_24"]      = [](Graph& g, const std::vector<Value>& v){ return g.reshape(v[0], {24}); };

    // broadcast [1,4] -> [3,4]
    T["broadcast__1x4_to_3x4"]     = [](Graph& g, const std::vector<Value>& v){ return g.broadcast_to(v[0], {3, 4}); };

    // broadcast scalar-row [3] -> [2,3]
    T["broadcast__3_to_2x3_row"]   = [](Graph& g, const std::vector<Value>& v){ return g.broadcast_to(v[0], {2, 3}); };

    // non-128-aligned: reshape [2,130] -> [260]
    T["reshape__2x130_to_260"]      = [](Graph& g, const std::vector<Value>& v){ return g.reshape(v[0], {260}); };

    // slice [4,4] -> [2,3]  (rows 1:3, cols 0:3)
    T["slice__4x4_to_2x3"]          = [](Graph& g, const std::vector<Value>& v){ return g.slice(v[0], {1, 0}, {3, 3}); };
    // pad [2,3] -> [4,5]  (low=(1,0), high=(1,2))
    T["pad__2x3_to_4x5"]            = [](Graph& g, const std::vector<Value>& v){ return g.pad(v[0], {1, 0}, {1, 2}); };
}

}  // namespace tpu
