// op_table_gather.hpp — builders for row gather/scatter. Mirrors cases_gather.py.
// in0 = table (float), in1 = ids (int32). gather_rows looks up rows of the table.
#pragma once
#include "op_table_base.hpp"

namespace tpu {

inline void register_gather(OpBuilderMap& T) {
    T["gather_rows__5x4_ids3"]              = [](Graph& g, const std::vector<Value>& v){ return g.gather_rows(v[0], v[1]); };
    T["gather_rows_dupes__5x4_ids4"]        = [](Graph& g, const std::vector<Value>& v){ return g.gather_rows(v[0], v[1]); };
    T["gather_rows__rank2ids__5x4_ids2x3"]  = [](Graph& g, const std::vector<Value>& v){ return g.gather_rows(v[0], v[1]); };
    T["gather_rows__larger__10x8_ids6"]     = [](Graph& g, const std::vector<Value>& v){ return g.gather_rows(v[0], v[1]); };
    T["gather_rows__rank3table__5x2x3_ids4"]= [](Graph& g, const std::vector<Value>& v){ return g.gather_rows(v[0], v[1]); };
    T["gather_rows__dupes_accum__5x4_ids4"] = [](Graph& g, const std::vector<Value>& v){ return g.gather_rows(v[0], v[1]); };
}

}  // namespace tpu
