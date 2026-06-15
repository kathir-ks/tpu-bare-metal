// op_table.hpp — master op-table: merges every per-family builder header into one
// map keyed by fixture key. The conformance / gradient suites parse a fixture's key,
// look up the builder here, feed it the fixture inputs as graph Inputs, and compare
// the on-device result to the golden output/gradient.
//
// Coverage is extended in the per-family files (op_table_<family>.hpp + the matching
// tests/oracle/cases_<family>.py). Add a new family by including it and calling its
// register_<family>() below.
#pragma once

#include "op_table_base.hpp"
#include "op_table_unary.hpp"
#include "op_table_binary.hpp"
#include "op_table_reduction.hpp"
#include "op_table_matmul.hpp"
#include "op_table_shape.hpp"
#include "op_table_gather.hpp"
#include "op_table_misc.hpp"

namespace tpu {

inline const OpBuilderMap& op_table() {
    static const OpBuilderMap T = [] {
        OpBuilderMap m;
        register_unary(m);
        register_binary(m);
        register_reduction(m);
        register_matmul(m);
        register_shape(m);
        register_gather(m);
        register_misc(m);
        return m;
    }();
    return T;
}

inline bool op_table_has(const std::string& key) { return op_table().count(key) > 0; }

}  // namespace tpu
