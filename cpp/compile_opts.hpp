// compile_opts.hpp — native construction of xla.CompileOptionsProto.
//
// Replaces the JAX-generated compile_opts*.pb blobs: builds the serialized
// proto for any (num_replicas, num_partitions, use_spmd_partitioning).
// Field numbers and ordering follow docs/compile_options_format.md so that
// the default configs re-encode byte-identically to the original fixtures
// (which remain in the repo as oracle test data).
#pragma once

#include "proto_writer.hpp"
#include "debug_opts_blob.h"

namespace tpu {

struct CompileOptsCfg {
    int  num_replicas        = 1;
    int  num_partitions      = 1;
    bool use_spmd_partitioning = false;
    bool include_debug_options = true;  // embed default DebugOptions blob
};

inline std::string make_compile_options(const CompileOptsCfg& cfg) {
    ProtoWriter ebo;  // ExecutableBuildOptionsProto
    ebo.field_varint(1, -1);  // device_ordinal
    if (cfg.include_debug_options)
        ebo.field_bytes(3, k_debug_options_blob, k_debug_options_blob_len);
    ebo.field_varint(4, cfg.num_replicas);
    ebo.field_varint(5, cfg.num_partitions);
    if (cfg.use_spmd_partitioning)
        ebo.field_bool(6, true);
    static const unsigned char zero_byte[] = {0x00};
    ebo.field_bytes(12, zero_byte, 1);  // defaults, per fixture (payload = 0x00)
    ebo.field_bytes(18, zero_byte, 1);
    ebo.field_varint(23, 1);

    ProtoWriter co;  // CompileOptionsProto
    co.field_msg(3, ebo);
    return co.bytes();
}

inline std::string make_compile_options(int num_replicas, int num_partitions = 1,
                                         bool use_spmd = false) {
    CompileOptsCfg cfg;
    cfg.num_replicas = num_replicas;
    cfg.num_partitions = num_partitions;
    cfg.use_spmd_partitioning = use_spmd;
    return make_compile_options(cfg);
}

}  // namespace tpu
