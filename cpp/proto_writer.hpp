// proto_writer.hpp — minimal protobuf wire-format encoder (no libprotobuf).
//
// Supports the subset needed to build xla.CompileOptionsProto natively:
// varint fields (including negative int64), length-delimited submessages and
// raw bytes. See docs/compile_options_format.md for the field map.
#pragma once

#include <cstdint>
#include <string>

namespace tpu {

class ProtoWriter {
  public:
    const std::string& bytes() const { return buf_; }

    void varint(uint64_t v) {
        while (v >= 0x80) { buf_ += (char)(0x80 | (v & 0x7f)); v >>= 7; }
        buf_ += (char)v;
    }

    // field with wire type 0 (varint). Negative int64 encodes as 2^64+v.
    void field_varint(int num, int64_t v) {
        varint(((uint64_t)num << 3) | 0);
        varint((uint64_t)v);
    }
    void field_bool(int num, bool v) { field_varint(num, v ? 1 : 0); }

    // field with wire type 2 (length-delimited), raw payload.
    void field_bytes(int num, const void* data, size_t len) {
        varint(((uint64_t)num << 3) | 2);
        varint(len);
        buf_.append((const char*)data, len);
    }
    void field_msg(int num, const ProtoWriter& sub) {
        field_bytes(num, sub.buf_.data(), sub.buf_.size());
    }

  private:
    std::string buf_;
};

}  // namespace tpu
