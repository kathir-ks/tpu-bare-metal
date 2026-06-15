// fixture.hpp — hermetic reader for the binary golden-vector format written by the
// offline JAX oracle (tests/oracle/fixture.py). No Python at test time: load a .fix
// file into named tensors, then assert against the framework's on-device results.
//
// Format (little-endian) — must match tests/oracle/fixture.py exactly:
//   "FIXV" | u32 version=1 | u32 n_tensors |
//   repeated: u32 name_len | name | u32 dtype(0=f32,1=i32) | u32 rank | i64 dims[rank] | data
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "tpu.hpp"

namespace tpu {

struct FixTensor {
    DType                dtype = DType::F32;
    Shape                dims;
    std::vector<float>   f32;   // populated when dtype==F32
    std::vector<int32_t> i32;   // populated when dtype==S32
    int64_t count() const { return num_elements(dims); }
};

class Fixture {
  public:
    explicit Fixture(const std::string& path) { load(path); }

    bool has(const std::string& name) const { return tensors_.count(name) > 0; }

    const FixTensor& at(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) throw Error("fixture: missing tensor '" + name + "'");
        return it->second;
    }

    // scalar f32 (e.g. tolerances); default if absent.
    float scalar(const std::string& name, float dflt) const {
        if (!has(name)) return dflt;
        const auto& t = at(name);
        return t.f32.empty() ? dflt : t.f32[0];
    }

    int num_inputs() const {
        int n = 0;
        while (has("in" + std::to_string(n))) n++;
        return n;
    }

  private:
    std::map<std::string, FixTensor> tensors_;

    static uint32_t rd_u32(FILE* f) {
        uint32_t v; if (fread(&v, 4, 1, f) != 1) throw Error("fixture: short read (u32)");
        return v;
    }
    static int64_t rd_i64(FILE* f) {
        int64_t v; if (fread(&v, 8, 1, f) != 1) throw Error("fixture: short read (i64)");
        return v;
    }

    void load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw Error("fixture: cannot open " + path);
        char magic[4];
        if (fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "FIXV", 4) != 0) {
            fclose(f); throw Error("fixture: bad magic in " + path);
        }
        uint32_t ver = rd_u32(f);
        if (ver != 1) { fclose(f); throw Error("fixture: unsupported version"); }
        uint32_t n = rd_u32(f);
        for (uint32_t t = 0; t < n; t++) {
            uint32_t nl = rd_u32(f);
            std::string name(nl, '\0');
            if (nl && fread(&name[0], 1, nl, f) != nl) { fclose(f); throw Error("fixture: short name"); }
            uint32_t dt = rd_u32(f);
            uint32_t rank = rd_u32(f);
            FixTensor ft;
            ft.dims.resize(rank);
            int64_t cnt = 1;
            for (uint32_t i = 0; i < rank; i++) { ft.dims[i] = rd_i64(f); cnt *= ft.dims[i]; }
            if (dt == 0) {
                ft.dtype = DType::F32; ft.f32.resize(cnt);
                if (cnt && fread(ft.f32.data(), 4, cnt, f) != (size_t)cnt) { fclose(f); throw Error("fixture: short f32 data"); }
            } else if (dt == 1) {
                ft.dtype = DType::S32; ft.i32.resize(cnt);
                if (cnt && fread(ft.i32.data(), 4, cnt, f) != (size_t)cnt) { fclose(f); throw Error("fixture: short i32 data"); }
            } else {
                fclose(f); throw Error("fixture: unknown dtype code");
            }
            tensors_[name] = std::move(ft);
        }
        fclose(f);
    }
};

}  // namespace tpu
