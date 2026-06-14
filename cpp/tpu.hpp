// tpu.hpp — RAII C++ wrapper over the C PJRT framework (framework/tpu.h).
//
// This is layer 1 of the pure-C++ TPU stack. It exposes:
//   tpu::Context     — process-wide PJRT client (RAII over tpu_ctx_t)
//   tpu::Buffer      — a device-resident tensor (RAII over tpu_buf_t)
//   tpu::Executable  — a compiled StableHLO program (RAII over tpu_exec_t)
//
// Higher layers (graph.hpp, nn.hpp) build StableHLO text and feed it here.
//
// Errors are reported as exceptions (tpu::Error). Link with:
//   -I framework -I cpp  ... cpp/*.cpp framework/libtpu_fw.a -ldl -lm
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "tpu.h"
}

#include "compile_opts.hpp"   // native CompileOptionsProto construction

namespace tpu {

// ── Errors ──────────────────────────────────────────────────────────────────
struct Error : std::runtime_error {
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

inline void check(bool ok, const std::string& what) {
    if (!ok) throw Error(what);
}

// ── DType ───────────────────────────────────────────────────────────────────
enum class DType : int {
    Invalid = TPU_DTYPE_INVALID,
    Pred    = TPU_DTYPE_PRED,
    S8      = TPU_DTYPE_S8,
    S16     = TPU_DTYPE_S16,
    S32     = TPU_DTYPE_S32,
    S64     = TPU_DTYPE_S64,
    U8      = TPU_DTYPE_U8,
    U16     = TPU_DTYPE_U16,
    U32     = TPU_DTYPE_U32,
    F16     = TPU_DTYPE_F16,
    F32     = TPU_DTYPE_F32,
    F64     = TPU_DTYPE_F64,
    BF16    = TPU_DTYPE_BF16,
};

inline size_t dtype_size(DType d) { return tpu_dtype_size((tpu_dtype_t)d); }

// StableHLO element-type spelling, e.g. "f32", "i32", "bf16", "i1".
inline const char* mlir_dtype(DType d) {
    switch (d) {
        case DType::Pred: return "i1";
        case DType::S8:   return "i8";
        case DType::S16:  return "i16";
        case DType::S32:  return "i32";
        case DType::S64:  return "i64";
        case DType::U8:   return "ui8";
        case DType::U16:  return "ui16";
        case DType::U32:  return "ui32";
        case DType::F16:  return "f16";
        case DType::F32:  return "f32";
        case DType::F64:  return "f64";
        case DType::BF16: return "bf16";
        default:          return "f32";
    }
}

using Shape = std::vector<int64_t>;

inline int64_t num_elements(const Shape& s) {
    int64_t n = 1;
    for (int64_t d : s) n *= d;
    return n;
}

// ── Buffer ──────────────────────────────────────────────────────────────────
class Buffer {
  public:
    Buffer() = default;
    explicit Buffer(tpu_buf_t* b) : buf_(b) {}
    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept : buf_(o.buf_) { o.buf_ = nullptr; }
    Buffer& operator=(Buffer&& o) noexcept {
        if (this != &o) { reset(); buf_ = o.buf_; o.buf_ = nullptr; }
        return *this;
    }
    ~Buffer() { reset(); }

    void reset() {
        if (buf_) { tpu_buf_free(buf_); buf_ = nullptr; }
    }

    tpu_buf_t* raw() const { return buf_; }
    explicit operator bool() const { return buf_ != nullptr; }

    size_t size_bytes() const { return tpu_buf_size(buf_); }
    DType  dtype() const { return (DType)tpu_buf_dtype(buf_); }

    Shape shape() const {
        int64_t dims[8];
        size_t  n = 0;
        check(tpu_buf_dims(buf_, dims, &n) == 0, "tpu_buf_dims failed");
        return Shape(dims, dims + n);
    }

    // Copy device buffer to a host vector of T.
    //
    // Size the host vector by the LOGICAL element count (product of dims), NOT
    // size_bytes(): size_bytes() reports the on-device tiled/padded byte size, so
    // for a tensor whose minor dim is not 128-aligned (e.g. a 2x2) it over-counts
    // (2x2 -> 2x128 = 256 floats).  tpu_download pins a dense row-major host
    // layout and delivers exactly the logical elements, so sizing by the logical
    // count returns them unpadded.  Aligned tensors are unaffected (padded ==
    // logical), so existing callers see identical results.
    template <typename T>
    std::vector<T> to_host() const {
        Shape s = shape();
        size_t n = 1;
        for (int64_t d : s) n *= (size_t)d;
        std::vector<T> out(n);
        check(tpu_download(buf_, out.data(), out.size() * sizeof(T)) == 0,
              "tpu_download failed");
        return out;
    }

    void download(void* dst, size_t n) const {
        check(tpu_download(buf_, dst, n) == 0, "tpu_download failed");
    }

  private:
    tpu_buf_t* buf_ = nullptr;
};

// ── Executable ──────────────────────────────────────────────────────────────
class Executable {
  public:
    Executable() = default;
    explicit Executable(tpu_exec_t* e) : exec_(e) {}
    Executable(const Executable&)            = delete;
    Executable& operator=(const Executable&) = delete;
    Executable(Executable&& o) noexcept : exec_(o.exec_) { o.exec_ = nullptr; }
    Executable& operator=(Executable&& o) noexcept {
        if (this != &o) { reset(); exec_ = o.exec_; o.exec_ = nullptr; }
        return *this;
    }
    ~Executable() { reset(); }

    void reset() {
        if (exec_) { tpu_exec_free(exec_); exec_ = nullptr; }
    }

    tpu_exec_t* raw() const { return exec_; }
    int num_outputs() const { return tpu_exec_num_outputs(exec_); }

    // device_order[r] = addressable device index the executable assigns to
    // replica r. Replica r's input buffers must live on that device.
    std::vector<int> device_order(int nreplicas) {
        std::vector<int> order(nreplicas, 0);
        check(tpu_exec_device_order(exec_, order.data(), (size_t)nreplicas) == 0,
              "tpu_exec_device_order failed");
        return order;
    }

    // Run on device dev_idx with the given inputs; returns all outputs.
    std::vector<Buffer> run(const std::vector<Buffer*>& args, int dev_idx = 0) {
        std::vector<tpu_buf_t*> raw_args;
        raw_args.reserve(args.size());
        for (auto* a : args) raw_args.push_back(a->raw());

        int no = num_outputs();
        std::vector<tpu_buf_t*> raw_out(no, nullptr);
        int rc = tpu_run(exec_, dev_idx, raw_args.data(), raw_args.size(),
                         raw_out.data(), (size_t)no);
        check(rc == 0, "tpu_run failed");
        std::vector<Buffer> out;
        out.reserve(no);
        for (int i = 0; i < no; i++) out.emplace_back(raw_out[i]);
        return out;
    }

    void save(const std::string& path) {
        check(tpu_exec_save(exec_, path.c_str()) == 0, "tpu_exec_save failed");
    }

    // Replicated (SPMD) execution across ndevs devices.
    // args[dev][i] are the input buffers on device dev. Returns out[dev][j].
    std::vector<std::vector<Buffer>> run_spmd(
            const std::vector<std::vector<Buffer*>>& args) {
        size_t ndevs = args.size();
        check(ndevs > 0, "run_spmd: no devices");
        size_t nargs = args[0].size();
        int no = num_outputs();

        std::vector<std::vector<tpu_buf_t*>> raw(ndevs);
        std::vector<tpu_buf_t* const*> arg_lists(ndevs);
        for (size_t d = 0; d < ndevs; d++) {
            raw[d].resize(nargs);
            for (size_t i = 0; i < nargs; i++) raw[d][i] = args[d][i]->raw();
            arg_lists[d] = raw[d].data();
        }

        tpu_buf_t** out_flat = nullptr;  // framework allocates [ndevs*noutputs]
        int rc = tpu_run_spmd(exec_, arg_lists.data(), ndevs, nargs,
                              &out_flat, (size_t)no);
        check(rc == 0, "tpu_run_spmd failed");

        std::vector<std::vector<Buffer>> out(ndevs);
        for (size_t d = 0; d < ndevs; d++) {
            out[d].reserve(no);
            for (int j = 0; j < no; j++) out[d].emplace_back(out_flat[d * no + j]);
        }
        // out_flat held tpu_buf_t* now owned by Buffer wrappers; free the array
        // shell only (tpu_spmd_outputs_free would also destroy the buffers).
        free(out_flat);
        return out;
    }

  private:
    tpu_exec_t* exec_ = nullptr;
};

// ── Context ─────────────────────────────────────────────────────────────────
class Context {
  public:
    explicit Context(const char* lib_path = nullptr) {
        ctx_ = tpu_init(lib_path);
        if (!ctx_) {
            const char* e = tpu_strerror(nullptr);
            throw Error(std::string("tpu_init failed: ") + (e ? e : "unknown"));
        }
    }
    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;
    ~Context() { if (ctx_) tpu_destroy(ctx_); }

    tpu_ctx_t* raw() const { return ctx_; }

    int num_devices() const { return tpu_num_devices(ctx_); }
    int num_addressable_devices() const { return tpu_num_addressable_devices(ctx_); }

    std::string topology() const {
        char* s = tpu_topology_string(ctx_);
        if (!s) return {};
        std::string out(s);
        free(s);
        return out;
    }

    tpu_device_info_t device_info(int idx) const {
        tpu_device_info_t info;
        check(tpu_device_info(ctx_, idx, &info) == 0, "tpu_device_info failed");
        return info;
    }

    std::string last_error() const {
        const char* e = tpu_strerror(ctx_);
        return e ? e : "";
    }

    // ── Compile StableHLO text (the "mlir" path) ────────────────────────────
    // With opts==nullptr, builds default options natively (num_replicas=1,
    // num_partitions=1) — PJRT rejects an empty CompileOptions as (0,0).
    Executable compile_mlir(const std::string& mlir,
                            const void* opts = nullptr, size_t opts_sz = 0) {
        std::string def;
        if (!opts) {
            def = make_compile_options(1);
            opts = def.data(); opts_sz = def.size();
        }
        tpu_exec_t* e = tpu_compile_buf(ctx_, mlir.data(), mlir.size(), "mlir",
                                        opts, opts_sz);
        if (!e) throw Error("compile_mlir failed: " + last_error());
        return Executable(e);
    }

    // Compile a replicated (data-parallel) StableHLO program with one replica
    // per addressable device (or an explicit count). Pair with
    // Graph::num_replicas and all_reduce_sum.
    Executable compile_mlir_dp(const std::string& mlir, int num_replicas = 0) {
        if (num_replicas <= 0) num_replicas = num_addressable_devices();
        std::string opts = make_compile_options(num_replicas);
        tpu_exec_t* e = tpu_compile_buf(ctx_, mlir.data(), mlir.size(), "mlir",
                                        opts.data(), opts.size());
        if (!e) throw Error("compile_mlir_dp failed: " + last_error());
        return Executable(e);
    }

    // Compile a binary HLO protobuf.
    Executable compile_hlo(const void* code, size_t code_sz,
                           const void* opts = nullptr, size_t opts_sz = 0) {
        tpu_exec_t* e = tpu_compile_buf(ctx_, code, code_sz, "hlo", opts, opts_sz);
        if (!e) throw Error("compile_hlo failed: " + last_error());
        return Executable(e);
    }

    // ── Upload host data → device buffer ────────────────────────────────────
    Buffer upload(const void* data, DType dtype, const Shape& dims, int dev_idx = 0) {
        tpu_buf_t* b = tpu_upload(ctx_, dev_idx, data, (tpu_dtype_t)dtype,
                                  dims.data(), dims.size());
        if (!b) throw Error("upload failed: " + last_error());
        return Buffer(b);
    }

    Buffer upload_f32(const float* data, const Shape& dims, int dev_idx = 0) {
        return upload(data, DType::F32, dims, dev_idx);
    }
    Buffer upload_f32(const std::vector<float>& data, const Shape& dims, int dev_idx = 0) {
        return upload(data.data(), DType::F32, dims, dev_idx);
    }
    Buffer upload_s32(const std::vector<int32_t>& data, const Shape& dims, int dev_idx = 0) {
        return upload(data.data(), DType::S32, dims, dev_idx);
    }

  private:
    tpu_ctx_t* ctx_ = nullptr;
};

}  // namespace tpu
