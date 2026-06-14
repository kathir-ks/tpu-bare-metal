// tensor.hpp — Dual-nature value-semantics Tensor handle for the eager/jit frontend.
//
// Layer 4 of the pure-C++ TPU stack.  A Tensor is a reference-counted handle to
// an Impl that is EITHER:
//   Concrete — owns a realized tpu::Buffer (device-resident); also carries an
//              optional tape Value for autograd bookkeeping.
//   Traced   — a (Graph*, node_id) pair recorded in an active jit trace; not yet
//              executed; to_host() throws.
//
// Value semantics: copy is cheap (aliases the same Impl); last handle drop frees
// the device buffer.  Assignment rebinds the handle (SSA; no in-place storage).
//
// This header is the contract Wave-B agents (autograd, jit, module, optim) build
// against.  Do not change public signatures without updating all dependents.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph.hpp"   // tpu::Value, tpu::Graph, tpu::Shape, tpu::DType, tpu::Op
#include "tpu.hpp"     // tpu::Buffer, tpu::Context

namespace tpu {

// Forward declarations for dispatch hooks (Wave B / eager.hpp fills these).
class Tape;    // eager tape — a Graph plus bookkeeping; defined in eager.hpp
class Trace;   // jit trace context; defined in jit.hpp (Wave B)

// ── Thread-local dispatch context (Decision 11) ─────────────────────────────
// Declared here (tensor.hpp is included everywhere); defined in eager.hpp.
// Both are null by default; eager.hpp manages their lifetimes.
extern thread_local Tape*  current_tape;   // set while a tape is active
extern thread_local Trace* active_trace;   // set while a jit trace is running

// ── Tensor::Impl ─────────────────────────────────────────────────────────────
// Not part of the public API; exposed only so autograd/jit agents can read Impl
// identity (for capture-set tracking) and rebind buffers (for optim).
struct TensorImpl {
    // ── Nature ────────────────────────────────────────────────────────────────
    bool is_traced = false;  // true → Traced; false → Concrete

    // Concrete path
    std::shared_ptr<Buffer> buf;   // device buffer (null until realized)
    // Tape record (optional even for Concrete; null in no_grad / no-grad path)
    // The Value points into tape_graph below; kept alive by the shared_ptr.
    std::shared_ptr<Graph>  tape_graph;   // shared ownership of the tape Graph
    std::optional<Value>    tape_value;   // node id within tape_graph

    // Traced path
    Graph* trace_graph = nullptr;  // non-owning; owned by the Trace object
    int    trace_node  = -1;       // node id within trace_graph

    // ── Metadata ──────────────────────────────────────────────────────────────
    Shape  shape_;
    DType  dtype_  = DType::F32;
    int    device_ = 0;

    // ── Autograd state ────────────────────────────────────────────────────────
    bool             requires_grad_ = false;
    std::shared_ptr<TensorImpl> grad_;  // accumulated gradient (same shape/dtype)

    // ── Helpers ──────────────────────────────────────────────────────────────
    bool has_tape_node()  const { return tape_value.has_value(); }
    Value tape_val()      const { return *tape_value; }

    bool has_trace_node() const { return is_traced && trace_graph != nullptr && trace_node >= 0; }
    Value trace_val()     const {
        if (!has_trace_node())
            throw Error("TensorImpl: not a traced tensor");
        return Value{trace_graph, trace_node};
    }
};

// ── Tensor ───────────────────────────────────────────────────────────────────
class Tensor {
public:
    // ── Null / uninitialized ──────────────────────────────────────────────────
    Tensor() = default;

    // ── Internal ctor used by eager.hpp and jit.hpp ─────────────────────────
    // Pass ownership of a pre-built Impl.
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

    // ── Copy / move (cheap: just share the Impl) ─────────────────────────────
    Tensor(const Tensor&)            = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&)                 = default;
    Tensor& operator=(Tensor&&)      = default;

    // ── Validity ─────────────────────────────────────────────────────────────
    bool valid() const { return impl_ != nullptr; }
    explicit operator bool() const { return valid(); }

    // ── Metadata ─────────────────────────────────────────────────────────────
    const Shape& shape()  const { check_valid(); return impl_->shape_; }
    DType         dtype()  const { check_valid(); return impl_->dtype_; }
    int           device() const { check_valid(); return impl_->device_; }
    int64_t       rank()   const { check_valid(); return (int64_t)impl_->shape_.size(); }

    int64_t numel() const {
        check_valid();
        return num_elements(impl_->shape_);
    }

    // ── Autograd flags ───────────────────────────────────────────────────────
    bool     requires_grad()             const { check_valid(); return impl_->requires_grad_; }
    Tensor&  requires_grad_(bool v)            { check_valid(); impl_->requires_grad_ = v; return *this; }

    // Accumulated gradient (set by backward()); null Tensor if not yet computed.
    Tensor grad() const {
        check_valid();
        if (!impl_->grad_) return Tensor{};
        return Tensor{impl_->grad_};
    }

    // Set gradient buffer from an already-constructed impl (used by autograd).
    void set_grad(std::shared_ptr<TensorImpl> g) { check_valid(); impl_->grad_ = std::move(g); }

    // Accumulate gradient: grad += delta.  Called by backward().
    // If grad is null, initializes it; otherwise adds element-wise on host.
    // NOTE: host-side accumulation is fine for the eager case (autograd Wave B
    // may replace this with a device add once it can dispatch ops without
    // recursion risk).
    void accumulate_grad(const std::vector<float>& delta);

    // backward() is declared here, implemented in autograd.hpp (Wave B).
    // The declaration must be visible to all users; the body needs the full
    // evaluator, so it's out-of-line in autograd.hpp via explicit instantiation.
    void backward();

    // ── Nature queries (for jit/autograd agents) ─────────────────────────────
    bool is_traced()   const { return impl_ && impl_->is_traced; }
    bool is_concrete() const { return impl_ && !impl_->is_traced; }
    bool has_tape_node() const { return impl_ && impl_->has_tape_node(); }

    // Access to tape / trace nodes (autograd / jit agents only).
    Value tape_value() const {
        if (!impl_ || !impl_->has_tape_node())
            throw Error("Tensor::tape_value: no tape node");
        return impl_->tape_val();
    }
    Value trace_value() const {
        if (!impl_ || !impl_->has_trace_node())
            throw Error("Tensor::trace_value: not a traced tensor");
        return impl_->trace_val();
    }

    // Raw Impl access — for jit capture-set identity tracking (identity by
    // Impl* pointer, not by buffer value).
    TensorImpl*              impl_ptr() const { return impl_.get(); }
    std::shared_ptr<TensorImpl> impl_shared() const { return impl_; }

    // ── Device buffer access (Concrete tensors only) ─────────────────────────
    // Returns a raw (non-owning) pointer for passing to Executable::run().
    Buffer* raw_buffer() const {
        check_concrete("raw_buffer");
        if (!impl_->buf)
            throw Error("Tensor::raw_buffer: buffer not realized");
        return impl_->buf.get();
    }

    // Share ownership of the buffer (for optim to rebind params after a step).
    std::shared_ptr<Buffer> shared_buffer() const {
        check_concrete("shared_buffer");
        return impl_->buf;
    }

    // Rebind this tensor's buffer to a new one (optimizer in-place update).
    // Leaves all other Impl fields (shape, dtype, tape node, grad) unchanged.
    void rebind_buffer(std::shared_ptr<Buffer> new_buf) {
        check_concrete("rebind_buffer");
        impl_->buf = std::move(new_buf);
    }

    // ── Download to host ─────────────────────────────────────────────────────
    // Forces reading the device buffer.  Throws if Traced (reads inside jit
    // are forbidden — they cause an implicit graph break).
    std::vector<float> to_host() const {
        if (is_traced())
            throw Error("Tensor::to_host: cannot read a traced tensor inside jit");
        check_concrete("to_host");
        if (!impl_->buf)
            throw Error("Tensor::to_host: buffer not realized");
        return impl_->buf->to_host<float>();
    }

    // ── Arithmetic operators (routed through the eager dispatch) ─────────────
    // Implementations are in eager.hpp (included after this class definition).
    Tensor operator+(const Tensor& o) const;
    Tensor operator-(const Tensor& o) const;
    Tensor operator*(const Tensor& o) const;
    Tensor operator/(const Tensor& o) const;
    Tensor operator-() const;

    // Scalar overloads (broadcast scalar to tensor shape).
    Tensor operator+(double s) const;
    Tensor operator-(double s) const;
    Tensor operator*(double s) const;
    Tensor operator/(double s) const;

    // Compound-assignment: rebind, not in-place storage mutation.
    Tensor& operator+=(const Tensor& o) { *this = *this + o; return *this; }
    Tensor& operator-=(const Tensor& o) { *this = *this - o; return *this; }
    Tensor& operator*=(const Tensor& o) { *this = *this * o; return *this; }
    Tensor& operator/=(const Tensor& o) { *this = *this / o; return *this; }

private:
    std::shared_ptr<TensorImpl> impl_;

    void check_valid() const {
        if (!impl_) throw Error("Tensor: null handle");
    }
    void check_concrete(const char* who) const {
        check_valid();
        if (impl_->is_traced)
            throw Error(std::string("Tensor::") + who + ": tensor is Traced (inside jit)");
    }
};

// Free-function scalar operators (s OP tensor).
inline Tensor operator+(double s, const Tensor& t) { return t + s; }
inline Tensor operator*(double s, const Tensor& t) { return t * s; }
inline Tensor operator-(double s, const Tensor& t);   // defined in eager.hpp

}  // namespace tpu

// ── eager.hpp is included here so operator bodies are visible everywhere ─────
// (eager.hpp includes tensor.hpp guard-protected, so the include is safe)
#include "eager.hpp"

namespace tpu {

// ── Tensor::accumulate_grad (host-side, no device recursion) ─────────────────
inline void Tensor::accumulate_grad(const std::vector<float>& delta) {
    check_valid();
    auto& ctx = global_context();
    if (!impl_->grad_) {
        // First accumulation: grad = 0 + delta = delta. Just upload delta.
        // (The previous version created a stray Buffer from upload(...).raw(),
        // which double-freed the device buffer — the temporary upload Buffer
        // freed the tpu_buf_t* at end of statement while the wrapper still owned
        // it. Fixed by uploading delta directly into the grad buffer.)
        impl_->grad_ = std::make_shared<TensorImpl>();
        impl_->grad_->shape_  = impl_->shape_;
        impl_->grad_->dtype_  = impl_->dtype_;
        impl_->grad_->device_ = impl_->device_;
        Buffer tmp = ctx.upload_f32(delta, impl_->shape_, impl_->device_);
        impl_->grad_->buf = std::make_shared<Buffer>(std::move(tmp));
    } else {
        // download existing grad, add delta, re-upload
        auto existing = impl_->grad_->buf->to_host<float>();
        for (size_t i = 0; i < existing.size() && i < delta.size(); ++i)
            existing[i] += delta[i];
        Buffer tmp = ctx.upload_f32(existing, impl_->shape_, impl_->device_);
        impl_->grad_->buf = std::make_shared<Buffer>(std::move(tmp));
    }
}

}  // namespace tpu
