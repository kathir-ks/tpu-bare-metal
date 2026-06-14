// eager.hpp — Eager dispatch, single-op kernel cache, and dispatch context.
//
// Layer 4b of the pure-C++ TPU stack (pair with tensor.hpp).
// Implements Decisions 2, 9, and 11 from the eager-jit-tensor-core design.
//
// What lives here:
//   global_context()     — lazy process-wide tpu::Context (Decision 9)
//   Tape                 — eager tape graph (thread-local current_tape)
//   Trace (forward decl) — jit trace seam (thread-local active_trace, Wave B)
//   NoGrad               — RAII scope guard (Decision 11)
//   EagerKernelCache     — (Op, shapes, dtypes, attrs, precision) → Executable
//   eager_dispatch()     — 1-op compile+execute path; returns a Concrete Tensor
//   from_host / zeros / ones / randn / full — leaf tensor constructors
//   matmul / relu / gelu / add / sub / mul / div — free-function op entry points
//   Tensor operator bodies (Decision 1/2 interface)
//
// Include order: graph.hpp → tpu.hpp → tensor.hpp → eager.hpp (tensor.hpp pulls
// us in at its end; never include eager.hpp directly before tensor.hpp).
#pragma once
#ifndef TPU_EAGER_HPP
#define TPU_EAGER_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// graph.hpp + tpu.hpp are already included via tensor.hpp (or the top of this
// TU).  Guard against double-include with the standard pragma.
#include "graph.hpp"
#include "tpu.hpp"

namespace tpu {

// ─────────────────────────────────────────────────────────────────────────────
// Forward declarations  (TensorImpl is defined in tensor.hpp which includes us;
// we need only the class name here since we use shared_ptr<TensorImpl>.)
// ─────────────────────────────────────────────────────────────────────────────
struct TensorImpl;
class  Tensor;      // defined in tensor.hpp (included before this point)
class  Tape;        // defined below
class  Trace;       // defined in jit.hpp (Wave B) — leave as forward decl seam

// ─────────────────────────────────────────────────────────────────────────────
// Thread-local dispatch context (Decision 11)  — definitions (declarations are
// in tensor.hpp so every #include sees them).
// ─────────────────────────────────────────────────────────────────────────────
inline thread_local Tape*  current_tape  = nullptr;
inline thread_local Trace* active_trace  = nullptr;

// ─────────────────────────────────────────────────────────────────────────────
// Precision knob (shared by eager and jit; default HIGHEST for correctness)
// ─────────────────────────────────────────────────────────────────────────────
inline std::string& default_dot_precision() {
    static std::string p = "HIGHEST";
    return p;
}
inline void set_dot_precision(const std::string& p) { default_dot_precision() = p; }

// ─────────────────────────────────────────────────────────────────────────────
// Process-wide PJRT Context (Decision 9)
// ─────────────────────────────────────────────────────────────────────────────
inline Context& global_context() {
    static Context ctx;   // constructed on first call; tpu_init path auto-selected
    return ctx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tape — an eager tape graph (the autograd record for one forward pass).
// current_tape is set to a Tape while we are recording; ops append nodes here.
// After loss.backward() (Wave B) the tape is reset and re-seeded.
// ─────────────────────────────────────────────────────────────────────────────
class Tape {
public:
    Tape()  { graph_ = std::make_shared<Graph>(); }

    // The underlying IR graph (shared ownership so TensorImpl tape_graph keeps
    // a reference; the nodes stay valid as long as any Tensor holds them).
    std::shared_ptr<Graph> graph() { return graph_; }

    // Record a new node on the tape; returns the Value handle.
    // Op and all fields are already set in the Node passed by the caller.
    // (graph_->add is private; we go through the public builder API instead,
    //  so callers must use the Graph builder methods and pass back the Value.)
    // This method exists for future extension (e.g. recording custom metadata).
    void record(const Value& v) { (void)v; /* node already appended to graph_ */ }

    // Reset: drop the old graph, create a fresh one.  Called after backward().
    // Also clears node_buffers_ and leaf records so the new step is clean.
    void reset() {
        graph_ = std::make_shared<Graph>();
        node_buffers_.clear();
        leaves_.clear();
        leaf_impls_.clear();
    }

    // ── Forward-activation buffer retention (Gotcha A) ───────────────────────
    // Each forward node's output Buffer is stored here keyed by its tape node
    // id.  The memoizing evaluator in autograd.hpp seeds its memo table from
    // this map so that forward activations are not recomputed during backward.
    void set_buffer(int node_id, std::shared_ptr<Buffer> buf) {
        node_buffers_[node_id] = std::move(buf);
    }
    std::shared_ptr<Buffer> get_buffer(int node_id) const {
        auto it = node_buffers_.find(node_id);
        return (it != node_buffers_.end()) ? it->second : nullptr;
    }

    // Collect every Value in the tape whose TensorImpl has requires_grad=true.
    // Used by backward() (Wave B) to identify leaf accumulation targets.
    // Leaves register themselves by calling tape.add_leaf().
    //
    // Extended (Gotcha A): also record the TensorImpl so backward() can call
    // accumulate_grad on the right Tensor object.
    void add_leaf(const Value& v, std::shared_ptr<TensorImpl> impl) {
        leaves_.push_back(v);
        leaf_impls_.push_back(std::move(impl));
    }
    // Compat overload for any caller that only has a Value (no grad accumulation
    // needed — e.g. non-requires_grad inputs that still need tape recording).
    void add_leaf_no_grad(const Value& v) {
        leaves_no_grad_.push_back(v);
    }

    const std::vector<Value>&                        leaves()      const { return leaves_; }
    const std::vector<std::shared_ptr<TensorImpl>>&  leaf_impls()  const { return leaf_impls_; }
    void clear_leaves() {
        leaves_.clear();
        leaf_impls_.clear();
        leaves_no_grad_.clear();
    }

private:
    std::shared_ptr<Graph>                          graph_;
    std::vector<Value>                              leaves_;        // requires_grad leaf nodes
    std::vector<std::shared_ptr<TensorImpl>>        leaf_impls_;    // parallel to leaves_
    std::vector<Value>                              leaves_no_grad_; // taped but no grad
    // Forward activation buffers retained for backward (keyed by tape node id).
    std::unordered_map<int, std::shared_ptr<Buffer>> node_buffers_;
};

// ─────────────────────────────────────────────────────────────────────────────
// NoGrad — RAII scope guard (Decision 11)
// Sets current_tape = nullptr for its lifetime: ops execute but do not tape.
// ─────────────────────────────────────────────────────────────────────────────
class NoGrad {
public:
    NoGrad()  : saved_(current_tape) { current_tape = nullptr; }
    ~NoGrad() { current_tape = saved_; }
    // Non-copyable, non-movable.
    NoGrad(const NoGrad&)            = delete;
    NoGrad& operator=(const NoGrad&) = delete;
private:
    Tape* saved_;
};

// ─────────────────────────────────────────────────────────────────────────────
// EagerKernelCache — (Op, in-shapes, in-dtypes, attrs, num_replicas, precision)
//                    → compiled Executable.
// Simple LRU with a configurable capacity cap.
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

// Serialise a cache key to a string for use in the unordered_map.
inline std::string make_cache_key(Op op,
                                   const std::vector<Shape>& in_shapes,
                                   const std::vector<DType>& in_dtypes,
                                   const std::vector<int64_t>& attrs,
                                   double fattr,
                                   int    num_replicas,
                                   const std::string& precision,
                                   const Shape& out_shape)
{
    std::ostringstream oss;
    oss << (int)op << '|';
    for (auto& sh : in_shapes) {
        for (auto d : sh) oss << d << ',';
        oss << ';';
    }
    oss << '|';
    for (auto dt : in_dtypes) oss << (int)dt << ',';
    oss << '|';
    for (auto a : attrs) oss << a << ',';
    // Output shape MUST be part of the key: for shape-determining ops (Broadcast,
    // Reshape, Iota, Constant) the result shape is NOT inferable from inputs+attrs,
    // so two such ops with identical inputs/attrs but different output shapes would
    // otherwise collide and return the wrong executable (a real cross-op bug).
    oss << '|';
    for (auto d : out_shape) oss << d << ',';
    oss << '|' << fattr << '|' << num_replicas << '|' << precision;
    return oss.str();
}

struct CacheEntry {
    std::string      key;
    Executable       exec;       // move-only; lives in the list node
};

}  // namespace detail

class EagerKernelCache {
public:
    explicit EagerKernelCache(size_t cap = 256) : cap_(cap) {}

    // Returns pointer to cached executable, or nullptr on miss.
    Executable* get(const std::string& key) {
        auto it = index_.find(key);
        if (it == index_.end()) return nullptr;
        // Move hit to front (LRU).
        lru_.splice(lru_.begin(), lru_, it->second);
        return &lru_.front().exec;
    }

    // Insert a new entry (after a cache miss + compile).  Takes ownership.
    void put(const std::string& key, Executable exec) {
        if (index_.count(key)) return;  // already present (shouldn't happen)
        if (lru_.size() >= cap_) {
            // evict LRU tail
            index_.erase(lru_.back().key);
            lru_.pop_back();
        }
        lru_.push_front(detail::CacheEntry{key, std::move(exec)});
        index_[key] = lru_.begin();
    }

    size_t size() const { return lru_.size(); }

private:
    size_t cap_;
    std::list<detail::CacheEntry>                                   lru_;
    std::unordered_map<std::string,
                       std::list<detail::CacheEntry>::iterator>     index_;
};

// Process-wide singleton cache (not thread-safe — v1 single-thread assumption).
inline EagerKernelCache& kernel_cache() {
    static EagerKernelCache cache;
    return cache;
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-op lowering helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

// Given the canonical node (already appended to some graph — could be tape or
// a one-shot graph), build a tiny 1-op Graph that reproduces exactly that node,
// compile it (with caching), execute it with the provided buffers, and return
// the output Buffer.
//
// strategy:
//   1. Build a fresh 1-op Graph with Input nodes matching each operand shape/dtype.
//   2. Replay the op using the Graph builder (not the tape graph) so we get a
//      clean StableHLO module with no unreferenced nodes.
//   3. emit() → make_compile_options() → compile → cache.
//   4. Run with operand buffers → return output Buffer.
//
// `node` is the node in the *tape* graph; we only read its fields.
// `operand_bufs` are the concrete device buffers for each input (in arg order).
// Returns the raw output buffer (caller wraps in Buffer / Tensor).

inline Buffer dispatch_node(const Node& node,
                             const std::vector<Buffer*>& operand_bufs,
                             int num_replicas = 1,
                             const std::string& precision = "HIGHEST")
{
    // ── Build cache key ──────────────────────────────────────────────────────
    std::vector<Shape> in_shapes;
    std::vector<DType> in_dtypes;
    for (auto* b : operand_bufs) {
        in_shapes.push_back(b->shape());
        in_dtypes.push_back(b->dtype());
    }
    std::string key = make_cache_key(node.op, in_shapes, in_dtypes,
                                      node.ints, node.fval,
                                      num_replicas, precision, node.shape);

    // ── Cache hit → execute immediately ────────────────────────────────────
    Executable* cached = kernel_cache().get(key);
    if (cached) {
        auto outs = cached->run(operand_bufs, 0);
        if (outs.empty()) throw Error("dispatch: executable produced no outputs");
        return std::move(outs[0]);
    }

    // ── Cache miss → build 1-op Graph, compile, cache ───────────────────────
    Graph g1;
    g1.dot_precision  = precision;
    g1.num_replicas   = num_replicas;

    // Add one Input per operand.
    std::vector<Value> inputs;
    inputs.reserve(operand_bufs.size());
    for (size_t i = 0; i < operand_bufs.size(); ++i) {
        std::string iname = "x" + std::to_string(i);
        inputs.push_back(g1.input(iname, in_shapes[i], in_dtypes[i]));
    }

    // Replay the op.
    Value out{nullptr, -1};
    switch (node.op) {
        // Binary elementwise
        case Op::Add: out = g1.add(inputs[0], inputs[1]); break;
        case Op::Sub: out = g1.sub(inputs[0], inputs[1]); break;
        case Op::Mul: out = g1.mul(inputs[0], inputs[1]); break;
        case Op::Div: out = g1.div(inputs[0], inputs[1]); break;
        case Op::Max: out = g1.max(inputs[0], inputs[1]); break;
        case Op::Min: out = g1.min(inputs[0], inputs[1]); break;
        // Unary
        case Op::Neg:         out = g1.neg(inputs[0]);         break;
        case Op::Exp:         out = g1.exp(inputs[0]);         break;
        case Op::Log:         out = g1.log(inputs[0]);         break;
        case Op::Sqrt:        out = g1.sqrt(inputs[0]);        break;
        case Op::Rsqrt:       out = g1.rsqrt(inputs[0]);       break;
        case Op::Tanh:        out = g1.tanh(inputs[0]);        break;
        case Op::Abs:         out = g1.abs(inputs[0]);         break;
        case Op::Logistic:    out = g1.logistic(inputs[0]);    break;
        case Op::StopGradient: out = g1.stop_gradient(inputs[0]); break;
        // Dot (batched matmul)
        case Op::Dot:
            out = g1.dot(inputs[0], inputs[1]);
            break;
        // Shape ops
        case Op::Reshape:
            out = g1.reshape(inputs[0], node.shape);
            break;
        case Op::Transpose:
            out = g1.transpose(inputs[0], node.ints);
            break;
        case Op::Broadcast:
            out = g1.broadcast_in_dim(inputs[0], node.shape, node.ints);
            break;
        // Reductions
        case Op::ReduceSum: {
            // node.ints = axes (keepdims already resolved in tape graph — the
            // tape node has the post-reduction shape; we always emit non-keepdims
            // because the shape is already correct).
            std::vector<int64_t> axes(node.ints.begin(), node.ints.end());
            // Determine keepdims by checking if input rank > output rank.
            // Since our 1-op graph Input has the actual pre-reduction shape,
            // we just use the known output shape from node.shape and emit
            // a keepdims=false reduction then reshape to match.
            out = g1.reduce_sum(inputs[0], axes, false);
            // If output shape doesn't match (keepdims was true in original),
            // reshape.
            if (g1.node(out.id).shape != node.shape)
                out = g1.reshape(out, node.shape);
            break;
        }
        case Op::ReduceMax: {
            std::vector<int64_t> axes(node.ints.begin(), node.ints.end());
            out = g1.reduce_max(inputs[0], axes, false);
            if (g1.node(out.id).shape != node.shape)
                out = g1.reshape(out, node.shape);
            break;
        }
        // Misc
        case Op::Convert:
            out = g1.convert(inputs[0], node.dtype);
            break;
        case Op::Compare:
            out = g1.compare(inputs[0], inputs[1], node.cmp);
            break;
        case Op::Select:
            out = g1.select(inputs[0], inputs[1], inputs[2]);
            break;
        case Op::Iota:
            out = g1.iota(node.shape, node.ints[0], node.dtype);
            break;
        case Op::Gather:
            out = g1.gather_rows(inputs[0], inputs[1]);
            break;
        case Op::ScatterAdd:
            out = g1.scatter_add_rows(inputs[0], inputs[1], inputs[2]);
            break;
        case Op::AllReduce:
            out = g1.all_reduce_sum(inputs[0]);
            break;
        case Op::Constant:
            // Constant ops have no operands; the value is in node.fval.
            out = g1.constant(node.fval, node.shape, node.dtype);
            break;
        default:
            throw Error("dispatch_node: unsupported op " + std::to_string((int)node.op));
    }

    if (!out.valid()) throw Error("dispatch_node: op build produced invalid Value");

    // Compile.
    std::string mlir  = g1.emit({out});
    std::string copts = make_compile_options(num_replicas);
    Executable exec   = global_context().compile_mlir(mlir, copts.data(), copts.size());

    // Execute.
    auto outs = exec.run(operand_bufs, 0);
    if (outs.empty()) throw Error("dispatch_node: executable produced no outputs");
    Buffer result = std::move(outs[0]);

    // Cache the compiled executable.
    kernel_cache().put(key, std::move(exec));

    return result;
}

}  // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// Tensor factory helpers (leaf constructors, Decision 11)
// New leaves attach to active_trace if a trace is open; otherwise to
// current_tape if taping; otherwise they are standalone Concrete tensors.
// ─────────────────────────────────────────────────────────────────────────────

// Internal: create a Concrete TensorImpl from a device Buffer.
inline std::shared_ptr<TensorImpl> make_concrete_impl(Buffer buf,
                                                        const Shape& shape,
                                                        DType dtype,
                                                        int device = 0)
{
    auto impl = std::make_shared<TensorImpl>();
    impl->is_traced = false;
    impl->buf       = std::make_shared<Buffer>(std::move(buf));
    impl->shape_    = shape;
    impl->dtype_    = dtype;
    impl->device_   = device;
    return impl;
}

// Internal: attach a tape node to an impl (if taping and requires_grad).
// Mutates impl in place.
inline void maybe_tape_leaf(TensorImpl& impl, Graph& tape_g,
                             std::shared_ptr<Graph> tape_shared,
                             const std::string& name)
{
    Value v = tape_g.input(name, impl.shape_, impl.dtype_);
    impl.tape_graph  = std::move(tape_shared);
    impl.tape_value  = v;
    // NOTE: caller must also call tape.add_leaf(v) if requires_grad.
}

// Upload a host float vector → Concrete Tensor.
inline Tensor from_host(const std::vector<float>& data,
                         const Shape& shape,
                         DType dtype = DType::F32,
                         int device = 0)
{
    if (dtype != DType::F32)
        throw Error("from_host: only F32 host upload supported in v1");
    Buffer buf = global_context().upload_f32(data, shape, device);
    auto impl  = make_concrete_impl(std::move(buf), shape, dtype, device);
    return Tensor{impl};
}

// Upload a host int32 vector → Concrete Tensor (S32).
inline Tensor from_host_s32(const std::vector<int32_t>& data,
                              const Shape& shape,
                              int device = 0)
{
    Buffer buf = global_context().upload_s32(data, shape, device);
    auto impl  = make_concrete_impl(std::move(buf), shape, DType::S32, device);
    return Tensor{impl};
}

// Constant tensor (all elements == val).
inline Tensor full(const Shape& shape, float val,
                    DType dtype = DType::F32, int device = 0)
{
    int64_t n = num_elements(shape);
    std::vector<float> data(n, val);
    Buffer buf = global_context().upload_f32(data, shape, device);
    auto impl  = make_concrete_impl(std::move(buf), shape, dtype, device);
    return Tensor{impl};
}

inline Tensor zeros(const Shape& shape, DType dtype = DType::F32, int device = 0) {
    return full(shape, 0.0f, dtype, device);
}
inline Tensor ones(const Shape& shape, DType dtype = DType::F32, int device = 0) {
    return full(shape, 1.0f, dtype, device);
}

// Random normal tensor (host-side generation → upload).
inline Tensor randn(const Shape& shape, float mean = 0.0f, float stddev = 1.0f,
                     uint64_t seed = 0, DType dtype = DType::F32, int device = 0)
{
    int64_t n = num_elements(shape);
    std::vector<float> data(n);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    for (auto& x : data) x = dist(rng);
    Buffer buf = global_context().upload_f32(data, shape, device);
    auto impl  = make_concrete_impl(std::move(buf), shape, dtype, device);
    return Tensor{impl};
}

// ─────────────────────────────────────────────────────────────────────────────
// eager_dispatch — core of the eager path (Decision 2)
//
// Given an Op and a list of Concrete-tensor operands, this function:
//   1. Appends a node to the active tape (if we are taping and at least one
//      operand requires_grad).
//   2. Builds a tiny 1-op Graph, compiles (or hits cache), executes.
//   3. Returns a Concrete Tensor with the result Buffer (and tape node if 1).
//
// Callers that need to pass extra per-op attributes (reshape target shape, axes,
// etc.) use the overloads below.
// ─────────────────────────────────────────────────────────────────────────────

// Internal: check whether taping is needed for this set of operands.
inline bool should_tape(const std::vector<const Tensor*>& operands) {
    if (!current_tape) return false;   // no_grad or no tape set
    for (auto* t : operands)
        if (t && t->requires_grad()) return true;
    return false;
}

// Core dispatch: op + operand Tensors + (optional) extra tape-node customizer.
// The customizer is called with a (Graph&, vector<Value> inputs) and must return
// the output Value to record on the tape; it is only called when taping.
// The `build_dispatch_node` function is always called and must return the Node
// that will be passed to dispatch_node().
//
// For simple ops the two are identical; for ops with attributes (reshape, etc.)
// the customizer builds the correct tape node while dispatch uses the same attrs.

// ── jit trace seam (Decisions 4 & 11) ────────────────────────────────────────
// When a jit trace is active, ops must record into the trace graph instead of
// executing on-device.  jit.hpp (Wave B / T4) registers this hook at static init;
// it stays null in non-jit builds, so the routing branch in eager_dispatch_impl
// is never taken and NO link dependency on jit.hpp is introduced.  The hook is
// handed the SAME `tape_builder` the eager path uses, so each op's shape/graph
// logic is written exactly once and eager/jit cannot drift.  The hook owns
// auto-lift, capture-set tracking, and Traced-Tensor construction (T4).
using TraceDispatchHook = Tensor (*)(
    Op op,
    const std::vector<const Tensor*>& operands,
    const std::function<Value(Graph&, const std::vector<Value>&)>& tape_builder);
inline TraceDispatchHook trace_dispatch_hook = nullptr;

inline Tensor eager_dispatch_impl(
    Op op,
    const std::vector<const Tensor*>& operands,
    // Returns the Node to dispatch (contains op, shape, dtype, ints, fval, cmp).
    std::function<Node(const std::vector<Buffer*>&)> build_node,
    // Returns the tape Value (called only when taping; receives g and inputs).
    std::function<Value(Graph&, const std::vector<Value>&)> tape_builder,
    int  num_replicas = 1,
    const std::string& precision = "HIGHEST")
{
    // jit trace seam: if a trace is active, record into the trace graph (no
    // execution).  Traced operands carry no buffers, so this MUST precede buffer
    // gathering.  Hook is null in non-jit builds (branch never taken).
    if (active_trace && trace_dispatch_hook) {
        return trace_dispatch_hook(op, operands, tape_builder);
    }

    // Gather device buffers from all operands.
    std::vector<Buffer*> bufs;
    bufs.reserve(operands.size());
    for (auto* t : operands) {
        if (!t || !t->is_concrete())
            throw Error("eager_dispatch: operand is not a Concrete tensor");
        bufs.push_back(t->raw_buffer());
    }

    // Build the node descriptor (op, shape, dtype, attrs).
    Node node = build_node(bufs);

    // Execute on-device (compile + cache).
    Buffer result_buf = detail::dispatch_node(node, bufs, num_replicas, precision);

    // Build result Concrete impl.
    auto impl = make_concrete_impl(std::move(result_buf), node.shape, node.dtype);

    // Tape recording (Decision 2 / 11).
    bool taping = should_tape(operands);
    if (taping && current_tape) {
        // We need inputs as graph Values on the tape.  Each operand is either:
        //   (a) already recorded as a tape node from the CURRENT tape, or
        //   (b) a leaf / not-yet-taped Concrete tensor, OR a tensor whose tape
        //       node belongs to a PREVIOUS (now reset) tape (Gotcha B: stale
        //       tape_value after reset).  In both cases, re-lift as fresh Input.
        Graph& g = *current_tape->graph();
        std::vector<Value> tape_inputs;
        tape_inputs.reserve(operands.size());
        for (auto* t : operands) {
            // Gotcha B: require that the existing tape node belongs to the
            // CURRENT tape's graph; otherwise treat as a new leaf this step.
            bool has_current_node = t->has_tape_node() &&
                t->impl_ptr()->tape_graph.get() == current_tape->graph().get();
            if (has_current_node) {
                tape_inputs.push_back(t->tape_value());
            } else {
                // First time this tensor is used on THIS tape → add as Input.
                // (This covers params created before the tape, AND params
                //  whose tape_value points at the previous tape after reset.)
                Value v = g.input("leaf_" + std::to_string(g.num_nodes()),
                                   t->shape(), t->dtype());
                // Tag the operand's impl so future uses find the same node.
                // We cast away const for this bookkeeping write (safe: the Impl
                // is mutable; only the Tensor handle is const-passed).
                auto leaf_impl = const_cast<Tensor*>(t)->impl_shared();
                leaf_impl->tape_graph = current_tape->graph();
                leaf_impl->tape_value = v;
                // Gotcha A: register the leaf's existing buffer so backward()
                // can find it in the node_buffers_ memo without recomputing.
                if (leaf_impl->buf)
                    current_tape->set_buffer(v.id, leaf_impl->buf);
                if (t->requires_grad())
                    current_tape->add_leaf(v, leaf_impl);
                else
                    current_tape->add_leaf_no_grad(v);
                tape_inputs.push_back(v);
            }
        }
        // Build the tape node using the caller's builder.
        Value tape_out = tape_builder(g, tape_inputs);
        impl->tape_graph = current_tape->graph();
        impl->tape_value = tape_out;
        // Propagate requires_grad: output requires grad if any input does.
        impl->requires_grad_ = true;
        // Gotcha A: register the result buffer at the output tape node id so
        // the backward evaluator can reuse it instead of recomputing the fwd op.
        current_tape->set_buffer(tape_out.id, impl->buf);
    }

    return Tensor{std::move(impl)};
}

// ─────────────────────────────────────────────────────────────────────────────
// Op-specific dispatch entry points
// ─────────────────────────────────────────────────────────────────────────────

// Binary elementwise helper (add/sub/mul/div/max/min).
inline Tensor binary_op(Op op, const Tensor& a, const Tensor& b,
                          int num_replicas = 1,
                          const std::string& precision = "HIGHEST")
{
    // Build a node with the broadcast shape.
    auto build = [op](const std::vector<Buffer*>& bufs) -> Node {
        Shape sa = bufs[0]->shape(), sb = bufs[1]->shape();
        // Broadcast shapes (numpy-style, trailing alignment).
        int64_t ra = (int64_t)sa.size(), rb = (int64_t)sb.size();
        int64_t R  = std::max(ra, rb);
        Shape rs(R);
        for (int64_t i = 0; i < R; ++i) {
            int64_t da = (i < R - ra) ? 1 : sa[i - (R - ra)];
            int64_t db = (i < R - rb) ? 1 : sb[i - (R - rb)];
            if (da == db)     rs[i] = da;
            else if (da == 1) rs[i] = db;
            else if (db == 1) rs[i] = da;
            else {
                std::string m = "binary_op(op=" + std::to_string((int)op) + "): incompatible broadcast shapes a=[";
                for (size_t k=0;k<sa.size();++k) m += std::to_string(sa[k]) + (k+1<sa.size()?",":"");
                m += "] b=[";
                for (size_t k=0;k<sb.size();++k) m += std::to_string(sb[k]) + (k+1<sb.size()?",":"");
                m += "]";
                throw Error(m);
            }
        }
        Node n;
        n.op    = op;
        n.shape = rs;
        n.dtype = bufs[0]->dtype();
        return n;
    };
    auto tape_build = [op](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.binary(op, ins[0], ins[1]);
    };
    return eager_dispatch_impl(op, {&a, &b}, build, tape_build,
                                num_replicas, precision);
}

// Unary elementwise helper.
inline Tensor unary_op(Op op, const Tensor& a,
                        int num_replicas = 1,
                        const std::string& precision = "HIGHEST")
{
    auto build = [op](const std::vector<Buffer*>& bufs) -> Node {
        Node n;
        n.op    = op;
        n.shape = bufs[0]->shape();
        n.dtype = bufs[0]->dtype();
        return n;
    };
    auto tape_build = [op](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.unary(op, ins[0]);
    };
    return eager_dispatch_impl(op, {&a}, build, tape_build,
                                num_replicas, precision);
}

// ── Public op entry points ────────────────────────────────────────────────────

inline Tensor add(const Tensor& a, const Tensor& b) {
    return binary_op(Op::Add, a, b, 1, default_dot_precision());
}
inline Tensor sub(const Tensor& a, const Tensor& b) {
    return binary_op(Op::Sub, a, b, 1, default_dot_precision());
}
inline Tensor mul(const Tensor& a, const Tensor& b) {
    return binary_op(Op::Mul, a, b, 1, default_dot_precision());
}
inline Tensor div_op(const Tensor& a, const Tensor& b) {
    return binary_op(Op::Div, a, b, 1, default_dot_precision());
}
inline Tensor maximum(const Tensor& a, const Tensor& b) {
    return binary_op(Op::Max, a, b, 1, default_dot_precision());
}
inline Tensor minimum(const Tensor& a, const Tensor& b) {
    return binary_op(Op::Min, a, b, 1, default_dot_precision());
}

inline Tensor neg(const Tensor& a)  { return unary_op(Op::Neg, a); }
inline Tensor exp_op(const Tensor& a) { return unary_op(Op::Exp, a); }
inline Tensor log_op(const Tensor& a) { return unary_op(Op::Log, a); }
inline Tensor sqrt_op(const Tensor& a){ return unary_op(Op::Sqrt, a); }
inline Tensor tanh_op(const Tensor& a){ return unary_op(Op::Tanh, a); }
inline Tensor abs_op(const Tensor& a) { return unary_op(Op::Abs, a); }
inline Tensor logistic(const Tensor& a){ return unary_op(Op::Logistic, a); }

// relu via max(x, 0).
inline Tensor relu(const Tensor& a) {
    Tensor zero = full(a.shape(), 0.0f, a.dtype(), a.device());
    return maximum(a, zero);
}

// gelu: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Mirrors the existing nn.hpp formulation.
inline Tensor gelu(const Tensor& x) {
    // x^3 = x * x * x
    Tensor x2   = mul(x, x);
    Tensor x3   = mul(x2, x);
    // coeff * x3
    Tensor c1   = full(x.shape(), 0.044715f, x.dtype(), x.device());
    Tensor term = mul(c1, x3);
    // x + coeff*x^3
    Tensor inner = add(x, term);
    // sqrt(2/pi) ≈ 0.7978845608
    Tensor sq2pi = full(x.shape(), 0.7978845608f, x.dtype(), x.device());
    Tensor scaled = mul(sq2pi, inner);
    // tanh(...)
    Tensor t    = tanh_op(scaled);
    // 1 + tanh(...)
    Tensor one  = ones(x.shape(), x.dtype(), x.device());
    Tensor t1   = add(one, t);
    // 0.5 * x * (1 + tanh(...))
    Tensor half = full(x.shape(), 0.5f, x.dtype(), x.device());
    Tensor hx   = mul(half, x);
    return mul(hx, t1);
}

// matmul: a[...,M,K] @ b[...,K,N] → [...,M,N]
inline Tensor matmul(const Tensor& a, const Tensor& b,
                      const std::string& precision = "")
{
    const std::string& prec = precision.empty() ? default_dot_precision() : precision;
    auto build = [](const std::vector<Buffer*>& bufs) -> Node {
        Shape A = bufs[0]->shape(), B = bufs[1]->shape();
        int64_t ra = (int64_t)A.size(), rb = (int64_t)B.size();
        if (ra < 2 || rb < 2) throw Error("matmul: operands must be rank >= 2");
        int64_t batch = ra - 2;
        if (rb - 2 != batch) throw Error("matmul: batch rank mismatch");
        if (A[ra-1] != B[rb-2]) throw Error("matmul: contracting dim mismatch");
        Shape out(A.begin(), A.begin() + batch);
        out.push_back(A[ra-2]);
        out.push_back(B[rb-1]);
        Node n;
        n.op    = Op::Dot;
        n.shape = out;
        n.dtype = bufs[0]->dtype();
        n.ints  = {batch};
        return n;
    };
    auto tape_build = [](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.dot(ins[0], ins[1]);
    };
    // Temporarily override precision for this call.
    std::string old_prec = default_dot_precision();
    default_dot_precision() = prec;
    Tensor result = eager_dispatch_impl(Op::Dot, {&a, &b}, build, tape_build);
    default_dot_precision() = old_prec;
    return result;
}

// reshape: produces a view with a new shape (same element count).
inline Tensor reshape(const Tensor& a, const Shape& new_shape) {
    Shape ns = new_shape;
    auto build = [ns](const std::vector<Buffer*>& bufs) -> Node {
        Node n;
        n.op    = Op::Reshape;
        n.shape = ns;
        n.dtype = bufs[0]->dtype();
        return n;
    };
    auto tape_build = [ns](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.reshape(ins[0], ns);
    };
    return eager_dispatch_impl(Op::Reshape, {&a}, build, tape_build);
}

// transpose last two axes.
inline Tensor transpose(const Tensor& a) {
    int64_t r = a.rank();
    if (r < 2) throw Error("transpose: rank < 2");
    std::vector<int64_t> perm(r);
    for (int64_t i = 0; i < r; ++i) perm[i] = i;
    std::swap(perm[r-1], perm[r-2]);
    Shape s = a.shape();
    std::swap(s[r-1], s[r-2]);
    auto build = [perm, s](const std::vector<Buffer*>& bufs) -> Node {
        Node n;
        n.op    = Op::Transpose;
        n.shape = s;
        n.dtype = bufs[0]->dtype();
        n.ints  = perm;
        return n;
    };
    auto tape_build = [perm](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.transpose(ins[0], perm);
    };
    return eager_dispatch_impl(Op::Transpose, {&a}, build, tape_build);
}

// reduce_sum over given axes.
inline Tensor reduce_sum(const Tensor& a, std::vector<int64_t> axes,
                          bool keepdims = false)
{
    // Compute output shape.
    Shape in = a.shape();
    for (auto& ax : axes) if (ax < 0) ax += (int64_t)in.size();
    std::sort(axes.begin(), axes.end());
    std::vector<bool> drop(in.size(), false);
    for (int64_t ax : axes) drop[ax] = true;
    Shape out;
    if (keepdims) {
        out = in;
        for (int64_t ax : axes) out[ax] = 1;
    } else {
        for (size_t i = 0; i < in.size(); ++i)
            if (!drop[i]) out.push_back(in[i]);
    }
    auto build = [axes, out](const std::vector<Buffer*>&) -> Node {
        Node n;
        n.op    = Op::ReduceSum;
        n.shape = out;
        n.dtype = DType::F32;  // set from buf below; placeholder
        n.ints  = axes;
        return n;
    };
    auto tape_build = [axes, keepdims](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.reduce_sum(ins[0], axes, keepdims);
    };
    // Override build to set correct dtype from buf.
    auto build2 = [axes, out](const std::vector<Buffer*>& bufs) -> Node {
        Node n;
        n.op    = Op::ReduceSum;
        n.shape = out;
        n.dtype = bufs[0]->dtype();
        n.ints  = axes;
        return n;
    };
    return eager_dispatch_impl(Op::ReduceSum, {&a}, build2, tape_build);
}

// scalar broadcast helper: broadcast a float scalar to tensor shape via full().
inline Tensor scalar_to_tensor(double val, const Tensor& ref) {
    return full(ref.shape(), (float)val, ref.dtype(), ref.device());
}

// ── Extra ops for transformer layers (T5: GPT on Tensor) ─────────────────────
// Each mirrors the eager_dispatch_impl pattern above so it works identically in
// eager and jit (trace) mode; the tape_builder reuses the Graph builder so the
// VJP rules in graph.cpp apply unchanged.

// transpose with an explicit permutation (general N-d transpose).
inline Tensor transpose_perm(const Tensor& a, const std::vector<int64_t>& perm) {
    Shape in = a.shape();
    if ((int64_t)perm.size() != (int64_t)in.size())
        throw Error("transpose_perm: perm rank mismatch");
    Shape s(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) s[i] = in[perm[i]];
    auto build = [perm, s](const std::vector<Buffer*>& bufs) -> Node {
        Node n; n.op = Op::Transpose; n.shape = s; n.dtype = bufs[0]->dtype(); n.ints = perm;
        return n;
    };
    auto tape_build = [perm](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.transpose(ins[0], perm);
    };
    return eager_dispatch_impl(Op::Transpose, {&a}, build, tape_build);
}

inline Tensor rsqrt_op(const Tensor& a)      { return unary_op(Op::Rsqrt, a); }
inline Tensor stop_gradient(const Tensor& a) { return unary_op(Op::StopGradient, a); }

// reduce_max over given axes (mirror of reduce_sum).
inline Tensor reduce_max(const Tensor& a, std::vector<int64_t> axes, bool keepdims = false) {
    Shape in = a.shape();
    for (auto& ax : axes) if (ax < 0) ax += (int64_t)in.size();
    std::sort(axes.begin(), axes.end());
    std::vector<bool> drop(in.size(), false);
    for (int64_t ax : axes) drop[ax] = true;
    Shape out;
    if (keepdims) { out = in; for (int64_t ax : axes) out[ax] = 1; }
    else { for (size_t i = 0; i < in.size(); ++i) if (!drop[i]) out.push_back(in[i]); }
    auto build = [axes, out](const std::vector<Buffer*>& bufs) -> Node {
        Node n; n.op = Op::ReduceMax; n.shape = out; n.dtype = bufs[0]->dtype(); n.ints = axes;
        return n;
    };
    auto tape_build = [axes, keepdims](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.reduce_max(ins[0], axes, keepdims);
    };
    return eager_dispatch_impl(Op::ReduceMax, {&a}, build, tape_build);
}

// reduce_mean = reduce_sum / count (count from the reduced axes).
inline Tensor reduce_mean(const Tensor& a, std::vector<int64_t> axes, bool keepdims = false) {
    Shape in = a.shape();
    int64_t cnt = 1;
    for (int64_t ax : axes) { int64_t k = ax < 0 ? ax + (int64_t)in.size() : ax; cnt *= in[k]; }
    Tensor s = reduce_sum(a, axes, keepdims);
    return mul(s, scalar_to_tensor(1.0 / (double)cnt, s));
}

// gather_rows(table[V, D...], ids) → ids.shape + [D...] (embedding lookup).
inline Tensor gather_rows(const Tensor& table, const Tensor& ids) {
    Shape ts = table.shape();
    if (ts.empty()) throw Error("gather_rows: table must be rank >= 1");
    Shape out = ids.shape();
    out.insert(out.end(), ts.begin() + 1, ts.end());
    auto build = [out](const std::vector<Buffer*>& bufs) -> Node {
        Node n; n.op = Op::Gather; n.shape = out; n.dtype = bufs[0]->dtype();
        return n;
    };
    auto tape_build = [](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.gather_rows(ins[0], ins[1]);
    };
    return eager_dispatch_impl(Op::Gather, {&table, &ids}, build, tape_build);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor operator bodies (Decision 1 — declared in tensor.hpp)
// ─────────────────────────────────────────────────────────────────────────────

inline Tensor Tensor::operator+(const Tensor& o) const { return add(*this, o); }
inline Tensor Tensor::operator-(const Tensor& o) const { return sub(*this, o); }
inline Tensor Tensor::operator*(const Tensor& o) const { return mul(*this, o); }
inline Tensor Tensor::operator/(const Tensor& o) const { return div_op(*this, o); }
inline Tensor Tensor::operator-()                const { return neg(*this); }

inline Tensor Tensor::operator+(double s) const { return add(*this, scalar_to_tensor(s, *this)); }
inline Tensor Tensor::operator-(double s) const { return sub(*this, scalar_to_tensor(s, *this)); }
inline Tensor Tensor::operator*(double s) const { return mul(*this, scalar_to_tensor(s, *this)); }
inline Tensor Tensor::operator/(double s) const { return div_op(*this, scalar_to_tensor(s, *this)); }

// Free-function scalar op where scalar is on the left.
inline Tensor operator-(double s, const Tensor& t) {
    return sub(scalar_to_tensor(s, t), t);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor::backward() is intentionally NOT defined here.  Its body lives in
// autograd.hpp (Wave B / T3), which owns the tape walk + memoizing evaluator.
// backward() is only odr-used when actually called, so non-autograd translation
// units (e.g. test_eager) link cleanly without a definition; any TU that calls
// backward() must include autograd.hpp.  This avoids both an ODR clash and any
// include-ordering requirement.
// ─────────────────────────────────────────────────────────────────────────────

}  // namespace tpu

#endif  // TPU_EAGER_HPP
