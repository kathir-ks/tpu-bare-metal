// autograd.hpp — Tape autograd for the eager frontend (T3 / Wave B).
//
// Implements Decision 3 of eager-jit-tensor-core/design.md.
//
// What lives here:
//   TapeScope       — RAII: install a fresh Tape as current_tape; resets on exit.
//   Tensor::backward() — scalar-rooted backward pass.
//   grad(fn)        — functional: run fn under a TapeScope, return grads.
//   value_and_grad(fn) — functional: return {value, grads}.
//
// Include after tensor.hpp (which already pulls in eager.hpp).
// Any TU that calls Tensor::backward() MUST include this header; other TUs
// (e.g. test_eager.cpp) link cleanly without it.
//
// Design notes:
//   * The memoizing evaluator seeds its memo from tape.node_buffers_ (forward
//     activation buffers registered during the forward pass in eager.hpp).
//     Backward-only nodes not in the memo are dispatched via
//     detail::dispatch_node(), which shares the same kernel cache as the
//     forward path.
//   * Graph::grad() is the ONLY VJP engine; this file only evaluates its output.
//   * Accumulation semantics match PyTorch: .grad sums across backward() calls
//     until zero_grad() clears it.  Tape is reset after each backward().
#pragma once
#ifndef TPU_AUTOGRAD_HPP
#define TPU_AUTOGRAD_HPP

#include "tensor.hpp"   // pulls in eager.hpp transitively

#include <cassert>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace tpu {

// ─────────────────────────────────────────────────────────────────────────────
// TapeScope — RAII guard that installs a fresh Tape as current_tape.
//
// Intended training-loop usage:
//
//   {
//       TapeScope scope;                   // current_tape = fresh Tape
//       Tensor loss = model(x, y);         // forward ops tape their nodes
//       loss.backward();                   // backward pass; tape reset inside
//   }                                      // scope dtor: restore old tape ptr
//
// TapeScope may be nested (saves/restores the outer tape pointer), but in v1
// only a single tape per thread is expected.  The scope does NOT call reset()
// on destruction — backward() already resets before the scope exits in the
// normal case, so the dtor only needs to restore the saved pointer.
// ─────────────────────────────────────────────────────────────────────────────
class TapeScope {
public:
    // Install a fresh tape.
    TapeScope() : tape_(std::make_unique<Tape>()), saved_(current_tape) {
        current_tape = tape_.get();
    }

    // Restore the previous tape pointer.
    ~TapeScope() {
        current_tape = saved_;
    }

    // Access the live tape (e.g. for inspecting leaf count in tests).
    Tape& tape() { return *tape_; }

    // Non-copyable, non-movable.
    TapeScope(const TapeScope&)            = delete;
    TapeScope& operator=(const TapeScope&) = delete;

private:
    std::unique_ptr<Tape> tape_;
    Tape*                 saved_;
};

// ─────────────────────────────────────────────────────────────────────────────
// detail::eval_node — memoizing backward evaluator
//
// Recursively evaluates a tape-graph node, seeding the memo from
// tape.node_buffers_ (forward activations).  Backward-only nodes that are
// not already in the memo are dispatched via detail::dispatch_node().
//
// Returns a shared_ptr<Buffer> for the given node id.
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

inline std::string shape_str(const Shape& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += std::to_string(s[i]) + (i+1<s.size()?",":"");
    return r + "]";
}

// Evaluate node `id` in `tape_graph`, filling `memo` as we go.
// `tape` supplies the forward-activation buffer cache.
// Modifies `memo` in place (accumulated as side effect of DFS).
inline std::shared_ptr<Buffer>
eval_node(int id,
          Graph& tape_graph,
          const Tape& tape,
          std::unordered_map<int, std::shared_ptr<Buffer>>& memo)
{
    // Memoization: if we already have a buffer for this node, return it.
    {
        auto it = memo.find(id);
        if (it != memo.end()) return it->second;
    }

    // Check forward activation cache first (avoids re-executing forward ops).
    {
        auto fwd_buf = tape.get_buffer(id);
        if (fwd_buf) {
            memo[id] = fwd_buf;
            return fwd_buf;
        }
    }

    // Evaluate all inputs first (DFS post-order).
    const Node& n = tape_graph.node(id);
    std::vector<Buffer*> input_bufs;
    input_bufs.reserve(n.inputs.size());
    for (int inp_id : n.inputs) {
        auto buf = eval_node(inp_id, tape_graph, tape, memo);
        if (!buf)
            throw Error("autograd evaluator: input node " +
                        std::to_string(inp_id) + " produced null buffer");
        input_bufs.push_back(buf.get());
    }

    // Dispatch the node (compile + cache + execute).
    Buffer result;
    try {
        result = dispatch_node(n, input_bufs);
    } catch (const std::exception& e) {
        std::string msg = std::string("eval_node id=") + std::to_string(id) +
                          " op=" + std::to_string((int)n.op) +
                          " out=" + shape_str(n.shape) + " | inputs(decl vs buf): ";
        for (size_t k = 0; k < n.inputs.size(); ++k) {
            int inp = n.inputs[k];
            msg += "node" + std::to_string(inp) +
                   " decl=" + shape_str(tape_graph.node(inp).shape) +
                   " buf=" + shape_str(input_bufs[k]->shape()) + "; ";
        }
        msg += "-> " + std::string(e.what());
        throw Error(msg);
    }

    auto result_shared = std::make_shared<Buffer>(std::move(result));
    memo[id] = result_shared;
    return result_shared;
}

}  // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// Tensor::backward() — scalar-rooted backward pass.
//
// Preconditions:
//   * *this is a scalar Concrete tensor with a tape node (i.e. created inside
//     a TapeScope with at least one requires_grad operand in its ancestry).
//   * current_tape is set (the TapeScope that produced *this is still alive).
//
// Steps (per design Decision 3):
//   1. Collect the requires_grad leaves from current_tape.
//   2. Call tape.graph()->grad(loss_value, leaf_values) to append backward nodes.
//   3. Run the memoizing evaluator over each grad Value, seeding from the
//      forward activation buffers in tape.node_buffers_.
//   4. Accumulate each grad buffer into the corresponding leaf Tensor's .grad.
//   5. Reset the tape (PyTorch semantics: tape is consumed after backward).
// ─────────────────────────────────────────────────────────────────────────────
inline void Tensor::backward() {
    check_valid();

    if (!impl_->has_tape_node())
        throw Error("Tensor::backward: tensor has no tape node "
                    "(call requires_grad_(true) on parameters before the forward pass)");

    if (!current_tape)
        throw Error("Tensor::backward: no active tape "
                    "(call backward() inside a TapeScope)");

    // Verify the tape node belongs to the current tape's graph.
    if (impl_->tape_graph.get() != current_tape->graph().get())
        throw Error("Tensor::backward: loss tape node is from a different tape "
                    "(likely the tape was reset before backward())");

    // Scalar check.
    if (impl_->shape_.size() != 0 && num_elements(impl_->shape_) != 1)
        throw Error("Tensor::backward: loss must be a scalar (0-d or 1-element tensor)");

    // ── Step 1: collect leaves ────────────────────────────────────────────────
    const std::vector<Value>&                        leaf_vals  = current_tape->leaves();
    const std::vector<std::shared_ptr<TensorImpl>>&  leaf_impls = current_tape->leaf_impls();

    if (leaf_vals.empty()) {
        // Nothing to differentiate — reset and return.
        current_tape->reset();
        return;
    }

    // ── Step 2: run Graph::grad to append backward nodes ─────────────────────
    Graph& tape_graph = *current_tape->graph();
    Value loss_val = impl_->tape_val();

    // Snapshot the tape before grad() appends backward nodes so we know the
    // boundary between forward and backward nodes.
    int fwd_node_count = tape_graph.num_nodes();
    (void)fwd_node_count;  // informational; evaluator uses memo regardless

    std::vector<Value> grad_vals = tape_graph.grad(loss_val, leaf_vals);
    // grad_vals[i] is the gradient of loss w.r.t. leaf_vals[i].

    // ── Step 3: memoizing evaluator ───────────────────────────────────────────
    // Seed memo from forward activation buffers.
    std::unordered_map<int, std::shared_ptr<Buffer>> memo;

    // Evaluate each gradient Value (DFS from each grad node, sharing memo).
    std::vector<std::shared_ptr<Buffer>> grad_bufs;
    grad_bufs.reserve(grad_vals.size());
    for (const Value& gv : grad_vals) {
        auto buf = detail::eval_node(gv.id, tape_graph, *current_tape, memo);
        grad_bufs.push_back(buf);
    }

    // ── Step 4: accumulate into leaf .grad fields ─────────────────────────────
    for (size_t i = 0; i < leaf_impls.size(); ++i) {
        TensorImpl* leaf = leaf_impls[i].get();
        if (!leaf) continue;

        std::shared_ptr<Buffer> gbuf = grad_bufs[i];
        if (!gbuf) continue;

        // Download grad buffer to host, then call accumulate_grad.
        // (accumulate_grad re-uploads on the device side.)
        // We wrap the TensorImpl in a temporary Tensor handle for the call.
        Tensor leaf_t{leaf_impls[i]};
        std::vector<float> gdata = gbuf->to_host<float>();
        leaf_t.accumulate_grad(gdata);
    }

    // ── Step 5: reset the tape (PyTorch semantics) ────────────────────────────
    current_tape->reset();
}

// ─────────────────────────────────────────────────────────────────────────────
// grad(fn, params) — functional interface.
//
// Runs `fn` under a TapeScope and returns the gradients of the returned scalar
// w.r.t. each Tensor in `params` (which must have requires_grad_=true).
//
// The scalar value is discarded; use value_and_grad() to retain it.
//
// Parameters:
//   fn     — callable () → Tensor (scalar), called once with current_tape set.
//   params — list of Tensors to differentiate w.r.t.; each must require grad.
//
// Returns: vector of grad Tensors, same order as params.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<Tensor>
grad(std::function<Tensor()> fn,
     const std::vector<Tensor*>& params)
{
    // Mark all params as requiring grad (caller may have already done this).
    for (auto* p : params) {
        if (!p->requires_grad())
            throw Error("grad(): parameter does not have requires_grad=true; "
                        "call requires_grad_(true) before grad()");
    }

    TapeScope scope;

    // Ensure params are seeded as leaves on the new tape when first used.
    // (They will be auto-lifted on first op encounter inside the tape; we just
    //  verify they are concrete.)
    for (auto* p : params) {
        if (!p->is_concrete())
            throw Error("grad(): parameter must be a Concrete tensor");
    }

    // Run forward.
    Tensor loss = fn();

    if (!loss.valid())
        throw Error("grad(): fn returned an invalid Tensor");
    if (!loss.has_tape_node())
        throw Error("grad(): fn returned a Tensor with no tape node "
                    "(parameters may not be used in fn, or no_grad is active)");

    // Collect tape leaves and their impls.
    const std::vector<Value>&                       leaf_vals  = scope.tape().leaves();
    const std::vector<std::shared_ptr<TensorImpl>>& leaf_impls = scope.tape().leaf_impls();

    if (leaf_vals.empty()) {
        // No grad leaves — return zero tensors.
        current_tape->reset();
        std::vector<Tensor> zeros_out;
        zeros_out.reserve(params.size());
        for (auto* p : params) zeros_out.push_back(zeros(p->shape(), p->dtype(), p->device()));
        return zeros_out;
    }

    // Run Graph::grad.
    Graph& tape_graph = *scope.tape().graph();
    Value loss_val = loss.tape_value();
    std::vector<Value> grad_vals = tape_graph.grad(loss_val, leaf_vals);

    // Memoizing evaluator.
    std::unordered_map<int, std::shared_ptr<Buffer>> memo;
    std::vector<std::shared_ptr<Buffer>> grad_bufs;
    grad_bufs.reserve(grad_vals.size());
    for (const Value& gv : grad_vals) {
        auto buf = detail::eval_node(gv.id, tape_graph, scope.tape(), memo);
        grad_bufs.push_back(buf);
    }

    // Build a leaf-impl → buffer index map for O(1) lookup.
    // leaf_impls[i] corresponds to grad_bufs[i].
    std::unordered_map<TensorImpl*, size_t> impl_to_idx;
    for (size_t i = 0; i < leaf_impls.size(); ++i)
        impl_to_idx[leaf_impls[i].get()] = i;

    // Match params to leaf indices by Impl identity.
    std::vector<Tensor> result;
    result.reserve(params.size());
    for (auto* p : params) {
        auto it = impl_to_idx.find(p->impl_ptr());
        if (it != impl_to_idx.end() && grad_bufs[it->second]) {
            // Build a Concrete impl sharing the grad buffer (no move: the buffer
            // may be referenced by other memo entries).
            auto gbuf = grad_bufs[it->second];
            auto g_impl = std::make_shared<TensorImpl>();
            g_impl->is_traced = false;
            g_impl->buf       = gbuf;
            g_impl->shape_    = p->shape();
            g_impl->dtype_    = p->dtype();
            g_impl->device_   = p->device();
            result.push_back(Tensor{g_impl});
        } else {
            // Parameter was not reached — zero gradient.
            result.push_back(zeros(p->shape(), p->dtype(), p->device()));
        }
    }

    // Reset the tape.
    scope.tape().reset();
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// value_and_grad(fn, params) — like grad() but also returns the scalar value.
//
// Returns: {loss_value, vector_of_grad_tensors}
// The loss Tensor is Concrete (no tape node after backward).
// ─────────────────────────────────────────────────────────────────────────────
inline std::pair<Tensor, std::vector<Tensor>>
value_and_grad(std::function<Tensor()> fn,
               const std::vector<Tensor*>& params)
{
    for (auto* p : params) {
        if (!p->requires_grad())
            throw Error("value_and_grad(): parameter does not have requires_grad=true");
        if (!p->is_concrete())
            throw Error("value_and_grad(): parameter must be a Concrete tensor");
    }

    TapeScope scope;
    Tensor loss = fn();

    if (!loss.valid())
        throw Error("value_and_grad(): fn returned an invalid Tensor");

    // Retain the loss value before the tape resets.
    // Download and re-upload so we own a clean Buffer with no tape linkage.
    std::vector<float> loss_data = loss.to_host();
    Tensor loss_out = from_host(loss_data, loss.shape(), loss.dtype(), loss.device());

    if (!loss.has_tape_node()) {
        // No differentiable path — return loss + zero grads.
        scope.tape().reset();
        std::vector<Tensor> zeros_out;
        zeros_out.reserve(params.size());
        for (auto* p : params) zeros_out.push_back(zeros(p->shape(), p->dtype(), p->device()));
        return {loss_out, zeros_out};
    }

    const std::vector<Value>&                       leaf_vals  = scope.tape().leaves();
    const std::vector<std::shared_ptr<TensorImpl>>& leaf_impls = scope.tape().leaf_impls();

    Graph& tape_graph = *scope.tape().graph();
    Value loss_val = loss.tape_value();
    std::vector<Value> grad_vals = tape_graph.grad(loss_val, leaf_vals);

    std::unordered_map<int, std::shared_ptr<Buffer>> memo;
    std::vector<std::shared_ptr<Buffer>> grad_bufs;
    grad_bufs.reserve(grad_vals.size());
    for (const Value& gv : grad_vals) {
        auto buf = detail::eval_node(gv.id, tape_graph, scope.tape(), memo);
        grad_bufs.push_back(buf);
    }

    std::unordered_map<TensorImpl*, size_t> impl_to_idx;
    for (size_t i = 0; i < leaf_impls.size(); ++i)
        impl_to_idx[leaf_impls[i].get()] = i;

    std::vector<Tensor> result;
    result.reserve(params.size());
    for (auto* p : params) {
        auto it = impl_to_idx.find(p->impl_ptr());
        if (it != impl_to_idx.end() && grad_bufs[it->second]) {
            auto gbuf = grad_bufs[it->second];
            auto g_impl = std::make_shared<TensorImpl>();
            g_impl->is_traced = false;
            g_impl->buf       = gbuf;
            g_impl->shape_    = p->shape();
            g_impl->dtype_    = p->dtype();
            g_impl->device_   = p->device();
            result.push_back(Tensor{g_impl});
        } else {
            result.push_back(zeros(p->shape(), p->dtype(), p->device()));
        }
    }

    scope.tape().reset();
    return {loss_out, result};
}

}  // namespace tpu

#endif  // TPU_AUTOGRAD_HPP
