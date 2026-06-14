// module.hpp — PyTorch-like Module / layer library built on Tensor + eager ops.
//
// Layer 5 (T5) of the eager-jit-tensor-core design.  Module carries named
// parameters and child modules; parameter init runs in each constructor; the
// whole tree is traversed by parameters() to hand a flat list to an optimizer.
//
// v1 scope (MLP proof): Module base, Linear, ReLU, GELU, Sequential,
//   cross_entropy, MLP.
//
// DEFERRED (follow-up tasks — stubs with TODOs):
//   Embedding  — needs gather_rows / from_host_s32 via eager; ops exist
//                (Op::Gather in dispatch_node) but the gather eager free-fn is
//                not yet exposed; add reduce_max helper below first.
//   RMSNorm    — needs reduce_mean (not yet a free fn in eager.hpp); trivial
//                to add once reduce_max is promoted.
//   Attention  — needs Embedding + RMSNorm + transpose_last2 + batched dot.
//   Block      — needs Attention + RMSNorm.
//   GPT        — needs Block + Embedding + cross_entropy.
//
// These are deferred purely because the eager.hpp free-function surface does
// not yet expose reduce_max / gather / reduce_mean; the IR ops exist and the
// dispatch_node switch already handles them.  Adding a thin free-function
// wrapper (mirroring reduce_sum) is all that is needed.

#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensor.hpp"   // pulls in eager.hpp automatically

namespace tpu {

// ─────────────────────────────────────────────────────────────────────────────
// Local eager helpers not yet exposed as free functions in eager.hpp.
// These mirror the reduce_sum / unary_op patterns exactly; they do NOT edit
// eager.hpp (constraint from the orchestrator).
// ─────────────────────────────────────────────────────────────────────────────
namespace module_detail {

// reduce_max over given axes (identical structure to reduce_sum in eager.hpp).
inline Tensor reduce_max(const Tensor& a, std::vector<int64_t> axes,
                          bool keepdims = false)
{
    Shape in = a.shape();
    for (auto& ax : axes) if (ax < 0) ax += (int64_t)in.size();
    std::sort(axes.begin(), axes.end());
    Shape out;
    if (keepdims) {
        out = in;
        for (int64_t ax : axes) out[ax] = 1;
    } else {
        std::vector<bool> drop(in.size(), false);
        for (int64_t ax : axes) drop[ax] = true;
        for (size_t i = 0; i < in.size(); ++i)
            if (!drop[i]) out.push_back(in[i]);
    }
    auto build2 = [axes, out](const std::vector<Buffer*>& bufs) -> Node {
        Node n;
        n.op    = Op::ReduceMax;
        n.shape = out;
        n.dtype = bufs[0]->dtype();
        n.ints  = axes;
        return n;
    };
    auto tape_build = [axes, keepdims](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.reduce_max(ins[0], axes, keepdims);
    };
    return eager_dispatch_impl(Op::ReduceMax, {&a}, build2, tape_build);
}

// broadcast_to: make a tensor match a target shape via broadcast_in_dim.
// Uses the existing Op::Broadcast dispatch path.
// `a` must have a shape that is a suffix of `target` (trailing dims match).
inline Tensor broadcast_to(const Tensor& a, const Shape& target) {
    const Shape& src = a.shape();
    // Compute broadcast_dims: mapping of src dim i → target dim.
    // src dims map to the LAST src.size() dims of target.
    int64_t ndiff = (int64_t)target.size() - (int64_t)src.size();
    std::vector<int64_t> bdims;
    bdims.reserve(src.size());
    for (int64_t i = 0; i < (int64_t)src.size(); ++i)
        bdims.push_back(ndiff + i);

    Shape tgt = target;
    auto build = [tgt, bdims](const std::vector<Buffer*>& bufs) -> Node {
        Node n;
        n.op    = Op::Broadcast;
        n.shape = tgt;
        n.dtype = bufs[0]->dtype();
        n.ints  = bdims;
        return n;
    };
    auto tape_build = [tgt, bdims](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.broadcast_in_dim(ins[0], tgt, bdims);
    };
    return eager_dispatch_impl(Op::Broadcast, {&a}, build, tape_build);
}

}  // namespace module_detail

// ─────────────────────────────────────────────────────────────────────────────
// cross_entropy — free function (not a Module; operates on Tensors).
//
// logits: [N, V]  (F32)
// targets: [N]    (S32, integer class indices)
// → scalar mean negative log-likelihood.
//
// Algorithm (numerically stable logsumexp):
//   m      = stop_grad(reduce_max(logits, axis=1, keepdims=true))   [N,1]
//   shifted = logits - m                                             [N,V]
//   lse    = log(reduce_sum(exp(shifted), axis=1, keepdims=true)) + m [N,1]
//   log_p  = logits - lse                                           [N,V]
//   For each row i pick log_p[i, targets[i]] via one-hot (no gather in v1).
//   loss   = mean(-log_p[i, targets[i]])
//
// One-hot path: iota + compare + convert (identical to nn.hpp cross_entropy).
// Gather path would need expose of gather_rows eager free-fn (TODO for follow-up).
// ─────────────────────────────────────────────────────────────────────────────
inline Tensor cross_entropy(const Tensor& logits, const Tensor& targets)
{
    // logits: [N, V],  targets: [N] S32
    if (logits.rank() != 2)
        throw Error("cross_entropy: logits must be rank 2 [N, V]");
    if (targets.rank() != 1)
        throw Error("cross_entropy: targets must be rank 1 [N]");

    int64_t N = logits.shape()[0];
    int64_t V = logits.shape()[1];

    // ── Numerically stable logsumexp ─────────────────────────────────────────
    // m = stop_grad(max over classes)  [N, 1]
    Tensor m_raw = module_detail::reduce_max(logits, {1}, /*keepdims=*/true);
    // stop_gradient: dispatch the StopGradient unary op.
    Tensor m = unary_op(Op::StopGradient, m_raw);

    // shifted = logits - m   (broadcast m [N,1] over [N,V])
    Tensor m_broad = module_detail::broadcast_to(m, {N, V});
    Tensor shifted = sub(logits, m_broad);

    // lse = log(sum(exp(shifted), axis=1, keepdims=true)) + m   [N, 1]
    Tensor eshift = exp_op(shifted);
    Tensor sumexp = reduce_sum(eshift, {1}, /*keepdims=*/true);
    Tensor lse    = add(log_op(sumexp), m);           // [N, 1]

    // log_p = logits - lse (broadcast lse [N,1] → [N,V])
    Tensor lse_broad = module_detail::broadcast_to(lse, {N, V});
    Tensor log_p     = sub(logits, lse_broad);        // [N, V]

    // ── One-hot gather of target log-probs ────────────────────────────────────
    // Build [N, V] one-hot matrix from targets using iota+compare+convert.
    // This mirrors nn::cross_entropy in nn.hpp exactly.
    //
    // iota_v:  [N, V] filled with 0..V-1 (class indices, S32)
    // tgt_col: [N, V] each row = targets[i] (broadcast targets [N] → [N,V])
    // onehot:  [N, V] F32, 1.0 where col == target[i]
    //
    // We drive this through eager_dispatch_impl for each op.
    // Note: targets is S32; iota produces S32.  Both paths stay host-compiled.

    // iota [N, V] along axis 1 (class axis)
    auto iota_build = [N, V](const std::vector<Buffer*>&) -> Node {
        Node n;
        n.op    = Op::Iota;
        n.shape = {N, V};
        n.dtype = DType::S32;
        n.ints  = {1};          // iota dimension
        n.fval  = 0;
        return n;
    };
    auto iota_tape = [N, V](Graph& g, const std::vector<Value>&) -> Value {
        return g.iota({N, V}, 1, DType::S32);
    };
    // Iota has no operands; pass a dummy scalar Tensor as a carrier so
    // eager_dispatch_impl gets a concrete device to work with.
    Tensor dummy = zeros({}, DType::F32, logits.device());
    Tensor iota_v = eager_dispatch_impl(Op::Iota, {&dummy}, iota_build, iota_tape);

    // Reshape targets [N] → [N, 1] then broadcast → [N, V]
    Tensor tgt_col1 = reshape(targets, {N, 1});
    Tensor tgt_col  = module_detail::broadcast_to(tgt_col1, {N, V});

    // compare iota_v == tgt_col  (both S32) → [N, V] i1
    auto cmp_build = [N, V](const std::vector<Buffer*>&) -> Node {
        Node n;
        n.op    = Op::Compare;
        n.shape = {N, V};
        n.dtype = DType::Pred;
        n.cmp   = Cmp::EQ;
        return n;
    };
    auto cmp_tape = [](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.compare(ins[0], ins[1], Cmp::EQ);
    };
    Tensor pred = eager_dispatch_impl(Op::Compare, {&iota_v, &tgt_col},
                                       cmp_build, cmp_tape);

    // convert PRED → F32
    auto conv_build = [N, V](const std::vector<Buffer*>&) -> Node {
        Node n;
        n.op    = Op::Convert;
        n.shape = {N, V};
        n.dtype = DType::F32;
        return n;
    };
    auto conv_tape = [](Graph& g, const std::vector<Value>& ins) -> Value {
        return g.convert(ins[0], DType::F32);
    };
    Tensor onehot = eager_dispatch_impl(Op::Convert, {&pred}, conv_build, conv_tape);

    // picked = sum(log_p * onehot, axis=1)  → [N]
    Tensor picked = reduce_sum(mul(log_p, onehot), {1});

    // nll = -picked  → [N]
    Tensor nll = neg(picked);

    // mean over N
    return reduce_sum(nll, {0}) * (1.0 / (double)N);
}

// ─────────────────────────────────────────────────────────────────────────────
// Module base class (Decision 6)
// ─────────────────────────────────────────────────────────────────────────────
class Module {
public:
    virtual ~Module() = default;

    // Register a parameter owned by this module.
    // `p` must remain valid for the lifetime of this Module (usually a member).
    void register_parameter(const std::string& name, Tensor& p) {
        params_.push_back({name, &p});
    }

    // Register a child module.
    // `m` must remain valid for the lifetime of this Module (usually a member).
    void register_module(const std::string& name, Module& m) {
        children_.push_back({name, &m});
    }

    // Flat list of all parameters (this + all descendants), depth-first.
    // Returns raw pointers so optimizers can rebind buffers.
    //
    // Weight tying (Decision: shared Tensor): if two registered parameters alias
    // the SAME underlying TensorImpl (e.g. a tied embedding/head), the tensor is
    // returned ONCE so the optimizer updates it once and gradients (which
    // accumulate across every use) are applied a single time.  De-dup is by Impl
    // identity, matching the jit auto-lift capture-set semantics.
    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> out;
        std::unordered_set<TensorImpl*> seen;
        collect_params(out, seen);
        return out;
    }

    // Total number of scalar elements across all parameters.
    int64_t param_count() const {
        int64_t total = 0;
        for (auto& [name, p] : params_)
            total += p->numel();
        for (auto& [name, m] : children_)
            total += m->param_count();
        return total;
    }

private:
    void collect_params(std::vector<Tensor*>& out, std::unordered_set<TensorImpl*>& seen) {
        for (auto& [name, p] : params_) {
            if (p->valid() && seen.insert(p->impl_ptr()).second)
                out.push_back(p);
        }
        for (auto& [name, m] : children_)
            m->collect_params(out, seen);
    }

    std::vector<std::pair<std::string, Tensor*>> params_;
    std::vector<std::pair<std::string, Module*>> children_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Initializer helpers (mirrors nn.hpp InitFn style; host-side generation)
// ─────────────────────────────────────────────────────────────────────────────
namespace init {

// Zero-filled float vector.
inline std::vector<float> zeros_data(int64_t n) {
    return std::vector<float>(n, 0.0f);
}

// Normal distribution N(0, stddev).
inline std::vector<float> normal_data(int64_t n, float stddev, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Glorot (fan-in scaled normal): std = sqrt(1 / fan_in).
// fan_in is first dimension of shape (for [in, out] matrices).
inline std::vector<float> glorot_data(const Shape& shape, std::mt19937& rng) {
    float fan_in = shape.empty() ? 1.0f : (float)shape[0];
    float std    = std::sqrt(1.0f / fan_in);
    return normal_data(num_elements(shape), std, rng);
}

}  // namespace init

// ─────────────────────────────────────────────────────────────────────────────
// Linear layer: y = x @ W + b   (x[...,in] → y[...,out])
//
// Parameters:
//   W  [in_features, out_features]  — Glorot init
//   b  [out_features]               — zero init (if bias=true)
//
// Forward: matmul(x, W) then broadcast-add b over leading dims.
// The implementation flattens leading dims into one batch dim (mirroring
// nn::linear) so rank-3+ inputs work correctly.
// ─────────────────────────────────────────────────────────────────────────────
class Linear : public Module {
public:
    Tensor W;
    Tensor b;
    bool   use_bias;

    Linear(int64_t in_features, int64_t out_features, bool bias = true,
           std::mt19937* rng = nullptr, unsigned seed = 1234)
        : use_bias(bias)
    {
        std::mt19937 local_rng(seed);
        std::mt19937& r = rng ? *rng : local_rng;

        // W: [in, out]  Glorot
        Shape ws{in_features, out_features};
        auto wd = init::glorot_data(ws, r);
        W = from_host(wd, ws);
        W.requires_grad_(true);
        register_parameter("W", W);

        // b: [out]  zeros
        if (use_bias) {
            Shape bs{out_features};
            auto bd = init::zeros_data(out_features);
            b = from_host(bd, bs);
            b.requires_grad_(true);
            register_parameter("b", b);
        }
    }

    // x[..., in] → out[..., out]
    Tensor forward(const Tensor& x) const {
        // Flatten all leading dims: [d0, d1, ..., in] → [d0*d1*..., in]
        const Shape& xs  = x.shape();
        int64_t in_feat  = xs.back();
        int64_t lead     = 1;
        for (size_t i = 0; i + 1 < xs.size(); ++i) lead *= xs[i];

        Tensor x2   = reshape(x, {lead, in_feat});
        Tensor y2   = matmul(x2, W);              // [lead, out]

        // Restore leading dims.
        Shape ys = xs;
        ys.back() = W.shape()[1];
        Tensor y = reshape(y2, ys);

        if (use_bias) {
            // b is [out]; broadcast over all leading dims.
            y = add(y, module_detail::broadcast_to(b, ys));
        }
        return y;
    }

    Tensor operator()(const Tensor& x) const { return forward(x); }
};

// ─────────────────────────────────────────────────────────────────────────────
// ReLU activation module / helper
// ─────────────────────────────────────────────────────────────────────────────
class ReLU : public Module {
public:
    // No parameters.
    Tensor forward(const Tensor& x) const { return relu(x); }
    Tensor operator()(const Tensor& x) const { return forward(x); }
};

// ─────────────────────────────────────────────────────────────────────────────
// GELU activation module / helper (tanh approximation, mirrors eager.hpp gelu)
// ─────────────────────────────────────────────────────────────────────────────
class GELU : public Module {
public:
    // No parameters.
    Tensor forward(const Tensor& x) const { return gelu(x); }
    Tensor operator()(const Tensor& x) const { return forward(x); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Sequential: compose layers that all share the same signature
//   Tensor forward(const Tensor&) or operator()(const Tensor&).
//
// Layers are stored as type-erased std::function objects so heterogeneous
// layer types (Linear, ReLU, lambdas, …) all fit in one container without
// requiring a common base forward signature in Module.
//
// Child modules that own parameters are also registered so parameters() walks
// them correctly.  Use add<LayerType>(args...) to build and register in one
// call.
// ─────────────────────────────────────────────────────────────────────────────
class Sequential : public Module {
public:
    using LayerFn = std::function<Tensor(const Tensor&)>;

    // Add a layer by moving a heap-allocated module; register it as a child.
    // The returned reference lets the caller name and inspect the layer.
    template <typename T>
    T& add_module(const std::string& name, std::unique_ptr<T> layer) {
        T* raw = layer.get();
        register_module(name, *raw);
        fns_.push_back([raw](const Tensor& x) { return raw->forward(x); });
        owned_.push_back(std::move(layer));
        return *raw;
    }

    // Convenience: construct-and-add in one call.
    template <typename T, typename... Args>
    T& add(const std::string& name, Args&&... args) {
        return add_module(name, std::make_unique<T>(std::forward<Args>(args)...));
    }

    Tensor forward(const Tensor& x) const {
        Tensor h = x;
        for (auto& fn : fns_) h = fn(h);
        return h;
    }
    Tensor operator()(const Tensor& x) const { return forward(x); }

private:
    std::vector<LayerFn>                          fns_;
    std::vector<std::unique_ptr<Module>>          owned_;
};

// ─────────────────────────────────────────────────────────────────────────────
// MLP — two-layer network: Linear → ReLU → Linear.
// Config: input_dim → hidden_dim → output_dim.
// Holds the two Linear layers as direct members (so their parameters are
// visible through register_module calls in the constructor).
// ─────────────────────────────────────────────────────────────────────────────
class MLP : public Module {
public:
    Linear fc1;
    Linear fc2;

    // hidden_act: "relu" or "gelu" (default relu)
    MLP(int64_t input_dim, int64_t hidden_dim, int64_t output_dim,
        bool bias = true, const std::string& hidden_act = "relu",
        std::mt19937* rng = nullptr, unsigned seed = 1234)
        : fc1(input_dim,  hidden_dim, bias, rng, seed)
        , fc2(hidden_dim, output_dim, bias, rng, seed + 1)
        , hidden_act_(hidden_act)
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    Tensor forward(const Tensor& x) const {
        Tensor h = fc1.forward(x);
        if (hidden_act_ == "gelu") h = gelu(h);
        else                        h = relu(h);
        return fc2.forward(h);
    }
    Tensor operator()(const Tensor& x) const { return forward(x); }

    // Expected parameter count (for sanity check).
    // fc1: W[in,hid] + b[hid]  +  fc2: W[hid,out] + b[out]
    static int64_t expected_param_count(int64_t in, int64_t hid, int64_t out,
                                         bool bias = true) {
        int64_t n = in * hid + hid * out;
        if (bias) n += hid + out;
        return n;
    }

private:
    std::string hidden_act_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Transformer layers (task 5.3) — re-expressed on Tensor.  Logic/shapes are
// ported verbatim from nn.hpp (embedding/rmsnorm/softmax/attention/block); only
// the surface changes from Value+TrainCtx to the eager/jit Tensor frontend, so
// the IR built is identical and the existing VJP rules apply unchanged.
// ─────────────────────────────────────────────────────────────────────────────

// Embedding: integer ids[...] → rows of a learned table[vocab, dim] → [..., dim].
class Embedding : public Module {
public:
    Tensor weight;   // [vocab, dim]

    Embedding(int64_t vocab, int64_t dim, std::mt19937* rng = nullptr,
              unsigned seed = 1234, float init_std = 0.02f) {
        std::mt19937 local(seed);
        std::mt19937& r = rng ? *rng : local;
        Shape ws{vocab, dim};
        weight = from_host(init::normal_data(num_elements(ws), init_std, r), ws);
        weight.requires_grad_(true);
        register_parameter("weight", weight);
    }

    Tensor forward(const Tensor& ids) const { return gather_rows(weight, ids); }
    Tensor operator()(const Tensor& ids) const { return forward(ids); }
};

// RMSNorm over the last dim with a learned scale[dim].
class RMSNorm : public Module {
public:
    Tensor scale;   // [dim]
    double eps;

    RMSNorm(int64_t dim, double eps_ = 1e-5) : eps(eps_) {
        scale = from_host(std::vector<float>((size_t)dim, 1.0f), {dim});
        scale.requires_grad_(true);
        register_parameter("scale", scale);
    }

    Tensor forward(const Tensor& x) const {
        int64_t last = x.rank() - 1;
        Tensor ms    = reduce_mean(mul(x, x), {last}, /*keepdims=*/true);  // [...,1]
        Tensor denom = rsqrt_op(add(ms, scalar_to_tensor(eps, ms)));       // [...,1]
        Tensor norm  = mul(x, denom);                                      // broadcast
        return mul(norm, scale);                                           // broadcast [dim]
    }
    Tensor operator()(const Tensor& x) const { return forward(x); }
};

// softmax over `axis` with max-subtraction (max is stop-grad'd for stability).
inline Tensor softmax(const Tensor& x, int64_t axis) {
    Tensor m = stop_gradient(reduce_max(x, {axis}, /*keepdims=*/true));
    Tensor e = exp_op(sub(x, m));
    Tensor s = reduce_sum(e, {axis}, /*keepdims=*/true);
    return div_op(e, s);
}

// Causal additive mask [T,T]: 0 on/below the diagonal, -1e30 above (constant,
// data-independent — lifts cleanly as a constant capture inside jit).
inline Tensor causal_mask(int64_t T, int device = 0) {
    std::vector<float> m((size_t)(T * T), 0.0f);
    for (int64_t i = 0; i < T; ++i)
        for (int64_t j = i + 1; j < T; ++j)
            m[i * T + j] = -1e30f;
    return from_host(m, {T, T}, DType::F32, device);
}

// Causal multi-head self-attention.  x[B,T,D] → [B,T,D].
class Attention : public Module {
public:
    Linear q, k, v, o;
    int64_t n_head, d_model, head_dim;

    Attention(int64_t d_model_, int64_t n_head_,
              std::mt19937* rng = nullptr, unsigned seed = 1234)
        : q(d_model_, d_model_, /*bias=*/false, rng, seed),
          k(d_model_, d_model_, /*bias=*/false, rng, seed + 1),
          v(d_model_, d_model_, /*bias=*/false, rng, seed + 2),
          o(d_model_, d_model_, /*bias=*/false, rng, seed + 3),
          n_head(n_head_), d_model(d_model_), head_dim(d_model_ / n_head_)
    {
        register_module("q", q); register_module("k", k);
        register_module("v", v); register_module("o", o);
    }

    Tensor forward(const Tensor& x) const {
        const Shape& xs = x.shape();
        int64_t B = xs[0], T = xs[1], D = d_model, H = n_head, hd = head_dim;
        auto split = [&](const Tensor& t) {           // [B,T,D] → [B,H,T,hd]
            return transpose_perm(reshape(t, {B, T, H, hd}), {0, 2, 1, 3});
        };
        Tensor Q = split(q.forward(x)), K = split(k.forward(x)), Vv = split(v.forward(x));
        // scores [B,H,T,T] = q @ k^T / sqrt(hd)
        Tensor scores = matmul(Q, transpose_perm(K, {0, 1, 3, 2}));
        scores = mul(scores, scalar_to_tensor(1.0 / std::sqrt((double)hd), scores));
        scores = add(scores, causal_mask(T, x.device()));   // broadcast [T,T] over [B,H,T,T]
        Tensor attn = softmax(scores, 3);                    // [B,H,T,T]
        Tensor out  = matmul(attn, Vv);                      // [B,H,T,hd]
        out = transpose_perm(out, {0, 2, 1, 3});             // [B,T,H,hd]
        out = reshape(out, {B, T, D});
        return o.forward(out);
    }
    Tensor operator()(const Tensor& x) const { return forward(x); }
};

// Pre-norm transformer block: x + attn(norm(x)); h + mlp(norm(h)).
class Block : public Module {
public:
    RMSNorm   n1, n2;
    Attention attn;
    Linear    fc1, fc2;

    Block(int64_t d_model, int64_t n_head, int64_t d_ff,
          std::mt19937* rng = nullptr, unsigned seed = 1234)
        : n1(d_model), n2(d_model),
          attn(d_model, n_head, rng, seed),
          fc1(d_model, d_ff, /*bias=*/true, rng, seed + 10),
          fc2(d_ff, d_model, /*bias=*/true, rng, seed + 11)
    {
        register_module("n1", n1); register_module("n2", n2);
        register_module("attn", attn);
        register_module("fc1", fc1); register_module("fc2", fc2);
    }

    Tensor forward(const Tensor& x) const {
        Tensor h = add(x, attn.forward(n1.forward(x)));
        Tensor m = fc2.forward(gelu(fc1.forward(n2.forward(h))));
        return add(h, m);
    }
    Tensor operator()(const Tensor& x) const { return forward(x); }
};

// ─────────────────────────────────────────────────────────────────────────────
// GPT composite (task 5.4) — decoder-only transformer on Module/Tensor.
//
// Matches the architecture of gpt.hpp / nn.hpp gpt_logits: learned token + (fixed
// block_size) positional embeddings, n_layer pre-norm blocks, final RMSNorm,
// untied output head.  Sizes wpe to block_size and requires the input T to equal
// block_size (the training regime); generation uses the same T.
// ─────────────────────────────────────────────────────────────────────────────
struct GPTModuleConfig {
    int64_t vocab      = 65;
    int64_t n_layer    = 4;
    int64_t n_head     = 4;
    int64_t d_model    = 128;
    int64_t d_ff       = 512;
    int64_t block_size = 64;
};

class GPT : public Module {
public:
    GPTModuleConfig cfg;
    Embedding wte;                              // token embedding [vocab, d_model]
    Tensor    wpe;                              // positional embedding [block_size, d_model]
    std::vector<std::unique_ptr<Block>> blocks;
    RMSNorm   lnf;
    Linear    head;                             // [d_model, vocab], no bias (untied)

    GPT(const GPTModuleConfig& c, std::mt19937* rng = nullptr, unsigned seed = 1234)
        : cfg(c),
          wte(c.vocab, c.d_model, rng, seed, 0.02f),
          lnf(c.d_model),
          head(c.d_model, c.vocab, /*bias=*/false, rng, seed + 1000)
    {
        std::mt19937 local(seed + 2000);
        std::mt19937& r = rng ? *rng : local;
        Shape ps{c.block_size, c.d_model};
        wpe = from_host(init::normal_data(num_elements(ps), 0.02f, r), ps);
        wpe.requires_grad_(true);

        register_module("wte", wte);
        register_parameter("wpe", wpe);
        for (int64_t l = 0; l < c.n_layer; ++l) {
            blocks.push_back(std::make_unique<Block>(
                c.d_model, c.n_head, c.d_ff, rng, seed + 100 * (unsigned)(l + 1)));
            register_module("h" + std::to_string(l), *blocks.back());
        }
        register_module("lnf", lnf);
        register_module("head", head);
    }

    // ids[B,T] (S32) → logits[B,T,vocab].
    Tensor logits(const Tensor& ids) const {
        const Shape& s = ids.shape();
        if (s.size() != 2) throw Error("GPT::logits: ids must be rank 2 [B,T]");
        int64_t T = s[1];
        if (T != cfg.block_size)
            throw Error("GPT::logits: T must equal block_size in v1");
        Tensor tok = wte.forward(ids);          // [B,T,D]
        Tensor h   = add(tok, wpe);             // wpe [T,D] broadcast over B
        for (auto& b : blocks) h = b->forward(h);
        h = lnf.forward(h);
        return head.forward(h);                 // [B,T,vocab]
    }

    // Mean next-token cross-entropy for ids[B,T] vs targets[B,T].
    Tensor loss(const Tensor& ids, const Tensor& targets) const {
        const Shape& s = ids.shape();
        int64_t B = s[0], T = s[1];
        Tensor lg   = logits(ids);                              // [B,T,V]
        Tensor flat = reshape(lg, {B * T, cfg.vocab});          // [B*T, V]
        Tensor tgt  = reshape(targets, {B * T});                // [B*T]
        return cross_entropy(flat, tgt);
    }

    int64_t expected_param_count() const {
        int64_t D = cfg.d_model, V = cfg.vocab, F = cfg.d_ff, L = cfg.n_layer;
        int64_t per_block = /*n1,n2*/ 2 * D
                          + /*qkvo*/  4 * D * D
                          + /*fc1*/   D * F + F
                          + /*fc2*/   F * D + D;
        return V * D                 // wte
             + cfg.block_size * D    // wpe
             + L * per_block         // blocks
             + D                     // lnf
             + D * V;                // head (untied, no bias)
    }
};

}  // namespace tpu
