// optim.hpp — Optimizer objects for the eager/jit frontend (T6 / Wave C).
//
// Layer 6 of the eager-jit-tensor-core design (Decision 7).
//
// What lives here:
//   AdamCfg       — hyperparameter pod (mirrors nn.hpp AdamCfg exactly).
//   Optimizer     — abstract base: holds params, zero_grad(), virtual step().
//   SGD           — stochastic gradient descent with optional momentum.
//   Adam          — Adam with bias-corrected moments, weight decay, ported
//                   exactly from the nn.hpp adam_update() formulation.
//
// Execution model (v1):
//   Eager step(): read param->grad(), compute update via eager Tensor ops,
//   rebind_buffer() the param (and m/v state) to the new Tensor's buffer.
//   No device kernel is ever re-executed for a param; only the Adam arithmetic
//   touches the device.
//
// JIT seam (T4 integration — NOT implemented here):
//   When inside a jit trace the optimizer step should be part of the traced
//   graph: params and m/v state tensors auto-lift as donation-aliased
//   inputs (requires_grad captured → differentiable + donation), so the
//   forward + backward + Adam update compile into one fused executable.
//   The seam: Adam::step() can detect active_trace (via eager.hpp) and
//   delegate to a trace-recording path.  In v1 only the eager path is wired;
//   the jit path is documented below under "JIT seam" to guide integration.
//
// Include after module.hpp (which transitively includes tensor.hpp+eager.hpp).
// Header-only, C++17.
#pragma once
#ifndef TPU_OPTIM_HPP
#define TPU_OPTIM_HPP

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "module.hpp"   // pulls tensor.hpp → eager.hpp → graph.hpp / tpu.hpp
#include "jit.hpp"      // Trace / trace_forward / CaptureEntry for JitAdamStep

namespace tpu {

// ─────────────────────────────────────────────────────────────────────────────
// AdamCfg — hyperparameter pod.
//
// Mirrors nn.hpp AdamCfg exactly so nn.hpp callers can pass the same struct.
// Defaults chosen to match the benchmarked GPT training path:
//   b1 = 0.9   (first-moment EMA)
//   b2 = 0.999 (second-moment EMA; set to 0.95 to match old GPT path if desired)
//   eps = 1e-8
//   weight_decay = 0.0
// ─────────────────────────────────────────────────────────────────────────────
struct AdamCfg {
    double b1           = 0.9;
    double b2           = 0.999;
    double eps          = 1e-8;
    double weight_decay = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Optimizer — abstract base class.
//
// Holds a flat list of parameter Tensor* (raw, non-owning) gathered from the
// Module at construction time.  The Module must outlive the Optimizer.
//
// zero_grad(): clears every param's .grad field.
//   Implementation: set_grad(nullptr) resets the accumulated gradient.
//   No new Tensor method is needed — set_grad(nullptr) is already part of the
//   tensor.hpp contract (line 122).
//
// step(): pure virtual; subclasses implement the update rule.
// ─────────────────────────────────────────────────────────────────────────────
class Optimizer {
public:
    // Construct from an explicit parameter list (e.g. model.parameters()).
    explicit Optimizer(std::vector<Tensor*> params)
        : params_(std::move(params))
    {}

    virtual ~Optimizer() = default;

    // Non-copyable (holds raw pointers to external Tensors).
    Optimizer(const Optimizer&)            = delete;
    Optimizer& operator=(const Optimizer&) = delete;

    // Zero all parameter gradients.
    // After this call, param->grad().valid() == false for every param.
    void zero_grad() {
        for (Tensor* p : params_) {
            if (p && p->valid())
                p->set_grad(nullptr);  // tensor.hpp line 122: set_grad(shared_ptr)
        }
    }

    // Apply the update rule.  Subclasses iterate params_ and rebind buffers.
    virtual void step() = 0;

    // Read-only access to the parameter list.
    const std::vector<Tensor*>& params() const { return params_; }

protected:
    std::vector<Tensor*> params_;
};

// ─────────────────────────────────────────────────────────────────────────────
// SGD — stochastic gradient descent with optional momentum.
//
// Update rule (per parameter p with gradient g):
//
//   Without momentum (momentum == 0):
//     p ← p − lr * g
//
//   With momentum (momentum > 0):
//     buf ← momentum * buf + g          (velocity buffer, init zeros)
//     p   ← p − lr * buf
//
// Momentum buffers are held as Tensors (lazily initialised to zeros on the
// first step, matching the param shape/dtype/device).
// ─────────────────────────────────────────────────────────────────────────────
class SGD : public Optimizer {
public:
    // params  — flat parameter list (e.g. from model.parameters()).
    // lr      — learning rate (positive scalar).
    // momentum — coefficient in [0, 1); 0 means vanilla SGD.
    SGD(std::vector<Tensor*> params, double lr, double momentum = 0.0)
        : Optimizer(std::move(params))
        , lr_(lr)
        , momentum_(momentum)
    {
        if (lr_ <= 0.0)
            throw std::invalid_argument("SGD: lr must be positive");
        if (momentum_ < 0.0 || momentum_ >= 1.0)
            throw std::invalid_argument("SGD: momentum must be in [0, 1)");
        // Momentum buffers are allocated lazily on first step.
        if (momentum_ > 0.0)
            vel_.resize(params_.size());   // Tensor default-ctor → invalid/null
    }

    // Eager step: update each param using its current .grad.
    // Skips params with null .grad (not yet differentiated or zero_grad'd).
    void step() override {
        for (size_t i = 0; i < params_.size(); ++i) {
            Tensor* p = params_[i];
            if (!p || !p->valid()) continue;

            Tensor g = p->grad();
            if (!g.valid()) continue;   // no gradient for this param this step

            if (momentum_ > 0.0) {
                // Lazily init velocity to zeros.
                if (!vel_[i].valid()) {
                    vel_[i] = zeros(p->shape(), p->dtype(), p->device());
                }
                // buf = momentum * buf + g
                Tensor m_s = scalar_to_tensor(momentum_, vel_[i]);
                Tensor new_vel = add(mul(m_s, vel_[i]), g);
                vel_[i].rebind_buffer(new_vel.shared_buffer());

                // p = p - lr * buf
                Tensor lr_s = scalar_to_tensor(lr_, vel_[i]);
                Tensor new_p = sub(*p, mul(lr_s, vel_[i]));
                p->rebind_buffer(new_p.shared_buffer());
            } else {
                // p = p - lr * g
                Tensor lr_s = scalar_to_tensor(lr_, g);
                Tensor new_p = sub(*p, mul(lr_s, g));
                p->rebind_buffer(new_p.shared_buffer());
            }
        }
    }

    double lr()       const { return lr_; }
    double momentum() const { return momentum_; }
    void   set_lr(double lr) { lr_ = lr; }

private:
    double  lr_;
    double  momentum_;
    std::vector<Tensor> vel_;   // momentum buffers; empty when momentum == 0
};

// ─────────────────────────────────────────────────────────────────────────────
// Adam — Adam optimiser (Kingma & Ba 2014).
//
// This ports the formula from nn.hpp adam_update() EXACTLY.  Do not redesign.
//
// Update rule (per parameter p with gradient g, step count t starting at 1):
//
//   if weight_decay > 0:
//     g ← g + weight_decay * p          (decoupled L2, applied to raw grad)
//
//   m ← b1 * m + (1 - b1) * g           (first-moment EMA, init 0)
//   v ← b2 * v + (1 - b2) * g²          (second-moment EMA, init 0)
//
//   bc1  = 1 - b1^t                      (bias correction: first moment)
//   bc2  = 1 - b2^t                      (bias correction: second moment)
//
//   mhat = m / bc1
//   vhat = v / bc2
//
//   p ← p − lr * mhat / (sqrt(vhat) + eps)
//
// Bias correction:
//   bc1 and bc2 are computed on the host (pure floating-point; no device op)
//   and broadcast to tensor shape via scalar_to_tensor().  step counter t is
//   incremented in step(), matching the nn.hpp Trainer's t_ counter semantics.
//
// State tensors m and v are held per-parameter in m_state_ / v_state_ and are
// lazily initialised to zeros on the first step.  They are rebound each step
// (same SSA semantics as params).
//
// JIT seam (integration note for T4 / orchestrator):
//   In the jit path, step() should NOT be called as a standalone eager update.
//   Instead, the traced step function (produced by jit(lambda)) must:
//     1. Auto-lift all params_ and m_state_ / v_state_ tensors as captured inputs
//        into the trace graph (they will appear alongside explicit x/y inputs).
//     2. Mark requires_grad params and their m/v state as donation-aliased
//        (Graph::arg_aliases / tf.aliasing_output) so XLA updates them in-place.
//     3. After execute(), call rebind_buffer() on each param and state tensor
//        with the corresponding output buffer — same as today's Trainer / DataParallelTrainer.
//     4. Increment step_ on the host after each execute (as nn.hpp does with t_).
//   The fused path produces the benchmarked numbers because the entire
//   forward+backward+Adam sequence is one compiled XLA computation with no
//   intermediate HBM round-trips.
// ─────────────────────────────────────────────────────────────────────────────
class Adam : public Optimizer {
public:
    // params — flat parameter list from model.parameters().
    // lr     — learning rate.
    // cfg    — AdamCfg hyperparameters (defaults: b1=0.9, b2=0.999, eps=1e-8).
    Adam(std::vector<Tensor*> params, double lr, AdamCfg cfg = {})
        : Optimizer(std::move(params))
        , lr_(lr)
        , cfg_(cfg)
        , step_(0)
    {
        if (lr_ <= 0.0)
            throw std::invalid_argument("Adam: lr must be positive");
        if (cfg_.b1 <= 0.0 || cfg_.b1 >= 1.0)
            throw std::invalid_argument("Adam: b1 must be in (0, 1)");
        if (cfg_.b2 <= 0.0 || cfg_.b2 >= 1.0)
            throw std::invalid_argument("Adam: b2 must be in (0, 1)");
        if (cfg_.eps <= 0.0)
            throw std::invalid_argument("Adam: eps must be positive");

        // Preallocate state slots (default-constructed Tensor = null/invalid).
        m_state_.resize(params_.size());
        v_state_.resize(params_.size());
    }

    // Eager Adam step.
    //
    // For each param with a non-null .grad:
    //   1. Lazily initialise m, v to zeros.
    //   2. Compute bias-correction scalars bc1, bc2 (host-side pow).
    //   3. Execute the Adam update via eager Tensor ops.
    //   4. rebind_buffer() param, m, v to the new computed Tensors.
    //
    // Increments step_ before computing bias corrections (t starts at 1).
    void step() override {
        ++step_;   // step counter: 1-indexed (matches nn.hpp t_ after increment)

        // Bias corrections — host-side scalar arithmetic (one pow per step).
        double bc1 = 1.0 - std::pow(cfg_.b1, (double)step_);
        double bc2 = 1.0 - std::pow(cfg_.b2, (double)step_);

        for (size_t i = 0; i < params_.size(); ++i) {
            Tensor* p = params_[i];
            if (!p || !p->valid()) continue;

            Tensor g = p->grad();
            if (!g.valid()) continue;   // param has no gradient this step

            // ── Lazily init m and v to zeros (same shape/dtype/device as param) ──
            if (!m_state_[i].valid()) {
                m_state_[i] = zeros(p->shape(), p->dtype(), p->device());
            }
            if (!v_state_[i].valid()) {
                v_state_[i] = zeros(p->shape(), p->dtype(), p->device());
            }

            Tensor& m = m_state_[i];
            Tensor& v = v_state_[i];

            // ── Weight decay: g ← g + weight_decay * p ──────────────────────────
            // Port of nn.hpp adam_update line:
            //   if (cfg_.weight_decay != 0.0)
            //     grad = g.add(grad, g.mul(p, g.scalar(cfg_.weight_decay)));
            if (cfg_.weight_decay != 0.0) {
                Tensor wd_s = scalar_to_tensor(cfg_.weight_decay, g);
                g = add(g, mul(wd_s, *p));
            }

            // ── First moment:  m ← b1 * m + (1 - b1) * g ───────────────────────
            // Port of:
            //   Value m = g.add(g.mul(mIn, g.scalar(cfg_.b1)),
            //                   g.mul(grad, g.scalar(1.0 - cfg_.b1)));
            Tensor b1_s      = scalar_to_tensor(cfg_.b1,        m);
            Tensor one_mb1_s = scalar_to_tensor(1.0 - cfg_.b1,  g);
            Tensor new_m     = add(mul(b1_s, m), mul(one_mb1_s, g));

            // ── Second moment: v ← b2 * v + (1 - b2) * g² ──────────────────────
            // Port of:
            //   Value v = g.add(g.mul(vIn, g.scalar(cfg_.b2)),
            //                   g.mul(g.mul(grad, grad), g.scalar(1.0 - cfg_.b2)));
            Tensor b2_s      = scalar_to_tensor(cfg_.b2,        v);
            Tensor one_mb2_s = scalar_to_tensor(1.0 - cfg_.b2,  g);
            Tensor g2        = mul(g, g);
            Tensor new_v     = add(mul(b2_s, v), mul(one_mb2_s, g2));

            // ── Bias-corrected estimates ─────────────────────────────────────────
            // Port of:
            //   Value mhat = g.div(m, bc1_);
            //   Value vhat = g.div(v, bc2_);
            // bc1_ and bc2_ in nn.hpp are scalar tensors; here we use
            // scalar_to_tensor so the division is a broadcast elementwise div.
            Tensor bc1_s = scalar_to_tensor(bc1, new_m);
            Tensor bc2_s = scalar_to_tensor(bc2, new_v);
            Tensor mhat  = div_op(new_m, bc1_s);
            Tensor vhat  = div_op(new_v, bc2_s);

            // ── Parameter update: p ← p − lr * mhat / (sqrt(vhat) + eps) ────────
            // Port of:
            //   Value upd  = g.div(mhat, g.add(g.sqrt(vhat), g.scalar(cfg_.eps)));
            //   Value newp = g.sub(p, g.mul(lr_, upd));
            Tensor eps_s  = scalar_to_tensor(cfg_.eps, vhat);
            Tensor denom  = add(sqrt_op(vhat), eps_s);
            Tensor upd    = div_op(mhat, denom);
            Tensor lr_s   = scalar_to_tensor(lr_, upd);
            Tensor new_p  = sub(*p, mul(lr_s, upd));

            // ── Rebind buffers (Decision 5: mutable bindings, SSA underneath) ───
            p->rebind_buffer(new_p.shared_buffer());
            m.rebind_buffer(new_m.shared_buffer());
            v.rebind_buffer(new_v.shared_buffer());
        }
    }

    // Accessors / mutators.
    double    lr()      const { return lr_; }
    AdamCfg   cfg()     const { return cfg_; }
    int64_t   step_count() const { return step_; }
    void      set_lr(double lr) { lr_ = lr; }
    void      reset_step()      { step_ = 0; }

    // State tensors (for inspection / jit integration).
    // Returns null Tensor if the state has not been initialised yet (before
    // the first step).
    Tensor m_state(size_t param_idx) const {
        if (param_idx >= m_state_.size()) return Tensor{};
        return m_state_[param_idx];
    }
    Tensor v_state(size_t param_idx) const {
        if (param_idx >= v_state_.size()) return Tensor{};
        return v_state_[param_idx];
    }

private:
    double  lr_;
    AdamCfg cfg_;
    int64_t step_;              // 1-indexed step counter for bias correction

    // Per-param Adam state (first moment m, second moment v).
    // Initialised lazily to zeros on the first step so we do not need the
    // PJRT context at construction time.
    std::vector<Tensor> m_state_;
    std::vector<Tensor> v_state_;
};

// ─────────────────────────────────────────────────────────────────────────────
// JitAdamStep — fully fused forward + backward + Adam-update training step.
//
// This is the benchmarked path: the entire (loss, grad, Adam) sequence compiles
// into ONE XLA executable with no intermediate HBM round-trips, and params + m/v
// state are donation-aliased (tf.aliasing_output) so XLA updates them in place.
// Mirrors the proven nn.hpp::Trainer layout exactly, but drives off auto-lifted
// Module parameters instead of declare_param.
//
// Usage:
//   JitAdamStep step([&](std::vector<Tensor> b){ return loss_of(b[0], b[1]); },
//                    model.parameters(), lr, AdamCfg{});
//   for (...) { float l = step({xb, yb}).to_host()[0]; }   // params updated in place
//
// Graph layout (mirrors Trainer):
//   inputs : [explicit args] ++ [captures (lift order)] ++ [m0,v0,m1,v1,...] ++ [lr,bc1,bc2]
//   outputs: [loss] ++ [new_p0..np-1] ++ [new_m0,new_v0,new_m1,new_v1,...]
//   donate : param/m/v inputs aliased to their updated outputs.
//
// Only the requires_grad captures (the model params) are differentiated and
// updated; non-param captures (e.g. a closed-over constant) are fed each call
// but neither differentiated nor donated.  lr/bc1/bc2 are scalar INPUTS, so the
// compiled graph is independent of the step counter (no recompile per step).
// num_replicas > 1 is deferred to the eager DataParallelTrainer path in v1.
// ─────────────────────────────────────────────────────────────────────────────
class JitAdamStep {
public:
    using FnType = std::function<Tensor(std::vector<Tensor>)>;

    JitAdamStep(FnType fn, std::vector<Tensor*> params, double lr,
                AdamCfg cfg = {}, int num_replicas = 1, std::string precision = "")
        : fn_(std::move(fn)), params_(std::move(params)), lr_(lr), cfg_(cfg),
          num_replicas_(num_replicas),
          precision_(precision.empty() ? default_dot_precision() : precision)
    {
        if (lr_ <= 0.0) throw std::invalid_argument("JitAdamStep: lr must be positive");
        if (num_replicas_ != 1)
            throw Error("JitAdamStep: num_replicas > 1 not supported in v1 (use DataParallelTrainer)");
        for (auto* p : params_)
            if (!p || !p->requires_grad())
                throw Error("JitAdamStep: every param must have requires_grad=true");
    }

    // Run one fused training step; updates params in place; returns scalar loss.
    Tensor operator()(std::vector<Tensor> args) {
        std::string sig = detail::make_sig_key(args);
        auto it = cache_.find(sig);
        if (it == cache_.end()) return compile_and_run(sig, std::move(args));
        return execute_cached(it->second, args);
    }
    Tensor operator()(Tensor a)            { return (*this)(std::vector<Tensor>{std::move(a)}); }
    Tensor operator()(Tensor a, Tensor b)  { return (*this)(std::vector<Tensor>{std::move(a), std::move(b)}); }

    int64_t step_count() const { return t_; }
    double  lr()         const { return lr_; }

private:
    struct Entry {
        Executable          exec;
        std::vector<Tensor> captured;     // ALL captures, lift order (buffer re-read)
        std::vector<Tensor> param_caps;   // requires_grad captures (alias user params)
        std::vector<Buffer> m_bufs;       // Adam first moment per param_cap
        std::vector<Buffer> v_bufs;       // Adam second moment per param_cap
        Shape  loss_shape;  DType loss_dtype = DType::F32;
        int    num_explicit_args = 0;
        int    np = 0;
    };

    FnType               fn_;
    std::vector<Tensor*> params_;
    double               lr_;
    AdamCfg              cfg_;
    int                  num_replicas_;
    std::string          precision_;
    int64_t              t_ = 0;
    std::unordered_map<std::string, Entry> cache_;

    // Append the in-graph Adam update for one param; returns (new_p,new_m,new_v).
    struct AdamNodes { Value p, m, v; };
    AdamNodes adam_update(Graph& g, Value p, Value grad, Value mIn, Value vIn,
                          Value lr, Value bc1, Value bc2) {
        if (cfg_.weight_decay != 0.0)
            grad = g.add(grad, g.mul(p, g.scalar(cfg_.weight_decay)));
        Value m = g.add(g.mul(mIn, g.scalar(cfg_.b1)),
                        g.mul(grad, g.scalar(1.0 - cfg_.b1)));
        Value v = g.add(g.mul(vIn, g.scalar(cfg_.b2)),
                        g.mul(g.mul(grad, grad), g.scalar(1.0 - cfg_.b2)));
        Value mhat = g.div(m, bc1);
        Value vhat = g.div(v, bc2);
        Value upd  = g.div(mhat, g.add(g.sqrt(vhat), g.scalar(cfg_.eps)));
        Value newp = g.sub(p, g.mul(lr, upd));
        return {newp, m, v};
    }

    Tensor compile_and_run(const std::string& sig, std::vector<Tensor> args) {
        fprintf(stderr, "[jit_adam_step] compiling for signature %s\n", sig.c_str());

        Trace trace;
        Tensor result = trace_forward(fn_, args, trace);

        Graph& g = *trace.graph();
        g.dot_precision = precision_;
        g.num_replicas  = num_replicas_;
        Value loss_val  = result.trace_value();

        Entry e;
        e.num_explicit_args = (int)args.size();
        e.loss_shape = result.shape();
        e.loss_dtype = result.dtype();

        // ALL captures (lift order) become Inputs fed each call.
        e.captured.reserve(trace.captures.size());
        for (auto& ce : trace.captures) e.captured.push_back(ce.tensor);

        // The requires_grad captures (in lift order) are the params we update.
        std::vector<Value> pvals;
        std::vector<int>   p_arg_index;
        for (auto& ce : trace.captures) {
            if (!ce.is_param) continue;
            e.param_caps.push_back(ce.tensor);
            pvals.push_back(ce.trace_val);
            p_arg_index.push_back(g.node(ce.trace_val.id).arg_index);
        }
        int np = (int)pvals.size();
        e.np = np;
        if (np == 0) throw Error("JitAdamStep: no requires_grad params were used in the step fn");

        // Backward: grads for every param (lift order).
        std::vector<Value> grads = g.grad(loss_val, pvals);

        // Adam-state Inputs created AFTER params: m0,v0,m1,v1,...
        std::vector<Value> mIns(np), vIns(np);
        std::vector<int>   m_arg_index(np), v_arg_index(np);
        for (int i = 0; i < np; ++i) {
            mIns[i] = g.input("__m" + std::to_string(i), e.param_caps[i].shape());
            vIns[i] = g.input("__v" + std::to_string(i), e.param_caps[i].shape());
            m_arg_index[i] = g.node(mIns[i].id).arg_index;
            v_arg_index[i] = g.node(vIns[i].id).arg_index;
        }
        // Step scalars as Inputs (created last): lr, bc1, bc2.
        Value lrIn  = g.input("__lr",  {}, DType::F32);
        Value bc1In = g.input("__bc1", {}, DType::F32);
        Value bc2In = g.input("__bc2", {}, DType::F32);

        // In-graph Adam update per param.
        std::vector<Value> new_p(np), new_m(np), new_v(np);
        for (int i = 0; i < np; ++i) {
            AdamNodes u = adam_update(g, pvals[i], grads[i], mIns[i], vIns[i],
                                      lrIn, bc1In, bc2In);
            new_p[i] = u.p; new_m[i] = u.m; new_v[i] = u.v;
        }

        // Outputs: [loss] ++ new_p... ++ [new_m0,new_v0,...]
        std::vector<Value> outs;
        outs.push_back(loss_val);
        for (int i = 0; i < np; ++i) outs.push_back(new_p[i]);
        for (int i = 0; i < np; ++i) { outs.push_back(new_m[i]); outs.push_back(new_v[i]); }

        // Donation: param/m/v inputs alias their updated outputs (in-place).
        const int base = 1;
        for (int i = 0; i < np; ++i) {
            g.arg_aliases[p_arg_index[i]] = base + i;
            g.arg_aliases[m_arg_index[i]] = base + np + 2 * i;
            g.arg_aliases[v_arg_index[i]] = base + np + 2 * i + 1;
        }

        std::string mlir  = g.emit(outs);
        std::string copts = make_compile_options(num_replicas_);
        e.exec = global_context().compile_mlir(mlir, copts.data(), copts.size());

        // Init Adam state to zeros on device (one buffer per param).
        for (int i = 0; i < np; ++i) {
            int64_t n = e.param_caps[i].numel();
            std::vector<float> z((size_t)n, 0.0f);
            e.m_bufs.push_back(global_context().upload_f32(z, e.param_caps[i].shape(), 0));
            e.v_bufs.push_back(global_context().upload_f32(z, e.param_caps[i].shape(), 0));
        }

        Entry& stored = (cache_[sig] = std::move(e));
        return execute_cached(stored, args);
    }

    Tensor execute_cached(Entry& e, const std::vector<Tensor>& args) {
        ++t_;
        float bc1 = 1.0f - (float)std::pow(cfg_.b1, (double)t_);
        float bc2 = 1.0f - (float)std::pow(cfg_.b2, (double)t_);
        float lrv = (float)lr_;

        // Scalar input buffers (NOT donated; recreated each step).
        Buffer lrb  = global_context().upload_f32(std::vector<float>{lrv},  {}, 0);
        Buffer bc1b = global_context().upload_f32(std::vector<float>{bc1}, {}, 0);
        Buffer bc2b = global_context().upload_f32(std::vector<float>{bc2}, {}, 0);

        // Feed order must match graph input registration order exactly:
        //   [explicit args] ++ [captures] ++ [m0,v0,m1,v1,...] ++ [lr,bc1,bc2]
        std::vector<Buffer*> bufs;
        for (auto& a : args) {
            if (!a.is_concrete())
                throw Error("JitAdamStep: explicit arg must be Concrete on execute");
            bufs.push_back(a.raw_buffer());
        }
        for (auto& ct : e.captured) bufs.push_back(ct.raw_buffer());
        for (int i = 0; i < e.np; ++i) { bufs.push_back(&e.m_bufs[i]); bufs.push_back(&e.v_bufs[i]); }
        bufs.push_back(&lrb); bufs.push_back(&bc1b); bufs.push_back(&bc2b);

        auto outs = e.exec.run(bufs, 0);
        if (outs.size() != (size_t)(1 + 3 * e.np))
            throw Error("JitAdamStep: unexpected output count");

        // Rebind params in place (capture.tensor aliases the user param Impl).
        const int base = 1;
        for (int i = 0; i < e.np; ++i)
            e.param_caps[i].rebind_buffer(std::make_shared<Buffer>(std::move(outs[base + i])));
        for (int i = 0; i < e.np; ++i) {
            e.m_bufs[i] = std::move(outs[base + e.np + 2 * i]);
            e.v_bufs[i] = std::move(outs[base + e.np + 2 * i + 1]);
        }

        return Tensor{make_concrete_impl(std::move(outs[0]), e.loss_shape, e.loss_dtype, 0)};
    }
};

}  // namespace tpu

#endif  // TPU_OPTIM_HPP
