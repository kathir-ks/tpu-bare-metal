// nn.hpp — neural-net layers, optimizers, and a Trainer, in pure C++.
//
// Layer 3 of the pure-C++ TPU stack. The Trainer compiles ONE StableHLO program
// for the whole train step (forward + autodiff + optimizer update) and runs it
// repeatedly, keeping parameters and optimizer state resident in TPU HBM across
// steps (outputs of step N feed back as inputs of step N+1). This mirrors how
// JAX functional training works, with no host<->device traffic except the batch
// and the scalar loss.
#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "graph.hpp"
#include "tpu.hpp"

namespace tpu {

// ── initializers ──────────────────────────────────────────────────────────────
using InitFn = std::function<void(std::vector<float>&, const Shape&, std::mt19937&)>;

inline InitFn zeros() {
    return [](std::vector<float>& d, const Shape&, std::mt19937&) {
        std::fill(d.begin(), d.end(), 0.0f);
    };
}
inline InitFn constant_init(float v) {
    return [v](std::vector<float>& d, const Shape&, std::mt19937&) {
        std::fill(d.begin(), d.end(), v);
    };
}
inline InitFn normal(float stddev) {
    return [stddev](std::vector<float>& d, const Shape&, std::mt19937& rng) {
        std::normal_distribution<float> dist(0.0f, stddev);
        for (auto& x : d) x = dist(rng);
    };
}
// fan-in scaled normal (good default for Linear weights [in, out]).
inline InitFn glorot() {
    return [](std::vector<float>& d, const Shape& s, std::mt19937& rng) {
        float fan_in = s.empty() ? 1.0f : (float)s[0];
        float std = std::sqrt(1.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, std);
        for (auto& x : d) x = dist(rng);
    };
}

// Passed to the model function during graph construction. c.param(...) resolves
// a parameter by name (declaring it on first use); repeated names return the
// same Value (weight tying). The resolver is supplied by whoever drives the
// build — the Trainer (which owns device buffers) or a Forward pass (which feeds
// the Trainer's buffers as graph inputs). This lets one model function build
// both training and inference graphs with identical parameter ordering.
class TrainCtx {
  public:
    Graph& g;
    using Resolver = std::function<Value(const std::string&, const Shape&, InitFn)>;
    TrainCtx(Graph& g_, Resolver r) : g(g_), resolve_(std::move(r)) {}
    Value param(const std::string& name, const Shape& shape, InitFn init) {
        return resolve_(name, shape, init);
    }
    // Optional debug outputs the model can request; appended to the step's
    // outputs (right after the loss) and retrievable via Trainer::last_debug().
    std::vector<Value> debug;

  private:
    Resolver resolve_;
};

// ── layer helpers (operate on the graph via TrainCtx) ─────────────────────────
namespace nn {

// x[..., in] @ W[in, out] (+ b[out]) → [..., out]
inline Value linear(TrainCtx& c, const Value& x, const std::string& name,
                    int64_t in, int64_t out, bool bias = true) {
    Graph& g = c.g;
    Value W = c.param(name + ".w", {in, out}, glorot());
    Shape xs = x.shape();
    int64_t lead = 1;
    for (size_t i = 0; i + 1 < xs.size(); i++) lead *= xs[i];
    Value x2 = g.reshape(x, {lead, in});
    Value y2 = g.dot(x2, W);                 // [lead, out]
    Shape ys = xs; ys.back() = out;
    Value y = g.reshape(y2, ys);
    if (bias) {
        Value b = c.param(name + ".b", {out}, zeros());
        y = g.add(y, b);                     // broadcast [out] over leading dims
    }
    return y;
}

// integer ids[B,T] → embedding rows from table[vocab, dim] → [B,T,dim].
// Implemented with stablehlo.gather; gradient is a scatter-add (exact, and
// O(N·dim) instead of the one-hot matmul's O(N·vocab) memory).
inline Value embedding(TrainCtx& c, const Value& ids, const std::string& name,
                       int64_t vocab, int64_t dim) {
    Value table = c.param(name, {vocab, dim}, normal(0.02f));
    return c.g.gather_rows(table, ids);
}

// Old one-hot @ table implementation (kept as a cross-check for the gather
// path; identical math, exact gradient, but O(N·vocab) memory).
inline Value embedding_onehot(TrainCtx& c, const Value& ids, const std::string& name,
                              int64_t vocab, int64_t dim) {
    Graph& g = c.g;
    Value table = c.param(name, {vocab, dim}, normal(0.02f));
    Shape is = ids.shape();
    int64_t n = num_elements(is);
    Value ids2 = g.reshape(ids, {n, 1});                       // [n,1] i32
    Value rng  = g.iota({n, vocab}, 1, DType::S32);            // [n,vocab]
    Value idsB = g.broadcast_to(ids2, {n, vocab});             // [n,vocab]
    Value onehot = g.convert(g.compare(rng, idsB, Cmp::EQ), DType::F32);
    Value out2 = g.dot(onehot, table);                         // [n,dim]
    Shape os = is; os.push_back(dim);
    return g.reshape(out2, os);
}

// GELU (tanh approximation).
inline Value gelu(Graph& g, const Value& x) {
    Value x3 = g.mul(g.mul(x, x), x);
    Value inner = g.mul(g.add(x, g.mul(x3, g.scalar(0.044715))), g.scalar(0.7978845608));
    return g.mul(g.mul(x, g.scalar(0.5)), g.add(g.tanh(inner), g.scalar(1.0)));
}

// RMSNorm over the last dim with a learned scale[dim].
inline Value rmsnorm(TrainCtx& c, const Value& x, const std::string& name,
                     int64_t dim, double eps = 1e-5) {
    Graph& g = c.g;
    Value scale = c.param(name, {dim}, constant_init(1.0f));
    int64_t last = x.rank() - 1;
    Value ms = g.reduce_mean(g.mul(x, x), {last}, true);      // [...,1]
    Value norm = g.mul(x, g.rsqrt(g.add(ms, g.scalar(eps))));
    return g.mul(norm, scale);
}

// softmax over `axis` with max-subtraction for stability (max is stop-grad'd).
inline Value softmax(Graph& g, const Value& x, int64_t axis) {
    Value m = g.stop_gradient(g.reduce_max(x, {axis}, true));
    Value e = g.exp(g.sub(x, m));
    Value s = g.reduce_sum(e, {axis}, true);
    return g.div(e, s);
}

// Causal multi-head self-attention. x[B,T,D] → [B,T,D].
inline Value attention(TrainCtx& c, const Value& x, const std::string& name,
                       int64_t B, int64_t T, int64_t D, int64_t H) {
    Graph& g = c.g;
    int64_t hd = D / H;
    Value q = linear(c, x, name + ".q", D, D, false);
    Value k = linear(c, x, name + ".k", D, D, false);
    Value v = linear(c, x, name + ".v", D, D, false);
    // [B,T,D] -> [B,H,T,hd]
    auto split_heads = [&](const Value& t) {
        Value r = g.reshape(t, {B, T, H, hd});
        return g.transpose(r, {0, 2, 1, 3});
    };
    q = split_heads(q); k = split_heads(k); v = split_heads(v);
    // scores [B,H,T,T] = q @ k^T / sqrt(hd)
    Value scores = g.dot(q, g.transpose_last2(k));
    scores = g.mul(scores, g.scalar(1.0 / std::sqrt((double)hd)));
    // causal mask: col>row → -inf
    Value row = g.iota({T, T}, 0, DType::S32);
    Value col = g.iota({T, T}, 1, DType::S32);
    Value masked = g.compare(col, row, Cmp::GT);               // [T,T] i1
    Value neg = g.constant(-1e30, {T, T});
    Value zero = g.constant(0.0, {T, T});
    Value maskv = g.select(masked, neg, zero);                 // [T,T]
    scores = g.add(scores, maskv);                             // broadcast over B,H
    Value attn = softmax(g, scores, 3);                        // [B,H,T,T]
    Value out = g.dot(attn, v);                                // [B,H,T,hd]
    out = g.transpose(out, {0, 2, 1, 3});                      // [B,T,H,hd]
    out = g.reshape(out, {B, T, D});
    return linear(c, out, name + ".o", D, D, false);
}

// Transformer block (pre-norm): x + attn(norm(x)); h + mlp(norm(h)).
inline Value block(TrainCtx& c, const Value& x, const std::string& name,
                   int64_t B, int64_t T, int64_t D, int64_t H, int64_t ff) {
    Graph& g = c.g;
    Value a = attention(c, rmsnorm(c, x, name + ".n1", D), name + ".attn", B, T, D, H);
    Value h = g.add(x, a);
    Value n2 = rmsnorm(c, h, name + ".n2", D);
    Value m = linear(c, gelu(g, linear(c, n2, name + ".fc1", D, ff)), name + ".fc2", ff, D);
    return g.add(h, m);
}

// Cross-entropy over logits[N, V] with integer targets[N] → scalar mean loss.
inline Value cross_entropy(Graph& g, const Value& logits, const Value& targets,
                           int64_t N, int64_t V) {
    Value m = g.stop_gradient(g.reduce_max(logits, {1}, true));
    Value sh = g.sub(logits, m);
    Value lse = g.add(g.log(g.reduce_sum(g.exp(sh), {1}, true)), m);  // [N,1] logsumexp
    Value logp = g.sub(logits, lse);                                  // [N,V]
    // gather target log-probs via one-hot
    Value tgt = g.reshape(targets, {N, 1});
    Value rng = g.iota({N, V}, 1, DType::S32);
    Value oh = g.convert(g.compare(rng, g.broadcast_to(tgt, {N, V}), Cmp::EQ), DType::F32);
    Value picked = g.reduce_sum(g.mul(logp, oh), {1});                // [N]
    Value nll = g.neg(picked);
    return g.reduce_mean(nll, {0});                                   // scalar
}

}  // namespace nn

// ── optimizer ─────────────────────────────────────────────────────────────────
struct AdamCfg { double b1 = 0.9, b2 = 0.999, eps = 1e-8, weight_decay = 0.0; };

// ── Trainer ───────────────────────────────────────────────────────────────────
// Model function: builds the forward graph and returns the scalar loss.
//   Value model(TrainCtx& c, Value x, Value y);
// x is the int32 input batch, y the int32 targets (shapes you choose).
using ModelFn = std::function<Value(TrainCtx&, Value, Value)>;

class Trainer {
  public:
    Trainer(Context& ctx, AdamCfg cfg = {}, unsigned seed = 1234)
        : ctx_(ctx), cfg_(cfg), rng_(seed) {}

    // Compile the train step. x_shape/y_shape are per-batch shapes.
    void build(ModelFn model,
               const Shape& x_shape, DType x_dt,
               const Shape& y_shape, DType y_dt,
               const std::string& precision = "HIGHEST") {
        g_ = Graph();
        g_.dot_precision = precision;

        // leading fixed inputs: x, y, lr, bc1, bc2
        x_  = g_.input("__x", x_shape, x_dt);
        y_  = g_.input("__y", y_shape, y_dt);
        lr_  = g_.input("__lr",  {}, DType::F32);
        bc1_ = g_.input("__bc1", {}, DType::F32);
        bc2_ = g_.input("__bc2", {}, DType::F32);

        TrainCtx c(g_, [this](const std::string& n, const Shape& s, InitFn i) {
            return declare_param(n, s, i);
        });
        Value loss = model(c, x_, y_);
        loss_ = loss;
        ndebug_ = c.debug.size();

        // params discovered during forward (in pdefs_ order); now create state.
        size_t np = pdefs_.size();
        std::vector<Value> pvals(np);
        for (size_t i = 0; i < np; i++) pvals[i] = Value{&g_, pnode_ids_[i]};
        std::vector<Value> grads = g_.grad(loss, pvals);

        // Adam state inputs (m,v per param) created AFTER params.
        m_ids_.resize(np); v_ids_.resize(np);
        std::vector<Value> outs;
        outs.push_back(loss);
        for (auto& d : c.debug) outs.push_back(d);   // debug outputs follow the loss
        for (size_t i = 0; i < np; i++) {
            Value mIn = g_.input("__m" + std::to_string(i), pdefs_[i].shape);
            Value vIn = g_.input("__v" + std::to_string(i), pdefs_[i].shape);
            m_ids_[i] = mIn.id; v_ids_[i] = vIn.id;
            auto upd = adam_update(g_, pvals[i], grads[i], mIn, vIn);
            new_p_.push_back(upd.p);
            new_m_.push_back(upd.m);
            new_v_.push_back(upd.v);
        }
        for (auto& p : new_p_) outs.push_back(p);
        for (size_t i = 0; i < np; i++) { outs.push_back(new_m_[i]); outs.push_back(new_v_[i]); }
        out_count_ = outs.size();

        // donate param/Adam-state inputs to their updated outputs so XLA
        // updates them in place (no per-step realloc). step() already replaces
        // the host-side Buffer handles with the outputs each step.
        size_t obase = 1 + ndebug_;
        for (size_t i = 0; i < np; i++) {
            g_.arg_aliases[g_.node(pnode_ids_[i]).arg_index] = (int)(obase + i);
            g_.arg_aliases[g_.node(m_ids_[i]).arg_index] = (int)(obase + np + 2 * i);
            g_.arg_aliases[g_.node(v_ids_[i]).arg_index] = (int)(obase + np + 2 * i + 1);
        }

        std::string mlir = g_.emit(outs);
        last_mlir_ = mlir;
        exec_ = ctx_.compile_mlir(mlir);

        // materialize parameter + state buffers on device.
        param_bufs_.clear(); m_bufs_.clear(); v_bufs_.clear();
        for (size_t i = 0; i < np; i++) {
            std::vector<float> host(num_elements(pdefs_[i].shape));
            pdefs_[i].init(host, pdefs_[i].shape, rng_);
            param_bufs_.push_back(ctx_.upload_f32(host, pdefs_[i].shape));
            std::vector<float> z(host.size(), 0.0f);
            m_bufs_.push_back(ctx_.upload_f32(z, pdefs_[i].shape));
            v_bufs_.push_back(ctx_.upload_f32(z, pdefs_[i].shape));
        }
        np_ = np;
    }

    // Run one optimization step on the given batch; returns the scalar loss.
    float step(const std::vector<int32_t>& x, const std::vector<int32_t>& y,
               float lr) {
        t_ += 1;
        float bc1 = 1.0f - (float)std::pow(cfg_.b1, (double)t_);
        float bc2 = 1.0f - (float)std::pow(cfg_.b2, (double)t_);

        Buffer xb = ctx_.upload(x.data(), x_.dtype(), x_.shape());
        Buffer yb = ctx_.upload(y.data(), y_.dtype(), y_.shape());
        float lrv = lr, b1v = bc1, b2v = bc2;
        Buffer lrb  = ctx_.upload(&lrv,  DType::F32, {});
        Buffer b1b  = ctx_.upload(&b1v,  DType::F32, {});
        Buffer b2b  = ctx_.upload(&b2v,  DType::F32, {});

        // args in graph-input order: x,y,lr,bc1,bc2, params..., m0,v0,m1,v1,...
        std::vector<Buffer*> args = {&xb, &yb, &lrb, &b1b, &b2b};
        for (auto& p : param_bufs_) args.push_back(&p);
        for (size_t i = 0; i < np_; i++) { args.push_back(&m_bufs_[i]); args.push_back(&v_bufs_[i]); }

        std::vector<Buffer> outs = exec_.run(args);
        float loss = outs[0].to_host<float>()[0];
        // outs: [loss, debug..., new_p..., new_m0,new_v0,...]
        size_t base = 1 + ndebug_;
        last_debug_.clear();
        for (size_t i = 0; i < ndebug_; i++) last_debug_.push_back(std::move(outs[1 + i]));
        for (size_t i = 0; i < np_; i++) param_bufs_[i] = std::move(outs[base + i]);
        for (size_t i = 0; i < np_; i++) {
            m_bufs_[i] = std::move(outs[base + np_ + 2 * i]);
            v_bufs_[i] = std::move(outs[base + np_ + 2 * i + 1]);
        }
        return loss;
    }

    // run forward-only inference would require a separate build; see examples.
    const std::string& last_mlir() const { return last_mlir_; }
    std::vector<Buffer>& last_debug() { return last_debug_; }
    Context& ctx() { return ctx_; }
    size_t num_params() const { return np_; }
    int64_t param_count() const {
        int64_t n = 0; for (auto& d : pdefs_) n += num_elements(d.shape); return n;
    }

    std::vector<std::string> param_names() const {
        std::vector<std::string> n; for (auto& d : pdefs_) n.push_back(d.name); return n;
    }
    std::vector<std::pair<std::string, Shape>> param_defs() const {
        std::vector<std::pair<std::string, Shape>> v;
        for (auto& d : pdefs_) v.push_back({d.name, d.shape});
        return v;
    }

    // ── checkpointing ────────────────────────────────────────────────────────
    // Simple binary format: [int32 nparams] then per param
    // [int32 name_len][name bytes][int32 nelem][float32 data...].
    void save_checkpoint(const std::string& path) {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) throw Error("cannot open " + path + " for writing");
        int32_t np = (int32_t)pdefs_.size();
        fwrite(&np, sizeof(np), 1, f);
        for (size_t i = 0; i < pdefs_.size(); i++) {
            auto host = param_bufs_[i].to_host<float>();
            int32_t nl = (int32_t)pdefs_[i].name.size();
            int32_t ne = (int32_t)host.size();
            fwrite(&nl, sizeof(nl), 1, f);
            fwrite(pdefs_[i].name.data(), 1, nl, f);
            fwrite(&ne, sizeof(ne), 1, f);
            fwrite(host.data(), sizeof(float), ne, f);
        }
        fclose(f);
    }

    void load_checkpoint(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw Error("cannot open " + path + " for reading");
        int32_t np = 0;
        if (fread(&np, sizeof(np), 1, f) != 1) { fclose(f); throw Error("bad checkpoint"); }
        for (int32_t k = 0; k < np; k++) {
            int32_t nl = 0; if (fread(&nl, sizeof(nl), 1, f) != 1) break;
            std::string name(nl, '\0'); if (fread(&name[0], 1, nl, f) != (size_t)nl) break;
            int32_t ne = 0; if (fread(&ne, sizeof(ne), 1, f) != 1) break;
            std::vector<float> host(ne); if (fread(host.data(), sizeof(float), ne, f) != (size_t)ne) break;
            auto it = pindex_.find(name);
            if (it == pindex_.end()) continue;
            param_bufs_[it->second] = ctx_.upload_f32(host, pdefs_[it->second].shape);
        }
        fclose(f);
    }

    // Used by TrainCtx::param.
    Value declare_param(const std::string& name, const Shape& shape, InitFn init) {
        auto it = pindex_.find(name);
        if (it != pindex_.end()) return Value{&g_, pnode_ids_[it->second]};
        Value v = g_.input(name, shape);
        pindex_[name] = pdefs_.size();
        pnode_ids_.push_back(v.id);
        pdefs_.push_back({name, shape, init});
        return v;
    }

    // Copy current parameter values back to host (name → values).
    std::vector<float> get_param(const std::string& name) {
        auto it = pindex_.find(name);
        if (it == pindex_.end()) throw Error("unknown param " + name);
        return param_bufs_[it->second].to_host<float>();
    }
    // Live device buffer for a parameter (used by Forward/inference). Pointer is
    // valid until the next step() (buffers are replaced in place each step).
    Buffer* param_buffer(const std::string& name) {
        auto it = pindex_.find(name);
        if (it == pindex_.end()) throw Error("unknown param " + name);
        return &param_bufs_[it->second];
    }
    Context& context() { return ctx_; }

  private:
    struct ParamDef { std::string name; Shape shape; InitFn init; };
    struct AdamOut { Value p, m, v; };

    AdamOut adam_update(Graph& g, Value p, Value grad, Value mIn, Value vIn) {
        if (cfg_.weight_decay != 0.0)
            grad = g.add(grad, g.mul(p, g.scalar(cfg_.weight_decay)));
        Value m = g.add(g.mul(mIn, g.scalar(cfg_.b1)),
                        g.mul(grad, g.scalar(1.0 - cfg_.b1)));
        Value v = g.add(g.mul(vIn, g.scalar(cfg_.b2)),
                        g.mul(g.mul(grad, grad), g.scalar(1.0 - cfg_.b2)));
        Value mhat = g.div(m, bc1_);
        Value vhat = g.div(v, bc2_);
        Value upd = g.div(mhat, g.add(g.sqrt(vhat), g.scalar(cfg_.eps)));
        Value newp = g.sub(p, g.mul(lr_, upd));
        return {newp, m, v};
    }

    Context& ctx_;
    AdamCfg  cfg_;
    std::mt19937 rng_;

    Graph g_;
    Value x_, y_, lr_, bc1_, bc2_, loss_;
    std::vector<ParamDef> pdefs_;
    std::map<std::string, size_t> pindex_;
    std::vector<int> pnode_ids_;
    std::vector<int> m_ids_, v_ids_;
    std::vector<Value> new_p_, new_m_, new_v_;
    size_t out_count_ = 0, np_ = 0, ndebug_ = 0;
    int64_t t_ = 0;

    Executable exec_;
    std::vector<Buffer> param_bufs_, m_bufs_, v_bufs_, last_debug_;
    std::string last_mlir_;
};

// ── Data-parallel Trainer (uses all addressable TPU chips) ───────────────────
// Replicates parameters + Adam state across `nd` devices. Each step shards the
// global batch across replicas; per-replica gradients are summed with
// all_reduce and averaged, so every replica applies the same update and weights
// stay bit-identical. Communication is one cross-replica sum per parameter.
class DataParallelTrainer {
  public:
    DataParallelTrainer(Context& ctx, AdamCfg cfg = {}, unsigned seed = 1234)
        : ctx_(ctx), cfg_(cfg), rng_(seed) {
        nd_ = ctx_.num_addressable_devices();
    }

    int num_replicas() const { return nd_; }

    // x_shape/y_shape are PER-REPLICA batch shapes. The global batch fed to
    // step() has nd * per_replica rows.
    void build(ModelFn model,
               const Shape& x_shape, DType x_dt,
               const Shape& y_shape, DType y_dt,
               const std::string& precision = "DEFAULT") {
        g_ = Graph();
        g_.dot_precision = precision;
        g_.num_replicas  = nd_;

        x_   = g_.input("__x", x_shape, x_dt);
        y_   = g_.input("__y", y_shape, y_dt);
        lr_  = g_.input("__lr",  {}, DType::F32);
        bc1_ = g_.input("__bc1", {}, DType::F32);
        bc2_ = g_.input("__bc2", {}, DType::F32);

        TrainCtx c(g_, [this](const std::string& n, const Shape& s, InitFn i) {
            return declare_param(n, s, i);
        });
        Value loss = model(c, x_, y_);

        size_t np = pdefs_.size();
        std::vector<Value> pvals(np);
        for (size_t i = 0; i < np; i++) pvals[i] = Value{&g_, pnode_ids_[i]};
        std::vector<Value> grads = g_.grad(loss, pvals);

        std::vector<Value> outs;
        outs.push_back(loss);
        double scale = 1.0 / (double)nd_;
        for (size_t i = 0; i < np; i++) {
            Value mIn = g_.input("__m" + std::to_string(i), pdefs_[i].shape);
            Value vIn = g_.input("__v" + std::to_string(i), pdefs_[i].shape);
            m_ids_.push_back(mIn.id); v_ids_.push_back(vIn.id);
            // average gradient across replicas
            Value gavg = g_.mul(g_.all_reduce_sum(grads[i]), g_.scalar(scale));
            auto upd = adam_update(g_, pvals[i], gavg, mIn, vIn);
            new_p_.push_back(upd.p); new_m_.push_back(upd.m); new_v_.push_back(upd.v);
        }
        for (auto& p : new_p_) outs.push_back(p);
        for (size_t i = 0; i < np; i++) { outs.push_back(new_m_[i]); outs.push_back(new_v_[i]); }

        // donate replicated param/Adam-state inputs to their updated outputs
        // (outs: [loss, new_p..., new_m0,new_v0,...]).
        for (size_t i = 0; i < np; i++) {
            g_.arg_aliases[g_.node(pnode_ids_[i]).arg_index] = (int)(1 + i);
            g_.arg_aliases[g_.node(m_ids_[i]).arg_index] = (int)(1 + np + 2 * i);
            g_.arg_aliases[g_.node(v_ids_[i]).arg_index] = (int)(1 + np + 2 * i + 1);
        }

        last_mlir_ = g_.emit(outs);
        exec_ = ctx_.compile_mlir_dp(last_mlir_);
        order_ = exec_.device_order(nd_);
        np_ = np;

        // replicate params + state on each device (identical init values).
        param_bufs_.resize(nd_);
        m_bufs_.resize(nd_);
        v_bufs_.resize(nd_);
        for (size_t i = 0; i < np; i++) {
            std::vector<float> host(num_elements(pdefs_[i].shape));
            pdefs_[i].init(host, pdefs_[i].shape, rng_);
            std::vector<float> z(host.size(), 0.0f);
            for (int r = 0; r < nd_; r++) {
                param_bufs_[r].push_back(ctx_.upload_f32(host, pdefs_[i].shape, order_[r]));
                m_bufs_[r].push_back(ctx_.upload_f32(z, pdefs_[i].shape, order_[r]));
                v_bufs_[r].push_back(ctx_.upload_f32(z, pdefs_[i].shape, order_[r]));
            }
        }
    }

    // global_x / global_y hold nd * per_replica rows (replica r = shard r).
    float step(const std::vector<int32_t>& global_x,
               const std::vector<int32_t>& global_y, float lr) {
        t_ += 1;
        float bc1 = 1.0f - (float)std::pow(cfg_.b1, (double)t_);
        float bc2 = 1.0f - (float)std::pow(cfg_.b2, (double)t_);
        size_t xper = num_elements(x_.shape());
        size_t yper = num_elements(y_.shape());

        // hold per-replica inputs alive through the run.
        std::vector<Buffer> xb(nd_), yb(nd_), lrb(nd_), b1b(nd_), b2b(nd_);
        std::vector<std::vector<Buffer*>> args(nd_);
        for (int r = 0; r < nd_; r++) {
            int dev = order_[r];
            std::vector<int32_t> xs(global_x.begin() + r * xper, global_x.begin() + (r + 1) * xper);
            std::vector<int32_t> ys(global_y.begin() + r * yper, global_y.begin() + (r + 1) * yper);
            xb[r] = ctx_.upload(xs.data(), x_.dtype(), x_.shape(), dev);
            yb[r] = ctx_.upload(ys.data(), y_.dtype(), y_.shape(), dev);
            float lrv = lr, b1v = bc1, b2v = bc2;
            lrb[r] = ctx_.upload(&lrv, DType::F32, {}, dev);
            b1b[r] = ctx_.upload(&b1v, DType::F32, {}, dev);
            b2b[r] = ctx_.upload(&b2v, DType::F32, {}, dev);
            args[r] = {&xb[r], &yb[r], &lrb[r], &b1b[r], &b2b[r]};
            for (auto& p : param_bufs_[r]) args[r].push_back(&p);
            for (size_t i = 0; i < np_; i++) { args[r].push_back(&m_bufs_[r][i]); args[r].push_back(&v_bufs_[r][i]); }
        }

        auto outs = exec_.run_spmd(args);   // outs[r] = [loss, new_p..., new_m,new_v]
        double loss_sum = 0;
        for (int r = 0; r < nd_; r++) {
            loss_sum += outs[r][0].to_host<float>()[0];
            for (size_t i = 0; i < np_; i++) param_bufs_[r][i] = std::move(outs[r][1 + i]);
            for (size_t i = 0; i < np_; i++) {
                m_bufs_[r][i] = std::move(outs[r][1 + np_ + 2 * i]);
                v_bufs_[r][i] = std::move(outs[r][1 + np_ + 2 * i + 1]);
            }
        }
        return (float)(loss_sum / nd_);
    }

    // parameters are identical across replicas; read replica 0 (on device 0).
    std::vector<float> get_param(const std::string& name) {
        auto it = pindex_.find(name);
        if (it == pindex_.end()) throw Error("unknown param " + name);
        return param_bufs_[0][it->second].to_host<float>();
    }
    Buffer* param_buffer(const std::string& name) {
        auto it = pindex_.find(name);
        if (it == pindex_.end()) throw Error("unknown param " + name);
        return &param_bufs_[0][it->second];  // replica 0 is on addressable device 0
    }
    Context& context() { return ctx_; }
    const std::string& last_mlir() const { return last_mlir_; }

    // ── checkpointing (same format as Trainer) ──────────────────────────────
    // Params are identical across replicas: save replica 0, restore to all.
    void save_checkpoint(const std::string& path) {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) throw Error("cannot open " + path + " for writing");
        int32_t np = (int32_t)pdefs_.size();
        fwrite(&np, sizeof(np), 1, f);
        for (size_t i = 0; i < pdefs_.size(); i++) {
            auto host = param_bufs_[0][i].to_host<float>();
            int32_t nl = (int32_t)pdefs_[i].name.size();
            int32_t ne = (int32_t)host.size();
            fwrite(&nl, sizeof(nl), 1, f);
            fwrite(pdefs_[i].name.data(), 1, nl, f);
            fwrite(&ne, sizeof(ne), 1, f);
            fwrite(host.data(), sizeof(float), ne, f);
        }
        fclose(f);
    }

    void load_checkpoint(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw Error("cannot open " + path + " for reading");
        int32_t np = 0;
        if (fread(&np, sizeof(np), 1, f) != 1) { fclose(f); throw Error("bad checkpoint"); }
        for (int32_t kk = 0; kk < np; kk++) {
            int32_t nl = 0; if (fread(&nl, sizeof(nl), 1, f) != 1) break;
            std::string name(nl, '\0'); if (fread(&name[0], 1, nl, f) != (size_t)nl) break;
            int32_t ne = 0; if (fread(&ne, sizeof(ne), 1, f) != 1) break;
            std::vector<float> host(ne);
            if (fread(host.data(), sizeof(float), ne, f) != (size_t)ne) break;
            auto it = pindex_.find(name);
            if (it == pindex_.end()) continue;
            for (int r = 0; r < nd_; r++)
                param_bufs_[r][it->second] =
                    ctx_.upload_f32(host, pdefs_[it->second].shape, order_[r]);
        }
        fclose(f);
    }
    int64_t param_count() const {
        int64_t n = 0; for (auto& d : pdefs_) n += num_elements(d.shape); return n;
    }

    Value declare_param(const std::string& name, const Shape& shape, InitFn init) {
        auto it = pindex_.find(name);
        if (it != pindex_.end()) return Value{&g_, pnode_ids_[it->second]};
        Value v = g_.input(name, shape);
        pindex_[name] = pdefs_.size();
        pnode_ids_.push_back(v.id);
        pdefs_.push_back({name, shape, init});
        return v;
    }

  private:
    struct ParamDef { std::string name; Shape shape; InitFn init; };
    struct AdamOut { Value p, m, v; };
    AdamOut adam_update(Graph& g, Value p, Value grad, Value mIn, Value vIn) {
        if (cfg_.weight_decay != 0.0)
            grad = g.add(grad, g.mul(p, g.scalar(cfg_.weight_decay)));
        Value m = g.add(g.mul(mIn, g.scalar(cfg_.b1)), g.mul(grad, g.scalar(1.0 - cfg_.b1)));
        Value v = g.add(g.mul(vIn, g.scalar(cfg_.b2)), g.mul(g.mul(grad, grad), g.scalar(1.0 - cfg_.b2)));
        Value mhat = g.div(m, bc1_), vhat = g.div(v, bc2_);
        Value upd = g.div(mhat, g.add(g.sqrt(vhat), g.scalar(cfg_.eps)));
        return {g.sub(p, g.mul(lr_, upd)), m, v};
    }

    Context& ctx_;
    AdamCfg  cfg_;
    std::mt19937 rng_;
    int nd_ = 1;
    Graph g_;
    Value x_, y_, lr_, bc1_, bc2_;
    std::vector<ParamDef> pdefs_;
    std::map<std::string, size_t> pindex_;
    std::vector<int> pnode_ids_, m_ids_, v_ids_, order_;
    std::vector<Value> new_p_, new_m_, new_v_;
    size_t np_ = 0;
    int64_t t_ = 0;
    Executable exec_;
    std::vector<std::vector<Buffer>> param_bufs_, m_bufs_, v_bufs_;
    std::string last_mlir_;
};

// ── Forward / inference ───────────────────────────────────────────────────────
// Builds a separate forward-only executable whose parameters are graph inputs,
// fed at run time from a trainer's live device buffers. Because it builds with
// the SAME model code, parameters appear in the same first-use order/names.
// Templated on trainer type so it works with Trainer and DataParallelTrainer
// (the latter exposes replica-0 buffers, which live on addressable device 0).
template <class TR>
class ForwardT {
  public:
    explicit ForwardT(TR& tr) : tr_(tr) {}

    // fwd_fn(c, x) returns the output Values (e.g. logits).
    void build(const Shape& x_shape, DType x_dt,
               std::function<std::vector<Value>(TrainCtx&, Value)> fwd_fn,
               const std::string& precision = "HIGHEST") {
        g_ = Graph();
        g_.dot_precision = precision;
        x_ = g_.input("__x", x_shape, x_dt);
        names_.clear(); idx_.clear(); ids_.clear();
        TrainCtx c(g_, [this](const std::string& n, const Shape& s, InitFn) {
            auto it = idx_.find(n);
            if (it != idx_.end()) return Value{&g_, ids_[it->second]};
            Value v = g_.input(n, s);
            idx_[n] = names_.size(); names_.push_back(n); ids_.push_back(v.id);
            return v;
        });
        outs_ = fwd_fn(c, x_);
        exec_ = tr_.context().compile_mlir(g_.emit(outs_));
    }

    std::vector<Buffer> run(const std::vector<int32_t>& x) {
        Buffer xb = tr_.context().upload(x.data(), x_.dtype(), x_.shape());
        std::vector<Buffer*> args = {&xb};
        for (auto& n : names_) args.push_back(tr_.param_buffer(n));
        return exec_.run(args);
    }

  private:
    TR&      tr_;
    Graph    g_;
    Value    x_;
    std::vector<Value> outs_;
    std::map<std::string, size_t> idx_;
    std::vector<std::string> names_;
    std::vector<int> ids_;
    Executable exec_;
};

using Forward   = ForwardT<Trainer>;
using ForwardDP = ForwardT<DataParallelTrainer>;

}  // namespace tpu
