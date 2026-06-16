// graph.hpp — A StableHLO computation graph builder with reverse-mode autodiff.
//
// Layer 2 of the pure-C++ TPU stack. You build a DAG of Values using natural
// C++ operators and method calls; the graph emits StableHLO MLIR text that the
// TPU's libtpu.so compiles directly (Context::compile_mlir). Autodiff produces
// gradient Values for any chosen parameters, all within the same graph, so a
// full train step (forward + backward + optimizer) compiles to one executable.
//
// Design notes:
//   * Node ids are assigned in creation order, which is always topological
//     (operands are created before their consumers). Emission and backprop both
//     rely on this.
//   * Elementwise binary ops perform numpy-style implicit broadcasting by
//     inserting explicit broadcast_in_dim nodes, so every elementwise node has
//     equal-shape operands and a trivial local gradient.
//   * dot() treats the last two axes as a matrix and all leading axes as batch
//     dims (which must match on both operands). This covers 2D matmul (no batch)
//     and batched matmul (attention) with one rule.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tpu.hpp"

namespace tpu {

enum class Op {
    Input, Constant,
    Add, Sub, Mul, Div, Max, Min,        // elementwise binary (equal shapes)
    Neg, Exp, Log, Sqrt, Rsqrt, Tanh, Abs, Logistic,  // unary
    Dot,                                  // batched matmul
    Transpose, Reshape, Broadcast,        // shape ops
    Slice, Pad,                           // sub-tensor extract / zero-pad (duals)
    ReduceSum, ReduceMax,                 // reductions (no keepdims)
    Select, Compare, Convert, Iota,       // misc
    Gather, ScatterAdd,                   // row lookup + its adjoint
    StopGradient,
    AllReduce,                            // cross-replica sum (data parallelism)
};

enum class Cmp { GT, GE, EQ, LT, LE, NE };

struct Node {
    Op           op;
    Shape        shape;
    DType        dtype;
    std::vector<int> inputs;
    // op-specific attributes
    std::vector<int64_t> ints;   // axes / perm / broadcast_dims / iota dim
    double       fval = 0;       // constant value
    Cmp          cmp  = Cmp::GT;
    std::string  name;          // for Input nodes: argument label (debug only)
    int          arg_index = -1; // for Input nodes: position in func signature
};

class Graph;

// Lightweight handle to a node in a Graph.
struct Value {
    Graph* g  = nullptr;
    int    id = -1;
    Value() = default;
    Value(Graph* g_, int id_) : g(g_), id(id_) {}
    bool valid() const { return g != nullptr && id >= 0; }
    const Shape& shape() const;
    DType dtype() const;
    int64_t rank() const { return (int64_t)shape().size(); }

    // operators (defined after Graph)
    Value operator+(const Value& o) const;
    Value operator-(const Value& o) const;
    Value operator*(const Value& o) const;
    Value operator/(const Value& o) const;
    Value operator-() const;
    Value operator+(double s) const;
    Value operator-(double s) const;
    Value operator*(double s) const;
    Value operator/(double s) const;
};

class Graph {
  public:
    Graph() = default;

    // dot_general precision: "HIGHEST" (accurate f32), "HIGH", or "DEFAULT"
    // (fast bf16 passes). Default HIGHEST so gradients are numerically exact;
    // lower it for throughput once a model is validated.
    std::string dot_precision = "HIGHEST";

    // Number of replicas for cross-replica all_reduce (data parallelism). Set to
    // the device count when compiling an SPMD/replicated executable.
    int num_replicas = 1;

    // Buffer donation: arg_index -> output_index. Emitted as
    // `{tf.aliasing_output = K : i32}` on the function argument, telling XLA to
    // reuse the input buffer for that output (in-place update). The donated
    // input buffer is consumed by execution and must not be reused by the host.
    std::map<int, int> arg_aliases;

    // ── leaves ──────────────────────────────────────────────────────────────
    Value input(const std::string& name, const Shape& shape, DType dt = DType::F32) {
        Node n; n.op = Op::Input; n.shape = shape; n.dtype = dt; n.name = name;
        n.arg_index = (int)args_.size();
        int id = add(n);
        args_.push_back(id);
        return {this, id};
    }

    // splat constant of `val` with shape/dtype.
    Value constant(double val, const Shape& shape = {}, DType dt = DType::F32) {
        Node n; n.op = Op::Constant; n.shape = shape; n.dtype = dt; n.fval = val;
        return {this, add(n)};
    }
    Value scalar(double val, DType dt = DType::F32) { return constant(val, {}, dt); }

    Value iota(const Shape& shape, int64_t dim, DType dt = DType::S32) {
        Node n; n.op = Op::Iota; n.shape = shape; n.dtype = dt; n.ints = {dim};
        return {this, add(n)};
    }

    // ── elementwise unary ────────────────────────────────────────────────────
    Value unary(Op op, const Value& a) {
        Node n; n.op = op; n.shape = node(a.id).shape; n.dtype = node(a.id).dtype;
        n.inputs = {a.id};
        return {this, add(n)};
    }
    Value exp(const Value& a)   { return unary(Op::Exp, a); }
    Value log(const Value& a)   { return unary(Op::Log, a); }
    Value sqrt(const Value& a)  { return unary(Op::Sqrt, a); }
    Value rsqrt(const Value& a) { return unary(Op::Rsqrt, a); }
    Value tanh(const Value& a)  { return unary(Op::Tanh, a); }
    Value abs(const Value& a)   { return unary(Op::Abs, a); }
    Value neg(const Value& a)   { return unary(Op::Neg, a); }
    Value logistic(const Value& a) { return unary(Op::Logistic, a); }
    Value stop_gradient(const Value& a) { return unary(Op::StopGradient, a); }
    // Sum a value across all replicas (data-parallel gradient reduction).
    Value all_reduce_sum(const Value& a) { return unary(Op::AllReduce, a); }

    // ── elementwise binary (with numpy broadcasting) ─────────────────────────
    Value binary(Op op, Value a, Value b) {
        Shape rs = broadcast_shape(node(a.id).shape, node(b.id).shape);
        a = broadcast_to(a, rs);
        b = broadcast_to(b, rs);
        Node n; n.op = op; n.shape = rs; n.dtype = node(a.id).dtype;
        n.inputs = {a.id, b.id};
        return {this, add(n)};
    }
    Value add(Value a, Value b) { return binary(Op::Add, a, b); }
    Value sub(Value a, Value b) { return binary(Op::Sub, a, b); }
    Value mul(Value a, Value b) { return binary(Op::Mul, a, b); }
    Value div(Value a, Value b) { return binary(Op::Div, a, b); }
    Value max(Value a, Value b) { return binary(Op::Max, a, b); }
    Value min(Value a, Value b) { return binary(Op::Min, a, b); }

    // compare → i1 tensor (broadcasted)
    Value compare(Value a, Value b, Cmp c) {
        Shape rs = broadcast_shape(node(a.id).shape, node(b.id).shape);
        a = broadcast_to(a, rs);
        b = broadcast_to(b, rs);
        Node n; n.op = Op::Compare; n.shape = rs; n.dtype = DType::Pred;
        n.inputs = {a.id, b.id}; n.cmp = c;
        return {this, add(n)};
    }

    // select(pred, on_true, on_false), broadcasting all three to a common shape.
    Value select(Value pred, Value a, Value b) {
        Shape rs = broadcast_shape(node(a.id).shape, node(b.id).shape);
        rs = broadcast_shape(rs, node(pred.id).shape);
        pred = broadcast_to(pred, rs);
        a = broadcast_to(a, rs);
        b = broadcast_to(b, rs);
        Node n; n.op = Op::Select; n.shape = rs; n.dtype = node(a.id).dtype;
        n.inputs = {pred.id, a.id, b.id};
        return {this, add(n)};
    }

    // ── shape ops ────────────────────────────────────────────────────────────
    Value reshape(const Value& a, const Shape& s) {
        const Shape& in = node(a.id).shape;
        if (num_elements(s) != num_elements(in))
            throw Error("reshape: element count mismatch (" +
                        std::to_string(num_elements(in)) + " -> " +
                        std::to_string(num_elements(s)) + ")");
        Node n; n.op = Op::Reshape; n.shape = s; n.dtype = node(a.id).dtype;
        n.inputs = {a.id};
        return {this, add(n)};
    }

    Value transpose(const Value& a, const std::vector<int64_t>& perm) {
        const Shape& in = node(a.id).shape;
        int64_t r = (int64_t)in.size();
        if ((int64_t)perm.size() != r)
            throw Error("transpose: perm length " + std::to_string(perm.size()) +
                        " != input rank " + std::to_string(r));
        std::vector<bool> seen(r, false);
        for (int64_t p : perm) {
            if (p < 0 || p >= r)
                throw Error("transpose: perm axis " + std::to_string(p) +
                            " out of range for rank " + std::to_string(r));
            if (seen[p]) throw Error("transpose: perm axis " + std::to_string(p) + " repeated");
            seen[p] = true;
        }
        Shape s(perm.size());
        for (size_t i = 0; i < perm.size(); i++) s[i] = in[perm[i]];
        Node n; n.op = Op::Transpose; n.shape = s; n.dtype = node(a.id).dtype;
        n.inputs = {a.id}; n.ints = perm;
        return {this, add(n)};
    }

    // transpose the last two axes (matrix transpose within batch).
    Value transpose_last2(const Value& a) {
        int64_t r = a.rank();
        std::vector<int64_t> perm(r);
        for (int64_t i = 0; i < r; i++) perm[i] = i;
        std::swap(perm[r - 1], perm[r - 2]);
        return transpose(a, perm);
    }

    // broadcast_in_dim with explicit broadcast_dimensions (operand axis i → result axis bd[i]).
    Value broadcast_in_dim(const Value& a, const Shape& target,
                           const std::vector<int64_t>& bd) {
        Node n; n.op = Op::Broadcast; n.shape = target; n.dtype = node(a.id).dtype;
        n.inputs = {a.id}; n.ints = bd;
        return {this, add(n)};
    }

    // numpy-style broadcast of `a` up to `target` (trailing alignment). No-op if equal.
    Value broadcast_to(const Value& a, const Shape& target) {
        const Shape& in = node(a.id).shape;
        if (in == target) return a;
        int64_t r = (int64_t)in.size(), R = (int64_t)target.size();
        std::vector<int64_t> bd(r);
        for (int64_t i = 0; i < r; i++) bd[i] = R - r + i;  // trailing alignment
        return broadcast_in_dim(a, target, bd);
    }

    // slice(a, start, limit): extract a[start_i : limit_i] along each dim (stride 1).
    // n.ints stores [start..., limit...]. VJP = pad with zeros (its dual).
    Value slice(const Value& a, const std::vector<int64_t>& start,
                const std::vector<int64_t>& limit) {
        Shape in = node(a.id).shape;
        DType dt = node(a.id).dtype;
        int64_t r = (int64_t)in.size();
        if ((int64_t)start.size() != r || (int64_t)limit.size() != r)
            throw Error("slice: start/limit length must equal input rank " + std::to_string(r));
        Shape s(r);
        for (int64_t i = 0; i < r; i++) {
            if (start[i] < 0 || limit[i] > in[i] || start[i] > limit[i])
                throw Error("slice: invalid [start,limit] for dim " + std::to_string(i));
            s[i] = limit[i] - start[i];
        }
        Node n; n.op = Op::Slice; n.shape = s; n.dtype = dt; n.inputs = {a.id};
        n.ints = start; n.ints.insert(n.ints.end(), limit.begin(), limit.end());
        return {this, add(n)};
    }

    // pad(a, low, high): zero-pad low_i before / high_i after each dim (interior 0).
    // n.ints stores [low..., high...]; inputs {a, zero}. VJP = slice (its dual).
    Value pad(const Value& a, const std::vector<int64_t>& low,
              const std::vector<int64_t>& high) {
        Shape in = node(a.id).shape;        // copy: scalar() below reallocates nodes_
        DType dt = node(a.id).dtype;
        int64_t r = (int64_t)in.size();
        if ((int64_t)low.size() != r || (int64_t)high.size() != r)
            throw Error("pad: low/high length must equal input rank " + std::to_string(r));
        Shape s(r);
        for (int64_t i = 0; i < r; i++) {
            if (low[i] < 0 || high[i] < 0) throw Error("pad: negative padding");
            s[i] = in[i] + low[i] + high[i];
        }
        Value zero = scalar(0.0, dt);
        Node n; n.op = Op::Pad; n.shape = s; n.dtype = dt;
        n.inputs = {a.id, zero.id};
        n.ints = low; n.ints.insert(n.ints.end(), high.begin(), high.end());
        return {this, add(n)};
    }

    // ── reductions (remove `axes`; use keepdims=true to keep them as size 1) ──
    Value reduce_sum(const Value& a, std::vector<int64_t> axes, bool keepdims = false) {
        return reduce(Op::ReduceSum, a, std::move(axes), keepdims);
    }
    Value reduce_max(const Value& a, std::vector<int64_t> axes, bool keepdims = false) {
        return reduce(Op::ReduceMax, a, std::move(axes), keepdims);
    }
    Value reduce_mean(const Value& a, std::vector<int64_t> axes, bool keepdims = false) {
        Shape in = node(a.id).shape;
        int64_t cnt = 1;
        for (int64_t ax : axes) cnt *= in[ax];
        Value s = reduce_sum(a, axes, keepdims);
        return mul(s, scalar(1.0 / (double)cnt, node(a.id).dtype));
    }

    Value convert(const Value& a, DType dt) {
        if (node(a.id).dtype == dt) return a;
        Node n; n.op = Op::Convert; n.shape = node(a.id).shape; n.dtype = dt;
        n.inputs = {a.id};
        return {this, add(n)};
    }

    // ── row gather / scatter ─────────────────────────────────────────────────
    // gather_rows(table[V,D...], ids) → ids.shape + [D...]: row lookup along
    // table's leading axis (embedding). ids is any-shape integer tensor.
    Value gather_rows(const Value& table, const Value& ids) {
        const Shape& ts = node(table.id).shape;
        if (ts.empty()) throw Error("gather_rows: table must be rank >= 1");
        Shape out = node(ids.id).shape;
        out.insert(out.end(), ts.begin() + 1, ts.end());
        Node n; n.op = Op::Gather; n.shape = out; n.dtype = node(table.id).dtype;
        n.inputs = {table.id, ids.id};
        return {this, add(n)};
    }

    // scatter_add_rows(operand[V,D...], ids, updates[ids.shape, D...]):
    // operand with updates[i] added into row ids[i] (duplicates accumulate).
    // This is the VJP of gather_rows; exposed for completeness.
    Value scatter_add_rows(const Value& operand, const Value& ids, const Value& updates) {
        Node n; n.op = Op::ScatterAdd; n.shape = node(operand.id).shape;
        n.dtype = node(operand.id).dtype;
        n.inputs = {operand.id, ids.id, updates.id};
        return {this, add(n)};
    }

    // ── batched matmul: a[...,M,K] @ b[...,K,N] → [...,M,N] ───────────────────
    Value dot(const Value& a, const Value& b) {
        const Shape& A = node(a.id).shape;
        const Shape& B = node(b.id).shape;
        int64_t ra = (int64_t)A.size(), rb = (int64_t)B.size();
        if (ra < 2 || rb < 2) throw Error("dot: operands must be rank >= 2");
        int64_t batch = ra - 2;
        if (rb - 2 != batch) throw Error("dot: batch rank mismatch");
        for (int64_t i = 0; i < batch; i++)
            if (A[i] != B[i]) throw Error("dot: batch dim mismatch");
        if (A[ra - 1] != B[rb - 2]) throw Error("dot: contracting dim mismatch");
        Shape out(A.begin(), A.begin() + batch);
        out.push_back(A[ra - 2]);   // M
        out.push_back(B[rb - 1]);   // N
        Node n; n.op = Op::Dot; n.shape = out; n.dtype = node(a.id).dtype;
        n.inputs = {a.id, b.id}; n.ints = {batch};
        return {this, add(n)};
    }

    // ── node access ──────────────────────────────────────────────────────────
    Node&       node(int id)       { return nodes_[id]; }
    const Node& node(int id) const { return nodes_[id]; }
    int num_nodes() const { return (int)nodes_.size(); }
    const std::vector<int>& args() const { return args_; }

    // ══ Autodiff ═════════════════════════════════════════════════════════════
    // Given a scalar `loss` and a set of parameter Values, return the gradient
    // Value for each (same order). Gradients are new nodes appended to the graph.
    std::vector<Value> grad(const Value& loss, const std::vector<Value>& params);

    // ══ Emission ═════════════════════════════════════════════════════════════
    // Emit a StableHLO module whose @main takes the registered inputs (in
    // registration order) and returns `outputs` (in order).
    std::string emit(const std::vector<Value>& outputs) const;

  private:
    std::vector<Node> nodes_;
    std::vector<int>  args_;

    int add(const Node& n) { nodes_.push_back(n); return (int)nodes_.size() - 1; }

    Value reduce(Op op, const Value& a, std::vector<int64_t> axes, bool keepdims) {
        Shape in = node(a.id).shape;   // copy: scalar()/add() below reallocate nodes_
        DType in_dt = node(a.id).dtype;
        int64_t rank = (int64_t)in.size();
        // normalize negative axes, bounds-check, & sort
        for (auto& ax : axes) {
            if (ax < 0) ax += rank;
            if (ax < 0 || ax >= rank)
                throw Error("reduce: axis " + std::to_string(ax) +
                            " out of range for rank " + std::to_string(rank));
        }
        std::sort(axes.begin(), axes.end());
        Shape reduced;
        std::vector<bool> drop(in.size(), false);
        for (int64_t ax : axes) drop[ax] = true;
        for (size_t i = 0; i < in.size(); i++) if (!drop[i]) reduced.push_back(in[i]);
        // init value for the reduction region (sum→0, max→-inf)
        double initv = (op == Op::ReduceMax) ? -3.402823466e+38 : 0.0;
        Value init = scalar(initv, in_dt);
        Node n; n.op = op; n.shape = reduced; n.dtype = in_dt;
        n.inputs = {a.id, init.id}; n.ints = axes;
        Value r{this, add(n)};
        if (!keepdims) return r;
        // reinsert size-1 dims
        Shape kept = in;
        for (int64_t ax : axes) kept[ax] = 1;
        return reshape(r, kept);
    }

    static Shape broadcast_shape(const Shape& a, const Shape& b) {
        int64_t ra = (int64_t)a.size(), rb = (int64_t)b.size();
        int64_t R = std::max(ra, rb);
        Shape out(R);
        for (int64_t i = 0; i < R; i++) {
            int64_t da = (i < R - ra) ? 1 : a[i - (R - ra)];
            int64_t db = (i < R - rb) ? 1 : b[i - (R - rb)];
            if (da == db)      out[i] = da;
            else if (da == 1)  out[i] = db;
            else if (db == 1)  out[i] = da;
            else throw Error("broadcast: incompatible shapes");
        }
        return out;
    }

    // emission helpers
    static std::string type_str(const Shape& s, DType dt);
    std::string emit_node(int id, const std::string& ssa) const;
};

// ── Value operator implementations ───────────────────────────────────────────
inline const Shape& Value::shape() const { return g->node(id).shape; }
inline DType Value::dtype() const { return g->node(id).dtype; }
inline Value Value::operator+(const Value& o) const { return g->add(*this, o); }
inline Value Value::operator-(const Value& o) const { return g->sub(*this, o); }
inline Value Value::operator*(const Value& o) const { return g->mul(*this, o); }
inline Value Value::operator/(const Value& o) const { return g->div(*this, o); }
inline Value Value::operator-() const { return g->neg(*this); }
inline Value Value::operator+(double s) const { return g->add(*this, g->scalar(s, dtype())); }
inline Value Value::operator-(double s) const { return g->sub(*this, g->scalar(s, dtype())); }
inline Value Value::operator*(double s) const { return g->mul(*this, g->scalar(s, dtype())); }
inline Value Value::operator/(double s) const { return g->div(*this, g->scalar(s, dtype())); }

inline Value operator+(double s, const Value& v) { return v.g->add(v.g->scalar(s, v.dtype()), v); }
inline Value operator*(double s, const Value& v) { return v.g->mul(v.g->scalar(s, v.dtype()), v); }
inline Value operator-(double s, const Value& v) { return v.g->sub(v.g->scalar(s, v.dtype()), v); }

}  // namespace tpu
