// graph.cpp — StableHLO emission and reverse-mode autodiff for Graph.
#include "graph.hpp"

#include <cstdio>

namespace tpu {

// ── formatting helpers ───────────────────────────────────────────────────────
static std::string fmt_float(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9e", v);
    return buf;
}

std::string Graph::type_str(const Shape& s, DType dt) {
    std::string t = "tensor<";
    for (int64_t d : s) { t += std::to_string(d); t += "x"; }
    t += mlir_dtype(dt);
    t += ">";
    return t;
}

static std::string join_i64(const std::vector<int64_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    return s;
}

static const char* cmp_tok(Cmp c) {
    switch (c) {
        case Cmp::GT: return "GT";
        case Cmp::GE: return "GE";
        case Cmp::EQ: return "EQ";
        case Cmp::LT: return "LT";
        case Cmp::LE: return "LE";
        case Cmp::NE: return "NE";
    }
    return "EQ";
}

// SSA name for a node: function args get %argN, everything else %vN.
static std::string ssa_name(const Node& n, int id) {
    if (n.op == Op::Input) return "%arg" + std::to_string(n.arg_index);
    return "%v" + std::to_string(id);
}

std::string Graph::emit_node(int id, const std::string& ssa) const {
    const Node& n = nodes_[id];
    auto T  = [&](int nid) { return type_str(nodes_[nid].shape, nodes_[nid].dtype); };
    auto NM = [&](int nid) { return ssa_name(nodes_[nid], nid); };
    std::string self_t = type_str(n.shape, n.dtype);
    std::ostringstream o;

    auto bin = [&](const char* name) {
        o << "  " << ssa << " = stablehlo." << name << " " << NM(n.inputs[0])
          << ", " << NM(n.inputs[1]) << " : " << self_t;
    };
    auto un = [&](const char* name) {
        o << "  " << ssa << " = stablehlo." << name << " " << NM(n.inputs[0])
          << " : " << self_t;
    };

    switch (n.op) {
        case Op::Constant: {
            std::string lit;
            if (n.dtype == DType::Pred)      lit = (n.fval != 0) ? "true" : "false";
            else if (n.dtype == DType::F32 || n.dtype == DType::F64 ||
                     n.dtype == DType::BF16 || n.dtype == DType::F16)
                lit = fmt_float(n.fval);
            else lit = std::to_string((int64_t)n.fval);
            o << "  " << ssa << " = stablehlo.constant dense<" << lit << "> : " << self_t;
            break;
        }
        case Op::Iota:
            o << "  " << ssa << " = stablehlo.iota dim = " << n.ints[0] << " : " << self_t;
            break;
        case Op::Add: bin("add"); break;
        case Op::Sub: bin("subtract"); break;
        case Op::Mul: bin("multiply"); break;
        case Op::Div: bin("divide"); break;
        case Op::Max: bin("maximum"); break;
        case Op::Min: bin("minimum"); break;
        case Op::Neg: un("negate"); break;
        case Op::Exp: un("exponential"); break;
        case Op::Log: un("log"); break;
        case Op::Sqrt: un("sqrt"); break;
        case Op::Rsqrt: un("rsqrt"); break;
        case Op::Tanh: un("tanh"); break;
        case Op::Abs: un("abs"); break;
        case Op::Logistic: un("logistic"); break;
        case Op::StopGradient:
            o << "  " << ssa << " = stablehlo.optimization_barrier " << NM(n.inputs[0])
              << " : " << self_t;
            break;
        case Op::AllReduce: {
            std::string et = mlir_dtype(n.dtype);
            std::string groups = "dense<[[";
            for (int r = 0; r < num_replicas; r++) {
                if (r) groups += ", ";
                groups += std::to_string(r);
            }
            groups += "]]> : tensor<1x" + std::to_string(num_replicas) + "xi64>";
            o << "  " << ssa << " = \"stablehlo.all_reduce\"(" << NM(n.inputs[0]) << ") ({\n"
              << "  ^bb0(%are_a: tensor<" << et << ">, %are_b: tensor<" << et << ">):\n"
              << "    %are_s = stablehlo.add %are_a, %are_b : tensor<" << et << ">\n"
              << "    stablehlo.return %are_s : tensor<" << et << ">\n"
              << "  }) {replica_groups = " << groups << "} : ("
              << T(n.inputs[0]) << ") -> " << self_t;
            break;
        }
        case Op::Convert:
            o << "  " << ssa << " = stablehlo.convert " << NM(n.inputs[0])
              << " : (" << T(n.inputs[0]) << ") -> " << self_t;
            break;
        case Op::Compare:
            o << "  " << ssa << " = stablehlo.compare " << cmp_tok(n.cmp) << ", "
              << NM(n.inputs[0]) << ", " << NM(n.inputs[1]) << " : ("
              << T(n.inputs[0]) << ", " << T(n.inputs[1]) << ") -> " << self_t;
            break;
        case Op::Select:
            o << "  " << ssa << " = stablehlo.select " << NM(n.inputs[0]) << ", "
              << NM(n.inputs[1]) << ", " << NM(n.inputs[2]) << " : "
              << T(n.inputs[0]) << ", " << self_t;
            break;
        case Op::Reshape:
            o << "  " << ssa << " = stablehlo.reshape " << NM(n.inputs[0])
              << " : (" << T(n.inputs[0]) << ") -> " << self_t;
            break;
        case Op::Transpose:
            o << "  " << ssa << " = stablehlo.transpose " << NM(n.inputs[0])
              << ", dims = [" << join_i64(n.ints) << "] : ("
              << T(n.inputs[0]) << ") -> " << self_t;
            break;
        case Op::Broadcast:
            o << "  " << ssa << " = stablehlo.broadcast_in_dim " << NM(n.inputs[0])
              << ", dims = [" << join_i64(n.ints) << "] : ("
              << T(n.inputs[0]) << ") -> " << self_t;
            break;
        case Op::Dot: {
            int64_t batch = n.ints[0];
            const Shape& A = nodes_[n.inputs[0]].shape;
            const Shape& B = nodes_[n.inputs[1]].shape;
            std::vector<int64_t> lb, rb;
            for (int64_t i = 0; i < batch; i++) { lb.push_back(i); rb.push_back(i); }
            int64_t lc = (int64_t)A.size() - 1;     // contracting dim of lhs
            int64_t rc = (int64_t)B.size() - 2;     // contracting dim of rhs
            o << "  " << ssa << " = stablehlo.dot_general " << NM(n.inputs[0]) << ", "
              << NM(n.inputs[1]) << ", ";
            if (batch > 0)
                o << "batching_dims = [" << join_i64(lb) << "] x [" << join_i64(rb) << "], ";
            o << "contracting_dims = [" << lc << "] x [" << rc << "]"
              << ", precision = [" << dot_precision << ", " << dot_precision << "] : ("
              << T(n.inputs[0]) << ", " << T(n.inputs[1]) << ") -> " << self_t;
            break;
        }
        case Op::ReduceSum:
        case Op::ReduceMax: {
            const char* rop = (n.op == Op::ReduceSum) ? "add" : "maximum";
            o << "  " << ssa << " = stablehlo.reduce(" << NM(n.inputs[0])
              << " init: " << NM(n.inputs[1]) << ") applies stablehlo." << rop
              << " across dimensions = [" << join_i64(n.ints) << "] : ("
              << T(n.inputs[0]) << ", " << T(n.inputs[1]) << ") -> " << self_t;
            break;
        }
        case Op::Input:
            break;  // function argument; no body line
    }
    return o.str();
}

std::string Graph::emit(const std::vector<Value>& outputs) const {
    std::ostringstream o;
    o << "module @m {\n";
    o << "  func.func public @main(";
    for (size_t i = 0; i < args_.size(); i++) {
        const Node& a = nodes_[args_[i]];
        if (i) o << ", ";
        o << "%arg" << a.arg_index << ": " << type_str(a.shape, a.dtype);
    }
    o << ") -> (";
    for (size_t i = 0; i < outputs.size(); i++) {
        if (i) o << ", ";
        o << type_str(nodes_[outputs[i].id].shape, nodes_[outputs[i].id].dtype);
    }
    o << ") {\n";

    for (int id = 0; id < (int)nodes_.size(); id++) {
        if (nodes_[id].op == Op::Input) continue;
        std::string line = emit_node(id, ssa_name(nodes_[id], id));
        if (!line.empty()) o << line << "\n";
    }

    o << "    return ";
    for (size_t i = 0; i < outputs.size(); i++) {
        if (i) o << ", ";
        o << ssa_name(nodes_[outputs[i].id], outputs[i].id);
    }
    o << " : ";
    for (size_t i = 0; i < outputs.size(); i++) {
        if (i) o << ", ";
        o << type_str(nodes_[outputs[i].id].shape, nodes_[outputs[i].id].dtype);
    }
    o << "\n  }\n}\n";
    return o.str();
}

// ══ Autodiff ═════════════════════════════════════════════════════════════════
std::vector<Value> Graph::grad(const Value& loss, const std::vector<Value>& params) {
    std::unordered_map<int, int> gmap;  // node id → grad node id

    auto V = [&](int nid) { return Value{this, nid}; };
    auto accum = [&](int target, Value contrib) {
        auto it = gmap.find(target);
        if (it == gmap.end()) gmap[target] = contrib.id;
        else gmap[target] = add(V(it->second), contrib).id;
    };

    // keptAxes for a reduce: axes of `full` (rank) not in reduced `axes`.
    auto kept_axes = [](int64_t rank, const std::vector<int64_t>& axes) {
        std::vector<int64_t> keep;
        for (int64_t i = 0; i < rank; i++)
            if (std::find(axes.begin(), axes.end(), i) == axes.end()) keep.push_back(i);
        return keep;
    };

    int last = loss.id;
    gmap[last] = constant(1.0, node(last).shape, node(last).dtype).id;

    for (int id = last; id >= 0; id--) {
        auto it = gmap.find(id);
        if (it == gmap.end()) continue;
        Node n = nodes_[id];                 // copy: we append nodes below
        Value G = V(it->second);

        switch (n.op) {
            case Op::Input:
            case Op::Constant:
            case Op::Iota:
            case Op::Compare:
            case Op::StopGradient:
            case Op::AllReduce:
                break;  // leaves / no gradient flow (all_reduce only on grads)

            case Op::Add:
                accum(n.inputs[0], G);
                accum(n.inputs[1], G);
                break;
            case Op::Sub:
                accum(n.inputs[0], G);
                accum(n.inputs[1], neg(G));
                break;
            case Op::Mul:
                accum(n.inputs[0], mul(G, V(n.inputs[1])));
                accum(n.inputs[1], mul(G, V(n.inputs[0])));
                break;
            case Op::Div: {
                Value a = V(n.inputs[0]), b = V(n.inputs[1]);
                accum(n.inputs[0], div(G, b));
                accum(n.inputs[1], neg(div(mul(G, a), mul(b, b))));
                break;
            }
            case Op::Max: {
                Value a = V(n.inputs[0]), b = V(n.inputs[1]);
                Value mask = compare(a, b, Cmp::GE);
                Value z = constant(0.0, n.shape, n.dtype);
                accum(n.inputs[0], select(mask, G, z));
                accum(n.inputs[1], select(mask, z, G));
                break;
            }
            case Op::Min: {
                Value a = V(n.inputs[0]), b = V(n.inputs[1]);
                Value mask = compare(a, b, Cmp::LE);
                Value z = constant(0.0, n.shape, n.dtype);
                accum(n.inputs[0], select(mask, G, z));
                accum(n.inputs[1], select(mask, z, G));
                break;
            }
            case Op::Neg:
                accum(n.inputs[0], neg(G));
                break;
            case Op::Exp:
                accum(n.inputs[0], mul(G, V(id)));        // y = exp(a)
                break;
            case Op::Log:
                accum(n.inputs[0], div(G, V(n.inputs[0])));
                break;
            case Op::Sqrt: {
                Value y = V(id);
                accum(n.inputs[0], div(G, mul(y, scalar(2.0, n.dtype))));
                break;
            }
            case Op::Rsqrt: {
                Value y = V(id), a = V(n.inputs[0]);     // y = a^{-1/2}
                accum(n.inputs[0], mul(G, mul(scalar(-0.5, n.dtype), div(y, a))));
                break;
            }
            case Op::Tanh: {
                Value y = V(id);
                accum(n.inputs[0], mul(G, sub(scalar(1.0, n.dtype), mul(y, y))));
                break;
            }
            case Op::Logistic: {
                Value y = V(id);
                accum(n.inputs[0], mul(G, mul(y, sub(scalar(1.0, n.dtype), y))));
                break;
            }
            case Op::Abs: {
                Value a = V(n.inputs[0]);
                Value sgn = select(compare(a, scalar(0.0, n.dtype), Cmp::GE),
                                   scalar(1.0, n.dtype), scalar(-1.0, n.dtype));
                accum(n.inputs[0], mul(G, sgn));
                break;
            }
            case Op::Dot: {
                Value a = V(n.inputs[0]), b = V(n.inputs[1]);
                accum(n.inputs[0], dot(G, transpose_last2(b)));
                accum(n.inputs[1], dot(transpose_last2(a), G));
                break;
            }
            case Op::Transpose: {
                std::vector<int64_t> inv(n.ints.size());
                for (size_t i = 0; i < n.ints.size(); i++) inv[n.ints[i]] = (int64_t)i;
                accum(n.inputs[0], transpose(G, inv));
                break;
            }
            case Op::Reshape:
                accum(n.inputs[0], reshape(G, node(n.inputs[0]).shape));
                break;
            case Op::Broadcast: {
                const Shape& os = node(n.inputs[0]).shape;  // operand
                const Shape& ts = n.shape;                  // target
                const std::vector<int64_t>& bd = n.ints;
                std::vector<int64_t> sum_axes;
                for (int64_t a = 0; a < (int64_t)ts.size(); a++) {
                    auto pos = std::find(bd.begin(), bd.end(), a);
                    if (pos == bd.end()) { sum_axes.push_back(a); continue; }
                    int64_t oi = (int64_t)(pos - bd.begin());
                    if (os[oi] == 1 && ts[a] > 1) sum_axes.push_back(a);
                }
                Value s = sum_axes.empty() ? G : reduce_sum(G, sum_axes, false);
                accum(n.inputs[0], reshape(s, os));
                break;
            }
            case Op::ReduceSum: {
                Value a = V(n.inputs[0]);
                auto keep = kept_axes(a.rank(), n.ints);
                accum(n.inputs[0], broadcast_in_dim(G, node(n.inputs[0]).shape, keep));
                break;
            }
            case Op::ReduceMax: {
                Value a = V(n.inputs[0]), y = V(id);
                auto keep = kept_axes(a.rank(), n.ints);
                Value yb = broadcast_in_dim(y, node(n.inputs[0]).shape, keep);
                Value Gb = broadcast_in_dim(G, node(n.inputs[0]).shape, keep);
                Value mask = convert(compare(a, yb, Cmp::EQ), n.dtype);
                accum(n.inputs[0], mul(Gb, mask));
                break;
            }
            case Op::Select: {
                Value pred = V(n.inputs[0]);
                Value z = constant(0.0, n.shape, n.dtype);
                accum(n.inputs[1], select(pred, G, z));
                accum(n.inputs[2], select(pred, z, G));
                break;
            }
            case Op::Convert:
                accum(n.inputs[0], convert(G, node(n.inputs[0]).dtype));
                break;
        }
    }

    std::vector<Value> out;
    out.reserve(params.size());
    for (const auto& p : params) {
        auto it = gmap.find(p.id);
        if (it != gmap.end()) out.push_back(V(it->second));
        else out.push_back(constant(0.0, node(p.id).shape, node(p.id).dtype));
    }
    return out;
}

}  // namespace tpu
