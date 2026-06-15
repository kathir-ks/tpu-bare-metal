// test_fd_grad.cpp — oracle-FREE gradient proof via central finite differences.
//
// A third, independent check on autodiff: for each op_table case, compare the C++
// analytic gradient (Graph::grad) against central finite differences of the C++
// FORWARD itself — no JAX, no oracle. This catches a class of error that oracle
// parity cannot: a conceptual mistake shared between the C++ VJP and the JAX analytic
// gradient would pass oracle parity but fail here, because finite differences only
// trust the forward pass.
//
// Non-smooth ops (abs/max/min/select — kinks where the subgradient convention and a
// symmetric difference legitimately disagree) and non-differentiable ops are skipped.
//
//   ./cpp_fd_grad      # arg1 = fixtures dir (default below)
#include "graph.hpp"
#include "fixture.hpp"
#include "op_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <dirent.h>
#include <string>
#include <vector>

using namespace tpu;

namespace {

const double EPS    = 1e-3;
const double FD_TOL = 2e-2;   // central differences are O(eps^2)+roundoff noisy; the
                              // existing gradcheck uses 1e-2 — fd over many ops needs a hair more.
const int MAX_SAMPLES = 6;    // elements probed per float input (bounds run time)

// Ops whose finite-difference gradient legitimately disagrees with the analytic rule
// (kinks / identity-forward-but-zero-grad). Matched as substrings of the case key.
bool is_nonsmooth(const std::string& key) {
    static const char* deny[] = {"abs", "max", "min", "select", "stop_gradient"};
    for (auto* d : deny) if (key.find(d) != std::string::npos) return true;
    return false;
}

std::vector<std::string> list_fixtures(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    for (dirent* e; (e = readdir(d)) != nullptr; ) {
        std::string n = e->d_name;
        if (n.size() > 4 && n.substr(n.size() - 4) == ".fix") out.push_back(n.substr(0, n.size() - 4));
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

double host_sum(const std::vector<float>& v) {
    double s = 0; for (float x : v) s += x; return s;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "tests/oracle/fixtures";
    Context ctx;

    auto keys = list_fixtures(dir);
    if (keys.empty()) { std::printf("no fixtures in %s\n", dir.c_str()); return 2; }

    std::printf("%-28s %-6s %12s %12s  %s\n", "op", "input", "max|Δ|", "fd_tol", "verdict");
    std::printf("%s\n", std::string(74, '-').c_str());

    int checked = 0, failed = 0, skipped = 0;
    for (const auto& key : keys) {
        if (!op_table_has(key) || is_nonsmooth(key)) { skipped++; continue; }
        Fixture fx(dir + "/" + key + ".fix");
        int nin = fx.num_inputs();
        std::vector<int> fidx;                       // float-input indices with an oracle grad
        for (int k = 0; k < nin; k++)
            if (fx.has("grad_in" + std::to_string(k)) && fx.at("in" + std::to_string(k)).dtype == DType::F32)
                fidx.push_back(k);
        if (fidx.empty()) { skipped++; continue; }

        // ── analytic gradient (HIGHEST): loss = sum(op(inputs)) ────────────────
        Graph g; g.dot_precision = "HIGHEST";
        std::vector<Value> ins;
        std::vector<Buffer> bufs;
        for (int k = 0; k < nin; k++) {
            const FixTensor& t = fx.at("in" + std::to_string(k));
            ins.push_back(g.input("in" + std::to_string(k), t.dims, t.dtype));
            if (t.dtype == DType::S32) bufs.push_back(ctx.upload_s32(t.i32, t.dims));
            else                       bufs.push_back(ctx.upload_f32(t.f32, t.dims));
        }
        Value out = op_table().at(key)(g, ins);
        std::vector<int64_t> axes; for (int64_t a = 0; a < out.rank(); a++) axes.push_back(a);
        Value loss = axes.empty() ? out : g.reduce_sum(out, axes);
        std::vector<Value> params; for (int k : fidx) params.push_back(ins[k]);
        std::vector<Value> grads = g.grad(loss, params);
        auto gexec = ctx.compile_mlir(g.emit(grads));
        std::vector<Buffer*> gargs; for (auto& b : bufs) gargs.push_back(&b);
        auto gres = gexec.run(gargs);

        // ── forward-only executable (reused for every perturbation) ────────────
        Graph gf; gf.dot_precision = "HIGHEST";
        std::vector<Value> fins;
        for (int k = 0; k < nin; k++) {
            const FixTensor& t = fx.at("in" + std::to_string(k));
            fins.push_back(gf.input("in" + std::to_string(k), t.dims, t.dtype));
        }
        Value fout = op_table().at(key)(gf, fins);
        auto fexec = ctx.compile_mlir(gf.emit({fout}));

        auto forward_sum = [&](const std::vector<std::vector<float>>& fdata,
                               const std::vector<const FixTensor*>& fts) -> double {
            std::vector<Buffer> b;
            for (int k = 0; k < nin; k++) {
                if (fts[k]->dtype == DType::S32) b.push_back(ctx.upload_s32(fts[k]->i32, fts[k]->dims));
                else                             b.push_back(ctx.upload_f32(fdata[k], fts[k]->dims));
            }
            std::vector<Buffer*> a; for (auto& x : b) a.push_back(&x);
            return host_sum(fexec.run(a)[0].to_host<float>());
        };

        std::vector<const FixTensor*> fts;
        std::vector<std::vector<float>> base(nin);
        for (int k = 0; k < nin; k++) { fts.push_back(&fx.at("in" + std::to_string(k))); base[k] = fts[k]->f32; }

        // ── compare analytic vs finite-difference on sampled elements ──────────
        for (size_t pi = 0; pi < fidx.size(); pi++) {
            int k = fidx[pi];
            std::vector<float> gA = gres[pi].to_host<float>();
            int64_t n = fx.at("in" + std::to_string(k)).count();
            int samples = (int)std::min<int64_t>(n, MAX_SAMPLES);
            double maxerr = 0;
            for (int s = 0; s < samples; s++) {
                int64_t i = (n <= MAX_SAMPLES) ? s : (s * n / samples);   // spread across elements
                std::vector<std::vector<float>> fp = base, fm = base;
                fp[k][i] += (float)EPS; fm[k][i] -= (float)EPS;
                double num = (forward_sum(fp, fts) - forward_sum(fm, fts)) / (2 * EPS);
                maxerr = std::max(maxerr, std::fabs(num - (double)gA[i]));
            }
            bool ok = maxerr <= FD_TOL; checked++; if (!ok) failed++;
            std::printf("%-28s in%-4d %12.3e %12.3e  %s\n", key.c_str(), k, maxerr, FD_TOL, ok ? "PASS" : "FAIL");
        }
    }
    std::printf("%s\n", std::string(74, '-').c_str());
    std::printf("fd-grad: %d checks, %d passed, %d failed, %d skipped (nonsmooth/non-diff)\n",
                checked, checked - failed, failed, skipped);
    return failed == 0 ? 0 : 1;
}
