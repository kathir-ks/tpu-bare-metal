// test_oracle_grad.cpp — per-op gradient parity against the JAX oracle.
//
// Generic, driven by the shared op_table + fixtures: for every *.fix that carries
// grad_in* tensors, build loss = sum(op(inputs)), take Graph::grad w.r.t. each float
// input, run on the TPU at HIGHEST precision, and assert the analytic gradient
// matches the oracle's golden gradient. This is the oracle-side proof (orthogonal to
// finite differences). Adding a case in tests/oracle/cases_*.py gives it gradient
// coverage automatically — no per-op C++ needed here.
//
//   ./cpp_oracle_grad     # arg1 = fixtures dir (default below)
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

std::vector<std::string> list_fixtures(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    for (dirent* e; (e = readdir(d)) != nullptr; ) {
        std::string n = e->d_name;
        if (n.size() > 4 && n.substr(n.size() - 4) == ".fix")
            out.push_back(n.substr(0, n.size() - 4));
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "tests/oracle/fixtures";
    Context ctx;

    auto keys = list_fixtures(dir);
    if (keys.empty()) {
        std::printf("no fixtures in %s — run `make oracle` first\n", dir.c_str());
        return 2;
    }

    std::printf("%-28s %-6s %12s %12s  %s\n", "op", "input", "measured", "tol", "verdict");
    std::printf("%s\n", std::string(74, '-').c_str());

    int checked = 0, failed = 0, skipped = 0;
    for (const auto& key : keys) {
        if (!op_table_has(key)) { skipped++; continue; }
        Fixture fx(dir + "/" + key + ".fix");

        // Which inputs are float params with an oracle gradient?
        int nin = fx.num_inputs();
        std::vector<int> param_idx;
        for (int k = 0; k < nin; k++)
            if (fx.has("grad_in" + std::to_string(k))) param_idx.push_back(k);
        if (param_idx.empty()) { skipped++; continue; }   // non-differentiable case

        // Build graph at HIGHEST (gradients want accuracy), inputs as graph Inputs.
        Graph g;
        g.dot_precision = "HIGHEST";
        std::vector<Value> ins;
        std::vector<Buffer> bufs;
        for (int k = 0; k < nin; k++) {
            const FixTensor& t = fx.at("in" + std::to_string(k));
            ins.push_back(g.input("in" + std::to_string(k), t.dims, t.dtype));
            if (t.dtype == DType::S32) bufs.push_back(ctx.upload_s32(t.i32, t.dims));
            else                       bufs.push_back(ctx.upload_f32(t.f32, t.dims));
        }
        Value out = op_table().at(key)(g, ins);
        // loss = sum over all axes of out → scalar
        std::vector<int64_t> all_axes;
        for (int64_t a = 0; a < out.rank(); a++) all_axes.push_back(a);
        Value loss = all_axes.empty() ? out : g.reduce_sum(out, all_axes);

        std::vector<Value> params;
        for (int k : param_idx) params.push_back(ins[k]);
        std::vector<Value> grads = g.grad(loss, params);

        // Compile [grad_param0, grad_param1, ...] and run.
        auto exec = ctx.compile_mlir(g.emit(grads));
        std::vector<Buffer*> args;
        for (auto& b : bufs) args.push_back(&b);
        auto res = exec.run(args);

        float tol_hi = fx.scalar("__tol_highest__", 1e-3f);
        double grad_tol = std::max(2e-3, (double)tol_hi * 5.0);

        for (size_t pi = 0; pi < param_idx.size(); pi++) {
            int k = param_idx[pi];
            std::vector<float> got = res[pi].to_host<float>();
            const FixTensor& gold = fx.at("grad_in" + std::to_string(k));
            double maxerr = 0;
            if ((int64_t)got.size() != gold.count()) {
                maxerr = 1e30;  // shape disagreement → hard fail
            } else {
                for (size_t i = 0; i < got.size(); i++)
                    maxerr = std::max(maxerr, (double)std::fabs(got[i] - gold.f32[i]));
            }
            bool ok = maxerr <= grad_tol;
            checked++;
            if (!ok) failed++;
            std::printf("%-28s in%-4d %12.3e %12.3e  %s\n",
                        key.c_str(), k, maxerr, grad_tol, ok ? "PASS" : "FAIL");
        }
    }
    std::printf("%s\n", std::string(74, '-').c_str());
    std::printf("oracle-grad: %d grads, %d passed, %d failed, %d fixtures skipped\n",
                checked, checked - failed, failed, skipped);
    return failed == 0 ? 0 : 1;
}
