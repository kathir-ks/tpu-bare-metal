// test_oracle_layers.cpp — per-layer forward + backward parity against the JAX oracle.
//
// For every *.fix in fixtures_layers/ with a builder in layer_table.hpp: build the
// layer via nn.hpp, feeding the fixture's params through a TrainCtx resolver, run
// forward + Graph::grad on the TPU at HIGHEST, and assert the output and all
// gradients (w.r.t. x and every param) match the oracle within tolerance.
//
//   ./cpp_oracle_layers   # arg1 = fixtures dir (default below)
#include "nn.hpp"
#include "fixture.hpp"
#include "layer_table.hpp"

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

double max_abs_err(const std::vector<float>& got, const FixTensor& gold) {
    if ((int64_t)got.size() != gold.count()) return 1e30;
    double m = 0;
    for (size_t i = 0; i < got.size(); i++)
        m = std::max(m, (double)std::fabs(got[i] - gold.f32[i]));
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "tests/oracle/fixtures_layers";
    Context ctx;

    auto keys = list_fixtures(dir);
    if (keys.empty()) {
        std::printf("no layer fixtures in %s — run `make oracle` first\n", dir.c_str());
        return 2;
    }

    std::printf("%-20s %-12s %12s %12s  %s\n", "layer", "what", "measured", "tol", "verdict");
    std::printf("%s\n", std::string(68, '-').c_str());

    int checked = 0, failed = 0, skipped = 0;
    for (const auto& key : keys) {
        if (!layer_table_has(key)) { skipped++; continue; }
        const LayerSpec& spec = layer_table().at(key);
        Fixture fx(dir + "/" + key + ".fix");
        float tol = fx.scalar("__tol_highest__", 2e-3f);
        float tol_grad = fx.scalar("__tol_grad__", tol);   // composites get a looser grad bound

        Graph g;
        g.dot_precision = "HIGHEST";

        // Forward inputs are args 0..k-1 (declared order); params append after, in
        // resolver-encounter order.
        std::vector<Value> in_vals;
        for (const auto& in : spec.inputs)
            in_vals.push_back(g.input(in.name, in.shape, in.is_int ? DType::S32 : DType::F32));

        std::vector<std::string> pnames;       // encounter order
        std::vector<Value>       pvals;
        TrainCtx::Resolver resolver =
            [&](const std::string& name, const Shape& shape, InitFn) -> Value {
                for (size_t i = 0; i < pnames.size(); i++)
                    if (pnames[i] == name) return pvals[i];   // weight tying: same Value
                Value p = g.input(name, shape, DType::F32);
                pnames.push_back(name);
                pvals.push_back(p);
                return p;
            };
        TrainCtx c(g, resolver);

        Value out = spec.build(c, in_vals);
        Value loss = spec.scalar_out ? out
                                     : [&]{
                                           std::vector<int64_t> axes;
                                           for (int64_t a = 0; a < out.rank(); a++) axes.push_back(a);
                                           return axes.empty() ? out : g.reduce_sum(out, axes);
                                       }();

        // grads: w.r.t. every float input, then every param (matches the oracle order)
        std::vector<Value> wrt;
        std::vector<std::string> wrt_name;
        for (size_t i = 0; i < spec.inputs.size(); i++)
            if (!spec.inputs[i].is_int) { wrt.push_back(in_vals[i]); wrt_name.push_back("grad_" + spec.inputs[i].name); }
        for (size_t i = 0; i < pnames.size(); i++) { wrt.push_back(pvals[i]); wrt_name.push_back("grad_" + pnames[i]); }
        std::vector<Value> grads = g.grad(loss, wrt);

        // outputs = [out, grads...]
        std::vector<Value> outputs = {out};
        for (auto& gr : grads) outputs.push_back(gr);
        auto exec = ctx.compile_mlir(g.emit(outputs));

        // upload inputs in arg order: declared inputs, then params in encounter order
        std::vector<Buffer> bufs;
        for (const auto& in : spec.inputs) {
            const FixTensor& it = fx.at(in.name);
            if (in.is_int) bufs.push_back(ctx.upload_s32(it.i32, it.dims));
            else           bufs.push_back(ctx.upload_f32(it.f32, it.dims));
        }
        for (const auto& nm : pnames) {
            const FixTensor& pt = fx.at(nm);
            bufs.push_back(ctx.upload_f32(pt.f32, pt.dims));
        }
        std::vector<Buffer*> args;
        for (auto& b : bufs) args.push_back(&b);
        auto res = exec.run(args);

        // res[0] = out ; res[1..] = grads in wrt order
        double e_out = max_abs_err(res[0].to_host<float>(), fx.at("out"));
        bool ok = e_out <= tol; checked++; if (!ok) failed++;
        std::printf("%-20s %-12s %12.3e %12.3e  %s\n", key.c_str(), "forward", e_out, (double)tol, ok ? "PASS" : "FAIL");
        for (size_t i = 0; i < wrt.size(); i++) {
            double e = max_abs_err(res[1 + i].to_host<float>(), fx.at(wrt_name[i]));
            bool gok = e <= tol_grad; checked++; if (!gok) failed++;
            std::printf("%-20s %-12s %12.3e %12.3e  %s\n", key.c_str(), wrt_name[i].c_str(), e, (double)tol_grad, gok ? "PASS" : "FAIL");
        }
    }
    std::printf("%s\n", std::string(68, '-').c_str());
    std::printf("oracle-layers: %d checks, %d passed, %d failed, %d skipped\n",
                checked, checked - failed, failed, skipped);
    return failed == 0 ? 0 : 1;
}
