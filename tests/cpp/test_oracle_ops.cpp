// test_oracle_ops.cpp — per-op forward parity against the JAX oracle.
//
// For every *.fix in the fixtures dir whose key is known to op_table.hpp: build the
// op in the graph, run it on the TPU at HIGHEST and DEFAULT precision, and assert
// the output matches the golden vector within that op's declared tolerance. Prints
// a measured-error-vs-threshold row per (op, precision); exits non-zero on any miss.
//
//   make oracle           # regenerate fixtures (JAX venv)
//   ./cpp_oracle_ops      # run on device   (arg1 = fixtures dir, default below)
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

// Build + run one op case at a given precision; returns max abs error vs golden.
double run_case(Context& ctx, const Fixture& fx, const std::string& key,
                const std::string& precision) {
    Graph g;
    g.dot_precision = precision;

    int nin = fx.num_inputs();
    std::vector<Value> ins;
    std::vector<Buffer> bufs;
    bufs.reserve(nin);
    for (int i = 0; i < nin; i++) {
        const FixTensor& t = fx.at("in" + std::to_string(i));
        ins.push_back(g.input("in" + std::to_string(i), t.dims, t.dtype));
        if (t.dtype == DType::S32) bufs.push_back(ctx.upload_s32(t.i32, t.dims));
        else                       bufs.push_back(ctx.upload_f32(t.f32, t.dims));
    }
    Value out = op_table().at(key)(g, ins);

    auto exec = ctx.compile_mlir(g.emit({out}));
    std::vector<Buffer*> args;
    for (auto& b : bufs) args.push_back(&b);
    auto res = exec.run(args);
    std::vector<float> got = res[0].to_host<float>();

    const FixTensor& gold = fx.at("out");
    if ((int64_t)got.size() != gold.count())
        throw Error("oracle_ops: size mismatch for " + key + " (" +
                    std::to_string(got.size()) + " vs " + std::to_string(gold.count()) + ")");
    double maxerr = 0;
    for (size_t i = 0; i < got.size(); i++)
        maxerr = std::max(maxerr, (double)std::fabs(got[i] - gold.f32[i]));
    return maxerr;
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

    std::printf("%-22s %-9s %12s %12s  %s\n", "op", "prec", "measured", "tol", "verdict");
    std::printf("%s\n", std::string(70, '-').c_str());

    int checked = 0, failed = 0, skipped = 0;
    for (const auto& key : keys) {
        if (!op_table_has(key)) { skipped++; continue; }
        Fixture fx(dir + "/" + key + ".fix");
        float tol_hi = fx.scalar("__tol_highest__", 1e-3f);
        float tol_df = fx.scalar("__tol_default__", 3e-2f);

        struct { const char* prec; float tol; } passes[] = {
            {"HIGHEST", tol_hi}, {"DEFAULT", tol_df}};
        for (auto& p : passes) {
            double err = run_case(ctx, fx, key, p.prec);
            bool ok = err <= p.tol;
            checked++;
            if (!ok) failed++;
            std::printf("%-22s %-9s %12.3e %12.3e  %s\n",
                        key.c_str(), p.prec, err, p.tol, ok ? "PASS" : "FAIL");
        }
    }
    std::printf("%s\n", std::string(70, '-').c_str());
    std::printf("oracle-ops: %d checks, %d passed, %d failed, %d fixtures skipped (no builder)\n",
                checked, checked - failed, failed, skipped);
    return failed == 0 ? 0 : 1;
}
