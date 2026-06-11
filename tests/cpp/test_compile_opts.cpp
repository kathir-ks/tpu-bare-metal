// test_compile_opts.cpp — native CompileOptionsProto encoder oracle test.
//
// 1. Byte-equality: make_compile_options(1|4) must match the JAX-generated
//    fixtures compile_opts.pb / compile_opts_n4.pb.
// 2. Hardware: compile+run a graph with natively encoded options for n=1
//    (single device) and n=4 (replicated all_reduce across all chips).
// 3. Minimal options (no embedded debug_options blob) must also compile.
#include "compile_opts.hpp"
#include "graph.hpp"
#include <cstdio>
#include <fstream>

using namespace tpu;

static std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw Error("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

static bool byte_check(const char* name, const std::string& got,
                       const std::string& want) {
    if (got == want) { printf("%-28s PASS (%zu bytes identical)\n", name, got.size()); return true; }
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) i++;
    printf("%-28s FAIL (sizes %zu vs %zu, first diff at %zu)\n",
           name, got.size(), want.size(), i);
    return false;
}

int main() {
    const char* dir = getenv("TPU_DATA_DIR");
    std::string base = dir ? std::string(dir) + "/" : "./";
    bool ok = true;

    // 1. byte equality vs fixtures
    ok &= byte_check("encode n=1 vs fixture", make_compile_options(1),
                     slurp(base + "compile_opts.pb"));
    ok &= byte_check("encode n=4 vs fixture", make_compile_options(4),
                     slurp(base + "compile_opts_n4.pb"));

    Context ctx;

    // 2a. compile + run with native n=1 options
    {
        Graph g;
        Value a = g.input("a", {4});
        Value b = g.input("b", {4});
        Value c = a * b + 1.0;
        std::string opts = make_compile_options(1);
        Executable e = ctx.compile_mlir(g.emit({c}), opts.data(), opts.size());
        float av[4] = {1, 2, 3, 4}, bv[4] = {10, 20, 30, 40};
        Buffer ab = ctx.upload_f32(av, {4}), bb = ctx.upload_f32(bv, {4});
        auto out = e.run({&ab, &bb});
        auto r = out[0].to_host<float>();
        bool good = r[0] == 11 && r[3] == 161;
        printf("%-28s %s\n", "native n=1 compile+run", good ? "PASS" : "FAIL");
        ok &= good;
    }

    // 2b. replicated n=4 all_reduce with native options
    {
        int nd = ctx.num_addressable_devices();
        Graph g;
        g.num_replicas = nd;
        Value x = g.input("x", {2});
        Value s = g.all_reduce_sum(x);
        std::string opts = make_compile_options(nd);
        Executable e = ctx.compile_mlir(g.emit({s}), opts.data(), opts.size());

        auto order = e.device_order(nd);
        std::vector<Buffer> bufs(nd);
        std::vector<std::vector<Buffer*>> args(nd);
        for (int r = 0; r < nd; r++) {
            float v[2] = {(float)(r + 1), (float)(10 * (r + 1))};
            bufs[r] = ctx.upload_f32(v, {2}, order[r]);
            args[r] = {&bufs[r]};
        }
        auto out = e.run_spmd(args);
        float want0 = 0, want1 = 0;
        for (int r = 0; r < nd; r++) { want0 += r + 1; want1 += 10 * (r + 1); }
        auto h = out[0][0].to_host<float>();
        bool good = h[0] == want0 && h[1] == want1;
        printf("%-28s %s (sum=[%g,%g])\n", "native n=4 all_reduce", good ? "PASS" : "FAIL",
               h[0], h[1]);
        ok &= good;
    }

    // 3. minimal options (no debug blob)
    {
        CompileOptsCfg cfg;
        cfg.include_debug_options = false;
        std::string opts = make_compile_options(cfg);
        Graph g;
        Value a = g.input("a", {4});
        Value c = a + 2.0;
        bool good = false;
        try {
            Executable e = ctx.compile_mlir(g.emit({c}), opts.data(), opts.size());
            float av[4] = {1, 2, 3, 4};
            Buffer ab = ctx.upload_f32(av, {4});
            auto out = e.run({&ab});
            good = out[0].to_host<float>()[0] == 3;
        } catch (const Error& e) {
            printf("minimal opts compile threw: %s\n", e.what());
        }
        printf("%-28s %s (%zu bytes)\n", "minimal opts (no debug)", good ? "PASS" : "FAIL",
               opts.size());
        ok &= good;
    }

    printf("test_compile_opts: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
