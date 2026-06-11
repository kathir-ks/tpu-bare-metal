// bench_matmul.cpp — matmul throughput at DEFAULT (bf16 passes) vs HIGHEST
// (accurate f32) dot_general precision. Validates that the precision policy
// actually changes TPU throughput (task 2.2).
#include "graph.hpp"
#include <chrono>
#include <cstdio>

using namespace tpu;

static double bench(Context& ctx, const std::string& precision, int64_t N, int iters) {
    Graph g;
    g.dot_precision = precision;
    Value a = g.input("a", {N, N});
    Value b = g.input("b", {N, N});
    // chain 8 matmuls so device compute dominates dispatch
    Value c = g.dot(a, b);
    for (int i = 0; i < 7; i++) c = g.dot(c, b);
    Value total = g.reduce_sum(c, {0, 1});  // scalar output: cheap sync download
    Executable e = ctx.compile_mlir(g.emit({total}));

    std::vector<float> hv(N * N, 0.001f);
    Buffer ab = ctx.upload_f32(hv, {N, N});
    Buffer bb = ctx.upload_f32(hv, {N, N});

    auto run1 = [&] {
        auto out = e.run({&ab, &bb});
        float sink;
        out[0].download(&sink, sizeof(sink));  // sync
    };
    run1();  // warmup
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) run1();
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count() / iters;
    double tflops = 8.0 * 2.0 * (double)N * N * N / s / 1e12;
    printf("%-8s N=%lld  %8.3f ms/run  %7.2f TFLOP/s\n", precision.c_str(),
           (long long)N, s * 1e3, tflops);
    return s;
}

int main() {
    Context ctx;
    const int64_t N = 4096;
    const int iters = 20;
    double hi = bench(ctx, "HIGHEST", N, iters);
    double df = bench(ctx, "DEFAULT", N, iters);
    printf("speedup DEFAULT vs HIGHEST: %.2fx -> %s\n", hi / df,
           (df < hi) ? "PASS" : "FAIL");
    return (df < hi) ? 0 : 1;
}
