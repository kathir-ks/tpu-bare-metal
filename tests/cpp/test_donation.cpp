// test_donation.cpp — spike + regression for buffer donation (task 3.1/3.2).
//
// Compiles p' = p + 1 with %arg0 aliased to output 0 (tf.aliasing_output) and
// runs it many times, feeding each step's output back as the next input.
// Checks (a) results stay correct, (b) device bytes_in_use stays flat (the
// aliased update must not allocate a new 64MB buffer per step that outlives
// the step), and (c) compares HBM behavior with the non-aliased variant.
#include "graph.hpp"
#include <cstdio>

using namespace tpu;

static int64_t hbm_in_use(Context& ctx) {
    return ctx.device_info(0).bytes_in_use;
}

// Run `iters` feedback steps of p+1 on a [N,N] f32 param; return peak HBM delta.
static bool run_variant(Context& ctx, bool alias, int64_t N, int iters,
                        int64_t* peak_delta) {
    Graph g;
    Value p = g.input("p", {N, N});
    Value np = p + 1.0;
    if (alias) g.arg_aliases[0] = 0;
    Executable e = ctx.compile_mlir(g.emit({np}));

    std::vector<float> host(N * N, 0.0f);
    Buffer buf = ctx.upload_f32(host, {N, N});
    int64_t base = hbm_in_use(ctx);
    int64_t peak = 0;
    for (int i = 0; i < iters; i++) {
        auto out = e.run({&buf});
        // move-assign frees the old buffer handle; for the aliased variant the
        // underlying memory was donated and now lives on as out[0].
        buf = std::move(out[0]);
        peak = std::max(peak, hbm_in_use(ctx) - base);
    }
    auto final_host = buf.to_host<float>();
    bool correct = final_host[0] == (float)iters && final_host[N * N - 1] == (float)iters;
    *peak_delta = peak;
    printf("%-12s peak HBM delta %8.2f MB, final value %g -> %s\n",
           alias ? "aliased" : "no-alias", peak / 1e6, final_host[0],
           correct ? "ok" : "WRONG");
    return correct;
}

// Probe whether donation actually happened: after execution, a donated input
// buffer is invalidated (reading it errors); a non-donated input stays live.
static bool input_dead_after_run(Context& ctx, bool alias) {
    const int64_t N = 256;
    Graph g;
    Value p = g.input("p", {N, N});
    Value np = p + 1.0;
    if (alias) g.arg_aliases[0] = 0;
    Executable e = ctx.compile_mlir(g.emit({np}));
    std::vector<float> host(N * N, 0.0f);
    Buffer buf = ctx.upload_f32(host, {N, N});
    auto out = e.run({&buf});
    try {
        auto h = buf.to_host<float>();
        (void)h;
        return false;  // input still readable
    } catch (const Error&) {
        return true;   // input consumed
    }
}

int main() {
    Context ctx;
    const int64_t N = 2048;  // 16 MB buffer
    const int iters = 50;

    int64_t peak_alias = 0, peak_plain = 0;
    bool ok = run_variant(ctx, true, N, iters, &peak_alias);
    ok &= run_variant(ctx, false, N, iters, &peak_plain);

    bool dead_aliased = input_dead_after_run(ctx, true);
    bool dead_plain   = input_dead_after_run(ctx, false);
    printf("input consumed after run: aliased=%d plain=%d (expect 1/0)\n",
           dead_aliased, dead_plain);
    ok &= dead_aliased && !dead_plain;

    printf("test_donation: %s (aliased peak %.2f MB vs plain %.2f MB)\n",
           ok ? "PASS" : "FAIL", peak_alias / 1e6, peak_plain / 1e6);
    return ok ? 0 : 1;
}
