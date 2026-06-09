// test_gradcheck.cpp — validate autodiff on real TPU hardware via finite differences.
//
// f(W) = sum( tanh(X @ W)^2 ) with X fixed. Compares analytic gradient (from
// Graph::grad) against central finite differences. Exercises dot, tanh, mul,
// reduce_sum, broadcast, and the backward rules for each.
#include "graph.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace tpu;

int main() {
    Context ctx;

    const int M = 2, K = 3, N = 4;
    std::vector<float> X = {0.1f, -0.2f, 0.3f, 0.4f, 0.5f, -0.6f};          // [2,3]
    std::vector<float> W = {0.2f, -0.1f, 0.0f, 0.3f,
                            -0.4f, 0.5f, 0.1f, -0.2f,
                            0.3f, 0.2f, -0.5f, 0.1f};                        // [3,4]

    // ── analytic: compile [loss, dW] ────────────────────────────────────────
    Graph g;
    auto x = g.input("x", {M, K});
    auto w = g.input("w", {K, N});
    auto h = g.tanh(g.dot(x, w));
    auto loss = g.reduce_sum(g.mul(h, h), {0, 1});
    auto dW = g.grad(loss, {w})[0];
    auto exec = ctx.compile_mlir(g.emit({loss, dW}));

    auto xb = ctx.upload_f32(X, {M, K});
    auto wb = ctx.upload_f32(W, {K, N});
    auto outs = exec.run({&xb, &wb});
    float L0 = outs[0].to_host<float>()[0];
    auto gW = outs[1].to_host<float>();
    printf("loss = %.6f\n", L0);

    // ── numeric: compile forward-only [loss], run with perturbed W ──────────
    Graph gf;
    auto xf = gf.input("x", {M, K});
    auto wf = gf.input("w", {K, N});
    auto hf = gf.tanh(gf.dot(xf, wf));
    auto lf = gf.reduce_sum(gf.mul(hf, hf), {0, 1});
    auto fwd = ctx.compile_mlir(gf.emit({lf}));

    auto eval = [&](const std::vector<float>& Wv) {
        auto wbb = ctx.upload_f32(Wv, {K, N});
        auto o = fwd.run({&xb, &wbb});
        return o[0].to_host<float>()[0];
    };

    const float eps = 1e-3f;
    double max_err = 0;
    printf(" idx   analytic     numeric      abserr\n");
    for (int i = 0; i < K * N; i++) {
        std::vector<float> Wp = W, Wm = W;
        Wp[i] += eps; Wm[i] -= eps;
        float num = (eval(Wp) - eval(Wm)) / (2 * eps);
        double err = std::fabs(num - gW[i]);
        max_err = std::max(max_err, err);
        printf("%4d  %10.5f  %10.5f  %10.6f\n", i, gW[i], num, err);
    }
    printf("max abs err = %.6f  -> %s\n", max_err,
           max_err < 1e-2 ? "PASS" : "FAIL");
    return max_err < 1e-2 ? 0 : 1;
}
