// test_gather.cpp — gather_rows forward + scatter-add gradient on TPU.
//
// Forward: out = gather(table[V,D], ids[B,T]) must equal row lookup.
// Backward: loss = sum(gather(table, ids) * coef) has the closed-form
// gradient d_table[v,d] = sum_{i : ids[i]==v} coef[i,d]; ids contains
// duplicates so scatter accumulation is exercised.
#include "graph.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace tpu;

int main() {
    Context ctx;

    const int64_t V = 5, D = 3, B = 2, T = 3;
    std::vector<float> table(V * D);
    for (size_t i = 0; i < table.size(); i++) table[i] = 0.1f * (float)i - 0.7f;
    std::vector<int32_t> ids = {1, 3, 1, 0, 4, 1};   // row 1 appears 3x
    std::vector<float> coef(B * T * D);
    for (size_t i = 0; i < coef.size(); i++) coef[i] = 0.05f * (float)i - 0.4f;

    Graph g;
    auto tb = g.input("table", {V, D});
    auto id = g.input("ids", {B, T}, DType::S32);
    auto cf = g.input("coef", {B, T, D});
    auto out = g.gather_rows(tb, id);                       // [B,T,D]
    auto loss = g.reduce_sum(g.mul(out, cf), {0, 1, 2});
    auto dT = g.grad(loss, {tb})[0];
    auto exec = ctx.compile_mlir(g.emit({out, loss, dT}));

    auto tbb = ctx.upload_f32(table, {V, D});
    auto idb = ctx.upload_s32(ids, {B, T});
    auto cfb = ctx.upload_f32(coef, {B, T, D});
    auto outs = exec.run({&tbb, &idb, &cfb});
    auto fwd = outs[0].to_host<float>();
    auto gT  = outs[2].to_host<float>();

    // forward check
    double max_fwd = 0;
    for (int64_t i = 0; i < B * T; i++)
        for (int64_t d = 0; d < D; d++)
            max_fwd = std::max(max_fwd,
                (double)std::fabs(fwd[i * D + d] - table[ids[i] * D + d]));
    printf("forward max err = %.2e\n", max_fwd);

    // gradient check vs closed form
    std::vector<float> want(V * D, 0.0f);
    for (int64_t i = 0; i < B * T; i++)
        for (int64_t d = 0; d < D; d++)
            want[ids[i] * D + d] += coef[i * D + d];
    double max_grad = 0;
    for (int64_t i = 0; i < V * D; i++)
        max_grad = std::max(max_grad, (double)std::fabs(gT[i] - want[i]));
    printf("gradient max err = %.2e (duplicate rows accumulated)\n", max_grad);

    bool ok = max_fwd < 1e-6 && max_grad < 1e-5;
    printf("test_gather: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
