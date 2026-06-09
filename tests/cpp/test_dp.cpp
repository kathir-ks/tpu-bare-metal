// test_dp.cpp — data-parallel training across all TPU chips. Each replica
// memorizes its own sequence; all_reduce-averaged gradients keep weights in
// sync. Validates DataParallelTrainer end-to-end.
#include "nn.hpp"
#include <cstdio>

using namespace tpu;

int main() {
    Context ctx;
    const int64_t T = 8, V = 16, D = 32;

    DataParallelTrainer tr(ctx, AdamCfg{0.9, 0.95, 1e-8, 0.0});
    int nd = tr.num_replicas();
    const int64_t Bpr = 1;  // per-replica batch
    printf("data-parallel over %d devices\n", nd);

    auto model = [&](TrainCtx& c, Value x, Value y) -> Value {
        Value emb = nn::embedding(c, x, "tok", V, D);
        Value logits = nn::linear(c, emb, "head", D, V);
        Value flat = c.g.reshape(logits, {Bpr * T, V});
        Value tgt  = c.g.reshape(y, {Bpr * T});
        return nn::cross_entropy(c.g, flat, tgt, Bpr * T, V);
    };
    printf("compiling DP train step...\n"); fflush(stdout);
    tr.build(model, {Bpr, T}, DType::S32, {Bpr, T}, DType::S32);
    printf("compiled. params=%lld\n", (long long)tr.param_count());

    // global batch: replica r memorizes sequence shifted by r.
    auto seqval = [&](int r, int64_t t) { return (int)((r * 3 + t) % V); };
    std::vector<int32_t> gx(nd * T), gy(nd * T);
    for (int r = 0; r < nd; r++)
        for (int64_t t = 0; t < T; t++) {
            gx[r * T + t] = seqval(r, t);
            gy[r * T + t] = seqval(r, t + 1);
        }

    float l0 = 0, lN = 0;
    for (int it = 0; it < 300; it++) {
        float l = tr.step(gx, gy, 2e-2f);
        if (it == 0) l0 = l;
        if (it % 50 == 0) printf("step %3d  loss %.5f\n", it, l);
        lN = l;
    }
    printf("l0=%.4f lN=%.5f -> %s\n", l0, lN, (lN < 0.05f && lN < l0) ? "PASS" : "FAIL");
    return (lN < 0.05f) ? 0 : 1;
}
