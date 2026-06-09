// test_gpt_smoke.cpp — tiny GPT regression test: train to memorize distinct
// sequences, then confirm the shared-weight inference path reproduces them.
// Exercises attention, blocks, RMSNorm, GELU, Adam, and the Trainer/Forward
// parameter-sharing + tiled-download path on real hardware.
#include "gpt.hpp"
#include <cstdio>

using namespace tpu;

int main() {
    Context ctx;
    GPTConfig cfg;
    cfg.vocab = 45; cfg.n_layer = 4; cfg.n_head = 4; cfg.d_model = 128;
    cfg.d_ff = 512; cfg.block_size = 64;
    const int64_t B = 16, T = cfg.block_size;

    Trainer tr(ctx, AdamCfg{0.9, 0.95, 1e-8, 0.0});
    printf("compiling tiny GPT train step...\n"); fflush(stdout);
    tr.build([&](TrainCtx& c, Value x, Value y) { return gpt_loss(c, x, y, cfg, B, T); },
             {B, T}, DType::S32, {B, T}, DType::S32, "DEFAULT");
    printf("compiled. params=%lld\n", (long long)tr.param_count());

    Forward fwd(tr);
    fwd.build({1, T}, DType::S32,
              [&](TrainCtx& c, Value x) -> std::vector<Value> {
                  return {gpt_logits(c, x, cfg, 1, T)};
              }, "DEFAULT");

    // B distinct memorizable sequences: element b is shifted by b.
    auto seqval = [&](int64_t b, int64_t t) { return (int)((b * 5 + t) % cfg.vocab); };
    std::vector<int32_t> x(B * T), y(B * T);
    for (int64_t b = 0; b < B; b++)
        for (int64_t t = 0; t < T; t++) {
            x[b * T + t] = seqval(b, t);
            y[b * T + t] = seqval(b, t + 1);
        }

    float l0 = 0, lN = 0;
    for (int it = 0; it < 200; it++) {
        float l = tr.step(x, y, 5e-3f);
        if (it == 0) l0 = l;
        if (it % 40 == 0) printf("step %3d loss %.4f\n", it, l);
        lN = l;
    }
    printf("l0=%.4f lN=%.4f\n", l0, lN);

    // inference must reproduce each memorized sequence.
    int correct = 0, total = 0;
    for (int64_t b = 0; b < B; b++) {
        std::vector<int32_t> seq(T); for (int64_t t = 0; t < T; t++) seq[t] = seqval(b, t);
        auto lg = fwd.run(seq)[0].to_host<float>();   // [1,T,V]
        for (int64_t t = 0; t < T; t++) {
            const float* row = &lg[t * cfg.vocab];
            int am = 0; for (int v = 1; v < cfg.vocab; v++) if (row[v] > row[am]) am = v;
            if (am == seqval(b, t + 1)) correct++;
            total++;
        }
    }
    printf("inference argmax acc: %d/%d\n", correct, total);
    bool ok = (lN < 0.05f) && (correct >= total - 2);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
