// train_gpt_dp.cpp — data-parallel char-level GPT across all TPU chips.
//
// Same model as train_gpt.cpp, but trained with DataParallelTrainer: each of the
// N addressable devices processes its own batch shard, gradients are summed with
// all_reduce and averaged, and weights stay synchronized. Effective batch =
// N * per-replica batch. Generation runs on a single device (ForwardDP) sharing
// the trained weights.
#include "gpt.hpp"
#include "corpus.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace tpu;

int main() {
    Context ctx;
    int nd = ctx.num_addressable_devices();
    printf("TPU: %d devices (data-parallel)\n", nd);

    std::string text = CORPUS;
    Vocab vocab; vocab.build(text);
    std::vector<int32_t> data = vocab.encode(text);
    printf("corpus: %zu chars, vocab=%d\n", text.size(), vocab.size());

    GPTConfig cfg;
    cfg.vocab = vocab.size();
    cfg.n_layer = 4; cfg.n_head = 4; cfg.d_model = 128; cfg.d_ff = 512;
    cfg.block_size = 64;
    const int64_t Bpr = 8, T = cfg.block_size;   // per-replica batch
    const int64_t Bglobal = Bpr * nd;

    DataParallelTrainer tr(ctx, AdamCfg{0.9, 0.95, 1e-8, 0.0});
    printf("compiling DP train step (effective batch = %lld)...\n", (long long)Bglobal);
    fflush(stdout);
    tr.build([&](TrainCtx& c, Value x, Value y) { return gpt_loss(c, x, y, cfg, Bpr, T); },
             {Bpr, T}, DType::S32, {Bpr, T}, DType::S32, "DEFAULT");
    printf("model: %lld params, compiled across %d chips\n",
           (long long)tr.param_count(), nd);

    ForwardDP fwd(tr);
    fwd.build({1, T}, DType::S32,
              [&](TrainCtx& c, Value x) -> std::vector<Value> {
                  return {gpt_logits(c, x, cfg, 1, T)};
              }, "DEFAULT");

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> pick(0, data.size() - T - 2);
    auto make_global_batch = [&](std::vector<int32_t>& xb, std::vector<int32_t>& yb) {
        xb.resize(Bglobal * T); yb.resize(Bglobal * T);
        for (int64_t b = 0; b < Bglobal; b++) {
            size_t s = pick(rng);
            for (int64_t t = 0; t < T; t++) {
                xb[b * T + t] = data[s + t];
                yb[b * T + t] = data[s + t + 1];
            }
        }
    };

    const int steps = 400;
    std::vector<int32_t> xb, yb;
    for (int it = 0; it <= steps; it++) {
        make_global_batch(xb, yb);
        float lr = 3e-3f * std::min(1.0f, (it + 1) / 50.0f);
        float loss = tr.step(xb, yb, lr);
        if (it % 25 == 0 || it == steps)
            printf("step %4d  lr %.4f  loss %.4f\n", it, lr, loss);
        fflush(stdout);
    }

    // generation (single device, shared weights)
    auto generate = [&](const std::string& prompt, int n, float temp) {
        std::vector<int32_t> ctxw(T, vocab.stoi[' ']);
        for (char ch : prompt) {
            if (!vocab.stoi.count(ch)) continue;
            ctxw.erase(ctxw.begin()); ctxw.push_back(vocab.stoi[ch]);
        }
        std::string out = prompt;
        for (int i = 0; i < n; i++) {
            auto logits = fwd.run(ctxw)[0].to_host<float>();
            const float* last = &logits[(T - 1) * cfg.vocab];
            float mx = -1e30f; for (int v = 0; v < cfg.vocab; v++) mx = std::max(mx, last[v]);
            std::vector<float> p(cfg.vocab); float sum = 0;
            for (int v = 0; v < cfg.vocab; v++) { p[v] = std::exp((last[v]-mx)/temp); sum += p[v]; }
            std::uniform_real_distribution<float> u(0, sum);
            float r = u(rng); int nxt = cfg.vocab - 1;
            for (int v = 0; v < cfg.vocab; v++) { r -= p[v]; if (r <= 0) { nxt = v; break; } }
            out.push_back(vocab.itos[nxt]);
            ctxw.erase(ctxw.begin()); ctxw.push_back(nxt);
        }
        return out;
    };

    printf("\n=== generated (temp=0.8) ===\n%s\n", generate("Alice ", 400, 0.8f).c_str());
    return 0;
}
