// train_gpt.cpp — train a char-level GPT on TPU and generate text. Pure C++.
//
//   make cpp_examples && TPU only needed at runtime
//   ./cpp_train_gpt
//
// Trains on an embedded public-domain corpus, prints decreasing loss, then
// autoregressively generates a sample from the trained weights.
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
    printf("TPU: %d devices\n", ctx.num_addressable_devices());

    std::string text = CORPUS;
    Vocab vocab; vocab.build(text);
    std::vector<int32_t> data; data.reserve(text.size());
    for (char ch : text) data.push_back(vocab.stoi[ch]);
    printf("corpus: %zu chars, vocab=%d\n", text.size(), vocab.size());

    GPTConfig cfg;
    cfg.vocab = vocab.size();
    cfg.n_layer = 4; cfg.n_head = 4; cfg.d_model = 128; cfg.d_ff = 512;
    cfg.block_size = 64;
    const int64_t B = 16, T = cfg.block_size;

    // ── build training graph ────────────────────────────────────────────────
    Trainer tr(ctx, AdamCfg{0.9, 0.95, 1e-8, 0.0});
    printf("compiling train step (this can take a minute)...\n"); fflush(stdout);
    tr.build([&](TrainCtx& c, Value x, Value y) { return gpt_loss(c, x, y, cfg, B, T); },
             {B, T}, DType::S32, {B, T}, DType::S32, "DEFAULT");
    printf("model: %lld params, train step compiled\n", (long long)tr.param_count());

    // ── inference graph (B=1) sharing the trainer's weights ─────────────────
    Forward fwd(tr);
    fwd.build({1, T}, DType::S32,
              [&](TrainCtx& c, Value x) -> std::vector<Value> {
                  return {gpt_logits(c, x, cfg, 1, T)};
              }, "DEFAULT");

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> pick(0, data.size() - T - 2);

    auto make_batch = [&](std::vector<int32_t>& xb, std::vector<int32_t>& yb) {
        xb.resize(B * T); yb.resize(B * T);
        for (int64_t b = 0; b < B; b++) {
            size_t s = pick(rng);
            for (int64_t t = 0; t < T; t++) {
                xb[b * T + t] = data[s + t];
                yb[b * T + t] = data[s + t + 1];
            }
        }
    };

    // ── train ────────────────────────────────────────────────────────────────
    const int steps = 800;
    std::vector<int32_t> xb, yb;
    for (int it = 0; it <= steps; it++) {
        make_batch(xb, yb);
        // simple warmup + cosine-ish decay
        float lr = 3e-3f * std::min(1.0f, (it + 1) / 50.0f);
        float loss = tr.step(xb, yb, lr);
        if (it % 50 == 0 || it == steps)
            printf("step %4d  lr %.4f  loss %.4f\n", it, lr, loss);
        fflush(stdout);
    }

    // ── teacher-forcing probe: does inference match training? ───────────────
    {
        std::vector<int32_t> probe(data.begin(), data.begin() + T);
        auto o = fwd.run(probe);
        auto logits = o[0].to_host<float>();   // [1,T,V]
        int correct = 0;
        for (int64_t t = 0; t < T; t++) {
            const float* row = &logits[t * cfg.vocab];
            int am = 0; for (int v = 1; v < cfg.vocab; v++) if (row[v] > row[am]) am = v;
            if (am == data[t + 1]) correct++;
        }
        printf("teacher-forcing argmax acc: %d/%lld\n", correct, (long long)T);
    }

    // ── generate ───────────────────────────────────────────────────────────
    auto generate = [&](const std::string& prompt, int n, float temp) {
        std::vector<int32_t> ctxw(T, vocab.stoi.count(' ') ? vocab.stoi[' '] : 0);
        std::string seed = prompt;
        for (char ch : seed) {
            if (!vocab.stoi.count(ch)) continue;
            ctxw.erase(ctxw.begin());
            ctxw.push_back(vocab.stoi[ch]);
        }
        std::string out = seed;
        for (int i = 0; i < n; i++) {
            auto o = fwd.run(ctxw);
            auto logits = o[0].to_host<float>();    // [1,T,V]
            const float* last = &logits[(T - 1) * cfg.vocab];
            // temperature softmax sample
            float mx = -1e30f;
            for (int v = 0; v < cfg.vocab; v++) mx = std::max(mx, last[v]);
            std::vector<float> p(cfg.vocab); float sum = 0;
            for (int v = 0; v < cfg.vocab; v++) { p[v] = std::exp((last[v]-mx)/temp); sum += p[v]; }
            std::uniform_real_distribution<float> u(0, sum);
            float r = u(rng); int nxt = cfg.vocab - 1;
            for (int v = 0; v < cfg.vocab; v++) { r -= p[v]; if (r <= 0) { nxt = v; break; } }
            out.push_back(vocab.itos[nxt]);
            ctxw.erase(ctxw.begin());
            ctxw.push_back(nxt);
        }
        return out;
    };

    printf("\n=== generated (temp=0.8) ===\n");
    printf("%s\n", generate("Alice ", 400, 0.8f).c_str());
    return 0;
}
