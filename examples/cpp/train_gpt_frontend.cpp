// train_gpt_frontend.cpp — train a char-level GPT end-to-end on TPU using the
// eager/jit Tensor frontend (module.hpp GPT + optim.hpp JitAdamStep), then
// generate text. The whole training loop runs through the *new* mutable,
// tensor-centric API — no Trainer, no hand-built graph.
//
//   make cpp_train_gpt_frontend
//   ./cpp_train_gpt_frontend
//
// Companion to examples/cpp/train_gpt.cpp (the nn.hpp Trainer path). Same model,
// same corpus, same hyperparameters — the point is to show the frontend trains a
// real model and to sanity-check throughput against the benchmarked path.
#include "module.hpp"      // GPT, Embedding/RMSNorm/Attention/Block, cross_entropy
#include "optim.hpp"       // Adam + JitAdamStep (fused fwd+bwd+Adam, donation)
#include "autograd.hpp"    // NoGrad
#include "jit.hpp"         // jit (B=1 inference forward)
#include "corpus.h"        // CORPUS + Vocab (shared with the Trainer-path example)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace tpu;

int main() {
    // ── corpus → char vocab → int32 token stream ────────────────────────────────
    std::string text = CORPUS;
    Vocab vocab; vocab.build(text);
    std::vector<int32_t> data; data.reserve(text.size());
    for (char ch : text) data.push_back(vocab.stoi[ch]);
    std::printf("corpus: %zu chars, vocab=%d\n", text.size(), vocab.size());

    // ── model: same shape as the Trainer-path train_gpt.cpp ─────────────────────
    GPTModuleConfig cfg;
    cfg.vocab = vocab.size();
    cfg.n_layer = 4; cfg.n_head = 4; cfg.d_model = 128; cfg.d_ff = 512;
    cfg.block_size = 64;
    const int64_t B = 16, T = cfg.block_size;

    std::mt19937 rng(42);
    GPT model(cfg, &rng, /*seed=*/1234);
    std::printf("model: %lld params (expected %lld)\n",
                (long long)model.param_count(), (long long)model.expected_param_count());

    // ── fused training step: forward + backward + Adam, params updated in place ─
    // DEFAULT precision (bf16 matmul) for speed, matching the benchmarked path.
    JitAdamStep step(std::function<Tensor(std::vector<Tensor>)>(
                         [&](std::vector<Tensor> b) { return model.loss(b[0], b[1]); }),
                     model.parameters(), /*lr=*/3e-3, AdamCfg{0.9, 0.95, 1e-8, 0.0},
                     /*num_replicas=*/1, /*precision=*/"DEFAULT");

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

    // ── train ───────────────────────────────────────────────────────────────────
    const int steps = 800;
    std::vector<int32_t> xb, yb;
    float first_loss = 0, last_loss = 0;
    double train_secs = 0;
    int timed_steps = 0;
    std::printf("compiling fused step on first call, then training %d steps...\n", steps);
    for (int it = 0; it <= steps; it++) {
        make_batch(xb, yb);
        Tensor x = from_host_s32(xb, {B, T});
        Tensor y = from_host_s32(yb, {B, T});

        auto t0 = std::chrono::steady_clock::now();
        float loss = step({x, y}).to_host()[0];   // forward+backward+Adam in one exe
        auto t1 = std::chrono::steady_clock::now();
        if (it >= 1) {   // skip step 0 (includes XLA compile)
            train_secs += std::chrono::duration<double>(t1 - t0).count();
            timed_steps++;
        }

        if (it == 0) first_loss = loss;
        last_loss = loss;
        if (it % 50 == 0 || it == steps)
            std::printf("step %4d  loss %.4f\n", it, loss);
        std::fflush(stdout);
    }

    double ms_per_step = 1000.0 * train_secs / std::max(1, timed_steps);
    double tok_per_s   = (double)(B * T) / (train_secs / std::max(1, timed_steps));
    std::printf("train: loss %.4f -> %.4f  |  %.1f ms/step  %.0f tok/s (post-compile, %d steps)\n",
                first_loss, last_loss, ms_per_step, tok_per_s, timed_steps);

    // ── inference forward (B=1) via jit; auto-lifts the trained params ──────────
    // JitAdamStep updates param buffers in place, and jit re-reads each captured
    // buffer per call — so this forward always sees the latest weights.
    auto fwd = jit(std::function<Tensor(Tensor)>(
                       [&](Tensor ids) { return model.logits(ids); }),
                   /*num_replicas=*/1, /*precision=*/"DEFAULT");

    // teacher-forcing argmax probe: does the model predict the corpus it trained on?
    {
        std::vector<int32_t> probe(data.begin(), data.begin() + T);
        Tensor lg = fwd(from_host_s32(probe, {1, T}));   // [1,T,V]
        std::vector<float> logits = lg.to_host();
        int correct = 0;
        for (int64_t t = 0; t < T; t++) {
            const float* row = &logits[t * cfg.vocab];
            int am = 0; for (int v = 1; v < cfg.vocab; v++) if (row[v] > row[am]) am = v;
            if (am == data[t + 1]) correct++;
        }
        std::printf("teacher-forcing argmax acc: %d/%lld\n", correct, (long long)T);
    }

    // ── autoregressive generation (temperature sampling) ────────────────────────
    auto generate = [&](const std::string& prompt, int n, float temp) {
        std::vector<int32_t> ctxw(T, vocab.stoi.count(' ') ? vocab.stoi[' '] : 0);
        for (char ch : prompt) {
            if (!vocab.stoi.count(ch)) continue;
            ctxw.erase(ctxw.begin());
            ctxw.push_back(vocab.stoi[ch]);
        }
        std::string out = prompt;
        for (int i = 0; i < n; i++) {
            Tensor lg = fwd(from_host_s32(ctxw, {1, T}));
            std::vector<float> logits = lg.to_host();
            const float* last = &logits[(T - 1) * cfg.vocab];
            float mx = -1e30f;
            for (int v = 0; v < cfg.vocab; v++) mx = std::max(mx, last[v]);
            std::vector<float> p(cfg.vocab); float sum = 0;
            for (int v = 0; v < cfg.vocab; v++) { p[v] = std::exp((last[v] - mx) / temp); sum += p[v]; }
            std::uniform_real_distribution<float> u(0, sum);
            float r = u(rng); int nxt = cfg.vocab - 1;
            for (int v = 0; v < cfg.vocab; v++) { r -= p[v]; if (r <= 0) { nxt = v; break; } }
            out.push_back(vocab.itos[nxt]);
            ctxw.erase(ctxw.begin());
            ctxw.push_back(nxt);
        }
        return out;
    };

    std::printf("\n=== generated (temp=0.8) ===\n%s\n", generate("Alice ", 400, 0.8f).c_str());
    return 0;
}
