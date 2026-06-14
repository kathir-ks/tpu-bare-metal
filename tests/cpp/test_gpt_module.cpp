// test_gpt_module.cpp — small GPT on the Module/Optimizer frontend (tasks 5.3/5.4/7.3).
//
//   [A] Architecture: param count matches the closed-form expectation.
//   [B] eager == jit forward parity for the FULL transformer (gather embedding,
//       causal attention, RMSNorm, softmax, cross-entropy) — validates the
//       Tensor-level layers build the same IR in both execution modes.
//   [C] Training: the fused JitAdamStep (forward+backward+Adam, donation in
//       place) memorizes a fixed sequence, driving cross-entropy from ~ln(V)
//       toward 0 — the same capability cpp_gpt_smoke proves for the nn.hpp path.
#include "module.hpp"
#include "optim.hpp"      // Adam + JitAdamStep
#include "autograd.hpp"   // value_and_grad / NoGrad

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace tpu;

int main() {
    std::printf("=== test_gpt_module (GPT on Module/Optimizer) ===\n");

    GPTModuleConfig cfg;
    cfg.vocab = 16; cfg.n_layer = 2; cfg.n_head = 2;
    cfg.d_model = 32; cfg.d_ff = 64; cfg.block_size = 16;

    std::mt19937 rng(2026);
    GPT model(cfg, &rng, /*seed=*/77);

    bool all_ok = true;

    // ── [A] Param-count sanity ──────────────────────────────────────────────────
    int64_t pc       = model.param_count();
    int64_t expected = model.expected_param_count();
    std::printf("[A] param_count=%ld expected=%ld  %s\n",
                (long)pc, (long)expected, pc == expected ? "PASS" : "FAIL");
    all_ok &= (pc == expected);
    std::printf("[A] trainable params (deduped) = %zu tensors\n", model.parameters().size());

    // ── Fixed sequence to memorize: seq[0..T], ids=seq[:T], targets=seq[1:T+1] ──
    const int64_t T = cfg.block_size;
    std::vector<int32_t> seq(T + 1);
    for (int64_t i = 0; i <= T; ++i) seq[i] = (int32_t)((i * 7 + 3) % cfg.vocab);
    std::vector<int32_t> ids_v(seq.begin(), seq.begin() + T);
    std::vector<int32_t> tgt_v(seq.begin() + 1, seq.begin() + T + 1);

    Tensor ids = from_host_s32(ids_v, {1, T});
    Tensor tgt = from_host_s32(tgt_v, {1, T});

    // ── [B] eager vs jit forward-loss parity ────────────────────────────────────
    float loss_eager;
    { NoGrad ng; loss_eager = model.loss(ids, tgt).to_host()[0]; }

    auto vgj = value_and_grad_jit(std::function<Tensor(std::vector<Tensor>)>(
                   [&](std::vector<Tensor> b) { return model.loss(b[0], b[1]); }),
                   model.parameters(), 1, "HIGHEST");
    auto [loss_jit, grads] = vgj({ids, tgt});
    float lj = loss_jit.to_host()[0];

    float pdiff = std::fabs(loss_eager - lj);
    std::printf("[B] forward loss eager=%.6f jit=%.6f |diff|=%.3e (ln V=%.4f)\n",
                loss_eager, lj, pdiff, std::log((double)cfg.vocab));
    bool okB = pdiff < 2e-3f && std::isfinite(loss_eager) && std::isfinite(lj);
    std::printf("[B] eager==jit forward parity: %s\n", okB ? "PASS" : "FAIL");
    all_ok &= okB;

    // ── [C] Train to memorize with the fused JitAdamStep ────────────────────────
    JitAdamStep step(std::function<Tensor(std::vector<Tensor>)>(
                         [&](std::vector<Tensor> b) { return model.loss(b[0], b[1]); }),
                     model.parameters(), /*lr=*/3e-3, AdamCfg{}, 1, "HIGHEST");

    float first = 0, last = 0;
    const int STEPS = 200;
    for (int s = 0; s < STEPS; ++s) {
        float l = step({ids, tgt}).to_host()[0];
        if (s == 0) first = l;
        last = l;
        if (s % 40 == 0 || s == STEPS - 1)
            std::printf("[C] step %3d  loss=%.6f\n", s, l);
    }
    std::printf("[C] loss %.6f -> %.6f\n", first, last);
    bool okC = std::isfinite(last) && last < 0.2f && first > 1.0f;
    std::printf("[C] JitAdamStep memorizes GPT: %s\n", okC ? "PASS" : "FAIL");
    all_ok &= okC;

    std::printf("\n%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
