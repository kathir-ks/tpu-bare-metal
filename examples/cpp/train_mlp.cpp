// train_mlp.cpp — v1 end-to-end proof for the eager/jit Tensor frontend.
//
// Demonstrates the whole stack working together on real TPU hardware:
//   * Module system (MLP = Linear → act → Linear)
//   * eager autograd (TapeScope + loss.backward())
//   * Optimizer (Adam, in-place param updates)
//   * jit transform (auto-lifts the model params, compiles the forward)
//
// It trains an MLP on a small synthetic regression task in EAGER mode and
// checks two things:
//   1. the loss decreases (the model learns), and
//   2. the v1 invariant: eager loss == jit loss for the same params
//      (same ops, same VJP-free forward, same XLA — they must agree).
#include "module.hpp"
#include "optim.hpp"
#include "autograd.hpp"   // TapeScope + Tensor::backward() body (required to call backward)
#include "jit.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace tpu;

int main() {
    std::printf("=== train_mlp (eager train + eager==jit parity) ===\n");

    const int N = 16, IN = 4, HID = 32, OUT = 2;

    // ── Synthetic data: a fixed smooth target function of X ─────────────────────
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Xv(N * IN), Yv(N * OUT);
    for (auto& v : Xv) v = nd(rng);
    for (int i = 0; i < N; ++i)
        for (int o = 0; o < OUT; ++o) {
            float s = 0.0f;
            for (int k = 0; k < IN; ++k) s += Xv[i * IN + k] * (0.3f * (k + 1) * (o + 1));
            Yv[i * OUT + o] = std::tanh(s);   // nonlinear target
        }

    Tensor X = from_host(Xv, {N, IN});
    Tensor Y = from_host(Yv, {N, OUT});   // constant target (no grad)

    MLP   model(IN, HID, OUT, /*bias=*/true, /*act=*/"gelu", &rng, /*seed=*/1234);
    Adam  opt(model.parameters(), /*lr=*/1e-2, AdamCfg{});

    // Mean-squared-error loss as a scalar Tensor.
    auto loss_of = [&](Tensor x) -> Tensor {
        Tensor pred = model.forward(x);
        Tensor diff = sub(pred, Y);
        Tensor sq   = mul(diff, diff);
        return reduce_sum(sq, {0, 1});       // scalar
    };

    // ── Eager training loop ─────────────────────────────────────────────────────
    float first_loss = 0.0f, last_loss = 0.0f;
    const int STEPS = 300;
    for (int step = 0; step < STEPS; ++step) {
        TapeScope scope;                     // installs a fresh tape
        Tensor loss = loss_of(X);
        float lv = loss.to_host()[0];
        if (step == 0) first_loss = lv;
        last_loss = lv;
        loss.backward();                     // fills param .grad over the tape
        opt.step();                          // Adam update (rebinds param buffers)
        opt.zero_grad();
        if (step % 50 == 0 || step == STEPS - 1)
            std::printf("  step %3d  loss = %.6f\n", step, lv);
    }
    std::printf("eager train: loss %.6f -> %.6f\n", first_loss, last_loss);

    // ── v1 invariant: eager loss == jit loss at the trained params ──────────────
    float eager_loss;
    { NoGrad ng; eager_loss = loss_of(X).to_host()[0]; }

    // jit the forward-loss; model params + Y are auto-lifted as captured inputs.
    // Use HIGHEST precision to match the eager dispatch default exactly.
    auto jitted = jit(std::function<Tensor(Tensor)>(
                          [&](Tensor x) { return loss_of(x); }),
                      /*num_replicas=*/1, /*precision=*/"HIGHEST");
    float jit_loss  = jitted(X).to_host()[0];
    float jit_loss2 = jitted(X).to_host()[0];   // second call = cache hit, no recompile

    float parity_err = std::fabs(eager_loss - jit_loss);
    std::printf("parity: eager=%.6f  jit=%.6f  jit(cache-hit)=%.6f  |diff|=%.3e\n",
                eager_loss, jit_loss, jit_loss2, parity_err);

    bool learned = last_loss < 0.5f * first_loss;
    bool parity  = parity_err < 1e-3f && std::fabs(jit_loss - jit_loss2) < 1e-6f;

    std::printf("learned=%s  parity=%s\n", learned ? "yes" : "no", parity ? "yes" : "no");
    bool ok = learned && parity;
    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
