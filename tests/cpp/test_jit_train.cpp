// test_jit_train.cpp — fused-jit training validation (tasks 4.5 + 6.4).
//
//   Test 1 (4.5): value_and_grad_jit produces the SAME loss and gradients as the
//                 eager value_and_grad() for the same model + params.
//   Test 2 (6.4): JitAdamStep (fused forward+backward+Adam, donation in place)
//                 tracks the eager Adam trajectory step-for-step.
//
// Both run on real TPU.  Precision is HIGHEST on both paths so the fused XLA
// program and the eager op-by-op path agree to f32 tolerance.
#include "module.hpp"
#include "optim.hpp"     // Adam + JitAdamStep (pulls jit.hpp)
#include "autograd.hpp"  // value_and_grad + Tensor::backward

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace tpu;

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    std::printf("=== test_jit_train (value_and_grad_jit + JitAdamStep) ===\n");

    const int N = 8, IN = 4, HID = 16, OUT = 3;

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Xv(N * IN), Yv(N * OUT);
    for (auto& v : Xv) v = nd(rng);
    for (auto& v : Yv) v = 0.2f * nd(rng);

    Tensor X = from_host(Xv, {N, IN});
    Tensor Y = from_host(Yv, {N, OUT});

    bool all_ok = true;

    // ── Test 1: value_and_grad_jit vs eager value_and_grad ──────────────────────
    {
        std::mt19937 mrng(123);
        MLP model(IN, HID, OUT, /*bias=*/true, /*act=*/"gelu", &mrng, /*seed=*/2024);
        auto params = model.parameters();

        auto loss_of = [&](Tensor x) -> Tensor {
            Tensor pred = model.forward(x);
            Tensor diff = sub(pred, Y);
            return reduce_sum(mul(diff, diff), {0, 1});
        };

        // Eager.
        auto [loss_e, grads_e] = value_and_grad(
            std::function<Tensor()>([&]() { return loss_of(X); }), params);

        // JIT (fused forward+backward).
        auto vgj = value_and_grad_jit(std::function<Tensor(Tensor)>(
                       [&](Tensor x) { return loss_of(x); }),
                       params, /*num_replicas=*/1, /*precision=*/"HIGHEST");
        auto [loss_j, grads_j] = vgj(X);
        auto [loss_j2, grads_j2] = vgj(X);   // cache hit

        float ld  = std::fabs(loss_e.to_host()[0] - loss_j.to_host()[0]);
        float ld2 = std::fabs(loss_j.to_host()[0] - loss_j2.to_host()[0]);
        std::printf("[1] loss eager=%.6f jit=%.6f |diff|=%.3e  cache-hit|diff|=%.3e\n",
                    loss_e.to_host()[0], loss_j.to_host()[0], ld, ld2);

        float gmax = 0.0f;
        for (size_t i = 0; i < params.size(); ++i)
            gmax = std::max(gmax, max_abs_diff(grads_e[i].to_host(), grads_j[i].to_host()));
        std::printf("[1] max grad |diff| across %zu params = %.3e\n", params.size(), gmax);

        bool ok1 = ld < 1e-3f && ld2 < 1e-6f && gmax < 1e-3f;
        std::printf("[1] value_and_grad_jit parity: %s\n", ok1 ? "PASS" : "FAIL");
        all_ok &= ok1;
    }

    // ── Test 2: JitAdamStep vs eager Adam trajectory ────────────────────────────
    {
        const double LR = 1e-2;
        const int    STEPS = 15;

        // Two models with the SAME init (same seed) → identical starting params.
        std::mt19937 ra(55), rb(55);
        MLP modelA(IN, HID, OUT, true, "gelu", &ra, 99);   // eager
        MLP modelB(IN, HID, OUT, true, "gelu", &rb, 99);   // jit

        Adam optA(modelA.parameters(), LR, AdamCfg{});

        auto lossA = [&](Tensor x) {
            Tensor d = sub(modelA.forward(x), Y);
            return reduce_sum(mul(d, d), {0, 1});
        };
        auto lossB = [&](Tensor x) {
            Tensor d = sub(modelB.forward(x), Y);
            return reduce_sum(mul(d, d), {0, 1});
        };

        JitAdamStep stepB(std::function<Tensor(std::vector<Tensor>)>(
                              [&](std::vector<Tensor> b) { return lossB(b[0]); }),
                          modelB.parameters(), LR, AdamCfg{}, 1, "HIGHEST");

        float first = 0, lastA = 0, lastB = 0, maxstep = 0;
        for (int s = 0; s < STEPS; ++s) {
            float la;
            { TapeScope scope;
              Tensor loss = lossA(X);
              la = loss.to_host()[0];
              loss.backward();
              optA.step();
              optA.zero_grad(); }

            float lb = stepB(X).to_host()[0];

            if (s == 0) first = la;
            lastA = la; lastB = lb;
            float d = std::fabs(la - lb);
            maxstep = std::max(maxstep, d);
            if (s < 3 || s == STEPS - 1)
                std::printf("[2] step %2d  eager=%.6f  jit=%.6f  |diff|=%.3e\n", s, la, lb, d);
        }
        std::printf("[2] first=%.6f  lastA=%.6f  lastB=%.6f  max per-step |diff|=%.3e\n",
                    first, lastA, lastB, maxstep);

        bool decreased = lastA < 0.6f * first && lastB < 0.6f * first;
        bool tracks    = maxstep < 2e-2f;
        std::printf("[2] JitAdamStep parity: decreased=%s tracks=%s -> %s\n",
                    decreased ? "yes" : "no", tracks ? "yes" : "no",
                    (decreased && tracks) ? "PASS" : "FAIL");
        all_ok &= (decreased && tracks);
    }

    std::printf("\n%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
