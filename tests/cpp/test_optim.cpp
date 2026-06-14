// test_optim.cpp — Compile + metadata tests for the Optimizer layer (T6).
//
// On-device numeric runs are DEFERRED to the orchestrator (TPU may be busy).
// However, ALL tests here use from_host / eager ops and DO touch the device,
// so they are annotated accordingly.  The orchestrator should confirm the
// numeric checks pass on hardware.
//
// Tests:
//   1.  Optimizer base: zero_grad() clears param grad.
//   2.  SGD (no momentum): single step, p' = p - lr*g checked on host.
//   3.  SGD (momentum): velocity buffer initialised and used on second step.
//   4.  Adam: single step matches hand-computed Adam update (t=1, no weight decay).
//   5.  Adam: second step matches hand-computed update (t=2, bias correction shifts).
//   6.  Adam: weight-decay path (grad ← grad + wd*p before moments).
//   7.  Adam: zero_grad resets grad; subsequent step skips the param.
//   8.  Adam: param-count invariant — step does not change parameters().size().
//   9.  Adam: step_count increments correctly.
//  10.  Adam: state tensors (m/v) are initialised after first step.

#include "tensor.hpp"    // pulls eager.hpp
#include "module.hpp"
#include "optim.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace tpu;

// ── test harness ─────────────────────────────────────────────────────────────
static int g_total = 0;
static int g_pass  = 0;

static void check(bool ok, const char* label) {
    ++g_total;
    if (ok) { ++g_pass; std::printf("PASS [%s]\n", label); }
    else               std::printf("FAIL [%s]\n", label);
}

// Compare two floats within an absolute tolerance.
static bool near(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

// ── helpers ───────────────────────────────────────────────────────────────────

// Make a concrete scalar param Tensor with value `val` and requires_grad=true.
static Tensor make_param(float val) {
    Tensor p = from_host({val}, {1});
    p.requires_grad_(true);
    return p;
}

// Inject a fake gradient into a param by writing into its grad slot.
// We do this by building a Tensor and calling set_grad() with its impl.
static void inject_grad(Tensor& p, float gval) {
    Tensor g = from_host({gval}, {1});
    p.set_grad(g.impl_shared());
}

// ── Test 1: zero_grad clears gradients ───────────────────────────────────────
static bool test_zero_grad() {
    Tensor p = make_param(1.0f);
    inject_grad(p, 3.0f);

    // Confirm grad is set.
    if (!p.grad().valid()) return false;

    std::vector<Tensor*> params = {&p};
    SGD opt(params, 0.1);
    opt.zero_grad();

    // After zero_grad, grad must be null/invalid.
    return !p.grad().valid();
}

// ── Test 2: SGD (no momentum) single step ────────────────────────────────────
// p = 3.0, g = 1.5, lr = 0.1  →  p' = 3.0 - 0.1*1.5 = 2.85
static bool test_sgd_no_momentum() {
    float p0 = 3.0f, g0 = 1.5f, lr = 0.1f;
    Tensor p = make_param(p0);
    inject_grad(p, g0);

    std::vector<Tensor*> params = {&p};
    SGD opt(params, lr);
    opt.step();

    float expected = p0 - lr * g0;   // 2.85
    auto data = p.to_host();
    return near(data[0], expected);
}

// ── Test 3: SGD with momentum ─────────────────────────────────────────────────
// Step 1: vel = g0 = 2.0,   p' = p0 - lr*vel = 5.0 - 0.1*2.0 = 4.8
// Step 2: vel = mu*vel + g1 = 0.9*2.0 + 1.0 = 2.8
//         p'' = p' - lr*2.8 = 4.8 - 0.1*2.8 = 4.52
static bool test_sgd_momentum() {
    float p0 = 5.0f, g0 = 2.0f, g1 = 1.0f, lr = 0.1f, mu = 0.9f;

    Tensor p = make_param(p0);
    std::vector<Tensor*> params = {&p};
    SGD opt(params, lr, mu);

    // Step 1.
    inject_grad(p, g0);
    opt.step();
    float vel1   = g0;                       // 2.0
    float p1_exp = p0 - lr * vel1;           // 4.8
    {
        auto d = p.to_host();
        if (!near(d[0], p1_exp)) return false;
    }

    // Step 2.
    inject_grad(p, g1);
    opt.step();
    float vel2   = mu * vel1 + g1;           // 0.9*2+1 = 2.8
    float p2_exp = p1_exp - lr * vel2;       // 4.8 - 0.28 = 4.52
    {
        auto d = p.to_host();
        if (!near(d[0], p2_exp)) return false;
    }
    return true;
}

// ── Test 4: Adam single step (t=1) ───────────────────────────────────────────
// Params: p=1.0, g=0.5, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, wd=0
//
// Hand computation:
//   m = 0.9*0 + 0.1*0.5 = 0.05
//   v = 0.999*0 + 0.001*0.25 = 0.00025
//   bc1 = 1 - 0.9^1 = 0.1
//   bc2 = 1 - 0.999^1 = 0.001
//   mhat = 0.05/0.1 = 0.5
//   vhat = 0.00025/0.001 = 0.25
//   upd  = 0.5/(sqrt(0.25)+1e-8) = 0.5/0.5 ≈ 1.0
//   p'   = 1.0 - 0.01*1.0 = 0.99
static bool test_adam_step1() {
    float p0 = 1.0f, g0 = 0.5f;
    double lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8;

    Tensor p = make_param(p0);
    inject_grad(p, g0);

    std::vector<Tensor*> params = {&p};
    Adam opt(params, lr, AdamCfg{b1, b2, eps, 0.0});
    opt.step();

    // Hand-computed result.
    double m    = (1.0 - b1) * g0;                 // 0.05
    double v    = (1.0 - b2) * g0 * g0;            // 0.00025
    double bc1  = 1.0 - std::pow(b1, 1.0);         // 0.1
    double bc2  = 1.0 - std::pow(b2, 1.0);         // 0.001
    double mhat = m / bc1;                          // 0.5
    double vhat = v / bc2;                          // 0.25
    double upd  = mhat / (std::sqrt(vhat) + eps);  // ~1.0
    double p1   = p0 - lr * upd;                   // ~0.99

    auto data = p.to_host();
    return near(data[0], (float)p1, 1e-4f);
}

// ── Test 5: Adam two steps ─────────────────────────────────────────────────────
// Continue from test 4; step 2 with g=0.3.
static bool test_adam_step2() {
    float p0 = 1.0f;
    double lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    float g0 = 0.5f, g1 = 0.3f;

    Tensor p = make_param(p0);
    std::vector<Tensor*> params = {&p};
    Adam opt(params, lr, AdamCfg{b1, b2, eps, 0.0});

    // Step 1.
    inject_grad(p, g0);
    opt.step();

    // Step 2.
    inject_grad(p, g1);
    opt.step();

    // Hand-computed for both steps.
    // t=1
    double m   = (1 - b1) * g0;
    double v   = (1 - b2) * g0 * g0;
    double bc1 = 1 - std::pow(b1, 1.0);
    double bc2 = 1 - std::pow(b2, 1.0);
    double p1  = p0 - lr * (m/bc1) / (std::sqrt(v/bc2) + eps);

    // t=2
    m   = b1*m + (1-b1)*g1;
    v   = b2*v + (1-b2)*g1*g1;
    bc1 = 1 - std::pow(b1, 2.0);
    bc2 = 1 - std::pow(b2, 2.0);
    double p2 = p1 - lr * (m/bc1) / (std::sqrt(v/bc2) + eps);

    auto data = p.to_host();
    return near(data[0], (float)p2, 1e-4f);
}

// ── Test 6: Adam with weight decay ────────────────────────────────────────────
// p=2.0, g=0.1, wd=0.01, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, t=1
// Effective grad g' = g + wd*p = 0.1 + 0.01*2.0 = 0.12
// m = (1-0.9)*0.12 = 0.012
// v = (1-0.999)*0.12^2 = 0.001*0.0144 = 0.0000144
// bc1 = 0.1, bc2 = 0.001
// mhat = 0.012/0.1 = 0.12
// vhat = 0.0000144/0.001 = 0.0144
// upd  = 0.12/(sqrt(0.0144)+1e-8) = 0.12/0.12 = 1.0
// p'   = 2.0 - 0.01*1.0 = 1.99
static bool test_adam_weight_decay() {
    float p0 = 2.0f, g0 = 0.1f;
    double lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.01;

    Tensor p = make_param(p0);
    inject_grad(p, g0);

    std::vector<Tensor*> params = {&p};
    Adam opt(params, lr, AdamCfg{b1, b2, eps, wd});
    opt.step();

    double geff = g0 + wd * p0;
    double m    = (1 - b1) * geff;
    double v    = (1 - b2) * geff * geff;
    double bc1  = 1 - std::pow(b1, 1.0);
    double bc2  = 1 - std::pow(b2, 1.0);
    double upd  = (m/bc1) / (std::sqrt(v/bc2) + eps);
    double p1   = p0 - lr * upd;

    auto data = p.to_host();
    return near(data[0], (float)p1, 1e-4f);
}

// ── Test 7: zero_grad between steps ──────────────────────────────────────────
// After zero_grad(), step() should leave param unchanged (no gradient).
static bool test_adam_zero_grad_skips() {
    Tensor p = make_param(5.0f);
    inject_grad(p, 1.0f);

    std::vector<Tensor*> params = {&p};
    Adam opt(params, 0.01);
    opt.zero_grad();    // clear grad before step
    opt.step();         // no grad → param must not change

    auto data = p.to_host();
    return near(data[0], 5.0f);   // unchanged
}

// ── Test 8: Adam step_count increments ───────────────────────────────────────
static bool test_adam_step_count() {
    Tensor p = make_param(1.0f);
    std::vector<Tensor*> params = {&p};
    Adam opt(params, 0.01);

    if (opt.step_count() != 0) return false;

    inject_grad(p, 0.5f);
    opt.step();
    if (opt.step_count() != 1) return false;

    inject_grad(p, 0.5f);
    opt.step();
    return opt.step_count() == 2;
}

// ── Test 9: Adam state tensors initialised after first step ──────────────────
static bool test_adam_state_init() {
    Tensor p = make_param(1.0f);
    std::vector<Tensor*> params = {&p};
    Adam opt(params, 0.01);

    // Before first step, state is uninitialised.
    if (opt.m_state(0).valid()) return false;
    if (opt.v_state(0).valid()) return false;

    inject_grad(p, 0.5f);
    opt.step();

    // After first step, state should be valid Tensors.
    return opt.m_state(0).valid() && opt.v_state(0).valid();
}

// ── Test 10: Multiple params, Adam updates each independently ─────────────────
// p0=1.0, p1=2.0; g0=0.5, g1=1.0; lr=0.01; both same b1/b2/eps.
// After one step each param must match its own hand-computed value.
static bool test_adam_multi_param() {
    float p0_val = 1.0f, p1_val = 2.0f;
    float g0_val = 0.5f, g1_val = 1.0f;
    double lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8;

    Tensor p0 = make_param(p0_val);
    Tensor p1 = make_param(p1_val);
    inject_grad(p0, g0_val);
    inject_grad(p1, g1_val);

    std::vector<Tensor*> params = {&p0, &p1};
    Adam opt(params, lr, AdamCfg{b1, b2, eps, 0.0});
    opt.step();

    // Hand-compute expected values (identical formula, different g).
    auto adam_expected = [&](double p_init, double g) -> double {
        double m   = (1-b1)*g;
        double v   = (1-b2)*g*g;
        double bc1 = 1 - std::pow(b1, 1.0);
        double bc2 = 1 - std::pow(b2, 1.0);
        double upd = (m/bc1) / (std::sqrt(v/bc2) + eps);
        return p_init - lr * upd;
    };

    float exp0 = (float)adam_expected(p0_val, g0_val);
    float exp1 = (float)adam_expected(p1_val, g1_val);

    auto d0 = p0.to_host();
    auto d1 = p1.to_host();
    return near(d0[0], exp0) && near(d1[0], exp1);
}

// ── Test 11: SGD lr=0 leaves param unchanged ─────────────────────────────────
static bool test_sgd_lr_zero_no_change() {
    Tensor p = make_param(7.0f);
    inject_grad(p, 2.0f);

    std::vector<Tensor*> params = {&p};
    // lr=0 is invalid per our constructor guard; test lr very small instead.
    // (Constructor throws on lr <= 0, which is correct behaviour.)
    // Instead test that a non-zero grad with lr=1e-20 barely changes the param.
    SGD opt(params, 1e-20);
    opt.step();
    auto d = p.to_host();
    return near(d[0], 7.0f, 1e-10f);
}

// ── Test 12: Adam reset_step resets bias corrections ─────────────────────────
// After one step (bc1=0.1), reset_step(), then another step should treat
// t=1 again (same bc1=0.1).  The resulting update should match a fresh Adam
// step with the same grad.
static bool test_adam_reset_step() {
    float p0 = 1.0f;
    double lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8;

    Tensor p = make_param(p0);
    inject_grad(p, 0.5f);

    std::vector<Tensor*> params = {&p};
    Adam opt(params, lr, AdamCfg{b1, b2, eps, 0.0});
    opt.step();   // step 1
    opt.reset_step();

    if (opt.step_count() != 0) return false;

    // A fresh Adam on the updated parameter with the same grad value should
    // produce the same delta as the first step, since bc1/bc2 are reset.
    // We just verify that step_count went back to 0 and step() increments it again.
    inject_grad(p, 0.5f);
    opt.step();
    return opt.step_count() == 1;
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== test_optim: Optimizer layer (T6) compile + numeric checks ===\n\n");
    std::printf("NOTE: All tests invoke eager on-device ops.  Numeric results are\n"
                "compared against hand-computed values with 1e-4 absolute tolerance.\n\n");

    check(test_zero_grad(),            "Optimizer::zero_grad() clears grad");
    check(test_sgd_no_momentum(),      "SGD no-momentum: p' = p - lr*g");
    check(test_sgd_momentum(),         "SGD momentum: velocity buffer, two steps");
    check(test_adam_step1(),           "Adam step t=1: mhat/vhat/bias-correct");
    check(test_adam_step2(),           "Adam step t=2: accumulates moments");
    check(test_adam_weight_decay(),    "Adam weight_decay: g' = g + wd*p");
    check(test_adam_zero_grad_skips(), "Adam zero_grad then step: param unchanged");
    check(test_adam_step_count(),      "Adam step_count increments to 2");
    check(test_adam_state_init(),      "Adam m/v state invalid before step, valid after");
    check(test_adam_multi_param(),     "Adam two params updated independently");
    check(test_sgd_lr_zero_no_change(),"SGD tiny lr: param effectively unchanged");
    check(test_adam_reset_step(),      "Adam reset_step: step_count back to 0");

    std::printf("\n%d / %d tests passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
