// test_autograd_eager.cpp — eager autograd gradcheck for the minimal op set.
//
// Tests Tensor::backward() / grad() / value_and_grad() against central finite
// differences, mirroring the methodology in test_gradcheck.cpp.
//
// Ops covered: add, sub, mul, matmul (dot), relu, gelu.
//
// Each test:
//   1. Construct an input Tensor W (requires_grad=true).
//   2. Compute a scalar loss eagerly inside a TapeScope.
//   3. Call backward() to get analytic gradients (W.grad()).
//   4. Perturb W element-wise and evaluate the loss forward-only to get
//      numerical gradients via central differences.
//   5. Compare analytic vs numerical; PASS if max abs err < threshold.
//
// On-device run is deferred (orchestrator gates on /dev/accel check).
#include "autograd.hpp"   // pulls tensor.hpp → eager.hpp transitively
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

using namespace tpu;

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

static bool check_grads(const std::vector<float>& analytic,
                        const std::vector<float>& numeric,
                        const char* label,
                        float tol = 1e-2f)
{
    if (analytic.size() != numeric.size()) {
        std::printf("FAIL [%s]: size mismatch %zu vs %zu\n",
                    label, analytic.size(), numeric.size());
        return false;
    }
    double max_err = 0.0;
    for (size_t i = 0; i < analytic.size(); ++i) {
        double err = std::fabs((double)analytic[i] - (double)numeric[i]);
        max_err = std::max(max_err, err);
    }
    bool ok = max_err < (double)tol;
    std::printf("%s [%s]: max_abs_err = %.6f (tol=%.4f)\n",
                ok ? "PASS" : "FAIL", label, max_err, (double)tol);
    if (!ok) {
        // Print first few mismatches for debugging.
        int printed = 0;
        for (size_t i = 0; i < analytic.size() && printed < 8; ++i) {
            double err = std::fabs((double)analytic[i] - (double)numeric[i]);
            if (err > (double)tol) {
                std::printf("  [%zu] analytic=%.6f numeric=%.6f err=%.6f\n",
                            i, analytic[i], numeric[i], err);
                ++printed;
            }
        }
    }
    return ok;
}

// Numerically differentiate a scalar function of a flat parameter vector.
// fn(w) must return a scalar Tensor using w (no grad, so run outside TapeScope).
static std::vector<float>
numerical_grad(const std::vector<float>& W,
               const Shape& shape,
               std::function<float(const std::vector<float>&)> fn,
               float eps = 1e-3f)
{
    std::vector<float> g(W.size());
    for (size_t i = 0; i < W.size(); ++i) {
        std::vector<float> Wp = W, Wm = W;
        Wp[i] += eps;
        Wm[i] -= eps;
        float fp = fn(Wp);
        float fm = fn(Wm);
        g[i] = (fp - fm) / (2.0f * eps);
    }
    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers: reset requires_grad param between steps.
// ─────────────────────────────────────────────────────────────────────────────

// Run backward and return the grad vector for W.
// Temporarily clears W.grad() before so accumulation starts fresh.
static std::vector<float>
analytic_grad_backward(Tensor& W,
                        std::function<Tensor(Tensor&)> loss_fn)
{
    // Clear any accumulated grad.
    W.set_grad(nullptr);

    TapeScope scope;
    Tensor loss = loss_fn(W);
    loss.backward();
    // After backward, scope dtor restores current_tape.
    // The tape was reset inside backward().
    Tensor g = W.grad();
    if (!g.valid())
        throw Error("analytic_grad_backward: W.grad() is null after backward");
    return g.to_host();
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: add — loss = sum(W + W) = 2 * sum(W),  dW[i] = 2
// ─────────────────────────────────────────────────────────────────────────────
static bool test_add() {
    std::vector<float> Wv = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f};
    Shape shape = {2, 3};

    Tensor W = from_host(Wv, shape);
    W.requires_grad_(true);

    // Analytic.
    auto analytic = analytic_grad_backward(W, [](Tensor& w) {
        Tensor s = add(w, w);
        return reduce_sum(s, {0, 1});
    });

    // Numerical.
    auto numeric = numerical_grad(Wv, shape, [&](const std::vector<float>& wv) {
        Tensor wt = from_host(wv, shape);
        Tensor s  = add(wt, wt);
        Tensor l  = reduce_sum(s, {0, 1});
        return l.to_host()[0];
    });

    return check_grads(analytic, numeric, "add");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: sub — loss = sum(W - W2) where W2 = -W (so loss = 2*sum(W))
// ─────────────────────────────────────────────────────────────────────────────
static bool test_sub() {
    std::vector<float> Wv = {0.2f, -0.3f, 0.1f, -0.4f};
    Shape shape = {2, 2};

    Tensor W = from_host(Wv, shape);
    W.requires_grad_(true);

    // Use a fixed W2 (not requires_grad).
    std::vector<float> W2v = {0.5f, 0.1f, -0.2f, 0.3f};
    Tensor W2 = from_host(W2v, shape);

    auto analytic = analytic_grad_backward(W, [&](Tensor& w) {
        Tensor d = sub(w, W2);
        return reduce_sum(d, {0, 1});
    });

    auto numeric = numerical_grad(Wv, shape, [&](const std::vector<float>& wv) {
        Tensor wt = from_host(wv, shape);
        Tensor d  = sub(wt, W2);
        Tensor l  = reduce_sum(d, {0, 1});
        return l.to_host()[0];
    });

    return check_grads(analytic, numeric, "sub");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: mul — loss = sum(W * W) = sum(W^2),  dW[i] = 2*W[i]
// ─────────────────────────────────────────────────────────────────────────────
static bool test_mul() {
    std::vector<float> Wv = {0.1f, -0.5f, 0.3f, 0.7f};
    Shape shape = {2, 2};

    Tensor W = from_host(Wv, shape);
    W.requires_grad_(true);

    auto analytic = analytic_grad_backward(W, [](Tensor& w) {
        Tensor s = mul(w, w);
        return reduce_sum(s, {0, 1});
    });

    auto numeric = numerical_grad(Wv, shape, [&](const std::vector<float>& wv) {
        Tensor wt = from_host(wv, shape);
        Tensor s  = mul(wt, wt);
        Tensor l  = reduce_sum(s, {0, 1});
        return l.to_host()[0];
    });

    return check_grads(analytic, numeric, "mul");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: matmul — f(W) = sum(tanh(X @ W)^2)
// Mirrors test_gradcheck.cpp to confirm eager autodiff matches the static path.
// ─────────────────────────────────────────────────────────────────────────────
static bool test_matmul() {
    const int M = 2, K = 3, N = 4;
    std::vector<float> Xv = {0.1f, -0.2f, 0.3f, 0.4f, 0.5f, -0.6f};
    std::vector<float> Wv = {0.2f, -0.1f, 0.0f, 0.3f,
                              -0.4f, 0.5f, 0.1f, -0.2f,
                              0.3f, 0.2f, -0.5f, 0.1f};
    Shape Xshape = {M, K}, Wshape = {K, N};

    Tensor X = from_host(Xv, Xshape);   // no grad
    Tensor W = from_host(Wv, Wshape);
    W.requires_grad_(true);

    auto analytic = analytic_grad_backward(W, [&](Tensor& w) {
        Tensor h = tanh_op(matmul(X, w));
        Tensor h2 = mul(h, h);
        return reduce_sum(h2, {0, 1});
    });

    auto numeric = numerical_grad(Wv, Wshape, [&](const std::vector<float>& wv) {
        Tensor wt = from_host(wv, Wshape);
        Tensor h  = tanh_op(matmul(X, wt));
        Tensor h2 = mul(h, h);
        return reduce_sum(h2, {0, 1}).to_host()[0];
    });

    return check_grads(analytic, numeric, "matmul(tanh)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: relu — f(W) = sum(relu(W))
// Gradient is 1 where W > 0, 0 elsewhere.
// ─────────────────────────────────────────────────────────────────────────────
static bool test_relu() {
    // Keep all inputs clear of 0: ReLU is non-differentiable at the kink, where
    // central differencing yields ~0.5 but the analytic subgradient is
    // relu'(0)=0 (matches PyTorch). That is a finite-difference artifact, not a
    // framework error — so the test data must avoid values within eps of 0.
    std::vector<float> Wv = {-0.3f, 0.4f, 0.6f, -0.1f, 0.7f, -0.5f};
    Shape shape = {2, 3};

    Tensor W = from_host(Wv, shape);
    W.requires_grad_(true);

    auto analytic = analytic_grad_backward(W, [](Tensor& w) {
        return reduce_sum(relu(w), {0, 1});
    });

    auto numeric = numerical_grad(Wv, shape, [&](const std::vector<float>& wv) {
        Tensor wt = from_host(wv, shape);
        return reduce_sum(relu(wt), {0, 1}).to_host()[0];
    });

    // Use slightly larger tolerance for relu at the kink (W≈0 case).
    return check_grads(analytic, numeric, "relu", 1e-2f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: gelu — f(W) = sum(gelu(W))
// ─────────────────────────────────────────────────────────────────────────────
static bool test_gelu() {
    std::vector<float> Wv = {-0.5f, 0.5f, 1.0f, -1.0f};
    Shape shape = {2, 2};

    Tensor W = from_host(Wv, shape);
    W.requires_grad_(true);

    auto analytic = analytic_grad_backward(W, [](Tensor& w) {
        return reduce_sum(gelu(w), {0, 1});
    });

    auto numeric = numerical_grad(Wv, shape, [&](const std::vector<float>& wv) {
        Tensor wt = from_host(wv, shape);
        return reduce_sum(gelu(wt), {0, 1}).to_host()[0];
    });

    return check_grads(analytic, numeric, "gelu", 1e-2f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: multi-step (tape re-seeding) — two separate backward() calls.
// Verifies Gotcha B: params correctly re-seed on a new tape each step.
// ─────────────────────────────────────────────────────────────────────────────
static bool test_multistep() {
    std::vector<float> Wv = {1.0f, 2.0f, 3.0f, 4.0f};
    Shape shape = {2, 2};

    Tensor W = from_host(Wv, shape);
    W.requires_grad_(true);

    bool ok = true;

    // Step 1: loss = sum(W * W)
    {
        W.set_grad(nullptr);
        TapeScope scope;
        Tensor loss = reduce_sum(mul(W, W), {0, 1});
        loss.backward();
        // Expected: dW = 2W
        auto g = W.grad().to_host();
        std::vector<float> expected;
        for (float v : Wv) expected.push_back(2.0f * v);
        ok &= check_grads(g, expected, "multistep/step1");
    }

    // Step 2: same W, same loss function — should give same result.
    // (Verifies that the tape node from step 1 doesn't contaminate step 2.)
    {
        W.set_grad(nullptr);
        TapeScope scope;
        Tensor loss = reduce_sum(mul(W, W), {0, 1});
        loss.backward();
        auto g = W.grad().to_host();
        std::vector<float> expected;
        for (float v : Wv) expected.push_back(2.0f * v);
        ok &= check_grads(g, expected, "multistep/step2");
    }

    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: value_and_grad functional interface
// ─────────────────────────────────────────────────────────────────────────────
static bool test_value_and_grad() {
    std::vector<float> Wv = {0.3f, -0.4f, 0.5f, -0.6f};
    Shape shape = {2, 2};

    Tensor W = from_host(Wv, shape);
    W.requires_grad_(true);

    auto [loss_t, grads] = value_and_grad(
        [&]() -> Tensor {
            return reduce_sum(mul(W, W), {0, 1});
        },
        {&W}
    );

    // loss should equal sum(W*W)
    float loss_val = loss_t.to_host()[0];
    float expected_loss = 0.0f;
    for (float v : Wv) expected_loss += v * v;

    bool ok = true;
    if (std::fabs(loss_val - expected_loss) > 1e-4f) {
        std::printf("FAIL [value_and_grad/loss]: got %.6f expected %.6f\n",
                    loss_val, expected_loss);
        ok = false;
    }

    // grad should be 2W
    auto g = grads[0].to_host();
    std::vector<float> expected_g;
    for (float v : Wv) expected_g.push_back(2.0f * v);
    ok &= check_grads(g, expected_g, "value_and_grad/grad");

    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: grad() functional interface
// ─────────────────────────────────────────────────────────────────────────────
static bool test_grad_fn() {
    std::vector<float> Wv = {0.2f, 0.8f, -0.3f, 0.5f};
    Shape shape = {2, 2};

    Tensor W = from_host(Wv, shape);
    W.requires_grad_(true);

    auto grads = grad(
        [&]() -> Tensor {
            // loss = sum(tanh(W)^2)
            Tensor t  = tanh_op(W);
            Tensor t2 = mul(t, t);
            return reduce_sum(t2, {0, 1});
        },
        {&W}
    );

    // Numerical reference.
    auto numeric = numerical_grad(Wv, shape, [&](const std::vector<float>& wv) {
        Tensor wt = from_host(wv, shape);
        Tensor t  = tanh_op(wt);
        Tensor t2 = mul(t, t);
        return reduce_sum(t2, {0, 1}).to_host()[0];
    });

    return check_grads(grads[0].to_host(), numeric, "grad_fn(tanh^2)");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== test_autograd_eager ===\n");

    bool all_ok = true;
    all_ok &= test_add();
    all_ok &= test_sub();
    all_ok &= test_mul();
    all_ok &= test_matmul();
    all_ok &= test_relu();
    all_ok &= test_gelu();
    all_ok &= test_multistep();
    all_ok &= test_value_and_grad();
    all_ok &= test_grad_fn();

    std::printf("\n%s\n", all_ok ? "ALL PASS" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
