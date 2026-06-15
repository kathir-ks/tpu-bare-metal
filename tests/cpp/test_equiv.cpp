// test_equiv.cpp — eager ↔ JIT numerical equivalence (capability: eager-jit-equivalence).
//
// The frontend's central invariant ("one IR, two execution policies"): eager and JIT
// build the SAME graph nodes via the SAME rules, so they must agree numerically. Here
// we run a matrix of compute functions BOTH ways on the TPU and compare values. Eager
// executes op-by-op; JIT fuses into one executable and re-reads captured buffers — the
// results must match within a tight equivalence tolerance (both use HIGHEST).
//
// Also checks JIT's per-call buffer re-read: after mutating a captured param in place,
// a previously-compiled JIT forward must reflect the new value without recompiling.
//
//   ./cpp_equiv     (device required)
#include "jit.hpp"      // jit() ; includes tensor/eager/graph/tpu
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace tpu;

static int g_pass = 0, g_fail = 0;
static const double TOL = 5e-4;   // HIGHEST both sides; XLA fusion vs op-by-op rounding

static double max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

// Run `f` eagerly and under jit on the same input; assert value equivalence.
static void check_equiv(const char* label, std::function<Tensor(Tensor)> f, Tensor x) {
    std::vector<float> ve = f(x).to_host();                 // eager path
    auto jf = jit(f);                                       // HIGHEST (default precision)
    std::vector<float> vj = jf(x).to_host();                // jit path (compiles on first call)
    double e = max_abs_err(ve, vj);
    bool ok = e <= TOL;
    std::printf("%-32s eager↔jit max|Δ| = %.3e  %s\n", label, e, ok ? "PASS" : "FAIL");
    if (ok) ++g_pass; else ++g_fail;
}

int main() {
    try { (void)global_context(); }
    catch (const std::exception& e) {
        std::printf("no device available: %s\n", e.what());
        return 2;
    }

    // Params (captured by the compute closures; auto-lifted into the jit trace).
    Tensor W  = from_host({ 0.1f,-0.2f, 0.3f, 0.0f,  0.4f, 0.1f,-0.1f, 0.2f,
                           -0.3f, 0.2f, 0.1f, 0.4f,  0.0f,-0.1f, 0.2f, 0.3f}, {4,4});
    Tensor W2 = from_host({ 0.2f, 0.1f, 0.0f,-0.1f, -0.2f, 0.3f, 0.1f, 0.0f,
                            0.1f,-0.1f, 0.2f, 0.3f,  0.0f, 0.2f,-0.3f, 0.1f}, {4,4});
    Tensor b  = from_host({ 0.1f, 0.2f, 0.3f, 0.4f,  0.0f,-0.1f, 0.2f, 0.1f}, {2,4});
    Tensor b2 = from_host({-0.1f, 0.0f, 0.1f, 0.2f,  0.3f, 0.1f,-0.2f, 0.0f}, {2,4});
    Tensor x  = from_host({ 1.f, 2.f, 3.f, 4.f,  5.f, 6.f, 7.f, 8.f}, {2,4});

    // ── compute-function matrix ─────────────────────────────────────────────────
    check_equiv("gelu(x@W + b)",
                [&](Tensor t){ return gelu(matmul(t, W) + b); }, x);
    check_equiv("relu(x@W2 + b2)",
                [&](Tensor t){ return relu(matmul(t, W2) + b2); }, x);
    check_equiv("tanh(elemwise square)",
                [&](Tensor t){ Tensor h = matmul(t, W) + b; return tanh_op(mul(h, h)); }, x);
    check_equiv("two-layer mlp",
                [&](Tensor t){ Tensor h = gelu(matmul(t, W) + b); return matmul(h, W2) + b2; }, x);
    check_equiv("residual: h + gelu(h@W2)",
                [&](Tensor t){ Tensor h = matmul(t, W) + b; return h + gelu(matmul(h, W2)); }, x);

    // ── JIT per-call buffer re-read reflects an in-place param update ────────────
    {
        Tensor Wm = from_host({1.f,0.f,0.f,0.f, 0.f,1.f,0.f,0.f,
                               0.f,0.f,1.f,0.f, 0.f,0.f,0.f,1.f}, {4,4});  // identity
        auto jf = jit(std::function<Tensor(Tensor)>([&](Tensor t){ return matmul(t, Wm); }));
        std::vector<float> before = jf(x).to_host();                  // ≈ x (identity)
        double e_id = max_abs_err(before, x.to_host());
        bool ok1 = e_id <= TOL;
        std::printf("%-32s identity matmul max|Δ| = %.3e  %s\n", "jit buffer reread (pre)", e_id, ok1 ? "PASS" : "FAIL");
        if (ok1) ++g_pass; else ++g_fail;

        // Mutate the captured param in place (2*identity) and re-invoke WITHOUT recompiling.
        Tensor Wm2 = from_host({2.f,0.f,0.f,0.f, 0.f,2.f,0.f,0.f,
                                0.f,0.f,2.f,0.f, 0.f,0.f,0.f,2.f}, {4,4});
        Wm.rebind_buffer(Wm2.shared_buffer());
        std::vector<float> after = jf(x).to_host();                   // should be ≈ 2x now
        std::vector<float> twox = x.to_host();
        for (auto& v : twox) v *= 2.0f;
        double e2 = max_abs_err(after, twox);
        bool ok2 = e2 <= TOL;
        std::printf("%-32s after in-place 2x update = %.3e  %s\n", "jit buffer reread (post)", e2, ok2 ? "PASS" : "FAIL");
        if (ok2) ++g_pass; else ++g_fail;
    }

    std::printf("\nequiv: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
