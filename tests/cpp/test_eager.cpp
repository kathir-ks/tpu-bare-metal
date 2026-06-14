// test_eager.cpp — Minimal proof that the eager dispatch machinery compiles and
// produces correct numerical results on-device.
//
// What this tests (no requires_grad, no tape — pure forward pass):
//   1. from_host: upload two 2×2 F32 matrices.
//   2. matmul: 2×2 @ 2×2 → 2×2.
//   3. relu:   max(x, 0) elementwise.
//   4. to_host: read back and compare against hand-computed expected values.
//
// Expected values (hand-computed):
//   A = [[1, 2], [3, 4]]   B = [[5, 6], [7, 8]]
//   C = A @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
//             = [[19, 22], [43, 50]]
//   D = relu(C) = [[19, 22], [43, 50]]  (all positive, so relu is identity here)
//
// Also tests that negative values ARE zeroed by relu:
//   E = from_host([[−1, 2], [3, −4]])
//   F = relu(E) = [[0, 2], [3, 0]]
#include "tensor.hpp"   // pulls in eager.hpp automatically
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace tpu;

static bool approx_eq(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

static bool check_vec(const std::vector<float>& got,
                       const std::vector<float>& expected,
                       const char* label,
                       float tol = 1e-4f)
{
    if (got.size() != expected.size()) {
        std::printf("FAIL [%s]: size mismatch %zu vs %zu\n",
                    label, got.size(), expected.size());
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!approx_eq(got[i], expected[i], tol)) {
            std::printf("FAIL [%s][%zu]: got %.6f expected %.6f\n",
                        label, i, got[i], expected[i]);
            ok = false;
        }
    }
    if (ok) std::printf("PASS [%s]\n", label);
    return ok;
}

int main() {
    // Eager dispatch needs the global context (lazy-init on first call).
    // We deliberately don't call global_context() directly — the first op
    // call triggers initialization.

    bool all_ok = true;

    // ── Test 1: matmul + relu (all-positive case) ────────────────────────────
    {
        std::vector<float> Av = {1.f, 2.f, 3.f, 4.f};  // [[1,2],[3,4]]
        std::vector<float> Bv = {5.f, 6.f, 7.f, 8.f};  // [[5,6],[7,8]]

        Tensor A = from_host(Av, {2, 2});
        Tensor B = from_host(Bv, {2, 2});

        Tensor C = matmul(A, B);   // [[19,22],[43,50]]
        Tensor D = relu(C);        // same (all positive)

        std::vector<float> D_host = D.to_host();
        std::vector<float> D_exp  = {19.f, 22.f, 43.f, 50.f};
        all_ok &= check_vec(D_host, D_exp, "matmul+relu(positive)");
    }

    // ── Test 2: relu zeros out negatives ────────────────────────────────────
    {
        std::vector<float> Ev = {-1.f, 2.f, 3.f, -4.f};  // [[−1,2],[3,−4]]
        Tensor E = from_host(Ev, {2, 2});
        Tensor F = relu(E);

        std::vector<float> F_host = F.to_host();
        std::vector<float> F_exp  = {0.f, 2.f, 3.f, 0.f};
        all_ok &= check_vec(F_host, F_exp, "relu(negatives zeroed)");
    }

    // ── Test 3: add + mul chain ──────────────────────────────────────────────
    {
        // a = [[1,2],[3,4]], b = [[2,2],[2,2]]
        // c = a + b = [[3,4],[5,6]]
        // d = c * a = [[3,8],[15,24]]
        Tensor a = from_host({1.f, 2.f, 3.f, 4.f}, {2, 2});
        Tensor b = from_host({2.f, 2.f, 2.f, 2.f}, {2, 2});

        Tensor c = add(a, b);
        Tensor d = mul(c, a);

        std::vector<float> d_host = d.to_host();
        std::vector<float> d_exp  = {3.f, 8.f, 15.f, 24.f};
        all_ok &= check_vec(d_host, d_exp, "add+mul chain");
    }

    // ── Test 4: kernel cache hit (same op+shape again) ───────────────────────
    {
        // Second matmul with same shapes → should be a cache hit.
        Tensor p = from_host({1.f, 0.f, 0.f, 1.f}, {2, 2});  // identity
        Tensor q = from_host({3.f, 7.f, 5.f, 2.f}, {2, 2});
        Tensor r = matmul(p, q);  // should == q

        std::vector<float> r_host = r.to_host();
        std::vector<float> r_exp  = {3.f, 7.f, 5.f, 2.f};
        all_ok &= check_vec(r_host, r_exp, "matmul(identity) cache-hit");

        size_t cache_sz = kernel_cache().size();
        std::printf("kernel_cache size after tests: %zu\n", cache_sz);
    }

    // ── Test 5: shape / metadata accessors ───────────────────────────────────
    {
        Tensor t = zeros({3, 4});
        bool ok = (t.shape() == Shape{3, 4})
               && (t.dtype() == DType::F32)
               && (t.rank() == 2)
               && (t.numel() == 12)
               && !t.requires_grad();
        if (!ok) { std::printf("FAIL [metadata]\n"); all_ok = false; }
        else       std::printf("PASS [metadata]\n");
    }

    std::printf("\n%s\n", all_ok ? "ALL PASS" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
