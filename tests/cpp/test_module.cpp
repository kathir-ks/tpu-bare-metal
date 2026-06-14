// test_module.cpp — Compile-time + metadata tests for the Module system (T5).
//
// On-device numeric run is DEFERRED to the orchestrator (TPU may be busy).
// All tests here are shape checks and parameter-count checks that do NOT
// require on-device execution (they only call from_host / Module constructors
// and inspect metadata).
//
// Tests:
//   1. Linear — forward output shape, parameter count.
//   2. Linear (no bias) — forward output shape, parameter count.
//   3. ReLU module — forward output shape.
//   4. GELU module — forward output shape.
//   5. Sequential — two-layer composition, output shape, parameter count.
//   6. MLP (ReLU) — forward output shape, expected parameter count.
//   7. MLP (GELU) — forward output shape.
//   8. MLP parameter list — recursive parameters() walk.
//   9. cross_entropy — output shape (scalar).
//  10. Module::param_count() matches manual sum.
//
// Note: shapes and parameter counts are checked via Tensor::shape() /
// Module::param_count() — both are host-side metadata queries, no TPU needed.
// The forward() calls that produce output Tensors DO upload host data and
// execute single-op kernels eagerly (matmul etc.), so the test requires a
// linked libtpu_fw.a; but since `make cpp_module` already links it, that is
// fine.  On-device execution is triggered lazily on the first op.

#include "tensor.hpp"   // pulls eager.hpp
#include "module.hpp"

#include <cstdio>
#include <cmath>
#include <vector>

using namespace tpu;

// ── helpers ──────────────────────────────────────────────────────────────────
static int g_total = 0;
static int g_pass  = 0;

static void check(bool ok, const char* label) {
    ++g_total;
    if (ok) { ++g_pass; std::printf("PASS [%s]\n", label); }
    else               std::printf("FAIL [%s]\n", label);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 — Linear: forward output shape (with bias)
// ─────────────────────────────────────────────────────────────────────────────
static bool test_linear_shape() {
    // Input: [batch=4, in=8]   Linear(8, 16, bias=true)   → [4, 16]
    int64_t B = 4, IN = 8, OUT = 16;
    Linear fc(IN, OUT, true);
    Tensor x = zeros({B, IN});
    Tensor y = fc.forward(x);
    return y.shape() == Shape{B, OUT};
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 — Linear param count (with bias)
// ─────────────────────────────────────────────────────────────────────────────
static bool test_linear_param_count_bias() {
    int64_t IN = 8, OUT = 16;
    Linear fc(IN, OUT, true);
    // W: [8, 16] = 128   b: [16] = 16   total = 144
    int64_t expected = IN * OUT + OUT;
    return fc.param_count() == expected;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 — Linear (no bias): forward output shape
// ─────────────────────────────────────────────────────────────────────────────
static bool test_linear_no_bias_shape() {
    int64_t B = 3, IN = 5, OUT = 7;
    Linear fc(IN, OUT, false);
    Tensor x = zeros({B, IN});
    Tensor y = fc.forward(x);
    return y.shape() == Shape{B, OUT};
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 — Linear (no bias): param count
// ─────────────────────────────────────────────────────────────────────────────
static bool test_linear_no_bias_param_count() {
    int64_t IN = 5, OUT = 7;
    Linear fc(IN, OUT, false);
    return fc.param_count() == IN * OUT;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5 — ReLU module output shape
// ─────────────────────────────────────────────────────────────────────────────
static bool test_relu_shape() {
    ReLU act;
    Tensor x = zeros({4, 8});
    Tensor y = act.forward(x);
    return y.shape() == Shape{4, 8};
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6 — GELU module output shape
// ─────────────────────────────────────────────────────────────────────────────
static bool test_gelu_shape() {
    GELU act;
    Tensor x = zeros({4, 8});
    Tensor y = act.forward(x);
    return y.shape() == Shape{4, 8};
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7 — Sequential: shape and param count
// ─────────────────────────────────────────────────────────────────────────────
static bool test_sequential_shape() {
    // Linear(8,16) → ReLU → Linear(16,4)
    Sequential seq;
    seq.add<Linear>("fc1", (int64_t)8, (int64_t)16, true);
    seq.add<ReLU>("relu");
    seq.add<Linear>("fc2", (int64_t)16, (int64_t)4, true);

    Tensor x = zeros({3, 8});
    Tensor y = seq.forward(x);
    return y.shape() == Shape{3, 4};
}

static bool test_sequential_param_count() {
    Sequential seq;
    seq.add<Linear>("fc1", (int64_t)8, (int64_t)16, true);
    seq.add<ReLU>("relu");
    seq.add<Linear>("fc2", (int64_t)16, (int64_t)4, true);

    // fc1: 8*16 + 16 = 144   fc2: 16*4 + 4 = 68   total = 212
    int64_t expected = (8 * 16 + 16) + (16 * 4 + 4);
    return seq.param_count() == expected;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8 — MLP (ReLU): forward output shape
// ─────────────────────────────────────────────────────────────────────────────
static bool test_mlp_relu_shape() {
    int64_t B = 4, IN = 8, HID = 32, OUT = 3;
    MLP mlp(IN, HID, OUT, true, "relu");
    Tensor x = zeros({B, IN});
    Tensor y = mlp.forward(x);
    return y.shape() == Shape{B, OUT};
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9 — MLP (GELU): forward output shape
// ─────────────────────────────────────────────────────────────────────────────
static bool test_mlp_gelu_shape() {
    int64_t B = 4, IN = 8, HID = 32, OUT = 3;
    MLP mlp(IN, HID, OUT, true, "gelu");
    Tensor x = zeros({B, IN});
    Tensor y = mlp.forward(x);
    return y.shape() == Shape{B, OUT};
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10 — MLP: parameter count matches expected formula
// ─────────────────────────────────────────────────────────────────────────────
static bool test_mlp_param_count() {
    int64_t IN = 8, HID = 32, OUT = 3;
    MLP mlp(IN, HID, OUT, true, "relu");
    int64_t expected = MLP::expected_param_count(IN, HID, OUT, true);
    // MLP::expected: IN*HID + HID + HID*OUT + OUT
    return mlp.param_count() == expected;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 11 — MLP: parameters() walk returns correct count
// ─────────────────────────────────────────────────────────────────────────────
static bool test_mlp_parameters_walk() {
    int64_t IN = 8, HID = 32, OUT = 3;
    MLP mlp(IN, HID, OUT, true, "relu");
    auto params = mlp.parameters();
    // fc1 has W + b = 2 params; fc2 has W + b = 2 params; total = 4
    return params.size() == 4;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 12 — MLP: parameter pointers are non-null and valid
// ─────────────────────────────────────────────────────────────────────────────
static bool test_mlp_param_pointers_valid() {
    int64_t IN = 8, HID = 32, OUT = 3;
    MLP mlp(IN, HID, OUT, true, "relu");
    auto params = mlp.parameters();
    for (auto* p : params)
        if (!p || !p->valid()) return false;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 13 — MLP: parameters all have requires_grad == true
// ─────────────────────────────────────────────────────────────────────────────
static bool test_mlp_params_require_grad() {
    int64_t IN = 8, HID = 32, OUT = 3;
    MLP mlp(IN, HID, OUT, true, "relu");
    for (auto* p : mlp.parameters())
        if (!p->requires_grad()) return false;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 14 — cross_entropy: output is a scalar (shape == {})
// ─────────────────────────────────────────────────────────────────────────────
static bool test_cross_entropy_shape() {
    int64_t N = 4, V = 10;
    // logits: [N, V]
    Tensor logits  = zeros({N, V});
    // targets: [N] S32  (all class 0)
    Tensor targets = from_host_s32(std::vector<int32_t>(N, 0), {N});
    Tensor loss    = cross_entropy(logits, targets);
    // loss should be a scalar: shape = {}
    return loss.shape() == Shape{};
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 15 — Linear rank-3 input: x[B,T,in] → y[B,T,out]
// ─────────────────────────────────────────────────────────────────────────────
static bool test_linear_rank3() {
    int64_t B = 2, T = 6, IN = 8, OUT = 16;
    Linear fc(IN, OUT, true);
    Tensor x = zeros({B, T, IN});
    Tensor y = fc.forward(x);
    return y.shape() == Shape{B, T, OUT};
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== test_module: Module system shape + param-count checks ===\n\n");

    check(test_linear_shape(),              "Linear forward shape [4,8]→[4,16]");
    check(test_linear_param_count_bias(),   "Linear(8,16,bias) param count = 144");
    check(test_linear_no_bias_shape(),      "Linear(no bias) forward shape");
    check(test_linear_no_bias_param_count(),"Linear(no bias) param count");
    check(test_relu_shape(),                "ReLU forward shape preserved");
    check(test_gelu_shape(),                "GELU forward shape preserved");
    check(test_sequential_shape(),          "Sequential forward shape [3,8]→[3,4]");
    check(test_sequential_param_count(),    "Sequential param count = 212");
    check(test_mlp_relu_shape(),            "MLP(relu) forward shape [4,8]→[4,3]");
    check(test_mlp_gelu_shape(),            "MLP(gelu) forward shape [4,8]→[4,3]");
    check(test_mlp_param_count(),           "MLP param count matches expected_param_count()");
    check(test_mlp_parameters_walk(),       "MLP parameters() returns 4 pointers");
    check(test_mlp_param_pointers_valid(),  "MLP parameter pointers are valid Tensors");
    check(test_mlp_params_require_grad(),   "MLP parameters have requires_grad==true");
    check(test_cross_entropy_shape(),       "cross_entropy output is scalar");
    check(test_linear_rank3(),              "Linear rank-3 input [2,6,8]→[2,6,16]");

    std::printf("\n%d / %d tests passed\n", g_pass, g_total);
    std::printf("\nNOTE: on-device numeric checks (loss value, backward pass) are\n"
                "deferred to the orchestrator once the TPU is available.\n");
    return (g_pass == g_total) ? 0 : 1;
}
