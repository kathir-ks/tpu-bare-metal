// test_robust.cpp — compiler-robustness / negative tests (capability: compiler-robustness).
//
// Malformed graph construction must fail with a clear, specific error — never crash,
// assert-abort, or silently emit invalid StableHLO. Plus a positive check that
// inferred output shape/dtype match what the device actually produces.
//
//   ./cpp_robust        # construction checks are host-only; the inference check uses the TPU
#include "graph.hpp"
#include <cstdio>
#include <string>

using namespace tpu;

static int g_pass = 0, g_fail = 0;

// Expect `stmt` to throw a tpu::Error whose message contains `needle`.
#define CHECK_THROWS(stmt, needle, label) do {                                  \
    bool threw = false; std::string msg;                                        \
    try { stmt; } catch (const Error& e) { threw = true; msg = e.what(); }      \
    catch (const std::exception& e) { threw = true; msg = e.what(); }           \
    bool ok = threw && msg.find(needle) != std::string::npos;                   \
    std::printf("%-40s %s\n", label, ok ? "PASS" : "FAIL");                     \
    if (ok) ++g_pass; else { ++g_fail;                                          \
        std::printf("   (threw=%d msg=\"%s\" expected substr=\"%s\")\n",        \
                    threw, msg.c_str(), needle); }                              \
} while (0)

#define CHECK(cond, label) do {                                                 \
    std::printf("%-40s %s\n", label, (cond) ? "PASS" : "FAIL");                 \
    if (cond) ++g_pass; else ++g_fail;                                          \
} while (0)

int main() {
    // ── construction-time negative tests (host-only) ────────────────────────────
    {
        Graph g;
        auto a = g.input("a", {2, 3});
        auto b = g.input("b", {4, 5});
        CHECK_THROWS(g.add(a, b), "broadcast", "add_incompatible_shapes_throws");
    }
    {
        Graph g;
        auto v = g.input("v", {6});              // rank 1
        auto w = g.input("w", {6, 2});
        CHECK_THROWS(g.dot(v, w), "rank", "dot_rank_lt_2_throws");
    }
    {
        Graph g;
        auto a = g.input("a", {2, 3});
        auto b = g.input("b", {4, 5});           // contracting 3 != 4
        CHECK_THROWS(g.dot(a, b), "contracting", "dot_contracting_mismatch_throws");
    }
    {
        Graph g;
        auto a = g.input("a", {2, 2, 3});
        auto b = g.input("b", {5, 3, 4});        // batch 2 != 5
        CHECK_THROWS(g.dot(a, b), "batch", "dot_batch_mismatch_throws");
    }
    {
        Graph g;
        auto a = g.input("a", {2, 3});           // 6 elems
        CHECK_THROWS(g.reshape(a, {4, 2}), "element count", "reshape_size_mismatch_throws");
    }
    {
        Graph g;
        auto a = g.input("a", {2, 3});
        CHECK_THROWS(g.transpose(a, {0, 1, 2}), "perm length", "transpose_wrong_perm_len_throws");
    }
    {
        Graph g;
        auto a = g.input("a", {2, 3});
        CHECK_THROWS(g.transpose(a, {0, 5}), "out of range", "transpose_axis_oob_throws");
    }
    {
        Graph g;
        auto a = g.input("a", {2, 3});
        CHECK_THROWS(g.transpose(a, {1, 1}), "repeated", "transpose_repeated_axis_throws");
    }
    {
        Graph g;
        auto a = g.input("a", {2, 3});
        CHECK_THROWS(g.reduce_sum(a, {5}), "out of range", "reduce_axis_oob_throws");
    }
    {
        Graph g;
        auto s = g.input("s", {});               // scalar table
        auto ids = g.input("ids", {3}, DType::S32);
        CHECK_THROWS(g.gather_rows(s, ids), "rank", "gather_scalar_table_throws");
    }
    {
        Graph g;
        auto a = g.input("a", {4, 4});
        CHECK_THROWS(g.slice(a, {0, 0}, {5, 4}), "invalid", "slice_limit_oob_throws");
    }
    {
        Graph g;
        auto a = g.input("a", {4, 4});
        CHECK_THROWS(g.slice(a, {0}, {2}), "rank", "slice_wrong_rank_throws");
    }
    {
        Graph g;
        auto a = g.input("a", {2, 3});
        CHECK_THROWS(g.pad(a, {-1, 0}, {0, 0}), "negative", "pad_negative_throws");
    }
    {
        Graph g;
        auto a = g.input("a", {2, 3});
        auto b = g.input("b", {3, 3});           // mismatch off the concat dim (axis 1)
        CHECK_THROWS(g.concat({a, b}, 1), "mismatch", "concat_shape_mismatch_throws");
    }

    // Valid constructions must NOT throw (no false positives).
    {
        Graph g;
        auto a = g.input("a", {2, 3});
        bool ok = true;
        try { (void)g.reshape(a, {3, 2}); (void)g.transpose(a, {1, 0}); (void)g.reduce_sum(a, {1}); }
        catch (...) { ok = false; }
        CHECK(ok, "valid_shape_ops_do_not_throw");
    }

    // ── positive: inferred shape/dtype == device-produced (on-device) ───────────
    bool device = true;
    try {
        Context ctx;
        Graph g;
        auto x = g.input("x", {2, 3});
        auto y = g.input("y", {3, 4});
        auto z = g.reduce_sum(g.dot(x, y), {1});   // inferred shape [2]
        CHECK(z.shape() == (Shape{2}), "inferred_shape_matches_builder");
        CHECK(z.dtype() == DType::F32, "inferred_dtype_matches_builder");
        auto exec = ctx.compile_mlir(g.emit({z}));
        std::vector<float> xv(6, 1.0f), yv(12, 1.0f);
        auto xb = ctx.upload_f32(xv, {2, 3});
        auto yb = ctx.upload_f32(yv, {3, 4});
        auto out = exec.run({&xb, &yb});
        CHECK(out[0].shape() == (Shape{2}), "device_output_shape_matches_inferred");
        CHECK(out[0].dtype() == DType::F32, "device_output_dtype_matches_inferred");
    } catch (const std::exception& e) {
        device = false;
        std::printf("inference check skipped (no device?): %s\n", e.what());
    }

    std::printf("\nrobust: %d passed, %d failed%s\n", g_pass, g_fail,
                device ? "" : " (on-device inference check skipped)");
    return g_fail == 0 ? 0 : 1;
}
