// test_plugin_probe.cpp — backend-agnostic PJRT plugin probe.
//
// Loads whatever plugin PJRT_PLUGIN_PATH (or argv[1]) points at, reports the
// API version and device inventory, and runs one tiny compile+execute. This is
// the smoke test for "the framework speaks generic PJRT, not just libtpu":
// point it at libtpu.so, xla_cuda_plugin.so, or a CPU plugin.
#include "graph.hpp"
#include <cstdio>

using namespace tpu;

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : nullptr;
    tpu_ctx_t* raw = tpu_init(path);
    if (!raw) {
        printf("init failed: %s\n", tpu_strerror(nullptr));
        return 1;
    }
    int maj = 0, min = 0;
    tpu_api_version(raw, &maj, &min);
    printf("PJRT API v%d.%d  devices=%d addressable=%d\n",
           maj, min, tpu_num_devices(raw), tpu_num_addressable_devices(raw));
    tpu_destroy(raw);

    // Full C++-stack path: compile + run y = x*x + 1 on device 0.
    Context ctx(path);
    Graph g;
    auto x = g.input("x", {8});
    auto y = g.add(g.mul(x, x), g.constant(1.0, {8}));
    auto exec = ctx.compile_mlir(g.emit({y}));
    std::vector<float> xs = {0, 1, 2, 3, 4, 5, 6, 7};
    auto xb = ctx.upload_f32(xs, {8});
    auto out = exec.run({&xb})[0].to_host<float>();
    bool ok = true;
    for (int i = 0; i < 8; i++) ok &= out[i] == xs[i] * xs[i] + 1.0f;
    printf("compile+run x*x+1: %s\n", ok ? "PASS" : "FAIL");
    printf("test_plugin_probe: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
