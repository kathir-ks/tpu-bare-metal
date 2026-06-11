// bench_gpt.cpp — ~100M-param GPT trained data-parallel on all TPU chips,
// instrumented for the C++-vs-JAX parity benchmark.
//
// Consumes bench/data/tokens.bin (uint16 LE) + meta.json from prepare_data.py.
// Batch order is a deterministic function of the step/row counters, mirrored
// exactly in bench/jax_baseline.py so both stacks see identical data.
//
// Usage: ./bench_gpt [steps=200] [batch_per_replica=8] [n_layer=12] [d_model=768] [n_head=12]
// (d_ff = 4*d_model). The default is the 98M parity config; e.g.
// `./bench_gpt 20 2 20 2048 16` is a ~1.04B feasibility run.
// Writes metrics to bench/results_cpp.json.
#include "gpt.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tpu;

// deterministic batch start offsets shared with the JAX baseline:
// row k (global, monotonically increasing) starts at (k*PRIME) % range.
static inline uint64_t batch_start(uint64_t k, uint64_t range) {
    return (k * 123456791ULL) % range;
}

static std::vector<uint16_t> load_tokens(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw Error("cannot open " + path + " (run bench/prepare_data.py)");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint16_t> t(sz / 2);
    if (fread(t.data(), 2, t.size(), f) != t.size()) { fclose(f); throw Error("short read"); }
    fclose(f);
    return t;
}

int main(int argc, char** argv) {
    const int   steps = argc > 1 ? atoi(argv[1]) : 200;
    const int64_t Bpr = argc > 2 ? atoi(argv[2]) : 8;

    Context ctx;
    int nd = ctx.num_addressable_devices();

    auto tokens = load_tokens("bench/data/tokens.bin");
    printf("tokens: %zu\n", tokens.size());

    GPTConfig cfg;
    cfg.vocab = 8192;
    cfg.n_layer = argc > 3 ? atoi(argv[3]) : 12;
    cfg.d_model = argc > 4 ? atoi(argv[4]) : 768;
    cfg.n_head  = argc > 5 ? atoi(argv[5]) : 12;
    cfg.d_ff    = 4 * cfg.d_model;
    cfg.block_size = 512;
    const int64_t T = cfg.block_size;
    const int64_t Bglobal = Bpr * nd;
    const uint64_t range = tokens.size() - T - 1;

    DataParallelTrainer tr(ctx, AdamCfg{0.9, 0.95, 1e-8, 0.0});
    printf("compiling %lldL d%lld ff%lld vocab%lld T%lld, batch %lldx%d...\n",
           (long long)cfg.n_layer, (long long)cfg.d_model, (long long)cfg.d_ff,
           (long long)cfg.vocab, (long long)T, (long long)Bpr, nd);
    fflush(stdout);

    auto c0 = std::chrono::steady_clock::now();
    tr.build([&](TrainCtx& c, Value x, Value y) { return gpt_loss(c, x, y, cfg, Bpr, T); },
             {Bpr, T}, DType::S32, {Bpr, T}, DType::S32, "DEFAULT");
    double compile_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - c0).count();

    int64_t nparams = tr.param_count();
    int64_t hbm = ctx.device_info(0).bytes_in_use;
    printf("params %lld (%.1fM)  compile %.1fs  HBM after init %.2f GB\n",
           (long long)nparams, nparams / 1e6, compile_s, hbm / 1e9);

    std::vector<int32_t> xb(Bglobal * T), yb(Bglobal * T);
    uint64_t k = 0;  // global row counter
    auto make_batch = [&] {
        for (int64_t b = 0; b < Bglobal; b++, k++) {
            uint64_t s = batch_start(k, range);
            for (int64_t t = 0; t < T; t++) {
                xb[b * T + t] = tokens[s + t];
                yb[b * T + t] = tokens[s + t + 1];
            }
        }
    };

    const int warmup = 10;
    std::vector<double> times;
    std::vector<float> losses;
    int64_t peak_hbm = hbm;
    for (int it = 0; it < steps; it++) {
        make_batch();
        float lr = 3e-4f * std::min(1.0f, (it + 1) / 100.0f);
        auto t0 = std::chrono::steady_clock::now();
        float loss = tr.step(xb, yb, lr);
        double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        if (it >= warmup) times.push_back(dt);
        losses.push_back(loss);
        if (it % 20 == 0 || it == steps - 1) {
            peak_hbm = std::max(peak_hbm, ctx.device_info(0).bytes_in_use);
            printf("step %4d  loss %.4f  %.0f ms\n", it, loss, dt * 1e3);
            fflush(stdout);
        }
    }

    std::sort(times.begin(), times.end());
    double med = times[times.size() / 2];
    double toks_per_s = (double)(Bglobal * T) / med;
    // analytic train FLOPs: 6 * params * tokens per step
    double mfu = 6.0 * (double)nparams * (double)(Bglobal * T) / med
                 / ((double)nd * 275e12);

    FILE* f = fopen("bench/results_cpp.json", "w");
    fprintf(f, "{\n  \"stack\": \"cpp\", \"params\": %lld, \"devices\": %d,\n"
               "  \"batch_global\": %lld, \"seq_len\": %lld, \"precision\": \"DEFAULT(bf16)\",\n"
               "  \"compile_s\": %.2f, \"median_step_s\": %.4f, \"tokens_per_s\": %.0f,\n"
               "  \"mfu\": %.4f, \"peak_hbm_bytes\": %lld,\n  \"losses\": [",
            (long long)nparams, nd, (long long)Bglobal, (long long)T,
            compile_s, med, toks_per_s, mfu, (long long)peak_hbm);
    for (size_t i = 0; i < losses.size(); i++)
        fprintf(f, "%s%.4f", i ? ", " : "", losses[i]);
    fprintf(f, "]\n}\n");
    fclose(f);

    printf("median step %.0f ms  %.0f tok/s  MFU %.1f%%  peak HBM %.2f GB\n",
           med * 1e3, toks_per_s, mfu * 100, peak_hbm / 1e9);

    // checkpoint round-trip at scale: forward loss (lr=0 step) on a fixed
    // batch must be identical before save and after restore.
    make_batch();
    float loss_a = tr.step(xb, yb, 0.0f);
    tr.save_checkpoint("/tmp/bench_gpt.ckpt");
    tr.load_checkpoint("/tmp/bench_gpt.ckpt");
    // rewind the row counter so the same batch is regenerated
    k -= (uint64_t)Bglobal; make_batch();
    float loss_b = tr.step(xb, yb, 0.0f);
    bool ckpt_ok = std::fabs(loss_a - loss_b) < 1e-5f;
    printf("ckpt roundtrip: %.6f vs %.6f -> %s\n", loss_a, loss_b,
           ckpt_ok ? "ok" : "MISMATCH");

    bool pass = losses.back() < losses.front() && ckpt_ok;
    printf("loss %.4f -> %.4f : %s\n", losses.front(), losses.back(),
           pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
