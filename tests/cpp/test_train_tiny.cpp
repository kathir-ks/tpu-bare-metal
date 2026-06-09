// test_train_tiny.cpp — validate the Trainer end-to-end on a tiny memorization
// task (embedding -> linear -> cross-entropy). Loss must fall toward zero.
#include "nn.hpp"
#include <cstdio>

using namespace tpu;

int main() {
    Context ctx;
    const int64_t B = 1, T = 8, V = 8, D = 16;

    // model: logits[B*T, V] = linear(embedding(ids))
    auto model = [&](TrainCtx& c, Value x, Value y) -> Value {
        Value emb = nn::embedding(c, x, "tok", V, D);          // [B,T,D]
        Value logits = nn::linear(c, emb, "head", D, V);        // [B,T,V]
        Value flat = c.g.reshape(logits, {B * T, V});
        Value tgt  = c.g.reshape(y, {B * T});
        return nn::cross_entropy(c.g, flat, tgt, B * T, V);
    };

    Trainer tr(ctx, AdamCfg{});
    tr.build(model, {B, T}, DType::S32, {B, T}, DType::S32);
    printf("params: %zu tensors, %lld scalars\n", tr.num_params(),
           (long long)tr.param_count());

    std::vector<int32_t> x = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int32_t> ytgt = {1, 2, 3, 4, 5, 6, 7, 0};

    float loss0 = 0;
    for (int it = 0; it < 300; it++) {
        float loss = tr.step(x, ytgt, 3e-2f);
        if (it == 0) loss0 = loss;
        if (it % 50 == 0 || it == 299)
            printf("step %3d  loss %.5f\n", it, loss);
    }
    float lossN = tr.step(x, ytgt, 0.0f);
    printf("loss0=%.4f lossN=%.4f -> %s\n", loss0, lossN,
           (lossN < 0.1f && lossN < loss0) ? "PASS" : "FAIL");
    return (lossN < 0.1f) ? 0 : 1;
}
