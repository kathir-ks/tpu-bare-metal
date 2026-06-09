// test_ckpt.cpp — train a tiny model, checkpoint it, restore into a fresh
// trainer, and confirm the restored loss matches without any further training.
#include "nn.hpp"
#include <cstdio>
#include <cmath>

using namespace tpu;

int main() {
    Context ctx;
    const int64_t B = 1, T = 8, V = 8, D = 16;
    auto model = [&](TrainCtx& c, Value x, Value y) -> Value {
        Value emb = nn::embedding(c, x, "tok", V, D);
        Value logits = nn::linear(c, emb, "head", D, V);
        return nn::cross_entropy(c.g, c.g.reshape(logits, {B*T, V}),
                                 c.g.reshape(y, {B*T}), B*T, V);
    };
    std::vector<int32_t> x = {0,1,2,3,4,5,6,7}, y = {1,2,3,4,5,6,7,0};
    const char* path = "/tmp/tpu_ckpt_test.bin";

    float trained_loss;
    {
        Trainer tr(ctx, AdamCfg{});
        tr.build(model, {B,T}, DType::S32, {B,T}, DType::S32);
        for (int i = 0; i < 250; i++) tr.step(x, y, 3e-2f);
        trained_loss = tr.step(x, y, 0.0f);
        tr.save_checkpoint(path);
        printf("trained loss = %.5f, saved checkpoint\n", trained_loss);
    }

    // fresh trainer: random init would give loss ~ln(8)=2.08; after load, ~trained.
    Trainer tr2(ctx, AdamCfg{});
    tr2.build(model, {B,T}, DType::S32, {B,T}, DType::S32);
    float fresh = tr2.step(x, y, 0.0f);
    tr2.load_checkpoint(path);
    float restored = tr2.step(x, y, 0.0f);
    printf("fresh loss = %.5f, restored loss = %.5f\n", fresh, restored);

    bool ok = (restored < 0.01f) && (std::fabs(restored - trained_loss) < 0.05f) && (fresh > 1.0f);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
