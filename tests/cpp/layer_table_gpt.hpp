// layer_table_gpt.hpp — builder for the composite GPT loss (cpp/gpt.hpp). Mirrors
// tests/oracle/layers_gpt.py. The capstone: embedding + N blocks + final rmsnorm +
// untied head + cross_entropy, verified end-to-end forward + grad vs the JAX oracle.
#pragma once
#include "layer_table_base.hpp"
#include "gpt.hpp"

namespace tpu {

inline void register_gpt(LayerMap& T) {
    // Tiny config — must match tests/oracle/layers_gpt.py (_V,_NL,_H,_D,_FF,_T,_B).
    T["gpt__1L_v10_d8_t3"] = LayerSpec{
        { {"ids", {1, 3}, true}, {"targets", {1, 3}, true} },
        /*scalar_out=*/true,
        [](TrainCtx& c, const std::vector<Value>& v) {
            GPTConfig cfg;
            cfg.vocab = 10; cfg.n_layer = 1; cfg.n_head = 2;
            cfg.d_model = 8; cfg.d_ff = 16; cfg.block_size = 3;
            return gpt_loss(c, v[0], v[1], cfg, /*B=*/1, /*T=*/3);
        }};
}

}  // namespace tpu
