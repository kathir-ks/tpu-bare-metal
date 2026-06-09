// gpt.hpp — a GPT-style decoder-only transformer, in pure C++.
//
// Builds the model graph from nn.hpp primitives. The same gpt_logits() function
// is used for both the training graph (wrapped in cross-entropy) and the
// inference graph, guaranteeing identical parameter names/order so a Forward
// pass can feed a Trainer's live weights.
#pragma once

#include <string>
#include "nn.hpp"

namespace tpu {

struct GPTConfig {
    int64_t vocab      = 65;
    int64_t n_layer    = 4;
    int64_t n_head     = 4;
    int64_t d_model    = 128;
    int64_t d_ff       = 512;
    int64_t block_size = 64;
};

// ids[B,T] (int32) → logits[B,T,vocab]. Learned token + positional embeddings,
// n_layer pre-norm transformer blocks, final RMSNorm, untied output head.
inline Value gpt_logits(TrainCtx& c, const Value& ids, const GPTConfig& cfg,
                        int64_t B, int64_t T) {
    Graph& g = c.g;
    Value tok = nn::embedding(c, ids, "wte", cfg.vocab, cfg.d_model);   // [B,T,D]
    Value pos = c.param("wpe", {T, cfg.d_model}, normal(0.02f));        // [T,D]
    Value h = g.add(tok, pos);                                          // broadcast over B
    for (int64_t l = 0; l < cfg.n_layer; l++)
        h = nn::block(c, h, "h" + std::to_string(l),
                      B, T, cfg.d_model, cfg.n_head, cfg.d_ff);
    h = nn::rmsnorm(c, h, "lnf", cfg.d_model);
    return nn::linear(c, h, "head", cfg.d_model, cfg.vocab, false);     // [B,T,vocab]
}

// Training loss: mean cross-entropy of next-token prediction.
inline Value gpt_loss(TrainCtx& c, const Value& ids, const Value& targets,
                      const GPTConfig& cfg, int64_t B, int64_t T) {
    Value logits = gpt_logits(c, ids, cfg, B, T);
    Value flat = c.g.reshape(logits, {B * T, cfg.vocab});
    Value tgt  = c.g.reshape(targets, {B * T});
    return nn::cross_entropy(c.g, flat, tgt, B * T, cfg.vocab);
}

}  // namespace tpu
