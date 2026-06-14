# C++ API Reference

Public surface of the `cpp/` headers. All types live in namespace `tpu` (with NN
helpers in `tpu::nn`). Include what you need; everything is header-only except
`graph.cpp`.

There are two layers, both live:
- the **graph builder** (`graph.hpp` / `nn.hpp` / `gpt.hpp`) — compile-only, define-then-run;
- the **eager/jit Tensor frontend** (`tensor.hpp` / `eager.hpp` / `autograd.hpp` / `jit.hpp` / `module.hpp` / `optim.hpp`) — define-by-run with `jit` on demand (documented at the end of this file).

```cpp
// graph builder
#include "tpu.hpp"     // Context, Buffer, Executable, DType, Shape
#include "graph.hpp"   // Graph, Value, Op
#include "nn.hpp"      // layers, AdamCfg, Trainer, DataParallelTrainer, Forward
#include "gpt.hpp"     // GPTConfig, gpt_logits, gpt_loss

// eager/jit Tensor frontend
#include "tensor.hpp"  // Tensor
#include "autograd.hpp"// TapeScope, backward, value_and_grad
#include "jit.hpp"     // jit, value_and_grad_jit
#include "module.hpp"  // Module, Linear, MLP, Embedding, RMSNorm, Attention, Block, GPT
#include "optim.hpp"   // SGD, Adam, JitAdamStep
```

---

## `tpu.hpp` — device wrapper

### `enum class DType`
`F32, S32, …` — maps to `tpu_dtype_t`. `dtype_size(d)` returns the element size.

### `using Shape = std::vector<int64_t>`
`num_elements(shape)` returns the product of dims (1 for a scalar `{}`).

### `class Context`
```cpp
Context(const char* plugin_path = nullptr);   // any PJRT plugin: NULL →
        // PJRT_PLUGIN_PATH env → LIBTPU_PATH env → built-in libtpu default
int  num_devices() const;
int  num_addressable_devices() const;
std::string topology() const;
tpu_device_info_t device_info(int idx) const;
std::string last_error() const;

Executable compile_mlir(const std::string& mlir, const void* opts = nullptr,
                        size_t opts_sz = 0);   // default opts = embedded 1-replica blob
Executable compile_mlir_dp(const std::string& mlir);             // 4-replica blob
Executable compile_hlo(const void* code, size_t code_sz, ...);

Buffer upload(const void* data, DType dtype, const Shape& dims, int dev_idx = 0);
Buffer upload_f32(const std::vector<float>&  data, const Shape& dims, int dev = 0);
Buffer upload_s32(const std::vector<int32_t>& data, const Shape& dims, int dev = 0);
```

### `class Buffer` (move-only)
```cpp
Shape  shape() const;
size_t size_bytes() const;
template <class T> std::vector<T> to_host() const;   // download as typed vector
void   download(void* dst, size_t n) const;
explicit operator bool() const;
```

### `class Executable` (move-only)
```cpp
int num_outputs() const;
std::vector<Buffer> run(const std::vector<Buffer*>& args, int dev_idx = 0);
std::vector<std::vector<Buffer>> run_spmd(                       // [device][output]
    const std::vector<std::vector<Buffer*>>& args_per_device);
std::vector<int> device_order(int nreplicas);    // addressable dev idx per replica
void save(const std::string& path);
```

---

## `graph.hpp` — graph builder + autodiff

### `class Graph`
Members you may set: `std::string dot_precision = "HIGHEST";` and `int num_replicas = 1;`.

**Leaves**
```cpp
Value input(const std::string& name, const Shape& shape, DType dt = DType::F32);
Value constant(double val, const Shape& shape = {}, DType dt = DType::F32);
Value scalar(double val, DType dt = DType::F32);
Value iota(const Shape& shape, int64_t dim, DType dt = DType::S32);
```

**Elementwise** — `add sub mul div max min` (binary, NumPy broadcasting);
`exp log sqrt rsqrt tanh logistic abs neg` (unary).

**Structure**
```cpp
Value reshape(const Value& a, const Shape& s);
Value transpose(const Value& a, const std::vector<int64_t>& perm);
Value broadcast_in_dim(const Value& a, const Shape& target, const std::vector<int64_t>& bd);
Value broadcast_to(const Value& a, const Shape& target);
Value convert(const Value& a, DType dt);
```

**Reductions** — `reduce_sum / reduce_max / reduce_mean(a, axes, keepdims=false)`.

**Linear algebra** — `dot(a, b)`: last two dims are the matrix, leading dims batch.

**Gather / scatter**
```cpp
Value gather_rows(const Value& table, const Value& ids);  // table[V,D...], int ids
        // → ids.shape + [D...]; row lookup (embedding). VJP is a scatter-add.
Value scatter_add_rows(const Value& operand, const Value& ids, const Value& updates);
        // operand with updates[i] accumulated into row ids[i] (duplicates add)
```

**Control / misc**
```cpp
Value compare(Value a, Value b, Cmp c);   // Cmp::{GT,GE,EQ,LT,LE,NE}
Value select(Value pred, Value a, Value b);
Value stop_gradient(const Value& a);
Value all_reduce_sum(const Value& a);      // cross-replica sum (data parallel)
```

**Autodiff & emission**
```cpp
std::vector<Value> grad(const Value& loss, const std::vector<Value>& params);
std::string        emit(const std::vector<Value>& outputs) const;   // StableHLO text
```

Example:
```cpp
tpu::Graph g;
auto a = g.input("a", {2,2});
auto b = g.input("b", {2,2});
auto loss = g.reduce_sum(g.mul(g.dot(a,b), g.dot(a,b)), {0,1});
auto grads = g.grad(loss, {a,b});                 // {dL/da, dL/db}
std::string mlir = g.emit({loss, grads[0], grads[1]});
```

---

## `nn.hpp` — layers, optimizer, training

### Layers (free functions)
```cpp
nn::linear(TrainCtx& c, Value x, name, in, out);
nn::embedding(TrainCtx& c, Value ids, name, vocab, d_model);   // one-hot @ table
nn::gelu(Graph& g, Value x);
nn::rmsnorm(TrainCtx& c, Value x, name, dim);
nn::softmax(Graph& g, Value x, int64_t axis);
nn::attention(TrainCtx& c, Value x, name, n_head, d_model, T);  // multi-head causal
nn::block(TrainCtx& c, Value x, name, cfg…);                    // pre-norm block
nn::cross_entropy(Graph& g, Value logits, Value targets, vocab);
```

### `struct AdamCfg { double b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0; };`

### `class Trainer`
```cpp
Trainer(Context& ctx, AdamCfg cfg);
void  build(ModelFn model, Shape x_shape, DType x_dt, Shape y_shape, DType y_dt);
float step(const std::vector<int32_t>& x, const std::vector<int32_t>& y, float lr);
std::vector<std::string> param_names() const;
std::vector<std::pair<std::string,Shape>> param_defs() const;
Buffer& param_buffer(const std::string& name);
void  save_checkpoint(const std::string& path);
void  load_checkpoint(const std::string& path);
```
`ModelFn` is `Value model(TrainCtx& c, Value x, Value y)` returning the scalar loss.
Parameters and Adam moments stay resident in HBM; `step()` returns the scalar loss.

### `class DataParallelTrainer`
Same surface as `Trainer`, but `build`'s shapes are **per-replica** and `step`'s
`global_x/global_y` cover all replicas (global batch = `n_dev × per_replica`).
Gradients are `all_reduce`-summed and averaged; runs via `run_spmd`.

### `Forward` / `ForwardDP` (`ForwardT<TR>`)
```cpp
Forward fwd(trainer);
fwd.build(x_shape, x_dt);                 // inference-only executable
auto out = fwd.run(x);                    // binds the trainer's live weights by name
```

---

## `gpt.hpp` — GPT model

```cpp
struct GPTConfig { int vocab, n_layer, n_head, d_model, d_ff, block_size; };

Value gpt_logits(TrainCtx& c, const Value& ids, const GPTConfig& cfg, int B, int T);
Value gpt_loss  (TrainCtx& c, const Value& ids, const Value& targets,
                 const GPTConfig& cfg, int B, int T);
```

`gpt_loss` is the `ModelFn` you hand to `Trainer::build`; `gpt_logits` (wrapped in a
`Forward`) gives logits for generation. The same code path serves training and
inference.

---

## Minimal end-to-end

```cpp
tpu::Context ctx;
tpu::Trainer tr(ctx, tpu::AdamCfg{});
tpu::GPTConfig cfg{ /*vocab*/45, /*n_layer*/4, /*n_head*/4,
                    /*d_model*/128, /*d_ff*/512, /*block_size*/64 };

tr.build([&](tpu::TrainCtx& c, tpu::Value x, tpu::Value y){
    return tpu::gpt_loss(c, x, y, cfg, B, T);
}, {B,T}, tpu::DType::S32, {B,T}, tpu::DType::S32);

for (int i = 0; i < steps; ++i)
    float loss = tr.step(x_batch, y_batch, /*lr*/3e-4f);

tr.save_checkpoint("gpt.ckpt");
```

See `examples/cpp/train_gpt.cpp` for a complete trainer + sampler.

---

# Eager / JIT Tensor frontend

A PyTorch-shaped, mutable, object-oriented layer on top of the graph builder:
define-by-run eager execution, a tape autograd, a `jit` transform that compiles
on demand, plus `Module` and `Optimizer`. One IR, two execution policies — eager
and jit build the *same* graph nodes and so produce bit-identical numerics
(validated on-device: `cpp_jit_train`, `cpp_gpt_module`).

```cpp
#include "tensor.hpp"    // Tensor (Concrete | Traced handle)
#include "eager.hpp"     // op free-functions, leaf ctors, global_context()
#include "autograd.hpp"  // TapeScope, Tensor::backward, value_and_grad
#include "jit.hpp"       // jit, value_and_grad_jit
#include "module.hpp"    // Module, Linear, MLP, Embedding, RMSNorm, Attention, Block, GPT
#include "optim.hpp"     // SGD, Adam, JitAdamStep
```

## `tensor.hpp` — `class Tensor`

A reference-counted handle over a `TensorImpl` that is either **Concrete** (owns a
device `Buffer`) or **Traced** (a `(Graph*, node)` pair inside a live jit trace).
Copy is cheap (aliases the same Impl); the last handle frees the buffer. Bindings
are mutable (SSA underneath) — assignment rebinds, it does not mutate storage.

- `shape() / dtype() / device() / numel() / rank()`
- `requires_grad() / requires_grad_(bool)` — mark a leaf as a parameter.
- `grad()` — accumulated gradient (null `Tensor` until `backward()`); `set_grad`, `accumulate_grad`.
- `backward()` — scalar-rooted reverse pass (needs `autograd.hpp` + an active `TapeScope`).
- `to_host()` — download to `std::vector<float>` (throws if Traced — reads inside jit are a graph break).
- `is_concrete() / is_traced()`, `raw_buffer() / shared_buffer() / rebind_buffer()` (optimizer-facing).
- Operators `+ - * /` (tensor and scalar), routed through the eager dispatch.

## `eager.hpp` — leaves, ops, dispatch

Leaf constructors: `from_host(vec, shape)`, `from_host_s32`, `full / zeros / ones`,
`randn`. Ops (free functions, each works in eager **and** jit):
`add, sub, mul, div_op, maximum, minimum`, `neg, exp_op, log_op, sqrt_op,
rsqrt_op, tanh_op, abs_op, logistic`, `relu, gelu`, `matmul` (batched),
`reshape, transpose, transpose_perm`, `reduce_sum, reduce_max, reduce_mean`,
`stop_gradient`, `gather_rows`, `softmax`. `set_dot_precision("HIGHEST"|"DEFAULT")`
sets the global dot precision (default `HIGHEST` for correctness).

Eager dispatch lowers one op to a 1-op executable and caches it, keyed by
`(op, input shapes/dtypes, attrs, num_replicas, precision, output shape)`; the
second identical op is a pure execute. `global_context()` is the single lazy
process-wide PJRT `Context`.

## `autograd.hpp` — tape

```cpp
TapeScope scope;                 // install a fresh tape
Tensor loss = model.forward(x);  // ops record onto the tape when grad is needed
loss.backward();                 // fills param .grad; tape resets (PyTorch semantics)
```

- `NoGrad ng;` — RAII scope; ops execute but do not tape.
- `value_and_grad(fn, params)` / `grad(fn, params)` — functional interface; `fn` is
  `() -> Tensor` (scalar), `params` is `std::vector<Tensor*>`.
- The tape reuses `Graph::grad`'s VJP rules and a memoizing evaluator that seeds
  from retained forward activations.

## `jit.hpp` — compile on demand

```cpp
auto f = jit(std::function<Tensor(Tensor)>([&](Tensor x){ return loss_of(x); }),
             /*num_replicas*/1, /*precision*/"HIGHEST");
Tensor l = f(x);   // first call traces+compiles; later calls execute the cached exe
```

- **Auto-lift**: Concrete tensors touched during a trace (Module params, captured
  constants) are lifted into the graph as Inputs by **Impl identity** — shared
  (weight-tied) tensors lift once. Buffers are re-read every call; the capture set
  must be stable across calls of the same input signature.
- `value_and_grad_jit(fn, params)` — compiles **one** executable returning
  `[loss, grad_0, grad_1, …]` (fused forward+backward); grads come back in
  `params` order. Exact-matches eager `value_and_grad` (`cpp_jit_train` test).

## `module.hpp` — layers

`Module` base: `register_parameter`, `register_module`, `parameters()` (returns
`std::vector<Tensor*>`, **de-duplicated by Impl identity** so a tied weight is
returned once), `param_count()`.

Layers: `Linear`, `ReLU`, `GELU`, `Sequential`, `MLP`, and the transformer set
`Embedding` (gather), `RMSNorm`, `Attention` (causal MHA), `Block` (pre-norm),
plus `cross_entropy(logits[N,V], targets[N])` and `softmax`. `GPT` (with
`GPTModuleConfig`) composes them: token+positional embeddings → `n_layer` blocks →
final RMSNorm → untied head; `GPT::logits(ids)` and `GPT::loss(ids, targets)`.

## `optim.hpp` — optimizers

- `SGD(params, lr, momentum=0)`, `Adam(params, lr, AdamCfg{})` — eager `step()`
  rebinds param buffers; `zero_grad()` clears `.grad`. State `m`/`v` held as Tensors.
- `JitAdamStep(fn, params, lr, AdamCfg{})` — the **fully fused** training step:
  forward + backward + Adam compiled into one XLA executable with param/`m`/`v`
  donation-aliased for in-place HBM update. `step({x,y})` returns the scalar loss
  and updates params in place. lr/bias-correction are scalar inputs, so the step
  count never forces a recompile.

## eager vs jit — performance contract

| | eager | jit (`JittedCallable` / `value_and_grad_jit` / `JitAdamStep`) |
|---|---|---|
| Lowering | one executable **per op**, cached | one executable for the **whole** traced region |
| Activations | round-trip HBM↔kernel per op | stay in HBM across the fused program |
| Optimizer step | per-op Adam arithmetic, buffer rebinds | forward+backward+Adam in one exe, params donated in place |
| Use it for | debugging, dynamic control flow, REPL-style work | the training loop / any hot path (this is the benchmarked path) |
| Numerics | — | **identical** to eager at the same precision (validated: exact loss/grad parity) |

Rule of thumb: prototype eagerly, then wrap the step in `JitAdamStep` (or the
forward in `jit`) for throughput. Because both paths build the same IR, switching
is loss-preserving — there is no "port to graph mode" step.

```cpp
// Fused training loop (the fast path)
MLP model(IN, HID, OUT, true, "gelu", &rng);
JitAdamStep step([&](std::vector<Tensor> b){
    Tensor d = sub(model.forward(b[0]), b[1]);
    return reduce_sum(mul(d, d), {0,1});
}, model.parameters(), /*lr*/1e-2);

for (int i = 0; i < steps; ++i)
    float loss = step({xb, yb}).to_host()[0];   // params updated in place
```

See `examples/cpp/train_mlp.cpp` (eager train + eager==jit parity) and the tests
`cpp_jit_train` (fused fwd/bwd + JitAdamStep) and `cpp_gpt_module` (GPT trains via
the fused step) for complete, on-device-validated examples.
