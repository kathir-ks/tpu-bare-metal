CC      = gcc
AR      = ar
CFLAGS  = -Wall -O2
LDFLAGS = -ldl -lm

# ── Framework library ──────────────────────────────────────────────────────────
FWK_CFLAGS = $(CFLAGS) -I./framework
FWK_SRCS   = framework/tpu.c framework/tpu_spmd.c framework/tpu_serial.c
FWK_OBJS   = $(FWK_SRCS:.c=.o)
FWK_LIB    = framework/libtpu_fw.a

$(FWK_LIB): $(FWK_OBJS)
	$(AR) rcs $@ $^

framework/%.o: framework/%.c framework/tpu.h framework/tpu_pjrt.h
	$(CC) $(FWK_CFLAGS) -c -o $@ $<

# ── Original demos (unchanged) ─────────────────────────────────────────────────
tpu_pjrt_test: tpu_pjrt_test.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

tpu_compute: tpu_compute.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ── Example programs ───────────────────────────────────────────────────────────
EXAMPLES = ex01_roundtrip ex02_matmul ex03_spmd ex04_dev2dev \
           ex05_async ex06_serialize ex07_shard ex08_topology

$(EXAMPLES): %: examples/%.c $(FWK_LIB)
	$(CC) $(FWK_CFLAGS) -o $@ $< -L./framework -ltpu_fw $(LDFLAGS)

# ── Pure C++ stack (cpp/, examples/cpp/, tests/cpp/) ────────────────────────────
CXX        = g++
CXXFLAGS   = -std=c++17 -O2 -Wall -I./framework -I./cpp
CPP_OBJ    = cpp/graph.o
CPP_HDRS   = cpp/tpu.hpp cpp/graph.hpp cpp/nn.hpp cpp/gpt.hpp cpp/default_opts.h \
             cpp/proto_writer.hpp cpp/compile_opts.hpp cpp/debug_opts_blob.h \
             cpp/tensor.hpp cpp/eager.hpp cpp/autograd.hpp cpp/jit.hpp cpp/module.hpp cpp/optim.hpp \
             cpp/fixture.hpp

cpp/graph.o: cpp/graph.cpp cpp/graph.hpp cpp/tpu.hpp cpp/default_opts.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

CPP_TESTS    = cpp_gradcheck cpp_train_tiny cpp_gpt_smoke cpp_dp cpp_ckpt cpp_compile_opts cpp_donation cpp_gather cpp_plugin_probe cpp_eager cpp_autograd cpp_jit cpp_module cpp_optim cpp_jit_train cpp_gpt_module cpp_oracle_ops cpp_oracle_grad cpp_oracle_layers cpp_robust
CPP_EXAMPLES = cpp_train_gpt cpp_train_gpt_dp cpp_train_mlp cpp_train_gpt_frontend

cpp_gradcheck:  tests/cpp/test_gradcheck.cpp  $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_train_tiny: tests/cpp/test_train_tiny.cpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_gpt_smoke:  tests/cpp/test_gpt_smoke.cpp   $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_dp:         tests/cpp/test_dp.cpp          $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_ckpt:       tests/cpp/test_ckpt.cpp        $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_compile_opts: tests/cpp/test_compile_opts.cpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_donation:   tests/cpp/test_donation.cpp     $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_gather:     tests/cpp/test_gather.cpp       $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_plugin_probe: tests/cpp/test_plugin_probe.cpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
OP_TABLE_HDRS = tests/cpp/op_table.hpp tests/cpp/op_table_base.hpp \
                tests/cpp/op_table_unary.hpp tests/cpp/op_table_binary.hpp \
                tests/cpp/op_table_reduction.hpp tests/cpp/op_table_matmul.hpp \
                tests/cpp/op_table_shape.hpp tests/cpp/op_table_gather.hpp \
                tests/cpp/op_table_misc.hpp
cpp_oracle_ops: tests/cpp/test_oracle_ops.cpp $(OP_TABLE_HDRS) $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_oracle_grad: tests/cpp/test_oracle_grad.cpp $(OP_TABLE_HDRS) $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
LAYER_TABLE_HDRS = $(wildcard tests/cpp/layer_table*.hpp)
cpp_robust: tests/cpp/test_robust.cpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_oracle_layers: tests/cpp/test_oracle_layers.cpp $(LAYER_TABLE_HDRS) $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_eager:      tests/cpp/test_eager.cpp       $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
# ── Wave-B frontend layers (added to CPP_TESTS at integration once all exist) ──
cpp_autograd:   tests/cpp/test_autograd_eager.cpp cpp/autograd.hpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_jit:        tests/cpp/test_jit.cpp         cpp/jit.hpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_module:     tests/cpp/test_module.cpp      cpp/module.hpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_optim:      tests/cpp/test_optim.cpp       cpp/optim.hpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_jit_train:  tests/cpp/test_jit_train.cpp   cpp/optim.hpp cpp/jit.hpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_gpt_module: tests/cpp/test_gpt_module.cpp  cpp/module.hpp cpp/optim.hpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_train_gpt:  examples/cpp/train_gpt.cpp     $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_train_gpt_dp: examples/cpp/train_gpt_dp.cpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_train_mlp:  examples/cpp/train_mlp.cpp     $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
cpp_train_gpt_frontend: examples/cpp/train_gpt_frontend.cpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)

cpp: $(CPP_TESTS) $(CPP_EXAMPLES)

# ── Benchmarks (cpp vs JAX parity; see bench/) ─────────────────────────────────
bench_matmul: bench/bench_matmul.cpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)
bench_gpt: bench/bench_gpt.cpp $(CPP_OBJ) $(CPP_HDRS) $(FWK_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< $(CPP_OBJ) -L./framework -ltpu_fw $(LDFLAGS)

bench-cpp: bench_gpt
	./bench_gpt 200 8

# needs ~/venv-maxtext-py312 activated
bench-jax:
	python3 bench/jax_baseline.py --steps 200 --batch-per-replica 8

# ── Verification harness (capability: verification-harness) ─────────────────────
# Regenerate golden vectors from the JAX oracle (offline tooling; needs the venv).
# Runs from tests/oracle so `import fixture` / `tolerances` resolve.
oracle:
	cd tests/oracle && JAX_PLATFORMS=cpu python gen_ops.py
	cd tests/oracle && JAX_PLATFORMS=cpu python gen_layers.py

# Strict acceptance gate: build + run the verification suites, exit non-zero on any
# failure. On-device steps use a single TPU chip serially; check for an existing
# /dev/accel* holder before running (CLAUDE.md) — this target does not preempt.
VERIFY_BINS = cpp_oracle_ops cpp_oracle_grad cpp_oracle_layers cpp_robust
verify: $(VERIFY_BINS)
	@echo "── verification: oracle op forward parity ─────────────────────"
	@./cpp_oracle_ops
	@echo "── verification: oracle gradient parity ───────────────────────"
	@./cpp_oracle_grad
	@echo "── verification: oracle layer parity ──────────────────────────"
	@./cpp_oracle_layers
	@echo "── verification: compiler robustness (negative tests) ─────────"
	@./cpp_robust

# ── HLO generation ─────────────────────────────────────────────────────────────
hlo: gen_hlo.py
	python3 gen_hlo.py

# ── Phony targets ──────────────────────────────────────────────────────────────
all: tpu_pjrt_test tpu_compute $(FWK_LIB) $(EXAMPLES) cpp

lib: $(FWK_LIB)

examples: $(EXAMPLES)

clean:
	rm -f tpu_pjrt_test tpu_compute $(EXAMPLES)
	rm -f framework/*.o $(FWK_LIB)
	rm -f cpp/*.o $(CPP_TESTS) $(CPP_EXAMPLES) cpp_eager

.PHONY: all lib examples cpp hlo oracle verify clean
