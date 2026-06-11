#!/usr/bin/env python3
"""Generate HLO protobuf files and compile options.

IMPORTANT: Must run with the system python3 (3.10) that has JAX 0.6.2,
NOT the py312 venv. The HLO unique-ID format changed in later JAX versions
and is incompatible with the installed libtpu.so.

  python3 gen_hlo.py          # output to current dir
  TPU_DATA_DIR=~/tpu_direct python3 gen_hlo.py  # for legacy tpu_compute demo

Framework examples (ex02_matmul etc.) read from $TPU_DATA_DIR or '.'.
"""

import os
import jax
import jax.numpy as jnp
from jax._src.lib import xla_client


def main():
    out_dir = os.environ.get("TPU_DATA_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    x = jnp.ones((4, 4), dtype=jnp.float32)
    y = jnp.ones((4, 4), dtype=jnp.float32)

    # Matrix multiply: f(x, y) = x @ y
    lowered = jax.jit(lambda x, y: x @ y).lower(x, y)
    hlo = lowered.compiler_ir("hlo").as_serialized_hlo_module_proto()
    p = os.path.join(out_dir, "matmul.hlo.pb")
    with open(p, "wb") as f:
        f.write(hlo)
    print(f"{p}: {len(hlo)} bytes")

    # Element-wise add: f(x) = x + x
    lowered2 = jax.jit(lambda x: x + x).lower(x)
    hlo2 = lowered2.compiler_ir("hlo").as_serialized_hlo_module_proto()
    p = os.path.join(out_dir, "add.hlo.pb")
    with open(p, "wb") as f:
        f.write(hlo2)
    print(f"{p}: {len(hlo2)} bytes")

    # Compile options (num_replicas=1, num_partitions=1)
    opts = xla_client.CompileOptions()
    opts.num_replicas = 1
    opts.num_partitions = 1
    opts_bytes = opts.SerializeAsString()
    p = os.path.join(out_dir, "compile_opts.pb")
    with open(p, "wb") as f:
        f.write(opts_bytes)
    print(f"{p}: {len(opts_bytes)} bytes")

    # Compile options for 4-replica SPMD (ex03_spmd uses this)
    ndev = jax.device_count()
    opts4 = xla_client.CompileOptions()
    opts4.num_replicas = ndev
    opts4.num_partitions = 1
    opts4_bytes = opts4.SerializeAsString()
    p = os.path.join(out_dir, "compile_opts_n4.pb")
    with open(p, "wb") as f:
        f.write(opts4_bytes)
    print(f"{p}: {len(opts4_bytes)} bytes  (num_replicas={ndev})")

    # Save human-readable MLIR (always in the repo dir for reference)
    mlir = str(lowered.compiler_ir("stablehlo"))
    with open("matmul.mlir", "w") as f:
        f.write(mlir)
    print(f"matmul.mlir: {len(mlir)} bytes")


def dump_opts(num_replicas, num_partitions, use_spmd, path):
    """Oracle for the native C++ CompileOptions encoder (cpp/compile_opts.hpp):
    dump a JAX-generated blob for an arbitrary config to byte/semantic-compare."""
    opts = xla_client.CompileOptions()
    opts.num_replicas = num_replicas
    opts.num_partitions = num_partitions
    if use_spmd:
        opts.executable_build_options.use_spmd_partitioning = True
    b = opts.SerializeAsString()
    with open(path, "wb") as f:
        f.write(b)
    print(f"{path}: {len(b)} bytes (r={num_replicas} p={num_partitions} spmd={use_spmd})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--dump-opts":
        # gen_hlo.py --dump-opts <replicas> <partitions> <spmd 0|1> <out.pb>
        dump_opts(int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4])), sys.argv[5])
    else:
        main()
