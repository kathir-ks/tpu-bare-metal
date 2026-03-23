#!/usr/bin/env python3
"""Generate HLO protobuf files and compile options for tpu_compute.c"""

import jax
import jax.numpy as jnp
from jax._src.lib import xla_client


def main():
    x = jnp.ones((4, 4), dtype=jnp.float32)
    y = jnp.ones((4, 4), dtype=jnp.float32)

    # Matrix multiply: f(x, y) = x @ y
    lowered = jax.jit(lambda x, y: x @ y).lower(x, y)
    hlo = lowered.compiler_ir("hlo").as_serialized_hlo_module_proto()
    with open("matmul.hlo.pb", "wb") as f:
        f.write(hlo)
    print(f"matmul.hlo.pb: {len(hlo)} bytes")

    # Element-wise add: f(x) = x + x
    lowered2 = jax.jit(lambda x: x + x).lower(x)
    hlo2 = lowered2.compiler_ir("hlo").as_serialized_hlo_module_proto()
    with open("add.hlo.pb", "wb") as f:
        f.write(hlo2)
    print(f"add.hlo.pb: {len(hlo2)} bytes")

    # Compile options (num_replicas=1, num_partitions=1)
    opts = xla_client.CompileOptions()
    opts.num_replicas = 1
    opts.num_partitions = 1
    opts_bytes = opts.SerializeAsString()
    with open("compile_opts.pb", "wb") as f:
        f.write(opts_bytes)
    print(f"compile_opts.pb: {len(opts_bytes)} bytes")

    # Save human-readable MLIR
    mlir = str(lowered.compiler_ir("stablehlo"))
    with open("matmul.mlir", "w") as f:
        f.write(mlir)
    print(f"matmul.mlir: {len(mlir)} bytes")


if __name__ == "__main__":
    main()
