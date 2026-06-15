"""JAX oracle driver for NN-layer conformance fixtures (cases_layers.py LAYERS).

    source ~/venv-maxtext-py312/bin/activate
    JAX_PLATFORMS=cpu python tests/oracle/gen_layers.py   # or: make oracle

Writes one fixture per layer to fixtures_layers/. Each holds: x, every param (by
name), out, grad_x (loss = sum(out) unless scalar_out), and grad_<param> for each
param. The C++ suite (test_oracle_layers.cpp) builds the same layer via nn.hpp with
a resolver that feeds these exact params, and asserts forward+backward parity.
"""
import os
import importlib
import numpy as np
import jax
import jax.numpy as jnp

from fixture import FixtureWriter
from oracle_util import randf, randint

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
FIXDIR = os.path.join(HERE, "fixtures_layers")

# Layer-parity tolerances (HIGHEST, DEFAULT). Layers compose matmuls + transcendentals,
# so DEFAULT (bf16) is looser; HIGHEST stays tight.
TOL_HI, TOL_DF = 2e-3, 6e-2

# Layer modules contributing LAYERS dicts. Agents add their module here once.
LAYER_MODULES = [
    "layers_core",       # gelu, linear, rmsnorm
    "layers_embedding",  # embedding (gather)
    "layers_softmax",    # softmax
    "layers_attention",  # causal multi-head attention
    "layers_block",      # transformer block (composite)
    "layers_ce",         # cross_entropy (scalar out, int targets)
    "layers_gpt",        # composite GPT (whole model)
]


def collect_layers():
    layers = {}
    for name in LAYER_MODULES:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError:
            continue  # module not authored yet
        for key, spec in mod.LAYERS.items():
            if key in layers:
                raise ValueError(f"duplicate layer key {key!r} (module {name})")
            layers[key] = spec
    return layers


def _draw_input(key, spec_in):
    name, shape = spec_in["name"], spec_in["shape"]
    if spec_in.get("int", False):
        return randint(key, name, tuple(shape), high=spec_in["high"])
    return randf(key, name, *shape)


def main():
    os.makedirs(FIXDIR, exist_ok=True)
    all_layers = collect_layers()
    n = 0
    for key, spec in sorted(all_layers.items()):
        ins_spec = spec["inputs"]
        params = spec.get("params", {})
        scalar_out = spec.get("scalar_out", False)

        in_names = [s["name"] for s in ins_spec]
        in_is_float = [not s.get("int", False) for s in ins_spec]
        ivals = {s["name"]: _draw_input(key, s) for s in ins_spec}
        pnames = list(params.keys())
        pvals = {nm: randf(key, nm, *params[nm]) for nm in pnames}

        # Differentiable arguments = float inputs followed by params, in that order.
        diff_names = [nm for nm, isf in zip(in_names, in_is_float) if isf] + pnames
        diff_vals = [ivals[nm] for nm, isf in zip(in_names, in_is_float) if isf] + [pvals[nm] for nm in pnames]

        def assemble(diff_args):
            di = {diff_names[i]: diff_args[i] for i in range(len(diff_names))}
            ins = {nm: (di[nm] if nm in di else jnp.asarray(ivals[nm])) for nm in in_names}
            p = {nm: di[nm] for nm in pnames}
            return spec["fn"](ins, p)

        out = np.asarray(assemble([jnp.asarray(v) for v in diff_vals]), dtype=np.float32)

        def loss(*diff_args):
            o = assemble(list(diff_args))
            return o if scalar_out else jnp.sum(o)

        grads = jax.grad(loss, argnums=tuple(range(len(diff_vals))))(*[jnp.asarray(v) for v in diff_vals])
        grad_by_name = {diff_names[i]: np.asarray(grads[i], dtype=np.float32) for i in range(len(diff_names))}

        w = FixtureWriter()
        for nm in in_names:
            w.add(nm, ivals[nm])
        for nm in pnames:
            w.add(nm, pvals[nm])
        w.add("out", out)
        for nm in diff_names:
            w.add("grad_" + nm, grad_by_name[nm])
        w.add_scalar("__tol_highest__", TOL_HI)
        w.add_scalar("__tol_default__", TOL_DF)
        # Gradient tolerance: atomic layers earn the tight forward bound; composite
        # layers (block/gpt) accumulate TPU-HIGHEST-vs-f32 rounding (~1e-6 relative
        # per matmul) through a deep backward, amplified by rmsnorm 1/rms + softmax.
        # Default to the project's established gradcheck envelope (1e-2) for composites.
        w.add_scalar("__tol_grad__", spec.get("grad_tol", TOL_HI))
        w.save(os.path.join(FIXDIR, key + ".fix"))
        n += 1
        print(f"  layer {key:28s} inputs={in_names} params={pnames} out{list(out.shape)}{' [scalar]' if scalar_out else ''}")
    print(f"oracle: wrote {n} layer fixtures to {FIXDIR}")


if __name__ == "__main__":
    main()
