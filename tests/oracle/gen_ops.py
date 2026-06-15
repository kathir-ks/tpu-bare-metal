"""JAX oracle driver: collect CASES from every cases_*.py family module and emit
golden forward + gradient fixtures.

Run with the JAX venv active (see CLAUDE.md):
    source ~/venv-maxtext-py312/bin/activate
    python tests/oracle/gen_ops.py        # or: make oracle

Each family module (cases_unary.py, cases_binary.py, ...) owns a disjoint set of
ops so multiple authors can extend coverage without colliding. A CASE is one
fully-specified op instance: key <-> C++ builder in tests/cpp/op_table_<family>.hpp.

Fixtures hold: in0.., out, grad_in0.. (d sum(out)/d in_k for float inputs, when
diff != False), and the op's HIGHEST/DEFAULT tolerances. Forward parity uses `out`;
the generic gradient suite uses the grads.
"""
import os
import importlib
import numpy as np
import jax
import jax.numpy as jnp

from fixture import FixtureWriter
from tolerances import tol_for, matmul_default_tol
from oracle_util import grad_of_sum

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")  # oracle is the f32 truth

HERE = os.path.dirname(os.path.abspath(__file__))
FIXDIR = os.path.join(HERE, "fixtures")

# Family modules contributing CASES. Agents add their family here once.
FAMILIES = [
    "cases_unary",
    "cases_binary",
    "cases_reduction",
    "cases_matmul",
    "cases_shape",
    "cases_gather",
    "cases_misc",
]


def collect_cases():
    cases = {}
    for fam in FAMILIES:
        mod = importlib.import_module(fam)
        for key, spec in mod.CASES.items():
            if key in cases:
                raise ValueError(f"duplicate case key {key!r} (in {fam} and earlier)")
            spec = dict(spec)
            spec["_family"] = fam
            cases[key] = spec
    return cases


def main():
    os.makedirs(FIXDIR, exist_ok=True)
    cases = collect_cases()
    n = 0
    for key, spec in sorted(cases.items()):
        cat = spec["category"]
        raw = spec["inputs"]
        diff = spec.get("diff", True)
        ins = [jnp.asarray(v) for (_, v) in raw]
        float_idx = [k for k, (dt, _) in enumerate(raw) if dt == "f32"]

        out = np.asarray(spec["fn"](*ins), dtype=np.float32)
        hi, df = tol_for(cat)
        if cat == "matmul":
            # DEFAULT tolerance scales with contraction depth K (= last axis of in0).
            k = int(raw[0][1].shape[-1])
            df = matmul_default_tol(k)

        w = FixtureWriter()
        for k, (dt, v) in enumerate(raw):
            w.add(f"in{k}", v.astype(np.int32 if dt == "i32" else np.float32))
        w.add("out", out)
        if diff and float_idx:
            grads = grad_of_sum(spec["fn"], ins, float_idx)
            for j, k in enumerate(float_idx):
                w.add(f"grad_in{k}", np.asarray(grads[j], dtype=np.float32))
        w.add_scalar("__tol_highest__", hi)
        w.add_scalar("__tol_default__", df)
        w.save(os.path.join(FIXDIR, key + ".fix"))
        n += 1
        print(f"  {spec['_family']:16s} {key:28s} out{list(out.shape)} "
              f"tol(hi={hi},df={df}) {'grad' if diff and float_idx else 'no-grad'}")
    print(f"oracle: wrote {n} fixtures from {len(FAMILIES)} families to {FIXDIR}")


if __name__ == "__main__":
    main()
