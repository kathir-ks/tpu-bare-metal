"""Oracle cases — row gather/scatter (graph Ops: Gather via gather_rows, ScatterAdd).
gather_rows(table[V,D], ids) -> ids.shape + [D]. Gradient flows to `table` only
(ids are integer). Duplicate ids must accumulate in the scatter-add VJP.
"""
import jax.numpy as jnp
from oracle_util import randf, randint, f, i

CASES = {
    "gather_rows__5x4_ids3": dict(
        category="gather",
        # in0 = table[5,4] (float, differentiable), in1 = ids[3] (int)
        inputs=[f(randf("gat", 0, 5, 4)), i(randint("gat", 1, (3,), high=5))],
        fn=lambda table, ids: table[ids]),
    "gather_rows_dupes__5x4_ids4": dict(
        category="gather",
        inputs=[f(randf("gatd", 0, 5, 4)), i(jnp.array([1, 1, 3, 1], dtype=jnp.int32))],
        fn=lambda table, ids: table[ids]),
    # rank-2 ids: ids [2,3] into table [5,4] -> out [2,3,4]
    "gather_rows__rank2ids__5x4_ids2x3": dict(
        category="gather",
        inputs=[f(randf("gatr2", 0, 5, 4)), i(randint("gatr2", 1, (2, 3), high=5))],
        fn=lambda table, ids: table[ids]),
    # larger: ids [6] into table [10,8]
    "gather_rows__larger__10x8_ids6": dict(
        category="gather",
        inputs=[f(randf("gatl", 0, 10, 8)), i(randint("gatl", 1, (6,), high=10))],
        fn=lambda table, ids: table[ids]),
    # rank-3 table: table [5,2,3], ids [4] -> out [4,2,3]
    "gather_rows__rank3table__5x2x3_ids4": dict(
        category="gather",
        inputs=[f(randf("gat3d", 0, 5, 2, 3)), i(randint("gat3d", 1, (4,), high=5))],
        fn=lambda table, ids: table[ids]),
    # duplicate ids that accumulate in the VJP: ids [2,2,0,2] into [5,4]
    "gather_rows__dupes_accum__5x4_ids4": dict(
        category="gather",
        inputs=[f(randf("gatda", 0, 5, 4)), i(jnp.array([2, 2, 0, 2], dtype=jnp.int32))],
        fn=lambda table, ids: table[ids]),
}
