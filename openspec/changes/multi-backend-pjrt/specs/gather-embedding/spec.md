# gather-embedding

## ADDED Requirements

### Requirement: Row gather op
The graph builder SHALL provide `gather_rows(table, ids)` where `table` is `[V, D…]` and `ids` is an integer tensor of any shape, producing `ids.shape + [D…]`, emitted as `stablehlo.gather` with scalar indices (`index_vector_dim = rank(ids)`).

#### Scenario: Embedding lookup
- **GIVEN** table `[V, D]` and ids `[B, T]` (i32)
- **WHEN** `gather_rows(table, ids)` is emitted and run
- **THEN** output `[B, T, D]` equals exact row lookup (max error 0)

### Requirement: Scatter-add adjoint
The graph builder SHALL provide `scatter_add_rows(operand, ids, updates)` emitted as `stablehlo.scatter` with an add region and `unique_indices = false`, and `grad()` SHALL differentiate `gather_rows` into a scatter-add of the upstream gradient into a zero table.

#### Scenario: Duplicate indices accumulate
- **GIVEN** ids containing the same row v three times
- **WHEN** the gradient of `sum(gather_rows(table, ids) * coef)` w.r.t. `table` is computed on device
- **THEN** `d_table[v]` equals the sum of the three corresponding `coef` rows (exact)

### Requirement: Embedding via gather
`nn::embedding` SHALL use `gather_rows`, making embedding memory O(N·dim) and independent of vocabulary size. The legacy one-hot implementation SHALL remain available as `nn::embedding_onehot` for cross-checking.

#### Scenario: Training parity
- **GIVEN** the 98M-param benchmark config
- **WHEN** trained with gather-based embedding
- **THEN** the loss trajectory matches the one-hot baseline (identical first-step loss 9.5171) and all regression tests pass
