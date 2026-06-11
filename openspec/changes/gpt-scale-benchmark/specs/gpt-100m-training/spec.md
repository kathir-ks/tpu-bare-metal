# gpt-100m-training

## ADDED Requirements

### Requirement: 100M-parameter GPT trains on TPU v4
The C++ stack SHALL train a GPT-2-small-class model (~124M params: 12 layers, d_model 768, 12 heads, vocab ~32k, context ≥512) data-parallel across all 4 TPU v4 chips.

#### Scenario: Stable training run
- **WHEN** the 100M model trains for a fixed benchmark run (≥200 steps) on the prepared token dataset
- **THEN** training completes without OOM and loss decreases monotonically over the run (smoothed)

#### Scenario: Checkpoint round-trip at scale
- **WHEN** training is checkpointed and restored
- **THEN** the restored model resumes with loss equal to the pre-save loss within tolerance

### Requirement: Offline-tokenized dataset
Training SHALL consume a pre-tokenized binary token file produced offline, with no tokenizer dependency in the C++ runtime.

#### Scenario: Data loading
- **WHEN** the trainer starts with the token file path
- **THEN** batches are sampled and fed to the device without any Python at runtime

### Requirement: Gradient accumulation when needed
If the target effective batch does not fit in HBM, the trainer SHALL support gradient accumulation to reach it.

#### Scenario: Accumulated step equivalence
- **WHEN** an effective batch B is run as k accumulation micro-steps of B/k
- **THEN** the resulting parameter update matches a single batch-B step within numerical tolerance
