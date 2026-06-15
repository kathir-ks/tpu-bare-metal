## ADDED Requirements

### Requirement: Per-op forward parity against the oracle
Every graph op SHALL produce a forward output matching the reference-oracle golden output within the op's declared tolerance, verified at both the HIGHEST and DEFAULT precision policies (each with its own documented tolerance).

#### Scenario: Op forward at HIGHEST
- **WHEN** an op is executed on-device under the HIGHEST precision policy on the oracle's fixed-seed input
- **THEN** the output matches the golden forward fixture within the op's HIGHEST tolerance

#### Scenario: Op forward at DEFAULT
- **WHEN** an op is executed on-device under the DEFAULT (bf16) precision policy on the oracle's fixed-seed input
- **THEN** the output matches the golden forward fixture within the op's looser, documented DEFAULT tolerance

### Requirement: Per-layer forward and backward parity against the oracle
Every NN layer (`linear`, `embedding`, `gelu`, `rmsnorm`, `softmax`, `attention`, `block`, `cross_entropy`, gather/scatter-add) SHALL produce forward outputs and parameter/input gradients matching the oracle within the layer's declared tolerance.

#### Scenario: Layer forward+backward parity
- **WHEN** a layer is run forward and backward on the oracle's fixed-seed inputs
- **THEN** both its forward output and all its gradients match the golden fixtures within the layer's declared tolerance

### Requirement: Composite-model parity
A set of small composite models (at minimum a multi-layer GPT block stack and the cross-entropy training loss) SHALL match the oracle's forward loss and parameter gradients within tolerance, confirming layers compose correctly.

#### Scenario: GPT loss parity
- **WHEN** the composite GPT model computes loss and gradients on a fixed-seed batch
- **THEN** the loss and parameter gradients match the oracle fixtures within the model's declared tolerance

### Requirement: Conformance report prints measured error
The conformance suite SHALL print, for each checked op/layer/model, the measured maximum error alongside the declared tolerance, so that available tolerance margin is visible.

#### Scenario: Reporting
- **WHEN** the conformance suite completes
- **THEN** its output includes a row per checked item showing measured error and the threshold, and a PASS/FAIL verdict
