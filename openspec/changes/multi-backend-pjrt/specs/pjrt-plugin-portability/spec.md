# pjrt-plugin-portability

## ADDED Requirements

### Requirement: Plugin selection
The framework SHALL load any PJRT C API plugin chosen by, in order of precedence: explicit `tpu_init(path)` argument, `PJRT_PLUGIN_PATH` environment variable, `LIBTPU_PATH` environment variable (legacy), built-in libtpu default path.

#### Scenario: Non-TPU plugin via env var
- **GIVEN** `PJRT_PLUGIN_PATH` points at a conforming plugin (e.g. `xla_cuda_plugin.so`)
- **WHEN** `tpu_init(NULL)` is called
- **THEN** the plugin is dlopened and driven through `GetPjrtApi` → `Plugin_Initialize` → `Client_Create` with no libtpu-specific assumptions

### Requirement: API version negotiation
The framework SHALL read the plugin's PJRT API version from the `PJRT_Api` header and SHALL fail `tpu_init` with a descriptive message if the plugin's function table does not cover the highest index the framework calls.

#### Scenario: Too-old plugin
- **GIVEN** a plugin whose `PJRT_Api.struct_size` is smaller than required for `FN_BUF_TOHOST`
- **WHEN** `tpu_init` is called
- **THEN** it returns NULL and `tpu_strerror(NULL)` names the plugin, its version, and the size shortfall

#### Scenario: Version accessor
- **GIVEN** an initialized context
- **WHEN** `tpu_api_version(ctx, &major, &minor)` is called
- **THEN** it reports the loaded plugin's PJRT API version

### Requirement: Per-plugin failure transparency
When a plugin loads but cannot serve (e.g. no matching hardware), the framework SHALL surface the plugin's own error message through `tpu_strerror` rather than crashing or masking it.

#### Scenario: GPU plugin on a GPU-less host
- **GIVEN** `xla_cuda_plugin.so` on a machine without CUDA devices
- **WHEN** `tpu_init` reaches `Client_Create`
- **THEN** init fails cleanly and the error contains the plugin's message (e.g. "No visible GPU devices")

### Requirement: Backend smoke test
The repo SHALL provide a probe binary (`cpp_plugin_probe`) that, for a given plugin, reports API version and device counts and verifies one compile+execute through the full C++ stack.

#### Scenario: TPU probe
- **WHEN** `./cpp_plugin_probe` runs against libtpu
- **THEN** it prints `PJRT API v0.69 devices=4 addressable=4` (on the v4-8) and `compile+run x*x+1: PASS`
