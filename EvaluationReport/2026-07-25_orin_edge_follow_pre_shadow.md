# Orin Edge Follow Pre-Shadow Verification

Date: 2026-07-25  
Repository baseline: `1c4ed926e4b78cbf7779d1659cbdf9de2b418b23`  
Scope: Orin-side Follow only  
Evidence class: Orin offline verification; no real STM32 Shadow run yet

## Result

The Edge Follow software and current Orin environment are ready for a first
supervised real-machine `mode=shadow` run. Edge control remains prohibited.

Post-power-loss preflight on 2026-07-26 confirmed:

- `/dev/ttyTHS1` is the STM32 port and continuously carries 115200-baud CSV;
- `/dev/ttyTHS2` produced no data during a three-second passive probe;
- the `jetson16` user is now a member of `dialout`;
- `/dev/ttyTHS1` is readable and writable by the runtime user;
- no other `orin_state_sender.py` or serial owner was present;
- Orin `192.168.0.55` reached PC `192.168.0.220` with zero loss in a
  three-packet ping check.

The default `/dev/ttyTHS0` still does not exist on this Orin, so the first
Shadow command must explicitly pass `--serial-port /dev/ttyTHS1`. Do not change
the repository default based only on this one device.

## Correctness changes

- Execution waypoint tolerance now comes only from
  `excavation_cycle.json limits.waypoint_tolerance_m`.
- The deployed value is `0.25 m`.
- `trajectory_command.json target_threshold=0.03 m` remains separate planner
  metadata and is not used by `WaypointTracker.advance()`.
- Follow progress is computed from caller-provided monotonic time:

  ```text
  clamp((now - follow_started_monotonic) / tracking_timeout_s, 0, 1)
  ```

- The deployed tracking timeout comes only from
  `excavation_cycle.json limits.tracking_timeout_s=60.0`.
- First valid state starts a new Follow at progress zero.
- Monotonic time regression is rejected fail-safe.
- TIMEOUT and COMPLETED are sticky terminal results.
- TIMEOUT produces normalized and physical zero actions.
- Edge control treats TIMEOUT as an immediate zero command and never reuses the
  previous nonzero action.
- Nonzero Mission waypoint dwell is explicitly rejected at startup because this
  migration slice does not implement dwell behavior.

## Transient-fault behavior

- A sequence gap is accepted, so missing intermediate frames do not terminate
  Follow.
- Duplicate or older sequences are rejected for that frame.
- Invalid sensor state, E-stop, STM32 loss, runtime error or time regression is
  fail-safe for that frame.
- In control, a rejected frame sends zero immediately.
- A later valid state may resume an ACTIVE Follow.
- Rejections record exception type and consecutive rejection count.
- A terminal TIMEOUT or COMPLETED state cannot resume.
- Action Relay lease expiry, shutdown and Ctrl+C retain terminal zero semantics.

Robustness means process continuity, not blind motion continuity: old nonzero
commands are never held through invalid or missing state.

## Deployment provenance

Startup now rejects:

- a Mission ID mismatch;
- a Mission SHA-256 mismatch;
- a Mission phase/task-mode mismatch;
- a trajectory not marked `planning_scope=execution_strict`;
- a trajectory not marked `execution_eligible=true`.

Verified asset SHA-256:

```text
baae58034e14d923a1b28318c123500ab5e395a385284f50da068311275ce85a  machine_profile.json
c1dffdac1e460682c4d6c45187e0fbb6e5a8323a189ddbc4fd66e0696919ef62  waji.urdf
df90bd5a64b136043bf9731b834b8b5b0417205ae1f3739b876d53e188fe5884  policy.onnx
71a69e3c4ab584a091be4f5926a0dd4da21f49200f8e2d902dda85940eaeb292  trajectory_command.json
c0259aecb13e369937fba265d09de6e7d3c2d86c79480353070b141afd1e2ee2  excavation_cycle.json
```

## Runtime environment

```text
architecture: aarch64
L4T: R39.2
Python: 3.10.20
NumPy: 1.26.4
ONNX Runtime: 1.23.2
available providers: AzureExecutionProvider, CPUExecutionProvider
active session provider: CPUExecutionProvider
```

No `onnxruntime-gpu` package was installed or enabled.

## Offline performance

Pure ONNX inference, 50 warmups and 1000 measured calls:

```text
min:    0.082114 ms
mean:   0.085496 ms
median: 0.084483 ms
p95:    0.091126 ms
p99:    0.107949 ms
max:    0.153733 ms
```

An additional 605-frame, 10 Hz simulated Shadow sequence used all five real
deployment assets and injected one invalid sensor frame and one duplicate
sequence:

```text
ACTIVE: 598
rejected: 2
TIMEOUT: 5
progress monotonic violations: 0
max consecutive rejections: 1
final physical action zero: true

pure ONNX:
  count: 598
  mean: 0.082436 ms
  p95: 0.099348 ms
  p99: 0.124188 ms
  max: 0.652827 ms

full FK + observation + ONNX + audit loop:
  mean: 0.859654 ms
  p95: 0.913887 ms
  p99: 0.995528 ms
  max: 1.990193 ms
```

The 10 Hz input and loop rates above were simulated and are not real STM32
measurements.

## Passive real STM32 state preflight

A ten-second passive read used the same `open_serial()`,
`parse_stm32_csv_line()` and `build_machine_state_packet()` path as the runtime:

```text
port: /dev/ttyTHS1
accepted frames: 81
rejected frames: 0
sensor-invalid packets: 0
STM32-dead packets: 0
STM32 timestamp regressions: 0
STM32 interval: min 100 ms, mean 125 ms, p95 200 ms, max 200 ms
host interval: min 98.611 ms, mean 124.997 ms, p95 201.184 ms, max 201.609 ms
```

The observed input rate is about 8 Hz rather than the expected nominal 10 Hz.
The maximum observed 202 ms interval remains below the 300 ms state timeout,
leaving about 98 ms of measured margin. A real Shadow run of at least two
minutes is required to measure the tail distribution before any control
decision.

Current CSV rows contain three additional trailing fields whose meaning is not
defined by the local protocol documents. They were consistently `1,1,1` during
preflight. The deployed parser intentionally consumes the documented first 16
fields. The trailing-field meaning must be confirmed before edge control.

## Verification commands

```bash
python -W error::ResourceWarning -m unittest discover -s tests -v
python -m py_compile orin_state_sender.py orin_csv_replay.py edge_runtime/*.py
git diff --check
```

Result: 50 tests passed; compile and diff checks passed.

Shadow logs can be summarized with:

```bash
python -m edge_runtime.audit deploy/logs/edge_runtime.jsonl
```

## Real Shadow entry conditions

Before starting:

1. Use the confirmed STM32 port `/dev/ttyTHS1`.
2. Confirm the `jetson16` process still has read/write permission for it.
3. `pgrep -af orin_state_sender.py` shows no existing serial owner.
4. `deploy/edge_runtime.json` remains `mode=shadow`.
5. The command does not include `--control-enabled` or
   `--edge-motion-authorization`.
6. The machine area is supervised even though Shadow has no ONNX action sink.

Existing project logs predate Orin real-machine testing and must not be used as
Orin Shadow evidence.
