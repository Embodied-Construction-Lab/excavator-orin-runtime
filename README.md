# Excavator Orin Runtime

Jetson Orin runtime for the scale hydraulic excavator. It supports two execution
topologies:

```text
PC-controlled (existing)                  Orin edge Follow
PC policy_action                          STM32 state
      ↓ UDP                                    ↓ serial
Action Relay                            FK → 38D → ONNX
      ↓ serial                                 ↓ physical velocity
    STM32                              loopback Action Relay
                                                  ↓ serial
                                                STM32
```

In both topologies `orin_state_sender.py` remains the only serial owner. Edge
control binds the Action Relay to `127.0.0.1`, so a PC network packet cannot
compete with the local inference source.

The authoritative action order is:

```text
[boom, stick, bucket, swing]
```

The action array sent to STM32 contains physical velocity references:

- boom, stick and bucket: `m/s`
- swing: `rad/s`

The compatibility protocol currently retains
`action_type="normalized_velocity_command"`; the values must not be normalized again.

The active unified STM32 wire command is newline-delimited JSON with
`schema_version="stm32_velocity_command.v1"`. The Action Relay resumes the
STM32 command sequence from `stm32_control_telemetry.v2` and sends a velocity
zero before the first nonzero command, so changing from manual/ACT to RL does
not require a firmware flash.

## Files

- `orin_state_sender.py`: reads STM32 state, publishes `machine_state_v1`, validates action structure/timing/safety state and relays finite physical velocity commands to STM32 without magnitude checks.
- `orin_csv_replay.py`: validates and replays an exported physical-velocity CSV through the local Action Relay.
- `edge_runtime/`: dependency-light URDF FK, Unity-compatible 38D observation,
  ONNX inference, waypoint tracking, normalized-to-physical conversion, shadow
  auditing, loopback edge control and remote Follow RPC.
- `deploy/edge_runtime.example.json`: static Shadow/control deployment configuration.
- `deploy/edge_runtime.remote.example.json`: remote Follow server configuration;
  it intentionally contains no static trajectory.
- `tests/`: host-side protocol, relay, timeout, ordering and replay tests.

Historical joystick and `[swing, boom, stick, bucket]` rollout tools are intentionally excluded.
The old workspace-root `urdf/` project is also excluded. The deployed URDF is
copied from `AiryLidar/kinematics/waji_description/urdf/waji.urdf`.

## Current field RL Follow command

The example field setup uses PC `192.168.0.220`, Orin `192.168.0.55`, STM32
serial `/dev/ttyTHS1` at `460800` and behavior RPC TCP `18083`. After the
explicit start-pose/preposition check, start the Orin side first:

```bash
cd /home/jetson16/workspace_excavator/excavator-orin-runtime
conda activate excavator-orin

mkdir -p deploy/logs
test -f deploy/edge_runtime.remote.json || \
  cp deploy/edge_runtime.remote.example.json deploy/edge_runtime.remote.json
python -m json.tool deploy/edge_runtime.remote.json >/dev/null

run_tag=$(date +%Y%m%d_%H%M%S)

python orin_state_sender.py \
  --serial-port /dev/ttyTHS1 \
  --control-enabled \
  --pc-host 192.168.0.220 \
  --edge-config deploy/edge_runtime.remote.json \
  --edge-motion-authorization ALLOW_EDGE_MACHINE_MOTION \
  --print-every 100 \
  2>&1 | tee "deploy/logs/rl_follow_${run_tag}_stdout.log"
```

`deploy/edge_runtime.remote.json` must use `mode=remote_control`, behavior port
`18083`, and `allowed_client_host=192.168.0.220`. Expected startup output includes
`REMOTE EDGE CONTROL ARMED IDLE`. The complete PC command, preflight checks and
RViz button order are maintained in the `AiryLidar/README.md` section
“强化学习 Orin + PC 真机测试速查”.

The hybrid Mission Adapter may additionally pass an absolute
`--hardware-start-gate`. In that internal mode the process validates deployment
assets and loads ONNX first, logs `RL prewarm ready`, and waits without opening
the serial port, action socket, behavior RPC port, or publishing Machine State.
The Adapter creates the one-shot gate only after the previous Runtime has sent
terminal zero and `/dev/ttyTHS1` is confirmed released. This flag is an internal
handoff mechanism; normal standalone RL commands do not need it.

## Installation on Orin

```bash
git clone <your-private-remote> /home/jetson16/workspace_excavator/excavator-orin-runtime
cd /home/jetson16/workspace_excavator/excavator-orin-runtime

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
# Install the NVIDIA JetPack-compatible onnxruntime wheel for this Orin image.
python3 -c 'import numpy, onnxruntime; print(numpy.__version__, onnxruntime.__version__)'
python3 -m unittest discover -s tests -v
```

Do not install an arbitrary desktop `onnxruntime-gpu` wheel on Jetson. Use the
wheel compatible with the installed JetPack/CUDA image.

## Normal PC-controlled operation

The default action source is `--pc-host`:

```bash
python3 orin_state_sender.py \
  --control-enabled \
  --pc-host 192.168.0.220 \
  --print-every 100
```

Only one Runtime may own `/dev/ttyTHS1`. The defaults are `/dev/ttyTHS1` at
`460800` baud, matching the unified `F407/data_celect` firmware.

## Edge deployment assets

The model and machine artifacts are intentionally copied manually for the first
field deployment. They are not committed to this repository.

On the PC, stage the current authoritative files:

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj

mkdir -p /tmp/excavator-edge-assets
cp shared/machine_profile.json \
  /tmp/excavator-edge-assets/machine_profile.json
cp AiryLidar/kinematics/waji_description/urdf/waji.urdf \
  /tmp/excavator-edge-assets/waji.urdf
cp RLExcavator/Assets/AIModels/ScaleModelDeploy05_cale_v3_deadzone_reward_03_p003.onnx \
  /tmp/excavator-edge-assets/policy.onnx
cp AiryLidar/localmap/exports/live_latest/trajectory_command.simple_rrt.json \
  /tmp/excavator-edge-assets/trajectory_command.json
cp AiryLidar/mission/config/excavation_cycle.json \
  /tmp/excavator-edge-assets/excavation_cycle.json
cp AiryLidar/runtime_bridge/config/fixed_actions.json \
  /tmp/excavator-edge-assets/fixed_actions.json
sha256sum /tmp/excavator-edge-assets/*
```

Copy those six files to `<ORIN_REPO>/deploy/assets/`, then on Orin:

```bash
cd /home/jetson16/workspace_excavator/excavator-orin-runtime
mkdir -p deploy/assets deploy/logs
cp deploy/edge_runtime.example.json deploy/edge_runtime.json
python3 -m json.tool deploy/edge_runtime.json >/dev/null
python3 -m unittest discover -s tests -v
```

The trajectory snapshot must use `frame_id=machine_root_ros`. The URDF FK root
is `fk_root`; the current deployed frame adapter is the explicit identity
`machine_root_ros -> fk_root`.

`trajectory_controller_backend` is the strict algorithm-selection seam. Keep
the active field configuration explicit as `onnx_rl`. `cartesian_p` is a
deterministic reference controller for controlled ablation experiments; it
shares the same 38D input and normalized `[boom, stick, bucket, swing]` output
contract, but it is not field-qualified merely by selecting the name. Shadow
mode may select `cartesian_p` without motion authorization. In `control` or
`remote_control`, it additionally requires this independent exact launch opt-in:

```bash
--trajectory-controller-commissioning-authorization \
  ALLOW_CARTESIAN_P_MACHINE_MOTION
```

This literal is an auditable commissioning acknowledgement, not a secret. It
does not replace `--control-enabled` or `--edge-motion-authorization`; a missing
or incorrect value is rejected before controller construction or serial
ownership. The `onnx_rl` launch contract is unchanged. Legacy configs without
the backend field continue to mean `onnx_rl`; unknown names fail at
configuration load. `onnx_path` is mandatory for `onnx_rl` and may be omitted
for `cartesian_p`, so the classical ablation does not depend on an unrelated
model artifact.

The execution waypoint tolerance and Follow deadline have one authoritative
source:

```text
excavation_cycle.json limits.waypoint_tolerance_m
excavation_cycle.json limits.tracking_timeout_s
```

`trajectory_command.json target_threshold` remains planner metadata and is not
the execution waypoint tolerance. Startup rejects a trajectory whose Mission
ID, SHA, phase, execution scope or eligibility does not match the deployed
Mission asset.

## Edge shadow verification

Keep `mode` set to `shadow` in `deploy/edge_runtime.json`, then start:

```bash
cd /home/jetson16/workspace_excavator/excavator-orin-runtime
source .venv/bin/activate

python3 orin_state_sender.py \
  --pc-host <PC_IP> \
  --edge-config deploy/edge_runtime.json \
  --print-every 100
```

Shadow mode still publishes Machine State to the PC, but the edge runtime has no
action sink. It records each local Bucket Tip, 38D observation, normalized ONNX
action, slew-limited normalized command, physical action and inference time:

```bash
tail -n 3 deploy/logs/edge_runtime.jsonl | python3 -m json.tool --json-lines
```

Summarize a Shadow log without sending any action:

```bash
python3 -m edge_runtime.audit deploy/logs/edge_runtime.jsonl
```

The summary reports input and inference-loop frequency, pure ONNX and full-loop
latency distributions, sequence gaps/regressions, progress monotonicity,
consecutive rejected states, terminal status and whether the final computed
physical action is zero. A rejected state is fail-safe for that frame but does
not terminate the process: control sends zero, records the reason and can resume
on a later valid state. A sequence gap is accepted; a duplicate or older
sequence is rejected. `COMPLETED` and `TIMEOUT` are sticky terminal states.

Run shadow before enabling edge motion. Compare its actions with the current PC
log for the same state/trajectory snapshot.

## Edge control

Stop PC live-control/action publishing first. The PC may continue perception,
planning visualization and state monitoring.

Change only `mode` in `deploy/edge_runtime.json` from `shadow` to `control`, then:

```bash
python3 orin_state_sender.py \
  --control-enabled \
  --pc-host <PC_IP> \
  --edge-config deploy/edge_runtime.json \
  --edge-motion-authorization ALLOW_EDGE_MACHINE_MOTION \
  --print-every 100
```

In this mode the high-rate state → FK → observation → ONNX → action path no
longer crosses the PC network. The PC link carries monitoring and future
low-rate trajectory/mission updates only. Ctrl+C, invalid sensor state, action
lease expiry, trajectory completion and shutdown all produce a zero command.

`follow_action_slew_rate_per_s` limits only how quickly the command sent to the
actuator can approach a new ONNX target. It does not scale, negate or clip the
steady-state ONNX target. With the deployed value `2.0`, a 10 Hz state stream
allows at most about `0.2` normalized command change per update. Direction
reversals therefore pass through zero instead of jumping directly from `+1` to
`-1`. The first Follow sample is zero so a new behavior ramps from rest.
Terminal, cancellation, rejected-state and shutdown zeros bypass the limiter
and remain immediate.

This static control mode remains available for the existing staged rollout.

## Remote edge Follow

Remote behavior control starts Idle. It accepts immutable Follow Trajectory
Snapshots and the named `ExecuteDig`/`ExecuteDump` fixed actions over a low-rate
TCP behavior connection. It does not execute
`deploy/assets/trajectory_command.json` at startup.
The fixed-action asset uses `fixed_action_profile.v2`: each stage declares
absolute normalized actuator targets, so execution does not inherit the final
pose error of the preceding Follow behavior.

Copy and edit the remote example, including the PC allowlisted address:

```bash
cp deploy/edge_runtime.remote.example.json deploy/edge_runtime.remote.json
python3 -m json.tool deploy/edge_runtime.remote.json >/dev/null

python3 orin_state_sender.py \
  --control-enabled \
  --pc-host <PC_IP> \
  --edge-config deploy/edge_runtime.remote.json \
  --edge-motion-authorization ALLOW_EDGE_MACHINE_MOTION \
  --print-every 100
```

Motion remains gated by both `--control-enabled` and the exact authorization
token. In `remote_control`, the Action Relay bind and allowlist are forcibly
set to `127.0.0.1`; behavior RPC never writes serial, scales actions or changes
signs. Follow command slew limiting is an explicit Orin execution-runtime stage
after ONNX inference; the audit keeps both `normalized_action` (raw ONNX output)
and `commanded_normalized_action` (the value converted to physical velocity).

Each TCP JSON message uses a four-byte big-endian payload length followed by
UTF-8 JSON, with a maximum payload of 1 MiB and
`schema_version="orin_behavior_rpc.v1"`. The server accepts the commissioning
requests `start_follow`, `cancel_follow`, `start_fixed_action` and
`cancel_fixed_action`. The Mission path uses `start_cycle`,
`provide_dump_trajectory` and `cancel_cycle`; it emits
`status`, `accepted`, `rejected`, `feedback` and `result`. It recomputes the
canonical Trajectory Snapshot SHA-256 before
acceptance. For remote Follow, the accepted snapshot is authoritative for the
Mission identity, waypoint tolerance, dwell and tracking timeout; this permits
PC-planned multi-point demonstrations without copying each dynamic Mission to
Orin. The preloaded Mission still establishes the deployment frame, while
`target_threshold` and `tube_radius` come only from the preloaded machine
profile.

`start_cycle` makes Orin execute `FollowDig → ExecuteDig` locally and return
`DIG_LEG_COMPLETED` only after terminal zero is confirmed. The PC then replans
from the new state and sends `provide_dump_trajectory`; Orin executes
`FollowDump → ExecuteDump` locally and returns `SEQUENCE_COMPLETED`. This keeps
environment-dependent planning on PC while removing the network round trip
between Follow and its fixed action. An active-leg connection loss remains
fail-closed and stops that local behavior.

The fixed-action asset is loaded once before the serial port is opened. Its
machine-profile and URDF SHA-256 bindings must match the deployed assets.
Fixed-action feedback is closed locally from the same high-rate Machine State
stream; PC sends only the behavior name and never streams actuator velocity.

The server sends status immediately on each allowed TCP connection and then at
the configured rate, including while Idle. A client disconnect, cancellation,
completion, timeout, rejected machine state, exception or shutdown closes the
active `EdgeControlRunner`, which submits a zero through the existing loopback
Action Relay before the terminal result reports quiescence.

Before accepting each Goal, Orin rechecks its own latest Machine State and
rejects stale or unsafe state with `MOTION_NOT_READY`; PC readiness is not
trusted as the final authority. While Follow is active, a local watchdog checks
the same gate on every status tick. If Machine State stops or the gate closes,
the runner submits terminal zero and returns `MOTION_GATE_CLOSED` instead of
remaining active indefinitely.

Network status/feedback/result writes use a bounded writer queue. Serial
ingestion, FK, observation construction, ONNX inference and loopback action
delivery never call TCP `sendall` directly. A stalled client therefore cannot
block the local control loop; sustained backpressure closes that client, which
then triggers the same terminal-zero disconnect path. Event sequence assignment
is serialized across concurrent status and feedback producers.

## Local CSV replay

CSV replay still uses `orin_state_sender.py` for sensor freshness, `control_enabled`, E-stop,
STM32 liveness, packet expiry, sequence checking and terminal zero commands. The replay tool never
opens the serial port itself.

Start a loopback-only relay:

```bash
python3 orin_state_sender.py \
  --control-enabled \
  --action-bind-host 127.0.0.1 \
  --allowed-action-host 127.0.0.1 \
  --print-every 100
```

Preflight a CSV without sending motion:

```bash
python3 orin_csv_replay.py replays/execute_dump.pc_to_orin.csv \
  --max-duration-s 30
```

Expected output includes:

```text
PREVIEW ONLY: no UDP packets sent
```

After checking the file SHA and machine area, explicitly authorize replay:

```bash
python3 orin_csv_replay.py replays/execute_dump.pc_to_orin.csv \
  --max-duration-s 30 \
  --motion-authorization ALLOW_CSV_REPLAY
```

CSV action values are forwarded verbatim. Neither the replay tool nor the Orin Action Relay reads
a machine profile or applies physical range checks, speed limits, scaling or sign changes. The
relay maps values by the declared action names into the fixed STM32 field order; STM32 owns the
physical limits.

Ctrl+C, scheduler lag, normal completion and Action Relay shutdown all lead to zero commands.
进程管理器发送的 `SIGTERM` 会进入同一 `KeyboardInterrupt`/`finally` 清理路径，用于 RL→示教采集
切换时先写终态零命令再释放 `/dev/ttyTHS1`。
PC live actions and local CSV replay must not be enabled at the same time.

## Updating Orin

Stop the running bridge before changing code, then update to an intentional commit:

```bash
cd /home/jetson16/workspace_excavator/excavator-orin-runtime
git pull --ff-only
source .venv/bin/activate
python3 -m unittest discover -s tests -v
```

Restart the bridge only after the tests pass.
