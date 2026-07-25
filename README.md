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

## Files

- `orin_state_sender.py`: reads STM32 state, publishes `machine_state_v1`, validates action structure/timing/safety state and relays finite physical velocity commands to STM32 without magnitude checks.
- `orin_csv_replay.py`: validates and replays an exported physical-velocity CSV through the local Action Relay.
- `edge_runtime/`: dependency-light URDF FK, Unity-compatible 38D observation,
  ONNX inference, waypoint tracking, normalized-to-physical conversion, shadow
  auditing and loopback edge control.
- `deploy/edge_runtime.example.json`: the single edge deployment configuration.
- `tests/`: host-side protocol, relay, timeout, ordering and replay tests.

Historical joystick and `[swing, boom, stick, bucket]` rollout tools are intentionally excluded.
The old workspace-root `urdf/` project is also excluded. The deployed URDF is
copied from `AiryLidar/kinematics/waji_description/urdf/waji.urdf`.

## Installation on Orin

```bash
git clone <your-private-remote> ~/excavator-orin-runtime
cd ~/excavator-orin-runtime

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
  --pc-host 192.168.2.127 \
  --print-every 100
```

Only one `orin_state_sender.py` process may own `/dev/ttyTHS0`.

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
sha256sum /tmp/excavator-edge-assets/*
```

Copy those four files to `<ORIN_REPO>/deploy/assets/`, then on Orin:

```bash
cd ~/excavator-orin-runtime
mkdir -p deploy/assets deploy/logs
cp deploy/edge_runtime.example.json deploy/edge_runtime.json
python3 -m json.tool deploy/edge_runtime.json >/dev/null
python3 -m unittest discover -s tests -v
```

The trajectory snapshot must use `frame_id=machine_root_ros`. The URDF FK root
is `fk_root`; the current deployed frame adapter is the explicit identity
`machine_root_ros -> fk_root`.

## Edge shadow verification

Keep `mode` set to `shadow` in `deploy/edge_runtime.json`, then start:

```bash
cd ~/excavator-orin-runtime
source .venv/bin/activate

python3 orin_state_sender.py \
  --pc-host <PC_IP> \
  --edge-config deploy/edge_runtime.json \
  --print-every 100
```

Shadow mode still publishes Machine State to the PC, but the edge runtime has no
action sink. It records each local Bucket Tip, 38D observation, normalized ONNX
action, physical action and inference time:

```bash
tail -n 3 deploy/logs/edge_runtime.jsonl | python3 -m json.tool --json-lines
```

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

The first implementation loads one immutable trajectory snapshot at startup.
To test a newly planned trajectory, stop the process, replace
`deploy/assets/trajectory_command.json`, then restart. Live trajectory update is
the next migration slice; it must not reintroduce the high-rate state/action
round trip.

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
PC live actions and local CSV replay must not be enabled at the same time.

## Updating Orin

Stop the running bridge before changing code, then update to an intentional commit:

```bash
cd ~/excavator-orin-runtime
git pull --ff-only
source .venv/bin/activate
python3 -m unittest discover -s tests -v
```

Restart the bridge only after the tests pass.
