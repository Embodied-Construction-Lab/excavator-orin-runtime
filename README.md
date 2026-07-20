# Excavator Orin Runtime

Jetson Orin runtime for the scale hydraulic excavator. It owns the deployed bridge:

```text
PC policy_action or local CSV replay
              ↓ UDP JSON
       orin_state_sender.py
              ↓ serial physical velocity command
             STM32

STM32 state
    ↓ serial
orin_state_sender.py
    ↓ UDP machine_state_v1
    PC
```

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
- `tests/`: host-side protocol, relay, timeout, ordering and replay tests.

Historical joystick and `[swing, boom, stick, bucket]` rollout tools are intentionally excluded.

## Installation on Orin

```bash
git clone <your-private-remote> ~/excavator-orin-runtime
cd ~/excavator-orin-runtime

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

## Normal PC-controlled operation

The default action source is `--pc-host`:

```bash
python3 orin_state_sender.py \
  --control-enabled \
  --pc-host 192.168.2.127 \
  --print-every 100
```

Only one `orin_state_sender.py` process may own `/dev/ttyTHS0`.

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
