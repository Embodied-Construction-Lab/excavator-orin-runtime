# Orin Runtime Collaboration Rules

This repository owns only the Jetson Orin runtime between the PC policy layer and STM32.

## Invariants

1. The authoritative action order is `[boom, stick, bucket, swing]`.
2. PC/CSV action values are physical velocities: boom/stick/bucket in `m/s`, swing in `rad/s`.
3. The compatibility field `action_type="normalized_velocity_command"` must not cause another normalization step.
4. Never silently scale, reorder or invert an action. STM32 owns the final hardware direction adaptation.
5. Invalid, stale, expired or out-of-order actions, invalid sensor state, disabled control, E-stop, STM32 timeout and process shutdown must produce a zero command.
6. CSV replay and the Orin Action Relay perform no physical range or speed validation; STM32 owns those limits.
7. STM32 firmware does not belong in this repository.

## Verification

Run before committing:

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile orin_state_sender.py orin_csv_replay.py
```

Do not perform a live replay as part of automated tests.
