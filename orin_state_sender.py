#!/usr/bin/env python3
"""Read unified STM32 state and send machine_state_v1 to PC over UDP.

Default STM32 serial input is `stm32_control_telemetry.v2` CSV from the unified
`F407/data_celect` firmware. Legacy 19-field CSV remains readable for offline
compatibility:

    t,x1,x2,y1,y2,s_boom,s_stick,s_bucket,v_boom,v_stick,v_bucket,
    a_boom,a_stick,a_bucket,yaw,yaw_rate,rs485_ok,adc_ok,imu_ok

Linear positions and velocities are sent by STM32 in mm and mm/s.
Angles and angular velocity are sent in deg and deg/s. This script converts
them to m, m/s, rad, and rad/s before sending UDP JSON. A zero hardware status
flag makes sensor_valid false. Legacy 12-field and 16-field CSV rows remain
supported without explicit hardware flags.

Legacy binary frame mode is still available with --input-format binary:

    uint32_t t_ms;
    float x1, x2, y1, y2;
    float s_boom, s_stick, s_bucket;
    float v_boom, v_stick, v_bucket;
    float a_boom, a_stick, a_bucket;
    float yaw, yaw_rate;

Frame size: 64 bytes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import select
import signal
import socket
import struct
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from edge_runtime.resident_action_audit import AsyncResidentActionAudit
from edge_runtime.resident_motion import ControlMode, ZERO_ACTION
from edge_runtime.resident_protocol import ResidentActWorkerIdentity
from edge_runtime.resident_commands import (
    Stm32ResidentCommandEncoder,
    VELOCITY_COMMAND_SCHEMA_VERSION,
)
from edge_runtime.resident_control import (
    DEFAULT_MISSION_LEASE_MS,
    ResidentMotionControlServer,
)
from edge_runtime.resident_core import (
    AxisManualActionDeadzone,
    ManualActionDeadzoneContract,
    ResidentMotionCore,
)
from edge_runtime.resident_data_link import ResidentActDataLink
from edge_runtime.resident_fixed_cycle import (
    load_fixed_cycle_plan,
    load_fixed_cycle_registry,
)
from edge_runtime.resident_fixed_cycle_control import (
    ResidentFixedCycleControlServer,
)
from edge_runtime.resident_fixed_cycle_runtime import ResidentFixedCycleRuntime
from edge_runtime.resident_sink import ResidentTelemetry, ResidentWriteResult
from edge_runtime.resident_state import (
    ResidentActState,
    decode_resident_state,
    encode_resident_state,
)


LOGGER = logging.getLogger("orin_state_sender")

DEFAULT_SERIAL_PORT = "/dev/ttyTHS1"
DEFAULT_BAUDRATE = 460800
DEFAULT_PC_STATE_HOST = "192.168.2.127"
DEFAULT_PC_STATE_PORT = 18081
DEFAULT_ORIN_ACTION_BIND_HOST = "0.0.0.0"
DEFAULT_ORIN_ACTION_BIND_PORT = 18082
DEFAULT_MACHINE_ID = "scale_excavator_v1"
DEFAULT_EDGE_ACTION_TRANSPORT = "loopback_udp"
RESIDENT_EDGE_ACTION_TRANSPORT = "resident_sink"
DEFAULT_RESIDENT_STATE_AGE_MS = 100.0
DEFAULT_RESIDENT_TERMINAL_ACK_TIMEOUT_S = 0.6
DEFAULT_RESIDENT_RUNTIME_ROOT = Path.home() / ".local/run/excavator-resident"
DEFAULT_RESIDENT_ACT_SOCKET = DEFAULT_RESIDENT_RUNTIME_ROOT / "act.sock"
DEFAULT_RESIDENT_CONTROL_SOCKET = DEFAULT_RESIDENT_RUNTIME_ROOT / "control.sock"
DEFAULT_RESIDENT_FIXED_CYCLE_CONTROL_SOCKET = (
    DEFAULT_RESIDENT_RUNTIME_ROOT / "fixed-cycle.sock"
)
DEFAULT_RESIDENT_ACTION_AUDIT_PATH = (
    Path.home() / ".local/state/excavator-resident/resident-action-audit.jsonl"
)
V3A_FIXED_TRAJECTORY_COMMISSIONING_AUTHORIZATION = (
    "ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING"
)


def load_resident_manual_action_deadzone_contract(
    contract_path: Path | None,
) -> ManualActionDeadzoneContract | None:
    """Load directional ACT/manual deadzones from their own contract.

    A missing contract disables ACT early completion. An existing malformed
    contract fails closed before serial ownership.
    """

    if contract_path is None:
        return None
    try:
        document = json.loads(
            Path(contract_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read resident manual action deadzone contract: {exc}"
        ) from exc
    expected_fields = {
        "schema_version",
        "action_order",
        "axes",
        "source",
    }
    if not isinstance(document, dict) or set(document) != expected_fields:
        raise ValueError("resident manual action deadzone fields are invalid")
    if document["schema_version"] != "resident_manual_action_deadzone.v1":
        raise ValueError("resident manual action deadzone schema is invalid")
    if tuple(document["action_order"]) != POLICY_ACTION_NAMES:
        raise ValueError("resident manual action deadzone action_order is invalid")
    if not isinstance(document["source"], str) or not document["source"]:
        raise ValueError("resident manual action deadzone source is invalid")
    axes_by_name = document["axes"]
    if not isinstance(axes_by_name, dict) or set(axes_by_name) != set(
        POLICY_ACTION_NAMES
    ):
        raise ValueError("resident manual action deadzone axes are invalid")
    axes = []
    for axis_name in POLICY_ACTION_NAMES:
        axis = axes_by_name[axis_name]
        if not isinstance(axis, dict) or set(axis) != {
            "positive_normalized",
            "negative_normalized",
        }:
            raise ValueError(
                f"resident manual action deadzone axis {axis_name} is invalid"
            )
        axes.append(
            AxisManualActionDeadzone(
                _resident_deadzone_component(
                    axis["positive_normalized"],
                    field_name=(
                        "manual_action_deadzone."
                        f"axes.{axis_name}.positive_normalized"
                    ),
                ),
                _resident_deadzone_component(
                    axis["negative_normalized"],
                    field_name=(
                        "manual_action_deadzone."
                        f"axes.{axis_name}.negative_normalized"
                    ),
                ),
            )
        )
    return ManualActionDeadzoneContract(tuple(axes))


def _resident_deadzone_component(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) < 1.0
    ):
        raise ValueError(
            f"{field_name} must be finite and in [0, 1)"
        )
    return float(value)

STM32_FRAME_FORMAT = "<I15f"
STM32_FRAME_SIZE = struct.calcsize(STM32_FRAME_FORMAT)
STM32_TIMEOUT_S = 0.3
MM_TO_M = 0.001
POLICY_ACTION_NAMES = ("boom", "stick", "bucket", "swing")
ACTION_FUTURE_SKEW_MS = 50
EDGE_MOTION_AUTHORIZATION = "ALLOW_EDGE_MACHINE_MOTION"
CARTESIAN_P_COMMISSIONING_AUTHORIZATION = (
    "ALLOW_CARTESIAN_P_MACHINE_MOTION"
)
STM32_V2_SCHEMA_VERSION = "stm32_control_telemetry.v2"
STM32_VELOCITY_COMMAND_SCHEMA_VERSION = VELOCITY_COMMAND_SCHEMA_VERSION
STM32_V2_FIELDS = (
    "schema_version", "control_seq", "control_stamp_ms", "sensor_seq",
    "sensor_stamp_ms", "sensor_is_new", "command_rx_seq",
    "command_source_stamp_ms", "command_received_stamp_ms", "command_age_ms",
    "command_action_boom", "command_action_stick", "command_action_bucket",
    "command_action_swing", "boom_pos_mm", "stick_pos_mm", "bucket_pos_mm",
    "boom_vel_mmps", "stick_vel_mmps", "bucket_vel_mmps", "boom_angle_deg",
    "arm_angle_deg", "bucket_angle_deg", "swing_angle_deg", "swing_vel_degps",
    "boom_v_ref_mmps", "stick_v_ref_mmps", "bucket_v_ref_mmps",
    "swing_v_ref_degps", "pid_out_boom", "pid_out_stick", "pid_out_bucket",
    "pid_out_swing", "valve_boom_deg", "valve_stick_deg", "valve_bucket_deg",
    "swing_percent", "pump_percent", "pwm_boom", "pwm_stick", "pwm_bucket",
    "pwm_swing", "pwm_pump", "control_mode", "homing_complete",
    "command_valid", "command_timed_out", "control_enabled", "estop",
    "limit_mask", "rs485_ok", "dwj_ok", "imu_ok", "fault_flags",
    "dropped_command_frames",
)
STM32_V2_INTEGER_FIELDS = frozenset(
    {
        "control_seq", "control_stamp_ms", "sensor_seq", "sensor_stamp_ms",
        "sensor_is_new", "command_rx_seq", "command_source_stamp_ms",
        "command_received_stamp_ms", "command_age_ms", "pwm_boom",
        "pwm_stick", "pwm_bucket", "pwm_swing", "pwm_pump", "control_mode",
        "homing_complete", "command_valid", "command_timed_out",
        "control_enabled", "estop", "limit_mask", "rs485_ok", "dwj_ok",
        "imu_ok", "fault_flags", "dropped_command_frames",
    }
)


@dataclass(frozen=True)
class Stm32State:
    t_ms: int
    x1: float
    x2: float
    y1: float
    y2: float
    s_boom: float
    s_stick: float
    s_bucket: float
    v_boom: float
    v_stick: float
    v_bucket: float
    a_boom: float
    a_stick: float
    a_bucket: float
    yaw: float
    yaw_rate: float
    rs485_ok: bool = True
    adc_ok: bool = True
    imu_ok: bool = True
    command_rx_seq: int = 0
    command_received: bool = False
    stm32_control_enabled: bool = False
    hardware_motion_gate_available: bool = False
    command_valid: bool = False
    command_timed_out: bool = False
    stm32_control_mode: int = 3
    command_boom_mps: float = 0.0
    command_stick_mps: float = 0.0
    command_bucket_mps: float = 0.0
    command_swing_radps: float = 0.0
    stm32_estop: bool = False
    stm32_fault_flags: int = 0
    control_seq: int = 0
    sensor_seq: int = 0
    sensor_is_new: bool = True


@dataclass(frozen=True)
class DataCommand:
    t_s: float
    boom_v_ref_mps: float
    stick_v_ref_mps: float
    bucket_v_ref_mps: float
    swing_v_ref_radps: float


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def open_serial(port: str, baudrate: int, timeout_s: float):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required: pip install pyserial") from exc

    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        timeout=timeout_s,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        exclusive=True,
    )
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.05)
    ser.reset_input_buffer()
    return ser


def read_exact(ser, size: int) -> Optional[bytes]:
    data = bytearray()
    while len(data) < size:
        chunk = ser.read(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def parse_stm32_frame(frame: bytes) -> Stm32State:
    if len(frame) != STM32_FRAME_SIZE:
        raise ValueError(f"bad frame size: {len(frame)} != {STM32_FRAME_SIZE}")
    values = struct.unpack(STM32_FRAME_FORMAT, frame)
    return Stm32State(*values)


def parse_stm32_csv_line(line: str) -> Optional[Stm32State]:
    line = line.strip()
    if not line:
        return None

    try:
        fields = [field.strip() for field in next(csv.reader([line]))]
    except csv.Error as exc:
        raise ValueError(f"bad STM32 CSV line: {exc}") from exc

    if not fields:
        return None
    if tuple(fields) == STM32_V2_FIELDS:
        return None
    if fields[0] == STM32_V2_SCHEMA_VERSION:
        return _parse_stm32_v2_fields(fields)
    if any(ch.isalpha() or ch == "_" for ch in fields[0]):
        return None

    try:
        values = [float(field) for field in fields]
    except ValueError as exc:
        raise ValueError(f"bad STM32 CSV numeric field: {line!r}") from exc

    rs485_ok = True
    adc_ok = True
    imu_ok = True
    if len(values) >= 16:
        (
            t_ms,
            x1,
            x2,
            y1,
            y2,
            s_boom_mm,
            s_stick_mm,
            s_bucket_mm,
            v_boom_mm_s,
            v_stick_mm_s,
            v_bucket_mm_s,
            a_boom_deg,
            a_stick_deg,
            a_bucket_deg,
            yaw_deg,
            yaw_rate_deg_s,
        ) = values[:16]
        if len(values) in (17, 18):
            raise ValueError(
                "bad STM32 CSV hardware validity field count: %d" % len(values)
            )
        if len(values) >= 19:
            rs485_ok = _parse_hardware_status("rs485_ok", values[16])
            adc_ok = _parse_hardware_status("adc_ok", values[17])
            imu_ok = _parse_hardware_status("imu_ok", values[18])
    elif len(values) == 12:
        (
            t_ms,
            s_boom_mm,
            s_stick_mm,
            s_bucket_mm,
            v_boom_mm_s,
            v_stick_mm_s,
            v_bucket_mm_s,
            a_boom_deg,
            a_stick_deg,
            a_bucket_deg,
            yaw_deg,
            yaw_rate_deg_s,
        ) = values[:12]
        x1 = x2 = y1 = y2 = 0.0
    else:
        raise ValueError(f"bad STM32 CSV field count: {len(values)}")

    return Stm32State(
        t_ms=int(round(t_ms)),
        x1=x1,
        x2=x2,
        y1=y1,
        y2=y2,
        s_boom=s_boom_mm * MM_TO_M,
        s_stick=s_stick_mm * MM_TO_M,
        s_bucket=s_bucket_mm * MM_TO_M,
        v_boom=v_boom_mm_s * MM_TO_M,
        v_stick=v_stick_mm_s * MM_TO_M,
        v_bucket=v_bucket_mm_s * MM_TO_M,
        a_boom=math.radians(a_boom_deg),
        a_stick=math.radians(a_stick_deg),
        a_bucket=math.radians(a_bucket_deg),
        yaw=math.radians(yaw_deg),
        yaw_rate=math.radians(yaw_rate_deg_s),
        rs485_ok=rs485_ok,
        adc_ok=adc_ok,
        imu_ok=imu_ok,
    )


def _parse_stm32_v2_fields(fields: List[str]) -> Stm32State:
    if len(fields) != len(STM32_V2_FIELDS):
        raise ValueError(
            f"bad STM32 v2 CSV field count: {len(fields)} != {len(STM32_V2_FIELDS)}"
        )
    raw = dict(zip(STM32_V2_FIELDS, fields))
    parsed: Dict[str, float | int] = {}

    def finite_float(name: str) -> float:
        return float(parsed[name])

    def nonnegative_int(name: str) -> int:
        return int(parsed[name])

    for name in STM32_V2_FIELDS[1:]:
        try:
            value: float | int
            if name in STM32_V2_INTEGER_FIELDS:
                value = int(raw[name])
                if value < 0:
                    raise ValueError
            else:
                value = float(raw[name])
                if not math.isfinite(value):
                    raise ValueError
        except ValueError as exc:
            raise ValueError(f"bad STM32 v2 field {name}: {raw[name]!r}") from exc
        parsed[name] = value

    command_valid = _parse_hardware_status(
        "command_valid", finite_float("command_valid")
    )
    command_timed_out = _parse_hardware_status(
        "command_timed_out", finite_float("command_timed_out")
    )
    return Stm32State(
        t_ms=nonnegative_int("control_stamp_ms"),
        x1=finite_float("command_action_swing"),
        x2=finite_float("command_action_bucket"),
        y1=finite_float("command_action_stick"),
        y2=finite_float("command_action_boom"),
        s_boom=finite_float("boom_pos_mm") * MM_TO_M,
        s_stick=finite_float("stick_pos_mm") * MM_TO_M,
        s_bucket=finite_float("bucket_pos_mm") * MM_TO_M,
        v_boom=finite_float("boom_vel_mmps") * MM_TO_M,
        v_stick=finite_float("stick_vel_mmps") * MM_TO_M,
        v_bucket=finite_float("bucket_vel_mmps") * MM_TO_M,
        a_boom=math.radians(finite_float("boom_angle_deg")),
        a_stick=math.radians(finite_float("arm_angle_deg")),
        a_bucket=math.radians(finite_float("bucket_angle_deg")),
        yaw=math.radians(finite_float("swing_angle_deg")),
        yaw_rate=math.radians(finite_float("swing_vel_degps")),
        rs485_ok=_parse_hardware_status("rs485_ok", finite_float("rs485_ok")),
        adc_ok=_parse_hardware_status("dwj_ok", finite_float("dwj_ok")),
        imu_ok=_parse_hardware_status("imu_ok", finite_float("imu_ok")),
        command_rx_seq=nonnegative_int("command_rx_seq"),
        command_received=command_valid or command_timed_out,
        stm32_control_enabled=_parse_hardware_status(
            "control_enabled", finite_float("control_enabled")
        ),
        hardware_motion_gate_available=True,
        command_valid=command_valid,
        command_timed_out=command_timed_out,
        stm32_control_mode=nonnegative_int("control_mode"),
        command_boom_mps=finite_float("boom_v_ref_mmps") * MM_TO_M,
        command_stick_mps=finite_float("stick_v_ref_mmps") * MM_TO_M,
        command_bucket_mps=finite_float("bucket_v_ref_mmps") * MM_TO_M,
        command_swing_radps=math.radians(finite_float("swing_v_ref_degps")),
        stm32_estop=_parse_hardware_status("estop", finite_float("estop")),
        stm32_fault_flags=nonnegative_int("fault_flags"),
        control_seq=nonnegative_int("control_seq"),
        sensor_seq=nonnegative_int("sensor_seq"),
        sensor_is_new=_parse_hardware_status(
            "sensor_is_new", finite_float("sensor_is_new")
        ),
    )


def _parse_hardware_status(name: str, value: float) -> bool:
    if not math.isfinite(value) or value not in (0.0, 1.0):
        raise ValueError("%s must be 0 or 1" % name)
    return value == 1.0


def resident_telemetry_from_state(
    state: Stm32State,
    *,
    receive_monotonic_ns: int,
    runtime_control_enabled: bool,
    runtime_estop: bool,
    sensor_valid: bool,
    stm32_alive: bool,
) -> ResidentTelemetry:
    """Adapt parsed v2 telemetry to the resident final-write contract."""

    if state.stm32_control_mode == 1:
        mode = ControlMode.MANUAL_ACTION
        action = (state.y2, state.y1, state.x2, state.x1)
    elif state.stm32_control_mode == 2:
        mode = ControlMode.VELOCITY_REFERENCE
        action = (
            state.command_boom_mps,
            state.command_stick_mps,
            state.command_bucket_mps,
            state.command_swing_radps,
        )
    else:
        mode = None
        action = ZERO_ACTION
    return ResidentTelemetry(
        receive_monotonic_ns=receive_monotonic_ns,
        command_rx_seq=state.command_rx_seq,
        command_valid=state.command_valid,
        command_timed_out=state.command_timed_out,
        control_mode=mode,
        command_action=action,
        control_enabled=(
            bool(runtime_control_enabled) and state.stm32_control_enabled
        ),
        estop=bool(runtime_estop) or state.stm32_estop,
        sensor_valid=bool(sensor_valid),
        stm32_alive=bool(stm32_alive),
        fault_flags=state.stm32_fault_flags,
    )


def wait_for_resident_terminal_zero_ack(
    serial_reader,
    result: ResidentWriteResult,
    *,
    timeout_s: float = DEFAULT_RESIDENT_TERMINAL_ACK_TIMEOUT_S,
) -> bool:
    """Wait for the exact STM32 echo of the resident terminal zero.

    A resident Runtime owns the serial device until this function returns.  It
    therefore cannot hand the device to another process merely because the
    final zero was written; the matching command sequence, control mode and
    four zero axes must first be observed in fresh unified telemetry.
    """

    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("terminal zero ACK timeout must be finite and positive")
    if not result.write_performed:
        return (
            result.reason == "terminal_disarm"
            and result.command_seq is None
            and result.mode is None
            and result.effective_action == ZERO_ACTION
        )
    if (
        result.command_seq is None
        or result.mode is None
        or result.effective_action != ZERO_ACTION
    ):
        return False

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            raw_line = serial_reader.readline()
        except OSError:
            return False
        if not raw_line:
            continue
        try:
            state = parse_stm32_csv_line(raw_line.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            continue
        if state is None:
            continue
        frame = resident_telemetry_from_state(
            state,
            receive_monotonic_ns=time.monotonic_ns(),
            runtime_control_enabled=True,
            runtime_estop=False,
            sensor_valid=True,
            stm32_alive=True,
        )
        if (
            frame.command_rx_seq == result.command_seq
            and frame.command_valid
            and not frame.command_timed_out
            and frame.control_mode is result.mode
            and frame.command_action == ZERO_ACTION
        ):
            return True
    return False


def resident_terminal_zero_is_acknowledged(
    core: ResidentMotionCore,
    serial_reader,
    result: ResidentWriteResult,
) -> bool:
    """Reuse a strict ACK already consumed by the main telemetry loop."""

    if getattr(core, "terminal_zero_acknowledged", False) is True:
        return True
    return wait_for_resident_terminal_zero_ack(serial_reader, result)


def resident_act_state_from_state(
    state: Stm32State,
    *,
    receive_monotonic_ns: int,
    control_generation: int,
    runtime_control_enabled: bool,
    runtime_estop: bool,
    sensor_valid: bool,
    stm32_alive: bool,
) -> ResidentActState:
    """Adapt unified telemetry to the canonical 11D resident ACT contract."""

    return ResidentActState(
        state=(
            state.s_boom,
            state.s_stick,
            state.s_bucket,
            state.v_boom,
            state.v_stick,
            state.v_bucket,
            state.a_boom,
            state.a_stick,
            state.a_bucket,
            state.yaw,
            state.yaw_rate,
        ),
        receive_monotonic_ns=receive_monotonic_ns,
        state_monotonic_ns=receive_monotonic_ns,
        control_seq=state.control_seq,
        sensor_seq=state.sensor_seq,
        sensor_is_new=state.sensor_is_new,
        control_enabled=(
            bool(runtime_control_enabled) and state.stm32_control_enabled
        ),
        estop=bool(runtime_estop) or state.stm32_estop,
        rs485_ok=state.rs485_ok,
        dwj_ok=state.adc_ok,
        imu_ok=state.imu_ok,
        sensor_valid=bool(sensor_valid),
        stm32_alive=bool(stm32_alive),
        fault_flags=state.stm32_fault_flags,
        control_generation=control_generation,
    )


def publish_resident_act_state_if_active(
    *,
    state: Stm32State,
    receive_monotonic_ns: int,
    core: ResidentMotionCore,
    data_link: ResidentActDataLink,
    runtime_control_enabled: bool,
    runtime_estop: bool,
    sensor_valid: bool,
    stm32_alive: bool,
) -> bool:
    """Publish every ACT-era telemetry frame, including 20 Hz safety updates.

    ``sensor_is_new`` remains part of the strict frame so the worker performs
    inference only for the 10 Hz observation stream.  Publishing the
    intervening telemetry rows is still required: control/fault evidence can
    change on those rows and must reset the resident policy queue immediately.
    """

    control_generation = core.active_act_generation
    if control_generation is None:
        return False
    frame = resident_act_state_from_state(
        state,
        receive_monotonic_ns=receive_monotonic_ns,
        control_generation=control_generation,
        runtime_control_enabled=runtime_control_enabled,
        runtime_estop=runtime_estop,
        sensor_valid=sensor_valid,
        stm32_alive=stm32_alive,
    )
    data_link.publish(encode_resident_state(frame))
    return True


def resident_machine_state_control_enabled(
    *,
    runtime_control_enabled: bool,
    core: ResidentMotionCore,
) -> bool:
    """Expose the active resident policy gate in ``machine_state_v1``.

    Both RL and ACT are valid motion owners.  Reporting only the RL phase as
    enabled made live state appear disabled throughout an otherwise healthy
    ACT segment.
    """

    return (
        bool(runtime_control_enabled)
        and core.mission_lease_is_active
        and (core.rl_is_active or core.act_is_active)
    )


def resident_rl_behavior_authorization(
    resident_action_transport: bool,
    core: ResidentMotionCore,
) -> Callable[[], bool] | None:
    """Bind remote RL behaviors to the generation selected by the Mission."""

    if not resident_action_transport:
        return None
    return lambda: core.mission_lease_is_active and core.rl_is_active


def resident_rl_behavior_idle_probe(
    executor_supplier: Callable[[], object | None],
) -> Callable[[], bool]:
    """Report whether a resident RL behavior has fully released its runner."""

    if not callable(executor_supplier):
        raise ValueError("executor_supplier must be callable")

    def is_idle() -> bool:
        executor = executor_supplier()
        if executor is None:
            return True
        return getattr(executor, "busy") is False

    return is_idle


def resident_act_activation_gate(
    executor_supplier: Callable[[], object | None],
) -> Callable[[Callable[[], object]], object]:
    """Atomically claim ACT only while the remote RL executor is idle."""

    if not callable(executor_supplier):
        raise ValueError("executor_supplier must be callable")

    def run_when_idle(operation: Callable[[], object]) -> object:
        executor = executor_supplier()
        if executor is None:
            raise RuntimeError("resident RL behavior executor is unavailable")
        gate = getattr(executor, "run_when_idle", None)
        if not callable(gate):
            raise RuntimeError("resident RL behavior executor gate is unavailable")
        return gate(operation)

    return run_when_idle


def resident_owner_tick_requests_exit(
    core: ResidentMotionCore | None,
    *,
    initialized: bool,
    now_monotonic_ns: int,
    release_allowed: bool = True,
) -> bool:
    """Run owner watchdogs independently of telemetry and detect terminal stop."""

    if core is None or not initialized:
        return False
    core.tick(now_monotonic_ns=now_monotonic_ns)
    if core.is_operational:
        return False
    if not release_allowed:
        return False
    LOGGER.warning(
        "resident motion core disarmed; exiting owner to release serial, "
        "ACT data link, and control sockets"
    )
    return True


def validate_state(
    state: Optional[Stm32State],
    last_receive_monotonic_s: Optional[float],
) -> tuple[bool, bool, List[str]]:
    fault_flags: List[str] = []

    if state is None:
        fault_flags.append("sensor_invalid")
        stm32_alive = (
            last_receive_monotonic_s is not None
            and time.monotonic() - last_receive_monotonic_s <= STM32_TIMEOUT_S
        )
        if not stm32_alive:
            fault_flags.append("stm32_timeout")
        return stm32_alive, False, fault_flags

    float_values = [
        state.x1,
        state.x2,
        state.y1,
        state.y2,
        state.s_boom,
        state.s_stick,
        state.s_bucket,
        state.v_boom,
        state.v_stick,
        state.v_bucket,
        state.a_boom,
        state.a_stick,
        state.a_bucket,
        state.yaw,
        state.yaw_rate,
    ]
    if not all(math.isfinite(value) for value in float_values):
        fault_flags.append("sensor_invalid")
    if not state.rs485_ok:
        fault_flags.append("rs485_invalid")
    if not state.adc_ok:
        fault_flags.append("adc_invalid")
    if not state.imu_ok:
        fault_flags.append("imu_invalid")

    stm32_alive = (
        last_receive_monotonic_s is not None
        and time.monotonic() - last_receive_monotonic_s <= STM32_TIMEOUT_S
    )
    if not stm32_alive:
        fault_flags.append("stm32_timeout")

    return stm32_alive, not fault_flags, fault_flags


def build_machine_state_packet(
    state: Stm32State,
    seq: int,
    machine_id: str,
    control_enabled: bool,
    estop: bool,
    include_raw: bool,
    last_receive_monotonic_s: Optional[float],
) -> Dict:
    stm32_alive, sensor_valid, fault_flags = validate_state(
        state,
        last_receive_monotonic_s,
    )
    merged_control_enabled = bool(control_enabled)
    merged_estop = bool(estop)
    if state.hardware_motion_gate_available:
        merged_control_enabled = (
            merged_control_enabled and state.stm32_control_enabled
        )
        merged_estop = merged_estop or state.stm32_estop
        if state.stm32_fault_flags:
            fault_flags.append(
                f"stm32_fault_flags:0x{state.stm32_fault_flags:08x}"
            )
            sensor_valid = False

    packet: Dict = {
        "type": "machine_state_v1",
        "schema_version": "1.0",
        "seq": seq,
        "stamp_ms": now_ms(),
        "source": "orin",
        "machine_id": machine_id,
        "stm32_stamp_ms": state.t_ms,
        "safety": {
            "estop": merged_estop,
            "stm32_alive": stm32_alive,
            "sensor_valid": sensor_valid,
            "control_enabled": merged_control_enabled,
            "fault_flags": fault_flags,
        },
        "actuator_state": {
            "boom": {
                "position_m": state.s_boom,
                "velocity_mps": state.v_boom,
            },
            "stick": {
                "position_m": state.s_stick,
                "velocity_mps": state.v_stick,
            },
            "bucket": {
                "position_m": state.s_bucket,
                "velocity_mps": state.v_bucket,
            },
            "swing": {
                "position_rad": state.yaw,
                "velocity_rad_s": state.yaw_rate,
            },
        },
        "joint_state": {
            "position_rad": {
                "swing": state.yaw,
                "boom": state.a_boom,
                "arm": state.a_stick,
                "bucket": state.a_bucket,
            },
            "velocity_rad_s": {
                "swing": state.yaw_rate,
                "boom": 0.0,
                "arm": 0.0,
                "bucket": 0.0,
            },
        },
    }

    if include_raw:
        packet["raw_sensor"] = asdict(state)

    return packet


def format_float(value: float) -> str:
    return f"{value:.9g}"


def format_policy_action_tx_log(sequence: object, command: DataCommand) -> str:
    command_time_ms = int(round(command.t_s * 1000.0))
    return (
        f"STM32 TX policy_action seq={sequence} t_ms:{command_time_ms} "
        f"boom:{format_float(command.boom_v_ref_mps)}m/s "
        f"stick:{format_float(command.stick_v_ref_mps)}m/s "
        f"bucket:{format_float(command.bucket_v_ref_mps)}m/s "
        f"swing:{format_float(command.swing_v_ref_radps)}rad/s"
    )


class Stm32VelocityCommandEncoder:
    """Encode the unified STM32 physical-velocity command contract."""

    def __init__(self) -> None:
        self._encoder = Stm32ResidentCommandEncoder()

    @property
    def next_sequence(self) -> int:
        return self._encoder.next_sequence

    def synchronize(self, *, command_rx_seq: int, command_received: bool) -> int:
        return self._encoder.synchronize(
            command_rx_seq=command_rx_seq,
            command_received=command_received,
        )

    def encode(self, command: DataCommand) -> bytes:
        values = (
            command.t_s,
            command.boom_v_ref_mps,
            command.stick_v_ref_mps,
            command.bucket_v_ref_mps,
            command.swing_v_ref_radps,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("STM32 velocity command values must be finite")
        if command.t_s < 0.0:
            raise ValueError("STM32 velocity command time must be non-negative")
        return self._encoder.encode(
            mode=ControlMode.VELOCITY_REFERENCE,
            action=(
                command.boom_v_ref_mps,
                command.stick_v_ref_mps,
                command.bucket_v_ref_mps,
                command.swing_v_ref_radps,
            ),
            monotonic_ns=int(command.t_s * 1_000_000_000),
        )


def zero_data_command(command_time_s: float) -> DataCommand:
    return DataCommand(
        t_s=command_time_s,
        boom_v_ref_mps=0.0,
        stick_v_ref_mps=0.0,
        bucket_v_ref_mps=0.0,
        swing_v_ref_radps=0.0,
    )


def is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def policy_action_to_data_command(
    packet: object,
    command_time_s: float,
    receive_wall_ms: int,
    control_enabled: bool,
    estop: bool,
    sensor_valid: bool,
    stm32_alive: bool,
) -> tuple[DataCommand, List[str]]:
    reject_reasons: List[str] = []
    action_by_name = {name: 0.0 for name in POLICY_ACTION_NAMES}

    if estop:
        reject_reasons.append("estop")
    if not control_enabled:
        reject_reasons.append("control_disabled")
    if not sensor_valid:
        reject_reasons.append("sensor_invalid")
    if not stm32_alive:
        reject_reasons.append("stm32_timeout")

    if not isinstance(packet, dict):
        reject_reasons.append("policy_action_not_object")
        return zero_data_command(command_time_s), reject_reasons

    if packet.get("type") != "policy_action":
        reject_reasons.append("bad_policy_action_type")

    if packet.get("schema_version") != "1.0":
        reject_reasons.append("bad_schema_version")

    packet_seq = packet.get("seq")
    if (
        isinstance(packet_seq, bool)
        or not isinstance(packet_seq, int)
        or packet_seq < 0
        or packet_seq > 0xFFFFFFFFFFFFFFFF
    ):
        reject_reasons.append("bad_policy_action_seq")

    if packet.get("action_type") != "normalized_velocity_command":
        reject_reasons.append("bad_action_type")

    stamp_ms = packet.get("stamp_ms")
    valid_for_ms = packet.get("valid_for_ms")
    if not is_finite_number(stamp_ms):
        reject_reasons.append("bad_policy_action_stamp_ms")
    if not is_finite_number(valid_for_ms) or float(valid_for_ms) < 0.0:
        reject_reasons.append("bad_policy_action_valid_for_ms")
    if is_finite_number(stamp_ms) and is_finite_number(valid_for_ms):
        if receive_wall_ms - int(round(float(stamp_ms))) > float(valid_for_ms):
            reject_reasons.append("policy_action_expired")
        if int(round(float(stamp_ms))) - receive_wall_ms > ACTION_FUTURE_SKEW_MS:
            reject_reasons.append("policy_action_from_future")

    action_order = packet.get("action_order")
    action = packet.get("action")

    action_order_valid = (
        isinstance(action_order, list)
        and len(action_order) == len(POLICY_ACTION_NAMES)
        and all(isinstance(name, str) for name in action_order)
        and set(action_order) == set(POLICY_ACTION_NAMES)
    )
    if not action_order_valid:
        reject_reasons.append("bad_action_order")

    action_valid = isinstance(action, list) and len(action) == len(POLICY_ACTION_NAMES)
    if not action_valid:
        reject_reasons.append("bad_action")

    if action_order_valid and action_valid:
        for name, value in zip(action_order, action):
            if not is_finite_number(value):
                reject_reasons.append(f"bad_action_value_{name}")
                continue

            action_by_name[name] = float(value)

    if reject_reasons:
        return zero_data_command(command_time_s), reject_reasons

    return (
        DataCommand(
            t_s=command_time_s,
            boom_v_ref_mps=action_by_name["boom"],
            stick_v_ref_mps=action_by_name["stick"],
            bucket_v_ref_mps=action_by_name["bucket"],
            swing_v_ref_radps=action_by_name["swing"],
        ),
        reject_reasons,
    )


def send_udp_json(sock: socket.socket, packet: Dict, host: str, port: int) -> None:
    payload = json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sock.sendto(payload, (host, port))


def open_action_socket(bind_host: str, bind_port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_host, bind_port))
    sock.setblocking(False)
    return sock


class ActionRelay:
    """Receive the newest PC action independently of the STM32 state cadence."""

    def __init__(
        self,
        action_sock: socket.socket,
        ser: object,
        allowed_action_host: str,
        control_enabled: bool,
        estop: bool,
        poll_interval_s: float = 0.01,
        command_encoder: Optional[Stm32VelocityCommandEncoder] = None,
    ) -> None:
        if not allowed_action_host:
            raise ValueError("allowed_action_host must be non-empty")
        if poll_interval_s <= 0.0:
            raise ValueError("poll_interval_s must be positive")
        self._action_sock = action_sock
        self._ser = ser
        self._allowed_action_host = socket.gethostbyname(allowed_action_host)
        self._control_enabled = bool(control_enabled)
        self._estop = bool(estop)
        self._poll_interval_s = float(poll_interval_s)
        self._command_encoder = command_encoder or Stm32VelocityCommandEncoder()
        self._command_encoder_lock = threading.Lock()
        self._command_sequence_synchronized = False
        self._safety_lock = threading.Lock()
        self._sensor_valid = False
        self._stm32_alive = False
        self._last_state_monotonic_s: Optional[float] = None
        self._stop = threading.Event()
        self._healthy = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="orin-policy-action-relay",
            daemon=False,
        )
        self._started = False
        self._last_source: Optional[tuple[str, int]] = None
        self._last_seq: Optional[int] = None
        self._active_deadline_monotonic_s: Optional[float] = None

    def update_safety(self, sensor_valid: bool, stm32_alive: bool) -> None:
        with self._safety_lock:
            self._sensor_valid = bool(sensor_valid)
            self._stm32_alive = bool(stm32_alive)
            self._last_state_monotonic_s = time.monotonic()

    def synchronize_command_sequence(self, state: Stm32State) -> int:
        with self._command_encoder_lock:
            if self._command_sequence_synchronized:
                return self._command_encoder.next_sequence
            next_sequence = self._command_encoder.synchronize(
                command_rx_seq=state.command_rx_seq,
                command_received=state.command_received,
            )
            self._command_sequence_synchronized = True
            return next_sequence

    def start(self) -> None:
        if self._started:
            raise RuntimeError("ActionRelay has already been started")
        self._started = True
        self._healthy.set()
        self._thread.start()

    @property
    def is_healthy(self) -> bool:
        return self._healthy.is_set() and self._thread.is_alive()

    def close(self) -> None:
        stop_error = None
        if self._started:
            self._stop.set()
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                stop_error = RuntimeError("ActionRelay did not stop within one second")
        try:
            self._write_zero()
        finally:
            self._healthy.clear()
        if stop_error is not None:
            raise stop_error

    def _safety_snapshot(self) -> tuple[bool, bool]:
        with self._safety_lock:
            age_s = (
                time.monotonic() - self._last_state_monotonic_s
                if self._last_state_monotonic_s is not None
                else math.inf
            )
            stm32_alive = self._stm32_alive and age_s <= STM32_TIMEOUT_S
            sensor_valid = self._sensor_valid and stm32_alive
        return sensor_valid, stm32_alive

    @staticmethod
    def _sequence_from_payload(payload: bytes) -> Optional[int]:
        try:
            packet = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(packet, dict):
            return None
        sequence = packet.get("seq")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            return None
        return sequence

    def _receive_latest(self) -> Optional[tuple[bytes, tuple[str, int]]]:
        candidates = []
        while True:
            try:
                payload, addr = self._action_sock.recvfrom(8192)
            except BlockingIOError:
                break
            if addr[0] != self._allowed_action_host:
                LOGGER.warning(
                    "drop policy_action from unexpected host %s; expected %s",
                    addr[0],
                    self._allowed_action_host,
                )
                continue
            candidates.append((payload, addr))
        if not candidates:
            return None
        sequenced = [
            (self._sequence_from_payload(payload), index, payload, addr)
            for index, (payload, addr) in enumerate(candidates)
        ]
        valid = [candidate for candidate in sequenced if candidate[0] is not None]
        if not valid:
            return candidates[-1]
        _, _, payload, addr = max(valid, key=lambda candidate: (candidate[0], candidate[1]))
        return payload, addr

    def _write_zero(self) -> None:
        encoded = self._encode_command(zero_data_command(time.monotonic()))
        self._ser.write(encoded)
        self._ser.flush()
        self._active_deadline_monotonic_s = None

    def _encode_command(self, command: DataCommand) -> bytes:
        with self._command_encoder_lock:
            return self._command_encoder.encode(command)

    def _apply_payload(self, payload: bytes, addr: tuple[str, int]) -> None:
        try:
            packet = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            LOGGER.warning("drop invalid policy_action from %s: %s", addr, exc)
            self._write_zero()
            return
        sequence = packet.get("seq") if isinstance(packet, dict) else None
        if (
            self._last_source == addr
            and self._last_seq is not None
            and isinstance(sequence, int)
            and not isinstance(sequence, bool)
            and sequence <= self._last_seq
        ):
            LOGGER.warning(
                "drop duplicate/out-of-order policy_action seq=%s from %s; last_seq=%s",
                sequence,
                addr,
                self._last_seq,
            )
            self._write_zero()
            return
        sensor_valid, stm32_alive = self._safety_snapshot()
        received_wall_ms = now_ms()
        command, reject_reasons = policy_action_to_data_command(
            packet=packet,
            command_time_s=time.monotonic(),
            receive_wall_ms=received_wall_ms,
            control_enabled=self._control_enabled,
            estop=self._estop,
            sensor_valid=sensor_valid,
            stm32_alive=stm32_alive,
        )
        command_is_nonzero = any(
            value != 0.0
            for value in (
                command.boom_v_ref_mps,
                command.stick_v_ref_mps,
                command.bucket_v_ref_mps,
                command.swing_v_ref_radps,
            )
        )
        if not reject_reasons and command_is_nonzero and self._active_deadline_monotonic_s is None:
            # Claim VELOCITY_REFERENCE through a zero command before motion.
            self._write_zero()
        encoded = self._encode_command(command)
        self._ser.write(encoded)
        self._ser.flush()
        if reject_reasons:
            self._active_deadline_monotonic_s = None
            LOGGER.warning(
                "policy_action rejected from %s: %s; STM32 TX safe zero: %s",
                addr,
                ",".join(reject_reasons),
                encoded.decode("ascii").strip(),
            )
            return
        self._last_source = addr
        self._last_seq = sequence
        action = packet["action"]
        remaining_lease_ms = max(
            0,
            int(packet["stamp_ms"]) + int(packet["valid_for_ms"]) - received_wall_ms,
        )
        self._active_deadline_monotonic_s = (
            time.monotonic() + remaining_lease_ms / 1000.0
            if any(float(value) != 0.0 for value in action)
            else None
        )
        LOGGER.info("%s", format_policy_action_tx_log(sequence, command))

    def _active_command_must_stop(self) -> bool:
        if self._active_deadline_monotonic_s is None:
            return False
        sensor_valid, stm32_alive = self._safety_snapshot()
        return (
            not sensor_valid
            or not stm32_alive
            or time.monotonic() >= self._active_deadline_monotonic_s
        )

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self._active_command_must_stop():
                    self._write_zero()
                readable, _, _ = select.select(
                    [self._action_sock], [], [], self._poll_interval_s
                )
                if not readable:
                    continue
                latest = self._receive_latest()
                if latest is None:
                    continue
                self._apply_payload(*latest)
        except Exception:
            LOGGER.exception("policy_action relay failed")
        finally:
            try:
                self._write_zero()
            except Exception:
                LOGGER.exception("policy_action relay could not write terminal zero")
            self._healthy.clear()


def parse_bool_arg(value: str) -> bool:
    text = value.strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("must be true or false")


def edge_control_authorized(value: str) -> bool:
    return value == EDGE_MOTION_AUTHORIZATION


def edge_mode_controls_motion(mode: str) -> bool:
    return mode in ("control", "remote_control")


def validate_trajectory_controller_commissioning(
    edge_config: object,
    authorization: str,
) -> None:
    """Require an independent opt-in for an unqualified powered controller."""
    if not edge_mode_controls_motion(getattr(edge_config, "mode", "")):
        return
    backend = getattr(
        edge_config,
        "trajectory_controller_backend",
        "onnx_rl",
    )
    if backend != "cartesian_p":
        return
    if authorization != CARTESIAN_P_COMMISSIONING_AUTHORIZATION:
        raise RuntimeError(
            "cartesian_p powered mode requires exact commissioning authorization: "
            + CARTESIAN_P_COMMISSIONING_AUTHORIZATION
        )


def format_resident_fixed_cycle_ready_line(
    args: argparse.Namespace,
    edge_config: object,
    plan: object | None = None,
) -> str:
    backend = getattr(edge_config, "trajectory_controller_backend", None)
    if backend not in {"onnx_rl", "cartesian_p"}:
        raise ValueError("resident fixed cycle controller backend is invalid")
    mission_fields = ""
    if plan is not None:
        mission = getattr(plan, "mission", None)
        if mission is None:
            raise ValueError("resident fixed cycle Mission is missing")
        behavior_id = mission.act_worker_behavior_id or "none"
        model_sha256 = mission.act_worker_model_sha256 or "none"
        mission_fields = (
            f" mission_id={mission.mission_id}"
            f" mission_sha256={plan.mission_sha256}"
            f" act_worker_required={str(mission.requires_act_worker).lower()}"
            f" act_worker_behavior_id={behavior_id}"
            f" act_worker_model_sha256={model_sha256}"
        )
    return (
        "RESIDENT_FIXED_CYCLE_READY "
        f"control_socket={args.resident_fixed_cycle_control_socket} "
        f"act_socket={args.resident_act_socket} "
        f"trajectory_controller_backend={backend}"
        f"{mission_fields}"
    )


def validate_resident_fixed_cycle_backend(
    plan: object,
    edge_config: object,
) -> None:
    mission = getattr(plan, "mission", None)
    configured = getattr(edge_config, "trajectory_controller_backend", None)
    if mission is None or configured != mission.trajectory_controller_backend:
        raise RuntimeError(
            "resident Mission trajectory controller does not match edge config"
        )


def validate_resident_motion_request(args: argparse.Namespace, edge_config: object) -> None:
    if not args.resident_motion_core:
        return
    if edge_config is None:
        raise ValueError("resident motion core requires an edge config")
    if not edge_mode_controls_motion(getattr(edge_config, "mode", "")):
        raise ValueError("resident motion core requires an edge motion mode")
    if args.input_format != "csv":
        raise ValueError("resident motion core requires unified STM32 CSV telemetry")
    if not args.control_enabled:
        raise ValueError("resident motion core requires --control-enabled")
    audit_path = getattr(
        args,
        "resident_action_audit_path",
        DEFAULT_RESIDENT_ACTION_AUDIT_PATH,
    )
    if not isinstance(audit_path, Path) or not audit_path.is_absolute():
        raise ValueError("resident action audit path must be absolute")
    if (
        getattr(edge_config, "action_transport", DEFAULT_EDGE_ACTION_TRANSPORT)
        != RESIDENT_EDGE_ACTION_TRANSPORT
    ):
        raise ValueError(
            "resident motion core requires edge action_transport=resident_sink"
        )


def validate_resident_fixed_cycle_request(
    args: argparse.Namespace,
    edge_config: object | None,
) -> None:
    """Reject partial V3-A activation before model, serial, or sockets open."""

    commissioning_authorization = getattr(
        args,
        "resident_fixed_cycle_commissioning_authorization",
        "",
    )
    if (
        commissioning_authorization
        and commissioning_authorization
        != V3A_FIXED_TRAJECTORY_COMMISSIONING_AUTHORIZATION
    ):
        raise RuntimeError(
            "resident fixed cycle candidate requires exact commissioning "
            "authorization: "
            + V3A_FIXED_TRAJECTORY_COMMISSIONING_AUTHORIZATION
        )
    if getattr(args, "resident_fixed_cycle_plan", None) is None:
        if commissioning_authorization:
            raise ValueError(
                "fixed cycle commissioning authorization requires a plan"
            )
        return
    if not args.resident_motion_core:
        raise ValueError("resident fixed cycle requires --resident-motion-core")
    if edge_config is None or getattr(edge_config, "mode", None) != "remote_control":
        raise ValueError("resident fixed cycle requires edge mode=remote_control")
    if not edge_uses_resident_action_transport(edge_config):
        raise ValueError("resident fixed cycle requires action_transport=resident_sink")
    if not args.resident_fixed_cycle_control_socket.is_absolute():
        raise ValueError("resident fixed cycle control socket must be absolute")


def validate_resident_fixed_cycle_catalog_digest(
    plan: object,
    expected_digest: str,
) -> None:
    """Reject a stale PC/Orin point catalog before hardware acquisition."""

    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise RuntimeError("expected dig catalog must be a lowercase SHA-256")
    if getattr(plan, "source_catalog_sha256", None) != expected_digest:
        raise RuntimeError("PC dig catalog does not match the deployed Orin catalog")


def edge_uses_resident_action_transport(edge_config: object | None) -> bool:
    if edge_config is None:
        return False
    return edge_mode_controls_motion(getattr(edge_config, "mode", "")) and (
        getattr(edge_config, "action_transport", DEFAULT_EDGE_ACTION_TRANSPORT)
        == RESIDENT_EDGE_ACTION_TRANSPORT
    )


def _raise_keyboard_interrupt_on_termination(_signum, _frame) -> None:
    """Route process-manager SIGTERM through the existing terminal-zero cleanup."""
    raise KeyboardInterrupt


def wait_for_hardware_start_gate(
    path: str | Path,
    *,
    poll_interval_s: float = 0.05,
) -> None:
    """Finish model loading, then wait without opening the STM32 serial port.

    The gate is a one-shot file: the process that creates it transfers hardware
    ownership only after the previous Runtime has written terminal zero and
    proved that ``/dev/ttyTHS1`` is released.
    """

    gate = Path(path)
    if not gate.is_absolute():
        raise ValueError("hardware start gate must be an absolute path")
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive and finite")
    LOGGER.info("RL prewarm ready: waiting for hardware start gate: %s", gate)
    while True:
        try:
            gate.unlink()
        except FileNotFoundError:
            time.sleep(poll_interval_s)
        else:
            LOGGER.info("RL hardware start gate accepted: %s", gate)
            return


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge unified STM32 state to PC machine_state_v1 UDP JSON.",
    )
    parser.add_argument(
        "--serial-port",
        default=os.getenv("STM32_SERIAL_PORT", DEFAULT_SERIAL_PORT),
        help=f"STM32 serial port, default: {DEFAULT_SERIAL_PORT}",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=int(os.getenv("STM32_BAUDRATE", str(DEFAULT_BAUDRATE))),
        help=f"STM32 baudrate, default: {DEFAULT_BAUDRATE}",
    )
    parser.add_argument(
        "--pc-host",
        default=os.getenv("PC_STATE_HOST", DEFAULT_PC_STATE_HOST),
        help=f"PC UDP host, default: {DEFAULT_PC_STATE_HOST}",
    )
    parser.add_argument(
        "--pc-port",
        type=int,
        default=int(os.getenv("PC_STATE_PORT", str(DEFAULT_PC_STATE_PORT))),
        help=f"PC UDP port, default: {DEFAULT_PC_STATE_PORT}",
    )
    parser.add_argument(
        "--action-bind-host",
        default=os.getenv("ORIN_ACTION_BIND_HOST", DEFAULT_ORIN_ACTION_BIND_HOST),
        help=f"Orin UDP bind host for PC policy_action packets, default: {DEFAULT_ORIN_ACTION_BIND_HOST}",
    )
    parser.add_argument(
        "--action-bind-port",
        type=int,
        default=int(os.getenv("ORIN_ACTION_BIND_PORT", str(DEFAULT_ORIN_ACTION_BIND_PORT))),
        help=f"Orin UDP bind port for PC policy_action packets, default: {DEFAULT_ORIN_ACTION_BIND_PORT}",
    )
    parser.add_argument(
        "--allowed-action-host",
        default=os.getenv("ORIN_ALLOWED_ACTION_HOST"),
        help=(
            "Only accept policy_action from this host. Defaults to --pc-host; "
            "use 127.0.0.1 with a loopback-only bind for local CSV replay."
        ),
    )
    parser.add_argument(
        "--machine-id",
        default=os.getenv("MACHINE_ID", DEFAULT_MACHINE_ID),
        help=f"machine_id field, default: {DEFAULT_MACHINE_ID}",
    )
    parser.add_argument(
        "--control-enabled",
        nargs="?",
        const="true",
        default=os.getenv("ORIN_CONTROL_ENABLED", "false"),
        type=parse_bool_arg,
        metavar="{true,false}",
        help="Set safety.control_enabled. Use --control-enabled, --control-enabled true, or --control-enabled false.",
    )
    parser.add_argument(
        "--estop",
        action="store_true",
        help="Set safety.estop=true.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw STM32 fields under raw_sensor for debugging.",
    )
    parser.add_argument(
        "--input-format",
        choices=("csv", "binary"),
        default=os.getenv("STM32_INPUT_FORMAT", "csv"),
        help="STM32 serial input format. Default: csv. Use binary for legacy <I15f frames.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print one status line every N sent frames. Use 0 to disable.",
    )
    parser.add_argument(
        "--edge-config",
        type=Path,
        help=(
            "Run local FK/38D/ONNX Follow using one orin_edge_runtime.v1 "
            "shadow, static control, or remote_control config."
        ),
    )
    parser.add_argument(
        "--edge-motion-authorization",
        default="",
        help=(
            "Exact token required when edge config mode=control or "
            "remote_control: "
            + EDGE_MOTION_AUTHORIZATION
        ),
    )
    parser.add_argument(
        "--trajectory-controller-commissioning-authorization",
        default="",
        help=(
            "Independent exact authorization required when cartesian_p is "
            "selected in control or remote_control: "
            + CARTESIAN_P_COMMISSIONING_AUTHORIZATION
        ),
    )
    parser.add_argument(
        "--hardware-start-gate",
        type=Path,
        help=(
            "After loading edge assets and ONNX, wait for this absolute one-shot "
            "file before opening the STM32 serial port."
        ),
    )
    parser.add_argument(
        "--resident-motion-core",
        action="store_true",
        help=(
            "Use the experimental single-owner resident RL/ACT command core. "
            "Requires a motion edge config and unified STM32 CSV telemetry."
        ),
    )
    parser.add_argument(
        "--resident-act-socket",
        type=Path,
        default=Path(
            os.getenv("RESIDENT_ACT_SOCKET", str(DEFAULT_RESIDENT_ACT_SOCKET))
        ),
        help="Absolute Unix socket shared with the resident ACT worker.",
    )
    parser.add_argument(
        "--resident-control-socket",
        type=Path,
        default=Path(
            os.getenv(
                "RESIDENT_CONTROL_SOCKET",
                str(DEFAULT_RESIDENT_CONTROL_SOCKET),
            )
        ),
        help="Absolute Unix socket for low-rate resident policy handoff commands.",
    )
    parser.add_argument(
        "--resident-action-audit-path",
        type=Path,
        default=Path(
            os.getenv(
                "RESIDENT_ACTION_AUDIT_PATH",
                str(DEFAULT_RESIDENT_ACTION_AUDIT_PATH),
            )
        ),
        help="Absolute JSONL path for non-blocking resident action evidence.",
    )
    parser.add_argument(
        "--resident-fixed-cycle-plan",
        type=Path,
        help=(
            "Enable V3-B Orin-local declarative Mission execution with this "
            "strict resident_fixed_cycle_plan.v5 artifact."
        ),
    )
    parser.add_argument(
        "--resident-fixed-cycle-expected-dig-catalog-sha256",
        default="",
        help="PC source Dig Point Catalog SHA-256 used for stale-deployment rejection.",
    )
    parser.add_argument(
        "--resident-fixed-cycle-control-socket",
        type=Path,
        default=Path(
            os.getenv(
                "RESIDENT_FIXED_CYCLE_CONTROL_SOCKET",
                str(DEFAULT_RESIDENT_FIXED_CYCLE_CONTROL_SOCKET),
            )
        ),
        help="Absolute Unix socket for V3-A start/status/cancel commands.",
    )
    parser.add_argument(
        "--resident-fixed-cycle-commissioning-authorization",
        default="",
        help=(
            "Independent exact acknowledgement required to load candidate "
            "V3-A trajectories. Normal operation accepts field_validated "
            "artifacts only."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    allowed_action_host = args.allowed_action_host or args.pc_host
    edge_config = None
    edge_runtime = None
    remote_runtime_factory = None
    if args.edge_config is not None:
        from edge_runtime.shadow import (
            build_edge_follow_runtime,
            load_edge_runtime_config,
        )

        edge_config = load_edge_runtime_config(args.edge_config)
        if edge_mode_controls_motion(edge_config.mode):
            if not args.control_enabled:
                raise RuntimeError("edge control requires --control-enabled")
            if not edge_control_authorized(args.edge_motion_authorization):
                raise RuntimeError(
                    "edge control requires --edge-motion-authorization "
                    + EDGE_MOTION_AUTHORIZATION
                )
            validate_trajectory_controller_commissioning(
                edge_config,
                getattr(
                    args,
                    "trajectory_controller_commissioning_authorization",
                    "",
                ),
            )
            args.action_bind_host = "127.0.0.1"
            allowed_action_host = "127.0.0.1"
        # Validate copied assets and load ONNX before taking serial ownership.
        if edge_config.mode == "remote_control":
            from edge_runtime.fixed_actions import FixedActionRuntimeFactory
            from edge_runtime.remote import EdgeFollowRuntimeFactory

            remote_runtime_factory = EdgeFollowRuntimeFactory.from_config(
                edge_config
            )
            fixed_action_runtime_factory = FixedActionRuntimeFactory.from_config(
                edge_config
            )
        else:
            edge_runtime = build_edge_follow_runtime(edge_config)
    validate_resident_motion_request(args, edge_config)
    validate_resident_fixed_cycle_request(args, edge_config)
    resident_fixed_cycle_plan = None
    resident_fixed_cycle_registry = None
    fixed_cycle_plan_path = getattr(args, "resident_fixed_cycle_plan", None)
    if fixed_cycle_plan_path is not None:
        allow_candidate = (
            args.resident_fixed_cycle_commissioning_authorization
            == V3A_FIXED_TRAJECTORY_COMMISSIONING_AUTHORIZATION
        )
        resident_fixed_cycle_plan = load_fixed_cycle_plan(
            fixed_cycle_plan_path,
            allow_candidate=allow_candidate,
        )
        expected_catalog_digest = (
            args.resident_fixed_cycle_expected_dig_catalog_sha256
        )
        if expected_catalog_digest:
            validate_resident_fixed_cycle_catalog_digest(
                resident_fixed_cycle_plan,
                expected_catalog_digest,
            )
        resident_fixed_cycle_registry = load_fixed_cycle_registry(
            resident_fixed_cycle_plan,
            allow_candidate=allow_candidate,
        )
        validate_resident_fixed_cycle_backend(
            resident_fixed_cycle_plan,
            edge_config,
        )
        if resident_fixed_cycle_plan.validation_status == "candidate":
            LOGGER.warning(
                "V3-A FIXED TRAJECTORY COMMISSIONING: candidate plan=%s",
                resident_fixed_cycle_plan.plan_id,
            )

    if args.hardware_start_gate is not None:
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt_on_termination)
        try:
            wait_for_hardware_start_gate(args.hardware_start_gate)
        except KeyboardInterrupt:
            LOGGER.info("RL prewarm cancelled before hardware acquisition")
            return

    resident_action_transport = (
        args.resident_motion_core
        and edge_uses_resident_action_transport(edge_config)
    )
    resident_manual_action_deadzone_contract = None
    if resident_action_transport:
        if edge_config is None:
            raise RuntimeError("resident motion core requires an edge config")
        resident_manual_action_deadzone_contract = (
            load_resident_manual_action_deadzone_contract(
                getattr(edge_config, "manual_action_deadzone_path", None)
            )
        )
        if resident_manual_action_deadzone_contract is None:
            LOGGER.warning(
                "resident ACT deadzone contract unavailable in %s; early completion disabled",
                getattr(edge_config, "manual_action_deadzone_path", None),
            )
        LOGGER.info(
            "opening serial %s @ %d, sending UDP JSON to %s:%d, "
            "using resident action sink for %s; deadzone_contract_enabled=%s",
            args.serial_port,
            args.baudrate,
            args.pc_host,
            args.pc_port,
            edge_config.mode,
            resident_manual_action_deadzone_contract is not None,
        )
    else:
        LOGGER.info(
            "opening serial %s @ %d, sending UDP JSON to %s:%d, listening policy_action on %s:%d from %s",
            args.serial_port,
            args.baudrate,
            args.pc_host,
            args.pc_port,
            args.action_bind_host,
            args.action_bind_port,
            allowed_action_host,
        )

    ser = open_serial(args.serial_port, args.baudrate, timeout_s=STM32_TIMEOUT_S)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    action_sock = None
    action_relay = None
    resident_motion_core = None
    resident_data_link = None
    resident_control_server = None
    resident_fixed_cycle_runtime = None
    resident_fixed_cycle_control_server = None
    resident_action_audit = None
    resident_initialized = False
    resident_hardware_ready_logged = False
    behavior_executor = None
    if resident_action_transport:
        resident_action_audit = AsyncResidentActionAudit(
            args.resident_action_audit_path
        )
        LOGGER.info(
            "resident action audit enabled: path=%s",
            args.resident_action_audit_path,
        )
        resident_motion_core = ResidentMotionCore(
            ser,
            max_state_age_ms=DEFAULT_RESIDENT_STATE_AGE_MS,
            manual_action_deadzone_contract=resident_manual_action_deadzone_contract,
            action_audit=resident_action_audit,
        )
        resident_data_link = ResidentActDataLink(
            args.resident_act_socket,
            on_candidate=resident_motion_core.submit_act,
            on_connection_lost=(
                resident_motion_core.notify_act_worker_disconnected
            ),
            expected_worker_identity=(
                None
                if resident_fixed_cycle_plan is None
                or not resident_fixed_cycle_plan.mission.requires_act_worker
                else ResidentActWorkerIdentity(
                    behavior_id=(
                        resident_fixed_cycle_plan.mission.act_worker_behavior_id
                    ),
                    checkpoint_model_sha256=(
                        resident_fixed_cycle_plan.mission.act_worker_model_sha256
                    ),
                )
            ),
        )
        if resident_fixed_cycle_plan is None:
            resident_control_server = ResidentMotionControlServer(
                resident_motion_core,
                socket_path=str(args.resident_control_socket),
                act_worker_ready=lambda: resident_data_link.connected,
                rl_behavior_idle=resident_rl_behavior_idle_probe(
                    lambda: behavior_executor
                ),
                activate_act_while_rl_idle=resident_act_activation_gate(
                    lambda: behavior_executor
                ),
            )
    else:
        action_sock = open_action_socket(args.action_bind_host, args.action_bind_port)
        action_relay = ActionRelay(
            action_sock=action_sock,
            ser=ser,
            allowed_action_host=allowed_action_host,
            control_enabled=args.control_enabled,
            estop=args.estop,
        )
        action_relay.start()
    edge_runner = None
    edge_action_sender = None
    remote_executor = None
    remote_server = None
    if edge_config is not None:
        from edge_runtime.shadow import EdgeShadowObserver

        if edge_config.mode == "shadow":
            edge_runner = EdgeShadowObserver(
                runtime=edge_runtime,
                audit_path=edge_config.audit_path,
            )
            LOGGER.info(
                "edge inference shadow enabled: config=%s (audit only, no STM32 action sink)",
                args.edge_config,
            )
        elif edge_config.mode == "control":
            from edge_runtime.control import EdgeControlRunner

            if resident_action_transport:
                edge_action_sink = resident_motion_core.rl_action_sink
            else:
                edge_action_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                edge_action_sender.connect(("127.0.0.1", args.action_bind_port))
                edge_action_sink = edge_action_sender
            edge_runner = EdgeControlRunner(
                runtime=edge_runtime,
                action_sink=edge_action_sink,
                audit_path=edge_config.audit_path,
                valid_for_ms=edge_config.action_valid_for_ms,
                retain_action_authority=resident_action_transport,
            )
            if resident_action_transport:
                LOGGER.warning(
                    "EDGE CONTROL ARMED: local inference owns the resident STM32 command sink"
                )
            else:
                LOGGER.warning(
                    "EDGE CONTROL ARMED: local inference owns loopback action source on 127.0.0.1:%d",
                    args.action_bind_port,
                )
        else:
            from edge_runtime.control import (
                ActionSequence,
                EdgeControlRunner,
                FixedActionControlRunner,
            )
            from edge_runtime.cycle import EdgeCycleCoordinator
            from edge_runtime.remote import EdgeBehaviorExecutor, RemoteBehaviorServer

            if not resident_action_transport:
                edge_action_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                edge_action_sender.connect(("127.0.0.1", args.action_bind_port))
                resident_rl_action_sink = None
            else:
                resident_rl_action_sink = resident_motion_core.rl_action_sink
            edge_action_sequence = ActionSequence()

            def build_follow_sink():
                if resident_action_transport:
                    return resident_rl_action_sink
                return edge_action_sender

            def build_fixed_sink(_behavior: str):
                if resident_action_transport:
                    return resident_rl_action_sink
                return edge_action_sender

            def build_remote_runner(runtime):
                return EdgeControlRunner(
                    runtime=runtime,
                    action_sink=build_follow_sink(),
                    audit_path=edge_config.audit_path,
                    valid_for_ms=edge_config.action_valid_for_ms,
                    action_sequence=edge_action_sequence,
                    retain_action_authority=resident_action_transport,
                )

            def build_fixed_action_runner(runtime, behavior):
                return FixedActionControlRunner(
                    runtime=runtime,
                    behavior=behavior,
                    action_sink=build_fixed_sink(behavior),
                    audit_path=edge_config.audit_path,
                    valid_for_ms=edge_config.action_valid_for_ms,
                    action_sequence=edge_action_sequence,
                    retain_action_authority=resident_action_transport,
                )

            remote = edge_config.remote_behavior
            behavior_executor = EdgeBehaviorExecutor(
                runtime_factory=remote_runtime_factory,
                runner_factory=build_remote_runner,
                fixed_action_factory=fixed_action_runtime_factory,
                fixed_runner_factory=build_fixed_action_runner,
                sender_constructed=True,
                state_timeout_s=remote.status_timeout_s,
                resident_rl_authorized=resident_rl_behavior_authorization(
                    resident_action_transport,
                    resident_motion_core,
                ),
            )
            if resident_fixed_cycle_plan is not None:
                resident_fixed_cycle_runtime = ResidentFixedCycleRuntime(
                    plan=resident_fixed_cycle_plan,
                    registry=resident_fixed_cycle_registry,
                    core=resident_motion_core,
                    behavior_executor=behavior_executor,
                    act_worker_ready=lambda: resident_data_link.connected,
                )
                resident_fixed_cycle_control_server = (
                    ResidentFixedCycleControlServer(
                        resident_fixed_cycle_runtime,
                        socket_path=str(
                            args.resident_fixed_cycle_control_socket
                        ),
                    )
                )
                remote_executor = behavior_executor
                LOGGER.warning(
                    "V3-A RESIDENT FIXED CYCLE ARMED IDLE: plan=%s; "
                    "external behavior RPC disabled",
                    resident_fixed_cycle_plan.plan_id,
                )
            else:
                remote_executor = EdgeCycleCoordinator(behavior_executor)
                remote_server = RemoteBehaviorServer(
                    bind_host=remote.bind_host,
                    bind_port=remote.bind_port,
                    allowed_client_host=remote.allowed_client_host,
                    executor=remote_executor,
                    status_interval_s=1.0 / remote.status_hz,
                )
                try:
                    remote_server.start()
                except Exception:
                    if edge_action_sender is not None:
                        edge_action_sender.close()
                    if action_relay is not None:
                        action_relay.close()
                    if action_sock is not None:
                        action_sock.close()
                    sock.close()
                    ser.close()
                    raise
            if resident_action_transport and resident_fixed_cycle_plan is None:
                LOGGER.warning(
                    "REMOTE EDGE CONTROL ARMED IDLE: behavior RPC %s:%d from %s; "
                    "actions use the resident STM32 command sink",
                    remote.bind_host,
                    remote.bind_port,
                    remote.allowed_client_host,
                )
            else:
                LOGGER.warning(
                    "REMOTE EDGE CONTROL ARMED IDLE: behavior RPC %s:%d from %s; "
                    "actions remain loopback-only on 127.0.0.1:%d",
                    remote.bind_host,
                    remote.bind_port,
                    remote.allowed_client_host,
                    args.action_bind_port,
                )

    seq = 0
    last_receive_monotonic_s: Optional[float] = None
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt_on_termination)

    try:
        if resident_data_link is not None:
            resident_data_link.start()
        if resident_control_server is not None:
            resident_control_server.start()
        if resident_fixed_cycle_control_server is not None:
            resident_fixed_cycle_control_server.start()
        if resident_action_transport:
            if resident_fixed_cycle_control_server is not None:
                LOGGER.info(
                    "%s",
                    format_resident_fixed_cycle_ready_line(
                        args,
                        edge_config,
                        resident_fixed_cycle_plan,
                    ),
                )
            else:
                LOGGER.info(
                    "RESIDENT_CONTROL_READY control_socket=%s act_socket=%s",
                    args.resident_control_socket,
                    args.resident_act_socket,
                )
        while True:
            if resident_owner_tick_requests_exit(
                resident_motion_core,
                initialized=resident_initialized,
                now_monotonic_ns=time.monotonic_ns(),
                release_allowed=(
                    resident_fixed_cycle_runtime is None
                    or resident_fixed_cycle_runtime.owner_release_ready
                ),
            ):
                break
            if args.input_format == "binary":
                frame = read_exact(ser, STM32_FRAME_SIZE)
                if frame is None:
                    LOGGER.warning("STM32 serial timeout, no complete %d-byte frame", STM32_FRAME_SIZE)
                    if action_relay is not None:
                        action_relay.update_safety(sensor_valid=False, stm32_alive=False)
                    elif resident_motion_core is not None and resident_initialized:
                        resident_motion_core.invalidate_telemetry(
                            receive_monotonic_ns=time.monotonic_ns(),
                            stm32_alive=False,
                        )
                    continue

                try:
                    state = parse_stm32_frame(frame)
                except ValueError as exc:
                    LOGGER.warning("drop invalid STM32 frame: %s", exc)
                    if action_relay is not None:
                        action_relay.update_safety(sensor_valid=False, stm32_alive=True)
                    elif resident_motion_core is not None and resident_initialized:
                        resident_motion_core.invalidate_telemetry(
                            receive_monotonic_ns=time.monotonic_ns(),
                            stm32_alive=True,
                        )
                    continue
            else:
                raw_line = ser.readline()
                if not raw_line:
                    LOGGER.warning("STM32 serial timeout, no CSV line")
                    if action_relay is not None:
                        action_relay.update_safety(sensor_valid=False, stm32_alive=False)
                    elif resident_motion_core is not None and resident_initialized:
                        resident_motion_core.invalidate_telemetry(
                            receive_monotonic_ns=time.monotonic_ns(),
                            stm32_alive=False,
                        )
                    continue

                try:
                    state = parse_stm32_csv_line(raw_line.decode("utf-8", errors="replace"))
                except ValueError as exc:
                    LOGGER.warning("drop invalid STM32 CSV line: %s", exc)
                    if action_relay is not None:
                        action_relay.update_safety(sensor_valid=False, stm32_alive=True)
                    elif resident_motion_core is not None and resident_initialized:
                        resident_motion_core.invalidate_telemetry(
                            receive_monotonic_ns=time.monotonic_ns(),
                            stm32_alive=True,
                        )
                    continue

                if state is None:
                    if action_relay is not None:
                        action_relay.update_safety(sensor_valid=False, stm32_alive=True)
                    elif resident_motion_core is not None and resident_initialized:
                        resident_motion_core.invalidate_telemetry(
                            receive_monotonic_ns=time.monotonic_ns(),
                            stm32_alive=True,
                        )
                    continue

            last_receive_monotonic_s = time.monotonic()
            stm32_alive, sensor_valid, _ = validate_state(
                state,
                last_receive_monotonic_s,
            )
            if resident_motion_core is not None:
                receive_monotonic_ns = time.monotonic_ns()
                resident_frame = resident_telemetry_from_state(
                    state,
                    receive_monotonic_ns=receive_monotonic_ns,
                    runtime_control_enabled=args.control_enabled,
                    runtime_estop=args.estop,
                    sensor_valid=sensor_valid,
                    stm32_alive=stm32_alive,
                )
                if not resident_initialized:
                    resident_motion_core.initialize(resident_frame)
                    if resident_fixed_cycle_runtime is None:
                        resident_motion_core.activate_rl(
                            now_monotonic_ns=resident_frame.receive_monotonic_ns
                        )
                        resident_motion_core.renew_mission_lease(
                            lease_ms=DEFAULT_MISSION_LEASE_MS,
                            now_monotonic_ns=resident_frame.receive_monotonic_ns,
                        )
                    resident_initialized = True
                else:
                    resident_motion_core.observe_telemetry(resident_frame)
                if sensor_valid and not resident_hardware_ready_logged:
                    LOGGER.info("RESIDENT_HARDWARE_READY sensor_valid=True")
                    resident_hardware_ready_logged = True
                publish_resident_act_state_if_active(
                    state=state,
                    receive_monotonic_ns=receive_monotonic_ns,
                    core=resident_motion_core,
                    data_link=resident_data_link,
                    runtime_control_enabled=args.control_enabled,
                    runtime_estop=args.estop,
                    sensor_valid=sensor_valid,
                    stm32_alive=stm32_alive,
                )
            else:
                action_relay.synchronize_command_sequence(state)
            packet = build_machine_state_packet(
                state=state,
                seq=seq,
                machine_id=args.machine_id,
                control_enabled=(
                    resident_machine_state_control_enabled(
                        runtime_control_enabled=args.control_enabled,
                        core=resident_motion_core,
                    )
                    if resident_motion_core is not None
                    else args.control_enabled and action_relay.is_healthy
                ),
                estop=args.estop,
                include_raw=args.include_raw,
                last_receive_monotonic_s=last_receive_monotonic_s,
            )
            if action_relay is not None:
                action_relay.update_safety(
                    sensor_valid=packet["safety"]["sensor_valid"],
                    stm32_alive=packet["safety"]["stm32_alive"],
                )
            send_udp_json(sock, packet, args.pc_host, args.pc_port)
            if edge_runner is not None:
                if edge_config.mode == "shadow":
                    edge_runner.observe(packet)
                elif (
                    resident_motion_core is None
                    or resident_motion_core.rl_is_active
                ):
                    edge_runner.observe(
                        packet,
                        now_s=time.monotonic(),
                        action_stamp_ms=now_ms(),
                    )
            if remote_executor is not None:
                remote_executor.observe(packet)
            if resident_fixed_cycle_runtime is not None:
                resident_fixed_cycle_runtime.tick()

            if args.print_every and seq % args.print_every == 0:
                LOGGER.info(
                    "sent seq=%d stm32_t=%d sensor_valid=%s boom=%.6f stick=%.6f bucket=%.6f yaw=%.6f",
                    seq,
                    state.t_ms,
                    packet["safety"]["sensor_valid"],
                    state.s_boom,
                    state.s_stick,
                    state.s_bucket,
                    state.yaw,
                )

            seq = (seq + 1) & 0xFFFFFFFF

    except KeyboardInterrupt:
        LOGGER.info("stopped by user")
    finally:
        try:
            try:
                if remote_server is not None:
                    remote_server.close()
            finally:
                try:
                    if edge_config is not None and edge_config.mode == "control":
                        edge_runner.close(action_stamp_ms=now_ms())
                    elif edge_runner is not None:
                        edge_runner.close()
                finally:
                    if action_relay is not None:
                        action_relay.close()
                    elif resident_motion_core is not None and resident_initialized:
                        terminal_result = resident_motion_core.terminal_disarm(
                            now_monotonic_ns=time.monotonic_ns()
                        )
                        if resident_terminal_zero_is_acknowledged(
                            resident_motion_core,
                            ser,
                            terminal_result,
                        ):
                            LOGGER.info(
                                "resident terminal zero acknowledged: command_seq=%s mode=%s",
                                terminal_result.command_seq,
                                (
                                    None
                                    if terminal_result.mode is None
                                    else terminal_result.mode.value
                                ),
                            )
                        else:
                            LOGGER.warning(
                                "resident terminal zero was not acknowledged before serial close: "
                                "command_seq=%s mode=%s",
                                terminal_result.command_seq,
                                (
                                    None
                                    if terminal_result.mode is None
                                    else terminal_result.mode.value
                                ),
                            )
                            raise RuntimeError(
                                "resident terminal zero was not acknowledged"
                            )
        finally:
            if resident_fixed_cycle_control_server is not None:
                resident_fixed_cycle_control_server.close()
            if resident_control_server is not None:
                resident_control_server.close()
            if resident_data_link is not None:
                resident_data_link.close()
            if resident_action_audit is not None:
                resident_action_audit.close()
            if edge_action_sender is not None:
                edge_action_sender.close()
            if action_sock is not None:
                action_sock.close()
            sock.close()
            ser.close()
if __name__ == "__main__":
    main()
