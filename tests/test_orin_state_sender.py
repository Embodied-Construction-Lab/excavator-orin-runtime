import tempfile
import threading
import time
import unittest
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import orin_state_sender as orin
from edge_runtime.follow import EdgeFollowStep
from edge_runtime.resident_motion import ControlMode
from edge_runtime.resident_sink import ResidentWriteResult


LIVE_ROW = (
    "2560404,0.000,0.000,0.000,0.000,"
    "99.950,221.950,161.580,"
    "0.000,0.000,0.000,"
    "23.300,71.530,111.600,35.290,0.076,"
    "1,1,1"
)


class Stm32CsvSafetyFlagsTest(unittest.TestCase):
    def test_resident_rl_behavior_probe_tracks_the_mission_authority(self):
        core = SimpleNamespace(
            rl_is_active=False,
            mission_lease_is_active=False,
        )

        probe = orin.resident_rl_behavior_authorization(True, core)

        self.assertIsNotNone(probe)
        self.assertFalse(probe())
        core.rl_is_active = True
        self.assertFalse(probe())
        core.mission_lease_is_active = True
        self.assertTrue(probe())
        core.rl_is_active = False
        self.assertFalse(probe())
        self.assertIsNone(orin.resident_rl_behavior_authorization(False, core))

    def test_resident_rl_behavior_idle_probe_tracks_the_executor(self):
        executor = SimpleNamespace(busy=False)

        probe = orin.resident_rl_behavior_idle_probe(lambda: executor)

        self.assertTrue(probe())
        executor.busy = True
        self.assertFalse(probe())
        self.assertTrue(orin.resident_rl_behavior_idle_probe(lambda: None)())

    def test_resident_act_activation_gate_uses_executor_atomic_idle_lock(self):
        events = []

        class Executor:
            def run_when_idle(self, operation):
                events.append("gate_enter")
                result = operation()
                events.append("gate_exit")
                return result

        gate = orin.resident_act_activation_gate(lambda: Executor())

        self.assertEqual(gate(lambda: events.append("activate") or 9), 9)
        self.assertEqual(events, ["gate_enter", "activate", "gate_exit"])
        unavailable = orin.resident_act_activation_gate(lambda: None)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            unavailable(lambda: None)

    def test_open_serial_requests_kernel_exclusive_ownership(self):
        serial_port = mock.Mock()
        serial_module = SimpleNamespace(
            Serial=serial_port,
            EIGHTBITS=8,
            PARITY_NONE="N",
            STOPBITS_ONE=1,
        )
        opened = mock.Mock()
        serial_port.return_value = opened

        with (
            mock.patch.dict(sys.modules, {"serial": serial_module}),
            mock.patch.object(orin.time, "sleep"),
        ):
            result = orin.open_serial("/dev/ttyTHS1", 460800, 0.1)

        self.assertIs(result, opened)
        serial_port.assert_called_once_with(
            port="/dev/ttyTHS1",
            baudrate=460800,
            timeout=0.1,
            bytesize=8,
            parity="N",
            stopbits=1,
            exclusive=True,
        )
        opened.setDTR.assert_called_once_with(False)
        opened.setRTS.assert_called_once_with(False)
        opened.reset_input_buffer.assert_called_once_with()

    def test_terminal_zero_wait_ignores_bad_rows_and_requires_exact_ack(self):
        values = {field: "0" for field in orin.STM32_V2_FIELDS}
        values.update(
            schema_version=orin.STM32_V2_SCHEMA_VERSION,
            command_rx_seq="42",
            command_valid="1",
            command_timed_out="0",
            control_mode="2",
            control_enabled="1",
            rs485_ok="1",
            dwj_ok="1",
            imu_ok="1",
        )
        matching_row = (
            ",".join(values[field] for field in orin.STM32_V2_FIELDS) + "\n"
        ).encode("ascii")

        class SerialRows:
            def __init__(self):
                self.rows = iter((b"not,csv\n", matching_row))

            def readline(self):
                return next(self.rows, b"")

        result = ResidentWriteResult(
            accepted=False,
            write_performed=True,
            reason="terminal_disarm",
            command_seq=42,
            mode=ControlMode.VELOCITY_REFERENCE,
            effective_action=(0.0, 0.0, 0.0, 0.0),
        )

        self.assertTrue(
            orin.wait_for_resident_terminal_zero_ack(
                SerialRows(),
                result,
                timeout_s=0.5,
            )
        )

        values["boom_v_ref_mmps"] = "0.1"
        wrong_action = (
            ",".join(values[field] for field in orin.STM32_V2_FIELDS) + "\n"
        ).encode("ascii")

        class OneWrongRow:
            def readline(self):
                return wrong_action

        with mock.patch.object(
            orin.time,
            "monotonic",
            side_effect=(0.0, 0.1, 0.6),
        ):
            self.assertFalse(
                orin.wait_for_resident_terminal_zero_ack(
                    OneWrongRow(),
                    result,
                    timeout_s=0.5,
                )
            )

    def test_terminal_zero_wait_is_a_noop_when_no_command_was_written(self):
        result = ResidentWriteResult(
            accepted=False,
            write_performed=False,
            reason="terminal_disarm",
            command_seq=None,
            mode=None,
            effective_action=(0.0, 0.0, 0.0, 0.0),
        )
        serial = mock.Mock()

        self.assertTrue(
            orin.wait_for_resident_terminal_zero_ack(
                serial,
                result,
                timeout_s=0.5,
            )
        )
        serial.readline.assert_not_called()

    def test_terminal_zero_wait_rejects_unproven_no_write_failure(self):
        result = ResidentWriteResult(
            accepted=False,
            write_performed=False,
            reason="sink_faulted",
            command_seq=None,
            mode=None,
            effective_action=(0.0, 0.0, 0.0, 0.0),
        )
        serial = mock.Mock()

        self.assertFalse(
            orin.wait_for_resident_terminal_zero_ack(
                serial,
                result,
                timeout_s=0.5,
            )
        )
        serial.readline.assert_not_called()

    def test_hardware_start_gate_waits_until_one_shot_file_arrives(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "rl.start"
            finished = threading.Event()

            def wait_for_gate():
                orin.wait_for_hardware_start_gate(gate, poll_interval_s=0.001)
                finished.set()

            worker = threading.Thread(target=wait_for_gate)
            worker.start()
            time.sleep(0.02)

            self.assertFalse(finished.is_set())
            gate.touch()
            worker.join(timeout=1.0)

            self.assertTrue(finished.is_set())
            self.assertFalse(gate.exists())

    def test_hardware_start_gate_requires_absolute_path(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            orin.wait_for_hardware_start_gate(
                Path("relative.start"), poll_interval_s=0.001
            )

    def test_cli_accepts_hardware_start_gate_without_changing_default(self):
        default_args = orin.parse_args([])
        gated_args = orin.parse_args(
            ["--hardware-start-gate", "/tmp/excavator-rl-control/test.start"]
        )

        self.assertIsNone(default_args.hardware_start_gate)
        self.assertEqual(
            gated_args.hardware_start_gate,
            Path("/tmp/excavator-rl-control/test.start"),
        )

    def test_resident_motion_core_is_an_explicit_runtime_intent(self):
        default_args = orin.parse_args([])
        resident_args = orin.parse_args(["--resident-motion-core"])

        self.assertFalse(default_args.resident_motion_core)
        self.assertTrue(resident_args.resident_motion_core)
        self.assertTrue(resident_args.resident_act_socket.is_absolute())
        self.assertTrue(resident_args.resident_control_socket.is_absolute())

    def test_resident_state_publishes_every_act_telemetry_for_safety_updates(self):
        class Link:
            def __init__(self):
                self.payloads = []

            def publish(self, payload):
                self.payloads.append(payload)

        link = Link()
        state = replace(
            orin.Stm32State(*(0 for _ in range(16))),
            sensor_is_new=True,
            control_seq=11,
            sensor_seq=7,
            stm32_control_enabled=True,
            rs485_ok=True,
            adc_ok=True,
            imu_ok=True,
        )
        core = SimpleNamespace(
            active_act_generation=None,
        )

        self.assertFalse(
            orin.publish_resident_act_state_if_active(
                state=state,
                receive_monotonic_ns=2_000_000_000,
                core=core,
                data_link=link,
                runtime_control_enabled=True,
                runtime_estop=False,
                sensor_valid=True,
                stm32_alive=True,
            )
        )
        self.assertEqual(link.payloads, [])

        core.active_act_generation = 3
        self.assertTrue(
            orin.publish_resident_act_state_if_active(
                state=state,
                receive_monotonic_ns=2_000_000_000,
                core=core,
                data_link=link,
                runtime_control_enabled=True,
                runtime_estop=False,
                sensor_valid=True,
                stm32_alive=True,
            )
        )
        self.assertEqual(len(link.payloads), 1)
        decoded = orin.decode_resident_state(link.payloads[0])
        self.assertEqual(decoded.control_generation, 3)
        self.assertEqual(decoded.control_seq, 11)
        self.assertEqual(decoded.sensor_seq, 7)

        self.assertTrue(
            orin.publish_resident_act_state_if_active(
                state=replace(
                    state,
                    sensor_is_new=False,
                    stm32_control_enabled=False,
                ),
                receive_monotonic_ns=2_050_000_000,
                core=core,
                data_link=link,
                runtime_control_enabled=True,
                runtime_estop=False,
                sensor_valid=True,
                stm32_alive=True,
            )
        )
        self.assertEqual(len(link.payloads), 2)
        safety_update = orin.decode_resident_state(link.payloads[-1])
        self.assertFalse(safety_update.sensor_is_new)
        self.assertFalse(safety_update.control_enabled)

    def test_machine_state_control_remains_enabled_for_active_resident_act(self):
        act_active = SimpleNamespace(
            rl_is_active=False,
            act_is_active=True,
            mission_lease_is_active=True,
        )
        idle = SimpleNamespace(
            rl_is_active=False,
            act_is_active=False,
            mission_lease_is_active=True,
        )
        unleased = SimpleNamespace(
            rl_is_active=False,
            act_is_active=True,
            mission_lease_is_active=False,
        )

        self.assertTrue(
            orin.resident_machine_state_control_enabled(
                runtime_control_enabled=True,
                core=act_active,
            )
        )
        self.assertFalse(
            orin.resident_machine_state_control_enabled(
                runtime_control_enabled=False,
                core=act_active,
            )
        )
        self.assertFalse(
            orin.resident_machine_state_control_enabled(
                runtime_control_enabled=True,
                core=idle,
            )
        )
        self.assertFalse(
            orin.resident_machine_state_control_enabled(
                runtime_control_enabled=True,
                core=unleased,
            )
        )

    def test_resident_motion_core_requires_csv_motion_edge_runtime(self):
        base = SimpleNamespace(
            resident_motion_core=True,
            input_format="csv",
            control_enabled=True,
        )
        for config, expected in (
            (None, "edge config"),
            (SimpleNamespace(mode="shadow"), "motion mode"),
        ):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, expected):
                    orin.validate_resident_motion_request(base, config)

        orin.validate_resident_motion_request(
            base,
            SimpleNamespace(
                mode="remote_control",
                action_transport="resident_sink",
            ),
        )
        with self.assertRaisesRegex(ValueError, "CSV"):
            orin.validate_resident_motion_request(
                SimpleNamespace(
                    resident_motion_core=True,
                    input_format="binary",
                    control_enabled=True,
                ),
                SimpleNamespace(
                    mode="remote_control",
                    action_transport="resident_sink",
                ),
            )
        with self.assertRaisesRegex(ValueError, "resident_sink"):
            orin.validate_resident_motion_request(
                base,
                SimpleNamespace(
                    mode="remote_control",
                    action_transport="loopback_udp",
                ),
            )

    def test_cancelled_hardware_gate_opens_no_device_or_network_socket(self):
        args = SimpleNamespace(
            allowed_action_host=None,
            pc_host="192.168.50.1",
            edge_config=None,
            hardware_start_gate=Path(
                "/tmp/excavator-rl-control/hybrid_cancelled.start"
            ),
            resident_motion_core=False,
        )

        with (
            mock.patch.object(orin, "parse_args", return_value=args),
            mock.patch.object(orin.signal, "signal"),
            mock.patch.object(
                orin,
                "wait_for_hardware_start_gate",
                side_effect=KeyboardInterrupt,
            ),
            mock.patch.object(orin, "open_serial") as open_serial,
            mock.patch.object(orin, "open_action_socket") as open_action_socket,
            mock.patch.object(orin.socket, "socket") as open_udp_socket,
            mock.patch.object(orin, "send_udp_json") as send_machine_state,
        ):
            orin.main()

        open_serial.assert_not_called()
        open_action_socket.assert_not_called()
        open_udp_socket.assert_not_called()
        send_machine_state.assert_not_called()

    def test_resident_main_uses_one_core_without_opening_action_socket(self):
        values = {field: "0" for field in orin.STM32_V2_FIELDS}
        values.update(
            schema_version=orin.STM32_V2_SCHEMA_VERSION,
            control_seq="1",
            control_stamp_ms="100",
            sensor_seq="1",
            sensor_stamp_ms="100",
            sensor_is_new="1",
            control_mode="3",
            control_enabled="1",
            rs485_ok="1",
            dwj_ok="1",
            imu_ok="1",
        )
        row = (
            ",".join(values[field] for field in orin.STM32_V2_FIELDS) + "\n"
        ).encode("ascii")

        class OneFrameSerial:
            def __init__(self):
                self._rows = iter((row,))
                self.writes = []
                self.closed = False

            def readline(self):
                try:
                    return next(self._rows)
                except StopIteration as exc:
                    raise KeyboardInterrupt from exc

            def write(self, payload):
                self.writes.append(payload)
                return len(payload)

            def flush(self):
                return None

            def close(self):
                self.closed = True

        class Runtime:
            def __init__(self):
                self.step_count = 0

            def step(self, machine_state, *, now_s):
                self.step_count += 1
                return EdgeFollowStep(
                    source_seq=machine_state["seq"],
                    source_stamp_ms=machine_state["stamp_ms"],
                    waypoint_index=0,
                    completed=False,
                    bucket_tip_ros_m=(0.0, 0.0, 0.0),
                    bucket_pitch_rad=0.0,
                    observation=tuple(0.0 for _ in range(38)),
                    normalized_action=(0.1, 0.0, 0.0, 0.0),
                    physical_action=(0.01, 0.0, 0.0, 0.0),
                )

        class UdpSocket:
            def close(self):
                return None

        args = orin.parse_args(
            [
                "--edge-config",
                "/tmp/resident-edge.json",
                "--control-enabled",
                "--edge-motion-authorization",
                orin.EDGE_MOTION_AUTHORIZATION,
                "--resident-motion-core",
            ]
        )
        audit_directory = tempfile.TemporaryDirectory()
        self.addCleanup(audit_directory.cleanup)
        config = SimpleNamespace(
            mode="control",
            action_transport="resident_sink",
            audit_path=Path(audit_directory.name) / "resident-edge.jsonl",
            action_valid_for_ms=100,
        )
        serial = OneFrameSerial()
        runtime = Runtime()
        data_link = mock.Mock()
        data_link.connected = True
        control_server = mock.Mock()

        with (
            mock.patch.object(orin, "parse_args", return_value=args),
            mock.patch.object(orin, "open_serial", return_value=serial),
            mock.patch.object(orin, "open_action_socket") as open_action_socket,
            mock.patch.object(orin.socket, "socket", return_value=UdpSocket()),
            mock.patch.object(orin, "send_udp_json"),
            mock.patch("edge_runtime.shadow.load_edge_runtime_config", return_value=config),
            mock.patch("edge_runtime.shadow.build_edge_follow_runtime", return_value=runtime),
            mock.patch.object(
                orin,
                "ResidentActDataLink",
                return_value=data_link,
            ) as data_link_factory,
            mock.patch.object(
                orin,
                "ResidentMotionControlServer",
                return_value=control_server,
            ) as control_server_factory,
            mock.patch.object(
                orin,
                "wait_for_resident_terminal_zero_ack",
                return_value=True,
            ) as wait_for_terminal_ack,
            mock.patch.object(orin.LOGGER, "info") as logger_info,
            mock.patch.object(orin.signal, "signal"),
        ):
            orin.main()

        open_action_socket.assert_not_called()
        self.assertTrue(serial.closed)
        self.assertGreaterEqual(len(serial.writes), 1)
        first = orin.json.loads(serial.writes[0].decode("ascii"))
        self.assertEqual(first["schema_version"], "stm32_velocity_command.v1")
        self.assertEqual(first["boom_mps"], 0.0)
        self.assertEqual(runtime.step_count, 0)
        data_link_factory.assert_called_once()
        disconnect_callback = data_link_factory.call_args.kwargs[
            "on_connection_lost"
        ]
        self.assertEqual(
            disconnect_callback.__name__,
            "notify_act_worker_disconnected",
        )
        control_server_factory.assert_called_once()
        info_formats = [call.args[0] for call in logger_info.call_args_list]
        self.assertIn(
            "RESIDENT_CONTROL_READY control_socket=%s act_socket=%s",
            info_formats,
        )
        self.assertIn(
            "RESIDENT_HARDWARE_READY sensor_valid=True",
            info_formats,
        )
        idle_probe = control_server_factory.call_args.kwargs["rl_behavior_idle"]
        self.assertTrue(idle_probe())
        atomic_gate = control_server_factory.call_args.kwargs[
            "activate_act_while_rl_idle"
        ]
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            atomic_gate(lambda: None)
        data_link.start.assert_called_once_with()
        control_server.start.assert_called_once_with()
        control_server.close.assert_called_once_with()
        data_link.close.assert_called_once_with()
        wait_for_terminal_ack.assert_called_once()

    def test_resident_main_fails_when_terminal_zero_is_not_acknowledged(self):
        values = {field: "0" for field in orin.STM32_V2_FIELDS}
        values.update(
            schema_version=orin.STM32_V2_SCHEMA_VERSION,
            control_seq="1",
            control_stamp_ms="100",
            sensor_seq="1",
            sensor_stamp_ms="100",
            sensor_is_new="1",
            control_mode="3",
            control_enabled="1",
            rs485_ok="1",
            dwj_ok="1",
            imu_ok="1",
        )
        row = (
            ",".join(values[field] for field in orin.STM32_V2_FIELDS) + "\n"
        ).encode("ascii")

        class OneFrameSerial:
            def __init__(self):
                self._rows = iter((row,))

            def readline(self):
                try:
                    return next(self._rows)
                except StopIteration as exc:
                    raise KeyboardInterrupt from exc

            def write(self, payload):
                return len(payload)

            def flush(self):
                return None

            def close(self):
                return None

        class UdpSocket:
            def close(self):
                return None

        class Runtime:
            def step(self, machine_state, *, now_s):
                raise AssertionError(
                    "resident owner must not execute the legacy RL runtime"
                )

        args = orin.parse_args(
            [
                "--edge-config",
                "/tmp/resident-edge.json",
                "--control-enabled",
                "--edge-motion-authorization",
                orin.EDGE_MOTION_AUTHORIZATION,
                "--resident-motion-core",
            ]
        )
        config = SimpleNamespace(
            mode="control",
            action_transport="resident_sink",
            audit_path=Path("/tmp/resident-edge.jsonl"),
            action_valid_for_ms=100,
        )
        data_link = mock.Mock()
        data_link.connected = True

        with (
            mock.patch.object(orin, "parse_args", return_value=args),
            mock.patch.object(orin, "open_serial", return_value=OneFrameSerial()),
            mock.patch.object(orin.socket, "socket", return_value=UdpSocket()),
            mock.patch.object(orin, "send_udp_json"),
            mock.patch(
                "edge_runtime.shadow.load_edge_runtime_config",
                return_value=config,
            ),
            mock.patch(
                "edge_runtime.shadow.build_edge_follow_runtime",
                return_value=Runtime(),
            ),
            mock.patch.object(orin, "ResidentActDataLink", return_value=data_link),
            mock.patch.object(
                orin,
                "ResidentMotionControlServer",
                return_value=mock.Mock(),
            ),
            mock.patch.object(
                orin,
                "wait_for_resident_terminal_zero_ack",
                return_value=False,
            ),
            mock.patch(
                "edge_runtime.control.EdgeControlRunner.close",
                side_effect=RuntimeError("edge close failed"),
            ),
            mock.patch.object(orin.signal, "signal"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "terminal zero was not acknowledged",
            ) as captured:
                orin.main()
        self.assertIsNotNone(captured.exception.__context__)
        self.assertIn("edge close failed", str(captured.exception.__context__))

    def test_resident_main_exits_after_the_core_terminally_disarms(self):
        values = {field: "0" for field in orin.STM32_V2_FIELDS}
        values.update(
            schema_version=orin.STM32_V2_SCHEMA_VERSION,
            control_seq="1",
            control_stamp_ms="100",
            sensor_seq="1",
            sensor_stamp_ms="100",
            sensor_is_new="1",
            control_mode="3",
            control_enabled="1",
            rs485_ok="1",
            dwj_ok="1",
            imu_ok="1",
        )
        row = (
            ",".join(values[field] for field in orin.STM32_V2_FIELDS) + "\n"
        ).encode("ascii")

        class OneFrameOnlySerial:
            def __init__(self):
                self.read_count = 0
                self.closed = False

            def readline(self):
                self.read_count += 1
                if self.read_count > 1:
                    raise AssertionError(
                        "resident owner read another frame after terminal disarm"
                    )
                return row

            def write(self, payload):
                return len(payload)

            def flush(self):
                return None

            def close(self):
                self.closed = True

        terminal_result = ResidentWriteResult(
            accepted=False,
            write_performed=True,
            reason="terminal_disarm",
            command_seq=42,
            mode=ControlMode.VELOCITY_REFERENCE,
            effective_action=(0.0, 0.0, 0.0, 0.0),
        )

        class TerminalCore:
            renew_calls = []

            def __init__(self, *_args, **_kwargs):
                self.is_operational = True
                self.rl_action_sink = mock.Mock()
                self.rl_is_active = True
                self.act_is_active = False
                self.mission_lease_is_active = True
                self.active_act_generation = None

            def initialize(self, _frame):
                return 0

            def activate_rl(self, **_kwargs):
                return 1

            def renew_mission_lease(self, **kwargs):
                self.renew_calls.append(kwargs)

            def observe_telemetry(self, _frame):
                return None

            def tick(self, **_kwargs):
                self.is_operational = False
                return terminal_result

            def terminal_disarm(self, **_kwargs):
                return terminal_result

            def submit_act(self, _payload):
                return None

            def notify_act_worker_disconnected(self):
                return None

        class UdpSocket:
            def close(self):
                return None

        class TerminalRuntime:
            def step(self, machine_state, *, now_s):
                del now_s
                return EdgeFollowStep(
                    source_seq=machine_state["seq"],
                    source_stamp_ms=machine_state["stamp_ms"],
                    waypoint_index=0,
                    completed=False,
                    bucket_tip_ros_m=(0.0, 0.0, 0.0),
                    bucket_pitch_rad=0.0,
                    observation=tuple(0.0 for _ in range(38)),
                    normalized_action=(0.0, 0.0, 0.0, 0.0),
                    physical_action=(0.0, 0.0, 0.0, 0.0),
                )

        args = orin.parse_args(
            [
                "--edge-config",
                "/tmp/resident-edge.json",
                "--control-enabled",
                "--edge-motion-authorization",
                orin.EDGE_MOTION_AUTHORIZATION,
                "--resident-motion-core",
            ]
        )
        config = SimpleNamespace(
            mode="control",
            action_transport="resident_sink",
            audit_path=Path("/tmp/resident-edge.jsonl"),
            action_valid_for_ms=100,
        )
        serial = OneFrameOnlySerial()
        data_link = mock.Mock()
        data_link.connected = True

        with (
            mock.patch.object(orin, "parse_args", return_value=args),
            mock.patch.object(orin, "open_serial", return_value=serial),
            mock.patch.object(orin.socket, "socket", return_value=UdpSocket()),
            mock.patch.object(orin, "send_udp_json"),
            mock.patch(
                "edge_runtime.shadow.load_edge_runtime_config",
                return_value=config,
            ),
            mock.patch(
                "edge_runtime.shadow.build_edge_follow_runtime",
                return_value=TerminalRuntime(),
            ),
            mock.patch.object(orin, "ResidentMotionCore", TerminalCore),
            mock.patch.object(orin, "ResidentActDataLink", return_value=data_link),
            mock.patch.object(
                orin,
                "ResidentMotionControlServer",
                return_value=mock.Mock(),
            ),
            mock.patch.object(
                orin,
                "wait_for_resident_terminal_zero_ack",
                return_value=True,
            ) as wait_for_terminal_ack,
            mock.patch.object(orin.signal, "signal"),
        ):
            orin.main()

        self.assertEqual(serial.read_count, 1)
        self.assertTrue(serial.closed)
        data_link.close.assert_called_once_with()
        wait_for_terminal_ack.assert_called_once_with(
            serial,
            terminal_result,
        )
        self.assertEqual(len(TerminalCore.renew_calls), 1)
        self.assertEqual(
            TerminalCore.renew_calls[0]["lease_ms"],
            orin.DEFAULT_MISSION_LEASE_MS,
        )

    def test_sigterm_handler_enters_existing_keyboard_interrupt_cleanup_path(self):
        with self.assertRaises(KeyboardInterrupt):
            orin._raise_keyboard_interrupt_on_termination(None, None)

    def test_unified_firmware_defaults_match_field_serial_link(self):
        self.assertEqual(orin.DEFAULT_SERIAL_PORT, "/dev/ttyTHS1")
        self.assertEqual(orin.DEFAULT_BAUDRATE, 460800)

    def test_unified_v2_row_maps_state_safety_and_command_sequence(self):
        values = {field: "0" for field in orin.STM32_V2_FIELDS}
        values.update(
            schema_version="stm32_control_telemetry.v2",
            control_stamp_ms="1234",
            command_rx_seq="41",
            command_valid="1",
            control_enabled="1",
            control_mode="2",
            boom_v_ref_mmps="25.0",
            stick_v_ref_mmps="-30.0",
            bucket_v_ref_mmps="40.0",
            swing_v_ref_degps="10.0",
            boom_pos_mm="100.0",
            stick_pos_mm="120.0",
            bucket_pos_mm="80.0",
            boom_vel_mmps="2.0",
            stick_vel_mmps="-3.0",
            bucket_vel_mmps="4.0",
            boom_angle_deg="10.0",
            arm_angle_deg="20.0",
            bucket_angle_deg="30.0",
            swing_angle_deg="40.0",
            swing_vel_degps="5.0",
            rs485_ok="1",
            dwj_ok="1",
            imu_ok="1",
        )
        row = ",".join(values[field] for field in orin.STM32_V2_FIELDS)

        state = orin.parse_stm32_csv_line(row)

        self.assertIsNotNone(state)
        self.assertEqual(state.command_rx_seq, 41)
        self.assertEqual(state.control_seq, 0)
        self.assertEqual(state.sensor_seq, 0)
        self.assertFalse(state.sensor_is_new)
        self.assertTrue(state.command_received)
        self.assertTrue(state.command_valid)
        self.assertFalse(state.command_timed_out)
        self.assertTrue(state.stm32_control_enabled)
        self.assertEqual(state.stm32_control_mode, 2)
        self.assertAlmostEqual(state.command_boom_mps, 0.025)
        self.assertAlmostEqual(state.command_stick_mps, -0.03)
        self.assertAlmostEqual(state.command_bucket_mps, 0.04)
        self.assertAlmostEqual(state.command_swing_radps, 0.1745329252)
        self.assertFalse(state.stm32_estop)
        self.assertEqual(state.stm32_fault_flags, 0)
        self.assertAlmostEqual(state.s_boom, 0.1)
        self.assertAlmostEqual(state.v_stick, -0.003)
        self.assertAlmostEqual(state.yaw, 0.6981317008)

    def test_resident_act_state_adapter_preserves_canonical_11d_order(self):
        values = {field: "0" for field in orin.STM32_V2_FIELDS}
        values.update(
            schema_version=orin.STM32_V2_SCHEMA_VERSION,
            control_seq="41",
            control_stamp_ms="1234",
            sensor_seq="20",
            sensor_stamp_ms="1220",
            sensor_is_new="1",
            boom_pos_mm="101",
            stick_pos_mm="202",
            bucket_pos_mm="303",
            boom_vel_mmps="11",
            stick_vel_mmps="-22",
            bucket_vel_mmps="33",
            boom_angle_deg="10",
            arm_angle_deg="20",
            bucket_angle_deg="30",
            swing_angle_deg="-40",
            swing_vel_degps="5",
            control_enabled="1",
            rs485_ok="1",
            dwj_ok="1",
            imu_ok="1",
        )
        state = orin.parse_stm32_csv_line(
            ",".join(values[field] for field in orin.STM32_V2_FIELDS)
        )

        frame = orin.resident_act_state_from_state(
            state,
            receive_monotonic_ns=2_000_000_000,
            control_generation=7,
            runtime_control_enabled=True,
            runtime_estop=False,
            sensor_valid=True,
            stm32_alive=True,
        )

        self.assertEqual(frame.control_seq, 41)
        self.assertEqual(frame.sensor_seq, 20)
        self.assertTrue(frame.sensor_is_new)
        self.assertEqual(frame.control_generation, 7)
        self.assertEqual(
            frame.state[:6],
            (0.101, 0.202, 0.303, 0.011, -0.022, 0.033),
        )
        for actual, expected in zip(
            frame.state[6:],
            (
                orin.math.radians(10),
                orin.math.radians(20),
                orin.math.radians(30),
                orin.math.radians(-40),
                orin.math.radians(5),
            ),
        ):
            self.assertAlmostEqual(actual, expected)

    def test_unified_v2_rejects_corruption_in_any_telemetry_field(self):
        values = {field: "0" for field in orin.STM32_V2_FIELDS}
        values.update(
            schema_version="stm32_control_telemetry.v2",
            pid_out_boom="nan",
        )
        row = ",".join(values[field] for field in orin.STM32_V2_FIELDS)

        with self.assertRaisesRegex(ValueError, "pid_out_boom"):
            orin.parse_stm32_csv_line(row)

    def test_velocity_encoder_resumes_stm32_sequence_and_preserves_units(self):
        encoder = orin.Stm32VelocityCommandEncoder()
        self.assertEqual(
            encoder.synchronize(command_rx_seq=41, command_received=True),
            42,
        )

        payload = orin.json.loads(
            encoder.encode(
                orin.DataCommand(
                    t_s=1.234567,
                    boom_v_ref_mps=0.12,
                    stick_v_ref_mps=-0.08,
                    bucket_v_ref_mps=0.04,
                    swing_v_ref_radps=-0.15,
                )
            ).decode("ascii")
        )

        self.assertEqual(payload["schema_version"], "stm32_velocity_command.v1")
        self.assertEqual(payload["command_seq"], 42)
        self.assertEqual(payload["command_source_stamp_ms"], 1234)
        self.assertEqual(
            (
                payload["boom_mps"],
                payload["stick_mps"],
                payload["bucket_mps"],
                payload["swing_radps"],
            ),
            (0.12, -0.08, 0.04, -0.15),
        )

    def test_resident_telemetry_adapter_preserves_mode_specific_command_semantics(self):
        manual = replace(
            orin.Stm32State(*(0 for _ in range(16))),
            x1=-0.4,
            x2=0.3,
            y1=-0.2,
            y2=0.1,
            command_rx_seq=8,
            command_received=True,
            stm32_control_enabled=True,
            command_valid=True,
            stm32_control_mode=1,
        )
        manual_frame = orin.resident_telemetry_from_state(
            manual,
            receive_monotonic_ns=1_000_000_000,
            runtime_control_enabled=True,
            runtime_estop=False,
            sensor_valid=True,
            stm32_alive=True,
        )
        self.assertEqual(manual_frame.control_mode, ControlMode.MANUAL_ACTION)
        self.assertEqual(manual_frame.command_action, (0.1, -0.2, 0.3, -0.4))

        velocity = replace(
            orin.Stm32State(*(0 for _ in range(16))),
            command_rx_seq=9,
            command_received=True,
            stm32_control_enabled=True,
            command_valid=True,
            stm32_control_mode=2,
            command_boom_mps=0.025,
            command_stick_mps=-0.03,
            command_bucket_mps=0.04,
            command_swing_radps=-0.5,
        )
        velocity_frame = orin.resident_telemetry_from_state(
            velocity,
            receive_monotonic_ns=1_010_000_000,
            runtime_control_enabled=True,
            runtime_estop=False,
            sensor_valid=True,
            stm32_alive=True,
        )
        self.assertEqual(
            velocity_frame.control_mode,
            ControlMode.VELOCITY_REFERENCE,
        )
        self.assertEqual(
            velocity_frame.command_action,
            (0.025, -0.03, 0.04, -0.5),
        )

    def test_resident_telemetry_adapter_combines_runtime_and_hardware_safety(self):
        state = replace(
            orin.Stm32State(*(0 for _ in range(16))),
            stm32_control_enabled=True,
            command_valid=True,
            stm32_control_mode=2,
            stm32_estop=True,
            stm32_fault_flags=4,
        )
        frame = orin.resident_telemetry_from_state(
            state,
            receive_monotonic_ns=1_000_000_000,
            runtime_control_enabled=False,
            runtime_estop=False,
            sensor_valid=True,
            stm32_alive=True,
        )
        self.assertFalse(frame.control_enabled)
        self.assertTrue(frame.estop)
        self.assertEqual(frame.fault_flags, 4)

    def test_live_19_field_row_maps_all_hardware_validity_flags(self):
        state = orin.parse_stm32_csv_line(LIVE_ROW)

        self.assertIsNotNone(state)
        self.assertTrue(state.rs485_ok)
        self.assertTrue(state.adc_ok)
        self.assertTrue(state.imu_ok)
        packet = orin.build_machine_state_packet(
            state=state,
            seq=1,
            machine_id="scale_excavator_v1",
            control_enabled=False,
            estop=False,
            include_raw=True,
            last_receive_monotonic_s=time.monotonic(),
        )
        self.assertTrue(packet["safety"]["sensor_valid"])
        self.assertEqual(packet["safety"]["fault_flags"], [])
        self.assertTrue(packet["raw_sensor"]["rs485_ok"])
        self.assertTrue(packet["raw_sensor"]["adc_ok"])
        self.assertTrue(packet["raw_sensor"]["imu_ok"])

    def test_each_zero_hardware_flag_fails_closed_with_specific_fault(self):
        cases = {
            16: "rs485_invalid",
            17: "adc_invalid",
            18: "imu_invalid",
        }
        for field_index, expected_fault in cases.items():
            with self.subTest(expected_fault=expected_fault):
                fields = LIVE_ROW.split(",")
                fields[field_index] = "0"
                state = orin.parse_stm32_csv_line(",".join(fields))
                packet = orin.build_machine_state_packet(
                    state=state,
                    seq=2,
                    machine_id="scale_excavator_v1",
                    control_enabled=False,
                    estop=False,
                    include_raw=False,
                    last_receive_monotonic_s=time.monotonic(),
                )

                self.assertFalse(packet["safety"]["sensor_valid"])
                self.assertIn(expected_fault, packet["safety"]["fault_flags"])

    def test_machine_state_safety_merges_runtime_and_stm32_motion_gate(self):
        state = replace(
            orin.parse_stm32_csv_line(LIVE_ROW),
            stm32_control_enabled=False,
            hardware_motion_gate_available=True,
            stm32_estop=True,
            stm32_fault_flags=0x12,
        )

        packet = orin.build_machine_state_packet(
            state=state,
            seq=3,
            machine_id="scale_excavator_v1",
            control_enabled=True,
            estop=False,
            include_raw=False,
            last_receive_monotonic_s=time.monotonic(),
        )

        self.assertFalse(packet["safety"]["control_enabled"])
        self.assertTrue(packet["safety"]["estop"])
        self.assertIn("stm32_fault_flags:0x00000012", packet["safety"]["fault_flags"])
        self.assertFalse(packet["safety"]["sensor_valid"])

    def test_partial_or_non_boolean_hardware_flags_are_rejected(self):
        fields = LIVE_ROW.split(",")
        with self.assertRaisesRegex(ValueError, "validity field count"):
            orin.parse_stm32_csv_line(",".join(fields[:18]))

        fields[16] = "2"
        with self.assertRaisesRegex(ValueError, "rs485_ok"):
            orin.parse_stm32_csv_line(",".join(fields))

    def test_legacy_16_field_row_remains_supported(self):
        state = orin.parse_stm32_csv_line(",".join(LIVE_ROW.split(",")[:16]))

        self.assertTrue(state.rs485_ok)
        self.assertTrue(state.adc_ok)
        self.assertTrue(state.imu_ok)


if __name__ == "__main__":
    unittest.main()
