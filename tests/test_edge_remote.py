import hashlib
import io
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from edge_runtime.trajectory_controller import TrajectoryControllerDescriptor

from edge_runtime.follow import EdgeFollowStep
from edge_runtime.remote_transport import ConnectionEventStream
from edge_runtime.remote import (
    EdgeBehaviorExecutor,
    EdgeFollowRuntimeFactory,
    MAX_FRAME_BYTES,
    RemoteBehaviorServer,
    FollowTrajectorySnapshot,
    receive_message,
    send_message,
)


def trajectory_snapshot():
    return {
        "trajectory_id": "trajectory-1",
        "trajectory_sha256": (
            "f86ad57e8caa5551bfca79651e6eb2be9f3df3acadeb0521c3d6f52a277c1b6c"
        ),
        "frame_id": "machine_root_ros",
        "created_at_s": 100.0,
        "mission_id": "mission-1",
        "mission_sha256": "a" * 64,
        "mission_phase": "dig",
        "task_mode": "MoveToDig",
        "planning_scope": "execution_strict",
        "control_stage": "commissioning",
        "workspace_constraint": "disabled_by_operator",
        "execution_eligible": True,
        "source_bucket_tip_stamp_s": 99.5,
        "source_local_map_stamp_s": 99.6,
        "inputs_frozen_at_s": 99.7,
        "valid_until_s": 110.0,
        "input_source": "live",
        "map_source": "live_local_map",
        "clock_mode": "ros_clock",
        "waypoints": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "waypoint_tolerance_m": 0.25,
        "waypoint_dwell_s": 0.0,
        "tracking_timeout_s": 60.0,
    }


class FramedJsonProtocolTest(unittest.TestCase):
    def test_round_trip_uses_big_endian_length_and_rejects_oversize_frames(self):
        class MemorySocket:
            def __init__(self):
                self.buffer = io.BytesIO()

            def sendall(self, payload):
                self.buffer.write(payload)

            def recv(self, size):
                return self.buffer.read(size)

        stream = MemorySocket()
        message = {
            "schema_version": "orin_behavior_rpc.v1",
            "type": "cancel_follow",
            "session_id": "session-1",
            "seq": 1,
            "request_id": "cancel-1",
        }

        send_message(stream, message)
        encoded = stream.buffer.getvalue()
        self.assertEqual(int.from_bytes(encoded[:4], "big"), len(encoded) - 4)
        stream.buffer.seek(0)
        self.assertEqual(receive_message(stream), message)

        oversized = MemorySocket()
        oversized.buffer.write((MAX_FRAME_BYTES + 1).to_bytes(4, "big"))
        oversized.buffer.seek(0)
        with self.assertRaisesRegex(ValueError, "maximum"):
            receive_message(oversized)


class ConnectionEventStreamTest(unittest.TestCase):
    def test_writer_failure_logs_connection_and_last_event_context(self):
        class FailingSocket:
            def __init__(self):
                self.shutdown_called = threading.Event()

            def sendall(self, _payload):
                raise OSError("simulated-link-loss")

            def shutdown(self, _how):
                self.shutdown_called.set()

        connection = FailingSocket()
        with self.assertLogs("orin_edge_remote", level="ERROR") as captured:
            stream = ConnectionEventStream(
                connection,
                connection_id="rpc-test-001",
            )
            stream.start()
            stream.emit(
                {
                    "schema_version": "orin_behavior_rpc.v1",
                    "type": "feedback",
                }
            )
            self.assertTrue(connection.shutdown_called.wait(0.5))
            stream.close()

        diagnostic = "\n".join(captured.output)
        self.assertIn("connection_id=rpc-test-001", diagnostic)
        self.assertIn("reason=send_error", diagnostic)
        self.assertIn("event_type=feedback", diagnostic)
        self.assertIn("event_seq=0", diagnostic)
        self.assertIn("simulated-link-loss", diagnostic)

    def test_slow_network_writer_does_not_block_control_thread_emit(self):
        class SlowSocket:
            def __init__(self):
                self.release = threading.Event()
                self.send_started = threading.Event()
                self.frames = []

            def sendall(self, payload):
                self.send_started.set()
                self.release.wait(1.0)
                self.frames.append(payload)

            def shutdown(self, _how):
                self.release.set()

        connection = SlowSocket()
        stream = ConnectionEventStream(connection, max_pending_events=4)
        stream.start()
        self.addCleanup(stream.close)

        stream.emit({"schema_version": "orin_behavior_rpc.v1", "type": "status"})
        self.assertTrue(connection.send_started.wait(0.5))

        started_at = time.monotonic()
        stream.emit({"schema_version": "orin_behavior_rpc.v1", "type": "feedback"})
        self.assertLess(time.monotonic() - started_at, 0.05)

        connection.release.set()
        deadline = time.monotonic() + 0.5
        while len(connection.frames) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(len(connection.frames), 2)
        decoded = [
            json.loads(frame[4:].decode("utf-8"))
            for frame in connection.frames
        ]
        self.assertEqual([event["seq"] for event in decoded], [0, 1])

    def test_bounded_queue_closes_stalled_connection_on_backpressure(self):
        class StalledSocket:
            def __init__(self):
                self.release = threading.Event()
                self.send_started = threading.Event()
                self.shutdown_called = threading.Event()

            def sendall(self, _payload):
                self.send_started.set()
                self.release.wait(1.0)

            def shutdown(self, _how):
                self.shutdown_called.set()
                self.release.set()

        connection = StalledSocket()
        stream = ConnectionEventStream(connection, max_pending_events=1)
        stream.start()
        self.addCleanup(stream.close)
        stream.emit({"schema_version": "orin_behavior_rpc.v1", "type": "status"})
        self.assertTrue(connection.send_started.wait(0.5))
        stream.emit({"schema_version": "orin_behavior_rpc.v1", "type": "feedback"})

        stream.emit({"schema_version": "orin_behavior_rpc.v1", "type": "feedback"})

        self.assertTrue(stream.failed)
        self.assertTrue(connection.shutdown_called.wait(0.5))

    def test_concurrent_producers_preserve_one_contiguous_event_sequence(self):
        class RecordingSocket:
            def __init__(self):
                self.frames = []
                self.lock = threading.Lock()

            def sendall(self, payload):
                with self.lock:
                    self.frames.append(payload)

            def shutdown(self, _how):
                return None

        connection = RecordingSocket()
        stream = ConnectionEventStream(connection, max_pending_events=512)
        stream.start()
        producers = [
            threading.Thread(
                target=lambda: [
                    stream.emit(
                        {
                            "schema_version": "orin_behavior_rpc.v1",
                            "type": "feedback",
                        }
                    )
                    for _ in range(100)
                ]
            )
            for _ in range(4)
        ]
        for producer in producers:
            producer.start()
        for producer in producers:
            producer.join()
        stream.close()

        decoded = [
            json.loads(frame[4:].decode("utf-8"))
            for frame in connection.frames
        ]
        self.assertEqual(len(decoded), 400)
        self.assertEqual([event["seq"] for event in decoded], list(range(400)))


class FollowTrajectorySnapshotTest(unittest.TestCase):
    def test_digest_matches_pc_snapshot_digest_and_tampering_is_rejected(self):
        snapshot = FollowTrajectorySnapshot.from_mapping(
            trajectory_snapshot(),
            now_s=101.0,
        )

        self.assertEqual(snapshot.trajectory_id, "trajectory-1")
        self.assertEqual(snapshot.waypoints[1], (0.4, 0.5, 0.6))

        tampered = dict(trajectory_snapshot())
        tampered["waypoint_tolerance_m"] = 0.3
        with self.assertRaisesRegex(ValueError, "sha256"):
            FollowTrajectorySnapshot.from_mapping(tampered, now_s=101.0)

    def test_execution_validation_allows_only_bounded_clock_skew(self):
        snapshot = FollowTrajectorySnapshot.from_mapping(
            trajectory_snapshot(),
            now_s=99.8,
        )

        self.assertEqual(snapshot.trajectory_id, "trajectory-1")

        with self.assertRaisesRegex(ValueError, "from the future"):
            FollowTrajectorySnapshot.from_mapping(
                trajectory_snapshot(),
                now_s=99.4,
            )

    def test_two_level_tolerance_is_digest_bound_and_legacy_snapshot_falls_back(self):
        legacy = FollowTrajectorySnapshot.from_mapping(
            trajectory_snapshot(),
            now_s=101.0,
        )
        self.assertIsNone(legacy.intermediate_waypoint_tolerance_m)

        payload = trajectory_snapshot()
        payload["intermediate_waypoint_tolerance_m"] = 0.40
        digest_fields = {
            key: value
            for key, value in payload.items()
            if key not in {"trajectory_id", "trajectory_sha256"}
        }
        payload["trajectory_sha256"] = hashlib.sha256(
            json.dumps(
                digest_fields,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        snapshot = FollowTrajectorySnapshot.from_mapping(payload, now_s=101.0)
        self.assertEqual(snapshot.waypoint_tolerance_m, 0.25)
        self.assertEqual(snapshot.intermediate_waypoint_tolerance_m, 0.40)

        payload["intermediate_waypoint_tolerance_m"] = 0.41
        with self.assertRaisesRegex(ValueError, "sha256"):
            FollowTrajectorySnapshot.from_mapping(payload, now_s=101.0)


class EdgeFollowRuntimeFactoryTest(unittest.TestCase):
    def test_from_config_honors_legacy_remote_onnx_policy_patch(self):
        profile = {
            "machine_id": "scale_excavator_v1",
            "observation_schema": {
                "normalizers": {
                    "target_threshold": 0.03,
                    "tube_radius": 0.04,
                }
            },
        }
        mission = {
            "schema_version": "excavation_mission.v1",
            "mission_id": "mission-1",
            "frame_id": "machine_root_ros",
            "limits": {
                "waypoint_tolerance_m": 0.25,
                "waypoint_dwell_s": 0.0,
                "tracking_timeout_s": 60.0,
            },
        }
        loaded_paths = []

        class PatchedOnnxPolicy:
            def __init__(self, path):
                loaded_paths.append(path)

            def run(self, _observation):
                return (0.0, 0.0, 0.0, 0.0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "machine_profile.json"
            mission_path = root / "mission.json"
            onnx_path = root / "policy.onnx"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            mission_path.write_text(json.dumps(mission), encoding="utf-8")
            config = SimpleNamespace(
                machine_profile_path=profile_path,
                mission_path=mission_path,
                urdf_path=root / "machine.urdf",
                trajectory_controller_backend="onnx_rl",
                onnx_path=onnx_path,
                follow_action_slew_rate_per_s=None,
            )
            with (
                mock.patch(
                    "edge_runtime.remote.UrdfBucketTipKinematics.from_path",
                    return_value=SimpleNamespace(root_link="fk_root"),
                ),
                mock.patch(
                    "edge_runtime.remote.OnnxPolicy",
                    PatchedOnnxPolicy,
                ),
            ):
                factory = EdgeFollowRuntimeFactory.from_config(config)
                first = factory._controller_builder()
                second = factory._controller_builder()

        self.assertEqual(loaded_paths, [onnx_path])
        self.assertIsNot(first, second)
        self.assertEqual(first.descriptor.backend_id, "onnx_rl")
        self.assertEqual(second.descriptor.backend_id, "onnx_rl")

    def test_from_config_uses_one_preloaded_builder_for_fresh_controllers(self):
        profile = {
            "machine_id": "scale_excavator_v1",
            "observation_schema": {
                "normalizers": {
                    "target_threshold": 0.03,
                    "tube_radius": 0.04,
                }
            },
        }
        mission = {
            "schema_version": "excavation_mission.v1",
            "mission_id": "mission-1",
            "frame_id": "machine_root_ros",
            "limits": {
                "waypoint_tolerance_m": 0.25,
                "waypoint_dwell_s": 0.0,
                "tracking_timeout_s": 60.0,
            },
        }
        controllers = []

        class Controller:
            descriptor = TrajectoryControllerDescriptor(
                backend_id="cartesian_p",
                implementation="test.Controller",
            )

            def __init__(self):
                self.reset_count = 0
                controllers.append(self)

            def reset(self):
                self.reset_count += 1

        def build_controller():
            return Controller()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "machine_profile.json"
            mission_path = root / "mission.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            mission_path.write_text(json.dumps(mission), encoding="utf-8")
            config = SimpleNamespace(
                machine_profile_path=profile_path,
                mission_path=mission_path,
                urdf_path=root / "machine.urdf",
                trajectory_controller_backend="cartesian_p",
                onnx_path=None,
                follow_action_slew_rate_per_s=None,
            )
            with (
                mock.patch(
                    "edge_runtime.remote.UrdfBucketTipKinematics.from_path",
                    return_value=SimpleNamespace(root_link="fk_root"),
                ),
                mock.patch(
                    "edge_runtime.remote.build_trajectory_controller_builder",
                    return_value=build_controller,
                ) as build_deployment,
            ):
                factory = EdgeFollowRuntimeFactory.from_config(config)
                snapshot = FollowTrajectorySnapshot.from_mapping(
                    trajectory_snapshot(),
                    now_s=101.0,
                )
                factory.create(snapshot)
                factory.create(snapshot)

        build_deployment.assert_called_once_with(
            "cartesian_p",
            onnx_path=None,
        )
        self.assertEqual(len(controllers), 2)
        self.assertIsNot(controllers[0], controllers[1])
        self.assertEqual([item.reset_count for item in controllers], [1, 1])

    def test_controller_builder_creates_a_fresh_controller_for_each_behavior(self):
        controller_instances = []
        runtime_arguments = []

        def build_controller():
            controller = object()
            controller_instances.append(controller)
            return controller

        class RecordingRuntime:
            def __init__(self, **kwargs):
                runtime_arguments.append(kwargs)

        profile = {
            "machine_id": "scale_excavator_v1",
            "observation_schema": {
                "normalizers": {
                    "target_threshold": 0.03,
                    "tube_radius": 0.04,
                }
            },
        }
        mission = {
            "schema_version": "excavation_mission.v1",
            "mission_id": "mission-1",
            "frame_id": "machine_root_ros",
            "limits": {
                "waypoint_tolerance_m": 0.25,
                "waypoint_dwell_s": 0.0,
                "tracking_timeout_s": 60.0,
            },
        }
        factory = EdgeFollowRuntimeFactory(
            machine_profile=profile,
            kinematics=object(),
            controller_builder=build_controller,
            controller_backend="cartesian_p",
            mission=mission,
            mission_sha256="a" * 64,
            runtime_type=RecordingRuntime,
        )
        snapshot = FollowTrajectorySnapshot.from_mapping(
            trajectory_snapshot(),
            now_s=101.0,
        )

        factory.create(snapshot)
        factory.create(snapshot)

        self.assertEqual(len(controller_instances), 2)
        self.assertIsNot(controller_instances[0], controller_instances[1])
        self.assertIs(
            runtime_arguments[0]["controller"], controller_instances[0]
        )
        self.assertIs(
            runtime_arguments[1]["controller"], controller_instances[1]
        )
        self.assertEqual(factory.trajectory_controller_backend, "cartesian_p")

    def test_builds_each_runtime_from_preloaded_assets_and_profile_normalizers(self):
        calls = []

        class RecordingRuntime:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        profile = {
            "machine_id": "scale_excavator_v1",
            "observation_schema": {
                "normalizers": {
                    "target_threshold": 0.03,
                    "tube_radius": 0.04,
                }
            },
        }
        mission = {
            "schema_version": "excavation_mission.v1",
            "mission_id": "mission-1",
            "frame_id": "machine_root_ros",
            "limits": {
                "waypoint_tolerance_m": 0.25,
                "waypoint_dwell_s": 0.0,
                "tracking_timeout_s": 60.0,
            },
        }
        factory = EdgeFollowRuntimeFactory(
            machine_profile=profile,
            kinematics=object(),
            policy=object(),
            mission=mission,
            mission_sha256="a" * 64,
            runtime_type=RecordingRuntime,
            action_slew_rate_per_s=2.0,
        )
        snapshot = FollowTrajectorySnapshot.from_mapping(
            trajectory_snapshot(),
            now_s=101.0,
        )

        first = factory.create(snapshot)
        second = factory.create(snapshot)

        self.assertIsNot(first, second)
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0]["machine_profile"], profile)
        self.assertEqual(calls[0]["trajectory"]["target_threshold"], 0.03)
        self.assertEqual(calls[0]["trajectory"]["tube_radius"], 0.04)
        self.assertEqual(calls[0]["action_slew_rate_per_s"], 2.0)
        self.assertNotIn("target_threshold", trajectory_snapshot())
        self.assertNotIn("tube_radius", trajectory_snapshot())

    def test_remote_demo_trajectory_uses_snapshot_identity_and_limits(self):
        calls = []

        class RecordingRuntime:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        profile = {
            "machine_id": "scale_excavator_v1",
            "observation_schema": {
                "normalizers": {
                    "target_threshold": 0.03,
                    "tube_radius": 0.04,
                }
            },
        }
        deployment_mission = {
            "schema_version": "excavation_mission.v1",
            "mission_id": "field_cycle_001",
            "frame_id": "machine_root_ros",
            "limits": {
                "waypoint_tolerance_m": 0.03,
                "waypoint_dwell_s": 0.0,
                "tracking_timeout_s": 30.0,
            },
        }
        factory = EdgeFollowRuntimeFactory(
            machine_profile=profile,
            kinematics=object(),
            policy=object(),
            mission=deployment_mission,
            mission_sha256="f" * 64,
            runtime_type=RecordingRuntime,
        )
        values = trajectory_snapshot()
        values.update(
            {
                "mission_id": "field_demo_001",
                "mission_sha256": "d" * 64,
                "waypoint_tolerance_m": 0.25,
                "tracking_timeout_s": 60.0,
            }
        )
        values["trajectory_sha256"] = FollowTrajectorySnapshot(
            **{
                name: (
                    tuple(tuple(point) for point in value)
                    if name == "waypoints"
                    else value
                )
                for name, value in values.items()
                if name not in {"trajectory_sha256"}
            },
            trajectory_sha256="0" * 64,
        ).computed_sha256()
        snapshot = FollowTrajectorySnapshot.from_mapping(values, now_s=101.0)

        factory.create(snapshot)

        runtime_mission = calls[0]["mission"]
        self.assertEqual(runtime_mission["mission_id"], "field_demo_001")
        self.assertEqual(
            runtime_mission["limits"],
            {
                "waypoint_tolerance_m": 0.25,
                "waypoint_dwell_s": 0.0,
                "tracking_timeout_s": 60.0,
            },
        )


def start_request(session_id="session-1", request_id="start-1", seq=0):
    return {
        "schema_version": "orin_behavior_rpc.v1",
        "type": "start_follow",
        "session_id": session_id,
        "seq": seq,
        "request_id": request_id,
        "trajectory": trajectory_snapshot(),
    }


def fixed_action_request(
    behavior="ExecuteDump",
    session_id="fixed-session-1",
    request_id="fixed-start-1",
    seq=0,
):
    return {
        "schema_version": "orin_behavior_rpc.v1",
        "type": "start_fixed_action",
        "session_id": session_id,
        "seq": seq,
        "request_id": request_id,
        "behavior": behavior,
    }


def follow_step(*, result="ACTIVE"):
    return EdgeFollowStep(
        source_seq=4,
        source_stamp_ms=101250,
        waypoint_index=1,
        completed=result == "COMPLETED",
        bucket_tip_ros_m=(0.11, 0.22, 0.33),
        bucket_pitch_rad=0.4,
        observation=tuple(float(index) for index in range(38)),
        normalized_action=(0.1, 0.2, 0.3, 0.4),
        physical_action=(0.01, 0.02, 0.03, 0.04),
        waypoint_distance_m=0.12,
        follow_elapsed_s=2.5,
        result=result,
        trajectory_controller_backend="test_controller",
        trajectory_waypoints_ros_m=(
            (0.11, 0.22, 0.33),
            (0.5, 0.1, 0.2),
            (1.0, 0.0, 0.0),
        ),
    )


def safe_machine_state(*, seq=1):
    return {
        "seq": seq,
        "safety": {
            "control_enabled": True,
            "sensor_valid": True,
            "stm32_alive": True,
            "estop": False,
            "fault_flags": [],
        },
    }


class EdgeBehaviorExecutorTest(unittest.TestCase):
    def test_run_when_idle_serializes_resident_act_claim_with_behavior_start(self):
        authorized = [True]
        events = []
        executor = EdgeBehaviorExecutor(
            runtime_factory=SimpleNamespace(create=lambda snapshot: object()),
            runner_factory=lambda runtime: SimpleNamespace(
                action_datagrams=0,
                close=lambda **kwargs: None,
            ),
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: 7.0,
            sender_constructed=True,
            resident_rl_authorized=lambda: authorized[0],
        )
        executor.observe(safe_machine_state())

        def claim_act():
            authorized[0] = False
            return 17

        self.assertEqual(executor.run_when_idle(claim_act), 17)
        executor.start(start_request(), events.append)

        self.assertEqual(events[-1]["type"], "rejected")
        self.assertEqual(events[-1]["message"], "resident_rl_not_active")
        self.assertFalse(executor.busy)

    def test_run_when_idle_rejects_act_claim_during_active_rl_behavior(self):
        calls = []
        executor = EdgeBehaviorExecutor(
            runtime_factory=SimpleNamespace(create=lambda snapshot: object()),
            runner_factory=lambda runtime: SimpleNamespace(
                action_datagrams=0,
                close=lambda **kwargs: None,
            ),
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: 7.0,
            sender_constructed=True,
        )
        executor.observe(safe_machine_state())
        executor.start(start_request(), lambda _event: None)

        with self.assertRaisesRegex(RuntimeError, "behavior is active"):
            executor.run_when_idle(lambda: calls.append("act"))

        self.assertEqual(calls, [])

    def test_resident_mission_gate_rejects_follow_before_rl_activation(self):
        created = []

        class Factory:
            def create(self, snapshot):
                created.append(snapshot.trajectory_id)
                return object()

        events = []
        executor = EdgeBehaviorExecutor(
            runtime_factory=Factory(),
            runner_factory=lambda runtime: object(),
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: 7.0,
            sender_constructed=True,
            resident_rl_authorized=lambda: False,
        )
        executor.observe(safe_machine_state())

        executor.start(start_request(), events.append)

        self.assertEqual(events[-1]["type"], "rejected")
        self.assertEqual(events[-1]["reason_code"], "MOTION_NOT_READY")
        self.assertEqual(events[-1]["message"], "resident_rl_not_active")
        self.assertEqual(created, [])
        self.assertFalse(executor.busy)
        self.assertEqual(
            executor.status_event()["motion_gate_reason"],
            "resident_rl_not_active",
        )

    def test_resident_mission_gate_rejects_fixed_action_before_rl_activation(self):
        created = []

        class FixedFactory:
            profile = SimpleNamespace(validation_status="candidate")

            def create(self, behavior):
                created.append(behavior)
                return object()

        events = []
        executor = EdgeBehaviorExecutor(
            runtime_factory=object(),
            runner_factory=lambda runtime: object(),
            fixed_action_factory=FixedFactory(),
            fixed_runner_factory=lambda runtime, behavior: object(),
            monotonic_clock=lambda: 7.0,
            sender_constructed=True,
            resident_rl_authorized=lambda: False,
        )
        executor.observe(safe_machine_state())

        executor.handle(fixed_action_request(), events.append)

        self.assertEqual(events[-1]["type"], "rejected")
        self.assertEqual(events[-1]["reason_code"], "MOTION_NOT_READY")
        self.assertEqual(events[-1]["message"], "resident_rl_not_active")
        self.assertEqual(created, [])
        self.assertFalse(executor.busy)

    def test_resident_mission_gate_allows_follow_for_active_rl_generation(self):
        events = []
        executor = EdgeBehaviorExecutor(
            runtime_factory=SimpleNamespace(create=lambda snapshot: object()),
            runner_factory=lambda runtime: SimpleNamespace(
                action_datagrams=0,
                close=lambda **kwargs: None,
            ),
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: 7.0,
            sender_constructed=True,
            resident_rl_authorized=lambda: True,
        )
        executor.observe(safe_machine_state())

        executor.start(start_request(), events.append)

        self.assertEqual(events[-1]["type"], "accepted")
        self.assertTrue(executor.busy)

    def test_resident_mission_gate_stops_follow_before_observe_after_rl_revocation(self):
        authorized = [True]
        observed = []
        closed = []

        class Runner:
            action_datagrams = 0

            def observe(self, state, *, now_s, action_stamp_ms):
                observed.append(state)
                return follow_step()

            def close(self, *, action_stamp_ms):
                closed.append(action_stamp_ms)

        events = []
        executor = EdgeBehaviorExecutor(
            runtime_factory=SimpleNamespace(create=lambda snapshot: object()),
            runner_factory=lambda runtime: Runner(),
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: 7.0,
            action_stamp_clock=lambda: 101000,
            sender_constructed=True,
            resident_rl_authorized=lambda: authorized[0],
        )
        executor.observe(safe_machine_state())
        executor.start(start_request(), events.append)
        authorized[0] = False

        executor.observe(safe_machine_state(seq=2))

        self.assertEqual(observed, [])
        self.assertEqual(closed, [101000])
        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["outcome"], "FAILED")
        self.assertEqual(events[-1]["reason_code"], "MOTION_GATE_CLOSED")
        self.assertEqual(events[-1]["message"], "resident_rl_not_active")
        self.assertFalse(executor.busy)

    def test_status_reflects_latest_machine_safety_while_idle(self):
        executor = EdgeBehaviorExecutor(
            runtime_factory=object(),
            runner_factory=lambda runtime: object(),
            monotonic_clock=lambda: 7.0,
            sender_constructed=True,
        )
        executor.observe(
            {
                "safety": {
                    "control_enabled": True,
                    "sensor_valid": True,
                    "stm32_alive": True,
                    "estop": False,
                    "fault_flags": [],
                }
            }
        )

        status = executor.status_event()

        self.assertEqual(status["type"], "status")
        self.assertTrue(status["state_fresh"])
        self.assertTrue(status["control_enabled"])
        self.assertTrue(status["sensor_valid"])
        self.assertTrue(status["stm32_alive"])
        self.assertFalse(status["estop"])
        self.assertTrue(status["fault_free"])
        self.assertTrue(status["quiescent"])
        self.assertEqual(status["active_behavior"], "")
        self.assertEqual(status["action_datagrams"], 0)
        self.assertTrue(status["sender_constructed"])
        self.assertEqual(status["motion_gate_reason"], "ready")

    def test_start_feedback_completion_and_busy_are_serialized_and_quiescent(self):
        call_order = []

        class Factory:
            def create(self, snapshot):
                call_order.append(("create", snapshot.trajectory_id))
                return object()

        class Runner:
            action_datagrams = 3

            def __init__(self):
                self.steps = [follow_step(), follow_step(result="COMPLETED")]

            def observe(self, state, *, now_s, action_stamp_ms):
                call_order.append(("observe", state["seq"]))
                return self.steps.pop(0)

            def close(self, *, action_stamp_ms):
                call_order.append(("close", action_stamp_ms))

        runner = Runner()
        events = []
        executor = EdgeBehaviorExecutor(
            runtime_factory=Factory(),
            runner_factory=lambda runtime: runner,
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: 7.0,
            action_stamp_clock=lambda: 101000,
            sender_constructed=True,
        )
        executor.observe(
            safe_machine_state()
        )

        executor.start(start_request(), events.append)
        executor.start(
            start_request(session_id="session-2", request_id="start-2"),
            events.append,
        )
        active_status = executor.status_event()
        self.assertFalse(active_status["quiescent"])
        self.assertEqual(active_status["active_behavior"], "Follow")
        self.assertEqual(active_status["motion_gate_reason"], "behavior_active")
        executor.observe({"seq": 4})
        executor.observe({"seq": 5})

        self.assertEqual(events[0]["type"], "accepted")
        self.assertEqual(events[1]["type"], "rejected")
        self.assertEqual(events[1]["reason_code"], "BUSY")
        feedback = events[2]
        self.assertEqual(feedback["bucket_tip_stamp_s"], 101.25)
        self.assertEqual(feedback["bucket_tip"], [0.11, 0.22, 0.33])
        self.assertEqual(feedback["tracking_state"], "ACTIVE")
        self.assertEqual(feedback["action_datagrams"], 3)
        self.assertEqual(
            feedback["trajectory_waypoints"],
            [[0.11, 0.22, 0.33], [0.5, 0.1, 0.2], [1.0, 0.0, 0.0]],
        )
        self.assertEqual(feedback["waypoint_tolerance_m"], 0.0)
        self.assertEqual(
            feedback["trajectory_controller_backend"],
            "test_controller",
        )
        result = events[3]
        self.assertEqual(result["type"], "result")
        self.assertEqual(result["outcome"], "SUCCEEDED")
        self.assertTrue(result["quiescence_confirmed"])
        self.assertEqual(result["action_datagrams"], 3)
        self.assertEqual(
            result["trajectory_controller_backend"],
            "test_controller",
        )
        self.assertEqual(call_order[-1][0], "close")
        self.assertFalse(executor.busy)

        executor.start(start_request(), events.append)
        self.assertEqual(events[-1]["type"], "rejected")
        self.assertEqual(events[-1]["reason_code"], "OUT_OF_ORDER")

    def test_start_rejects_stale_machine_state_before_constructing_runner(self):
        now = [1.0]
        create_calls = []

        class Factory:
            def create(self, snapshot):
                create_calls.append(snapshot.trajectory_id)
                return object()

        executor = EdgeBehaviorExecutor(
            runtime_factory=Factory(),
            runner_factory=lambda runtime: object(),
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: now[0],
            sender_constructed=True,
            state_timeout_s=0.3,
        )
        executor.observe(safe_machine_state())
        now[0] = 2.0
        events = []

        executor.start(start_request(), events.append)

        self.assertEqual(events[-1]["type"], "rejected")
        self.assertEqual(events[-1]["reason_code"], "MOTION_NOT_READY")
        self.assertEqual(events[-1]["message"], "state_stale")
        self.assertEqual(create_calls, [])
        self.assertFalse(executor.busy)

    def test_watchdog_closes_active_follow_when_machine_state_becomes_stale(self):
        now = [1.0]
        call_order = []

        class Factory:
            def create(self, snapshot):
                return object()

        class Runner:
            action_datagrams = 3

            def close(self, *, action_stamp_ms):
                call_order.append("close")

        def emit(event):
            call_order.append(event["type"])
            events.append(event)

        events = []
        executor = EdgeBehaviorExecutor(
            runtime_factory=Factory(),
            runner_factory=lambda runtime: Runner(),
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: now[0],
            action_stamp_clock=lambda: 101000,
            sender_constructed=True,
            state_timeout_s=0.3,
        )
        executor.observe(safe_machine_state())
        executor.start(start_request(), emit)
        now[0] = 2.0

        executor.watchdog()

        self.assertEqual(call_order[-2:], ["close", "result"])
        self.assertEqual(events[-1]["outcome"], "FAILED")
        self.assertEqual(events[-1]["reason_code"], "MOTION_GATE_CLOSED")
        self.assertEqual(events[-1]["message"], "state_stale")
        self.assertTrue(events[-1]["quiescence_confirmed"])
        self.assertFalse(executor.busy)

    def test_cancel_closes_runner_before_emitting_cancelled_result(self):
        call_order = []

        class Factory:
            def create(self, snapshot):
                return object()

        class Runner:
            action_datagrams = 0

            def close(self, *, action_stamp_ms):
                call_order.append("close")

        events = []

        def emit(event):
            call_order.append(event["type"])
            events.append(event)

        executor = EdgeBehaviorExecutor(
            runtime_factory=Factory(),
            runner_factory=lambda runtime: Runner(),
            wall_clock=lambda: 101.0,
            action_stamp_clock=lambda: 101000,
            sender_constructed=True,
        )
        executor.observe(safe_machine_state())
        executor.start(start_request(), emit)

        executor.cancel(
            {
                "schema_version": "orin_behavior_rpc.v1",
                "type": "cancel_follow",
                "session_id": "session-1",
                "seq": 1,
                "request_id": "cancel-1",
            }
        )

        self.assertEqual(call_order[-2:], ["close", "result"])
        self.assertEqual(events[-1]["outcome"], "CANCELLED")
        self.assertTrue(events[-1]["quiescence_confirmed"])

    def test_fixed_action_feedback_completion_and_mutual_exclusion_use_same_executor(self):
        created = []
        closed = []

        class FixedFactory:
            profile = SimpleNamespace(validation_status="candidate")

            def create(self, behavior):
                created.append(behavior)
                return object()

        class FixedRunner:
            action_datagrams = 2

            def __init__(self):
                self.steps = [
                    SimpleNamespace(
                        phase="running",
                        step_index=0,
                        step_label="open_bucket",
                        max_error=0.4,
                        result="ACTIVE",
                        reason_code="",
                    ),
                    SimpleNamespace(
                        phase="done",
                        step_index=1,
                        step_label="recover_bucket",
                        max_error=0.0,
                        result="COMPLETED",
                        reason_code="SEQUENCE_COMPLETED",
                    ),
                ]

            def observe(self, state, *, now_s, action_stamp_ms):
                return self.steps.pop(0)

            def close(self, *, action_stamp_ms):
                closed.append(action_stamp_ms)

        events = []
        executor = EdgeBehaviorExecutor(
            runtime_factory=object(),
            runner_factory=lambda runtime: object(),
            fixed_action_factory=FixedFactory(),
            fixed_runner_factory=lambda runtime, behavior: FixedRunner(),
            monotonic_clock=lambda: 7.0,
            action_stamp_clock=lambda: 101000,
            sender_constructed=True,
        )
        executor.observe(safe_machine_state())

        executor.handle(fixed_action_request(), events.append)
        active_status = executor.status_event()
        executor.handle(
            start_request(session_id="follow-session", request_id="follow-start"),
            events.append,
        )
        executor.observe(safe_machine_state(seq=2))
        executor.observe(safe_machine_state(seq=3))

        self.assertEqual(created, ["ExecuteDump"])
        self.assertEqual(active_status["active_behavior"], "ExecuteDump")
        self.assertTrue(active_status["fixed_actions_available"])
        self.assertFalse(active_status["fixed_actions_validated"])
        self.assertEqual(events[0]["type"], "accepted")
        self.assertEqual(events[0]["behavior"], "ExecuteDump")
        self.assertEqual(events[1]["reason_code"], "BUSY")
        self.assertEqual(events[2]["type"], "feedback")
        self.assertEqual(events[2]["step_label"], "open_bucket")
        self.assertEqual(events[3]["type"], "result")
        self.assertEqual(events[3]["outcome"], "SUCCEEDED")
        self.assertEqual(events[3]["reason_code"], "SEQUENCE_COMPLETED")
        self.assertTrue(events[3]["quiescence_confirmed"])
        self.assertEqual(closed, [101000])

    def test_fixed_action_watchdog_before_first_observation_reports_not_started(self):
        now = [1.0]

        class FixedFactory:
            profile = SimpleNamespace(validation_status="candidate")

            def create(self, behavior):
                return object()

        class FixedRunner:
            action_datagrams = 0

            def close(self, *, action_stamp_ms):
                return None

        events = []
        executor = EdgeBehaviorExecutor(
            runtime_factory=object(),
            runner_factory=lambda runtime: object(),
            fixed_action_factory=FixedFactory(),
            fixed_runner_factory=lambda runtime, behavior: FixedRunner(),
            monotonic_clock=lambda: now[0],
            action_stamp_clock=lambda: 101000,
            sender_constructed=True,
            state_timeout_s=0.3,
        )
        executor.observe(safe_machine_state())
        executor.handle(fixed_action_request(), events.append)
        now[0] = 2.0

        executor.watchdog()

        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["outcome"], "FAILED")
        self.assertEqual(events[-1]["reason_code"], "MOTION_GATE_CLOSED")
        self.assertEqual(events[-1]["final_step_label"], "not_started")


class RemoteBehaviorServerTest(unittest.TestCase):
    def test_connection_lifecycle_logs_peer_and_event_summary(self):
        class Executor:
            def status_event(self):
                return {
                    "schema_version": "orin_behavior_rpc.v1",
                    "type": "status",
                }

            def watchdog(self):
                return None

            def handle(self, _request, _emit):
                return None

            def disconnect(self, _emit):
                return None

            def close(self, *, emit_result=False):
                return None

        server = RemoteBehaviorServer(
            bind_host="127.0.0.1",
            bind_port=0,
            allowed_client_host="127.0.0.1",
            executor=Executor(),
            status_interval_s=0.01,
        )
        client, server_side = socket.socketpair()
        client.settimeout(0.5)
        with self.assertLogs("orin_edge_remote", level="INFO") as captured:
            thread = threading.Thread(
                target=server.serve_connection,
                args=(server_side,),
                kwargs={
                    "connection_id": "rpc-test-002",
                    "peer": "127.0.0.1:43210",
                },
            )
            thread.start()
            try:
                self.assertEqual(receive_message(client)["type"], "status")
                send_message(
                    client,
                    {
                        "schema_version": "orin_behavior_rpc.v1",
                        "type": "diagnostic_probe",
                        "session_id": "session-test-002",
                        "seq": 7,
                        "request_id": "request-test-002",
                    },
                )
            finally:
                client.close()
                thread.join(timeout=1.0)
                server_side.close()

        self.assertFalse(thread.is_alive())
        diagnostic = "\n".join(captured.output)
        self.assertIn(
            "opened: connection_id=rpc-test-002 peer=127.0.0.1:43210",
            diagnostic,
        )
        self.assertIn(
            "closed: connection_id=rpc-test-002 peer=127.0.0.1:43210",
            diagnostic,
        )
        self.assertIn("reason=peer_eof", diagnostic)
        self.assertIn("events_sent=1", diagnostic)
        self.assertIn("last_event_type=status", diagnostic)
        self.assertIn("last_event_seq=0", diagnostic)
        self.assertIn(
            "request received: connection_id=rpc-test-002",
            diagnostic,
        )
        self.assertIn("type=diagnostic_probe", diagnostic)
        self.assertIn("session_id=session-test-002", diagnostic)
        self.assertIn("request_id=request-test-002", diagnostic)
        self.assertIn("request_seq=7", diagnostic)

    def test_fragmented_start_and_cancel_before_observe_return_finite_result(self):
        class Factory:
            def create(self, snapshot):
                return object()

        class Runner:
            action_datagrams = 0

            def close(self, *, action_stamp_ms):
                return None

        executor = EdgeBehaviorExecutor(
            runtime_factory=Factory(),
            runner_factory=lambda runtime: Runner(),
            wall_clock=lambda: 101.0,
            sender_constructed=True,
        )
        executor.observe(safe_machine_state())
        server = RemoteBehaviorServer(
            bind_host="127.0.0.1",
            bind_port=0,
            allowed_client_host="127.0.0.1",
            executor=executor,
            status_interval_s=0.01,
        )
        client, server_side = socket.socketpair()
        client.settimeout(1.0)
        thread = threading.Thread(
            target=server.serve_connection,
            args=(server_side,),
        )
        thread.start()
        try:
            self.assertEqual(receive_message(client)["type"], "status")
            payload = json.dumps(
                start_request(),
                separators=(",", ":"),
            ).encode("utf-8")
            frame = len(payload).to_bytes(4, "big") + payload
            client.sendall(frame[:2])
            time.sleep(0.03)
            client.sendall(frame[2:])
            accepted = receive_message(client)
            self.assertEqual(accepted["type"], "accepted")

            send_message(
                client,
                {
                    "schema_version": "orin_behavior_rpc.v1",
                    "type": "cancel_follow",
                    "session_id": "session-1",
                    "seq": 1,
                    "request_id": "start-1",
                },
            )
            result = receive_message(client)
            while result["type"] == "status":
                result = receive_message(client)
            self.assertEqual(result["type"], "result")
            self.assertEqual(result["outcome"], "CANCELLED")
            self.assertEqual(result["final_distance_m"], -1.0)
            self.assertTrue(result["quiescence_confirmed"])
        finally:
            client.close()
            thread.join(timeout=1.0)
            server_side.close()

    def test_status_connection_does_not_block_or_cancel_follow_connection(self):
        closed = threading.Event()

        class Factory:
            def create(self, snapshot):
                return object()

        class Runner:
            action_datagrams = 0

            def close(self, *, action_stamp_ms):
                closed.set()

        executor = EdgeBehaviorExecutor(
            runtime_factory=Factory(),
            runner_factory=lambda runtime: Runner(),
            wall_clock=lambda: 101.0,
            sender_constructed=True,
        )
        executor.observe(safe_machine_state())
        server = RemoteBehaviorServer(
            bind_host="127.0.0.1",
            bind_port=0,
            allowed_client_host="127.0.0.1",
            executor=executor,
        )
        server.start()
        self.addCleanup(server.close)
        status_client = socket.create_connection(
            ("127.0.0.1", server.bound_port),
            timeout=1.0,
        )
        follow_client = socket.create_connection(
            ("127.0.0.1", server.bound_port),
            timeout=1.0,
        )
        try:
            status_event = receive_message(status_client)
            follow_status = receive_message(follow_client)
            self.assertEqual(status_event["type"], "status")
            self.assertEqual(follow_status["type"], "status")
            self.assertEqual(status_event["seq"], 0)
            self.assertEqual(follow_status["seq"], 0)
            send_message(follow_client, start_request())
            event = receive_message(follow_client)
            while event["type"] == "status":
                event = receive_message(follow_client)
            self.assertEqual(event["type"], "accepted")

            status_client.close()
            self.assertTrue(executor.busy)
            self.assertFalse(closed.is_set())

            follow_client.close()
            self.assertTrue(closed.wait(1.0))
            self.assertFalse(executor.busy)
        finally:
            status_client.close()
            follow_client.close()

    def test_connection_accepts_one_goal_and_disconnect_closes_it_fail_safe(self):
        closed = threading.Event()

        class Factory:
            def create(self, snapshot):
                return object()

        class Runner:
            action_datagrams = 0

            def close(self, *, action_stamp_ms):
                closed.set()

        executor = EdgeBehaviorExecutor(
            runtime_factory=Factory(),
            runner_factory=lambda runtime: Runner(),
            wall_clock=lambda: 101.0,
            sender_constructed=True,
        )
        executor.observe(safe_machine_state())
        server = RemoteBehaviorServer(
            bind_host="127.0.0.1",
            bind_port=0,
            allowed_client_host="127.0.0.1",
            executor=executor,
        )
        client, server_side = socket.socketpair()
        client.settimeout(1.0)
        thread = threading.Thread(
            target=server.serve_connection,
            args=(server_side,),
        )
        thread.start()
        try:
            initial_status = receive_message(client)
            self.assertEqual(initial_status["type"], "status")
            send_message(client, start_request())
            accepted = receive_message(client)
            while accepted["type"] == "status":
                accepted = receive_message(client)
            self.assertEqual(accepted["type"], "accepted")
            self.assertTrue(executor.busy)
        finally:
            client.close()
            thread.join(timeout=1.0)
            server_side.close()

        self.assertFalse(thread.is_alive())
        self.assertTrue(closed.is_set())
        self.assertFalse(executor.busy)


if __name__ == "__main__":
    unittest.main()
