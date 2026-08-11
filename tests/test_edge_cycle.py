import unittest
from types import SimpleNamespace

from edge_runtime.cycle import EdgeCycleCoordinator, EdgeExcavationCycle
from edge_runtime.remote import EdgeBehaviorExecutor
from tests.test_edge_remote import (
    follow_step,
    safe_machine_state,
    trajectory_snapshot,
)


def trajectory(trajectory_id: str, mission_phase: str):
    return {
        "trajectory_id": trajectory_id,
        "mission_phase": mission_phase,
        "waypoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    }


class EdgeExcavationCycleTest(unittest.TestCase):
    def test_dig_half_cycle_advances_only_after_each_behavior_is_quiescent(self):
        cycle = EdgeExcavationCycle()

        follow = cycle.start(
            cycle_id="cycle-1",
            dig_trajectory=trajectory("dig-trajectory", "dig"),
        )

        self.assertEqual(follow.stage, "FOLLOW_DIG")
        self.assertEqual(follow.behavior, "Follow")
        self.assertEqual(follow.trajectory["trajectory_id"], "dig-trajectory")
        self.assertEqual(cycle.status.stage, "FOLLOW_DIG")

        execute_dig = cycle.record_behavior_result(
            outcome="SUCCEEDED",
            reason_code="SUCCEEDED",
            quiescence_confirmed=True,
        )

        self.assertEqual(execute_dig.stage, "EXECUTE_DIG")
        self.assertEqual(execute_dig.behavior, "ExecuteDig")
        self.assertIsNone(execute_dig.trajectory)

        next_directive = cycle.record_behavior_result(
            outcome="SUCCEEDED",
            reason_code="SEQUENCE_COMPLETED",
            quiescence_confirmed=True,
        )

        self.assertIsNone(next_directive)
        self.assertEqual(cycle.status.stage, "WAITING_FOR_DUMP_TRAJECTORY")
        self.assertEqual(cycle.status.active_behavior, "")
        self.assertFalse(cycle.status.terminal)

    def test_dump_trajectory_resumes_waiting_cycle_and_completes_locally(self):
        cycle = EdgeExcavationCycle()
        cycle.start(
            cycle_id="cycle-1",
            dig_trajectory=trajectory("dig-trajectory", "dig"),
        )
        cycle.record_behavior_result(
            outcome="SUCCEEDED",
            reason_code="SUCCEEDED",
            quiescence_confirmed=True,
        )
        cycle.record_behavior_result(
            outcome="SUCCEEDED",
            reason_code="SEQUENCE_COMPLETED",
            quiescence_confirmed=True,
        )

        follow_dump = cycle.provide_dump_trajectory(
            trajectory("dump-trajectory", "dump")
        )

        self.assertEqual(follow_dump.stage, "FOLLOW_DUMP")
        self.assertEqual(follow_dump.behavior, "Follow")
        self.assertEqual(
            follow_dump.trajectory["trajectory_id"],
            "dump-trajectory",
        )

        execute_dump = cycle.record_behavior_result(
            outcome="SUCCEEDED",
            reason_code="SUCCEEDED",
            quiescence_confirmed=True,
        )
        self.assertEqual(execute_dump.stage, "EXECUTE_DUMP")
        self.assertEqual(execute_dump.behavior, "ExecuteDump")

        directive = cycle.record_behavior_result(
            outcome="SUCCEEDED",
            reason_code="SEQUENCE_COMPLETED",
            quiescence_confirmed=True,
        )

        self.assertIsNone(directive)
        self.assertEqual(cycle.status.stage, "COMPLETED")
        self.assertTrue(cycle.status.terminal)
        self.assertEqual(cycle.status.outcome, "SUCCEEDED")
        self.assertEqual(cycle.status.reason_code, "SEQUENCE_COMPLETED")

    def test_child_failure_stops_the_cycle_without_issuing_another_behavior(self):
        cycle = EdgeExcavationCycle()
        cycle.start(
            cycle_id="cycle-1",
            dig_trajectory=trajectory("dig-trajectory", "dig"),
        )

        directive = cycle.record_behavior_result(
            outcome="FAILED",
            reason_code="TRACKING_TIMEOUT",
            quiescence_confirmed=True,
        )

        self.assertIsNone(directive)
        self.assertEqual(cycle.status.stage, "FAILED")
        self.assertTrue(cycle.status.terminal)
        self.assertEqual(cycle.status.outcome, "FAILED")
        self.assertEqual(cycle.status.reason_code, "TRACKING_TIMEOUT")
        with self.assertRaisesRegex(RuntimeError, "not waiting"):
            cycle.provide_dump_trajectory(
                trajectory("dump-trajectory", "dump")
            )

    def test_cancel_is_a_terminal_cycle_result(self):
        cycle = EdgeExcavationCycle()
        cycle.start(
            cycle_id="cycle-1",
            dig_trajectory=trajectory("dig-trajectory", "dig"),
        )

        cycle.cancel()

        self.assertEqual(cycle.status.stage, "CANCELLED")
        self.assertTrue(cycle.status.terminal)
        self.assertEqual(cycle.status.outcome, "CANCELLED")
        self.assertEqual(cycle.status.reason_code, "CANCELLED")

    def test_completed_cycle_can_be_replaced_by_the_next_demo_cycle(self):
        cycle = EdgeExcavationCycle()
        cycle.start(
            cycle_id="cycle-1",
            dig_trajectory=trajectory("dig-1", "dig"),
        )
        cycle.record_behavior_result(
            outcome="SUCCEEDED",
            reason_code="SUCCEEDED",
            quiescence_confirmed=True,
        )
        cycle.record_behavior_result(
            outcome="SUCCEEDED",
            reason_code="SEQUENCE_COMPLETED",
            quiescence_confirmed=True,
        )
        cycle.provide_dump_trajectory(trajectory("dump-1", "dump"))
        cycle.record_behavior_result(
            outcome="SUCCEEDED",
            reason_code="SUCCEEDED",
            quiescence_confirmed=True,
        )
        cycle.record_behavior_result(
            outcome="SUCCEEDED",
            reason_code="SEQUENCE_COMPLETED",
            quiescence_confirmed=True,
        )

        directive = cycle.start(
            cycle_id="cycle-2",
            dig_trajectory=trajectory("dig-2", "dig"),
        )

        self.assertEqual(directive.stage, "FOLLOW_DIG")
        self.assertEqual(cycle.status.cycle_id, "cycle-2")
        self.assertFalse(cycle.status.terminal)


class EdgeCycleCoordinatorTest(unittest.TestCase):
    def test_existing_behavior_executor_runs_both_local_legs_without_pc_action_round_trips(self):
        created = []
        closed = []

        class FollowFactory:
            def create(self, snapshot):
                created.append(("Follow", snapshot.trajectory_id))
                return object()

        class FollowRunner:
            action_datagrams = 4

            def observe(self, state, *, now_s, action_stamp_ms):
                return follow_step(result="COMPLETED")

            def close(self, *, action_stamp_ms):
                closed.append("Follow")

        class FixedFactory:
            profile = SimpleNamespace(validation_status="field_validated")

            def create(self, behavior):
                created.append((behavior, None))
                return object()

        class FixedRunner:
            action_datagrams = 3

            def __init__(self, behavior):
                self.behavior = behavior

            def observe(self, state, *, now_s, action_stamp_ms):
                return SimpleNamespace(
                    phase="done",
                    step_index=2,
                    step_label="lift_boom",
                    max_error=0.0,
                    result="COMPLETED",
                    reason_code="SEQUENCE_COMPLETED",
                )

            def close(self, *, action_stamp_ms):
                closed.append(self.behavior)

        behavior_executor = EdgeBehaviorExecutor(
            runtime_factory=FollowFactory(),
            runner_factory=lambda runtime: FollowRunner(),
            fixed_action_factory=FixedFactory(),
            fixed_runner_factory=lambda runtime, behavior: FixedRunner(behavior),
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: 7.0,
            action_stamp_clock=lambda: 101000,
            sender_constructed=True,
        )
        coordinator = EdgeCycleCoordinator(behavior_executor)
        coordinator.observe(safe_machine_state())

        coordinator.start(
            cycle_id="cycle-1",
            dig_trajectory=trajectory_snapshot(),
        )
        coordinator.observe(safe_machine_state(seq=2))

        self.assertEqual(coordinator.status.stage, "EXECUTE_DIG")
        self.assertTrue(behavior_executor.busy)

        coordinator.observe(safe_machine_state(seq=3))

        self.assertEqual(
            coordinator.status.stage,
            "WAITING_FOR_DUMP_TRAJECTORY",
        )
        self.assertFalse(behavior_executor.busy)
        self.assertEqual(
            created,
            [("Follow", "trajectory-1"), ("ExecuteDig", None)],
        )
        self.assertEqual(closed, ["Follow", "ExecuteDig"])

        coordinator.provide_dump_trajectory(trajectory_snapshot())
        coordinator.observe(safe_machine_state(seq=4))
        self.assertEqual(coordinator.status.stage, "EXECUTE_DUMP")
        coordinator.observe(safe_machine_state(seq=5))

        self.assertEqual(coordinator.status.stage, "COMPLETED")
        self.assertTrue(coordinator.status.terminal)
        self.assertEqual(
            created,
            [
                ("Follow", "trajectory-1"),
                ("ExecuteDig", None),
                ("Follow", "trajectory-1"),
                ("ExecuteDump", None),
            ],
        )
        self.assertEqual(
            closed,
            ["Follow", "ExecuteDig", "Follow", "ExecuteDump"],
        )

    def test_rpc_boundary_returns_after_each_orin_local_leg(self):
        class FollowFactory:
            def create(self, snapshot):
                return object()

        class FollowRunner:
            action_datagrams = 2

            def observe(self, state, *, now_s, action_stamp_ms):
                return follow_step(result="COMPLETED")

            def close(self, *, action_stamp_ms):
                return None

        class FixedFactory:
            profile = SimpleNamespace(validation_status="field_validated")

            def create(self, behavior):
                return object()

        class FixedRunner:
            action_datagrams = 3

            def observe(self, state, *, now_s, action_stamp_ms):
                return SimpleNamespace(
                    phase="done",
                    step_index=1,
                    step_label="complete",
                    max_error=0.0,
                    result="COMPLETED",
                    reason_code="SEQUENCE_COMPLETED",
                )

            def close(self, *, action_stamp_ms):
                return None

        executor = EdgeBehaviorExecutor(
            runtime_factory=FollowFactory(),
            runner_factory=lambda runtime: FollowRunner(),
            fixed_action_factory=FixedFactory(),
            fixed_runner_factory=lambda runtime, behavior: FixedRunner(),
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: 7.0,
            sender_constructed=True,
        )
        coordinator = EdgeCycleCoordinator(executor)
        coordinator.observe(safe_machine_state())
        dig_events = []

        coordinator.handle(
            {
                "schema_version": "orin_behavior_rpc.v1",
                "type": "start_cycle",
                "session_id": "pc-cycle-session",
                "seq": 0,
                "request_id": "dig-leg-request",
                "cycle_id": "cycle-1",
                "dig_trajectory": trajectory_snapshot(),
            },
            dig_events.append,
        )
        coordinator.observe(safe_machine_state(seq=2))
        coordinator.observe(safe_machine_state(seq=3))

        self.assertEqual(dig_events[0]["type"], "accepted")
        self.assertEqual(dig_events[0]["stage"], "FOLLOW_DIG")
        self.assertEqual(dig_events[-1]["type"], "result")
        self.assertEqual(dig_events[-1]["outcome"], "SUCCEEDED")
        self.assertEqual(
            dig_events[-1]["reason_code"],
            "DIG_LEG_COMPLETED",
        )
        self.assertEqual(
            dig_events[-1]["completed_stage"],
            "EXECUTE_DIG",
        )
        self.assertTrue(dig_events[-1]["quiescence_confirmed"])
        self.assertEqual(dig_events[-1]["action_datagrams"], 5)

        dump_events = []
        coordinator.handle(
            {
                "schema_version": "orin_behavior_rpc.v1",
                "type": "provide_dump_trajectory",
                "session_id": "pc-cycle-session",
                "seq": 1,
                "request_id": "dump-leg-request",
                "cycle_id": "cycle-1",
                "dump_trajectory": trajectory_snapshot(),
            },
            dump_events.append,
        )
        coordinator.observe(safe_machine_state(seq=4))
        coordinator.observe(safe_machine_state(seq=5))

        self.assertEqual(dump_events[0]["type"], "accepted")
        self.assertEqual(dump_events[0]["stage"], "FOLLOW_DUMP")
        self.assertEqual(dump_events[-1]["type"], "result")
        self.assertEqual(dump_events[-1]["outcome"], "SUCCEEDED")
        self.assertEqual(
            dump_events[-1]["reason_code"],
            "SEQUENCE_COMPLETED",
        )
        self.assertEqual(
            dump_events[-1]["completed_stage"],
            "EXECUTE_DUMP",
        )
        self.assertTrue(dump_events[-1]["quiescence_confirmed"])
        self.assertEqual(dump_events[-1]["action_datagrams"], 10)

    def test_cancel_cycle_closes_the_active_local_behavior_and_reports_quiescence(self):
        closed = []

        class FollowFactory:
            def create(self, snapshot):
                return object()

        class FollowRunner:
            action_datagrams = 2

            def close(self, *, action_stamp_ms):
                closed.append(action_stamp_ms)

        executor = EdgeBehaviorExecutor(
            runtime_factory=FollowFactory(),
            runner_factory=lambda runtime: FollowRunner(),
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: 7.0,
            action_stamp_clock=lambda: 101000,
            sender_constructed=True,
        )
        coordinator = EdgeCycleCoordinator(executor)
        coordinator.observe(safe_machine_state())
        events = []
        coordinator.handle(
            {
                "schema_version": "orin_behavior_rpc.v1",
                "type": "start_cycle",
                "session_id": "pc-cycle-session",
                "seq": 0,
                "request_id": "dig-leg-request",
                "cycle_id": "cycle-1",
                "dig_trajectory": trajectory_snapshot(),
            },
            events.append,
        )

        coordinator.handle(
            {
                "schema_version": "orin_behavior_rpc.v1",
                "type": "cancel_cycle",
                "session_id": "pc-cycle-session",
                "seq": 1,
                "request_id": "dig-leg-request",
                "cycle_id": "cycle-1",
            },
            events.append,
        )

        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["outcome"], "CANCELLED")
        self.assertEqual(events[-1]["reason_code"], "CANCELLED")
        self.assertTrue(events[-1]["quiescence_confirmed"])
        self.assertEqual(coordinator.status.stage, "CANCELLED")
        self.assertFalse(executor.busy)
        self.assertEqual(closed, [101000])

    def test_cancel_cycle_while_waiting_for_dump_trajectory_is_quiescent(self):
        class FollowFactory:
            def create(self, snapshot):
                return object()

        class FollowRunner:
            action_datagrams = 2

            def observe(self, state, *, now_s, action_stamp_ms):
                return follow_step(result="COMPLETED")

            def close(self, *, action_stamp_ms):
                return None

        class FixedFactory:
            profile = SimpleNamespace(validation_status="field_validated")

            def create(self, behavior):
                return object()

        class FixedRunner:
            action_datagrams = 3

            def observe(self, state, *, now_s, action_stamp_ms):
                return SimpleNamespace(
                    phase="done",
                    step_index=1,
                    step_label="complete",
                    max_error=0.0,
                    result="COMPLETED",
                    reason_code="SEQUENCE_COMPLETED",
                )

            def close(self, *, action_stamp_ms):
                return None

        executor = EdgeBehaviorExecutor(
            runtime_factory=FollowFactory(),
            runner_factory=lambda runtime: FollowRunner(),
            fixed_action_factory=FixedFactory(),
            fixed_runner_factory=lambda runtime, behavior: FixedRunner(),
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: 7.0,
            sender_constructed=True,
        )
        coordinator = EdgeCycleCoordinator(executor)
        coordinator.observe(safe_machine_state())
        coordinator.handle(
            {
                "schema_version": "orin_behavior_rpc.v1",
                "type": "start_cycle",
                "session_id": "pc-cycle-session",
                "seq": 0,
                "request_id": "dig-leg-request",
                "cycle_id": "cycle-1",
                "dig_trajectory": trajectory_snapshot(),
            },
            lambda _event: None,
        )
        coordinator.observe(safe_machine_state(seq=2))
        coordinator.observe(safe_machine_state(seq=3))
        self.assertEqual(
            coordinator.status.stage,
            "WAITING_FOR_DUMP_TRAJECTORY",
        )
        events = []

        coordinator.handle(
            {
                "schema_version": "orin_behavior_rpc.v1",
                "type": "cancel_cycle",
                "session_id": "pc-cycle-session",
                "seq": 1,
                "request_id": "cancel-cycle-request",
                "cycle_id": "cycle-1",
            },
            events.append,
        )

        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["outcome"], "CANCELLED")
        self.assertTrue(events[-1]["quiescence_confirmed"])
        self.assertEqual(events[-1]["action_datagrams"], 5)
        self.assertEqual(coordinator.status.stage, "CANCELLED")

    def test_cycle_start_surfaces_motion_gate_rejection_before_acceptance(self):
        executor = EdgeBehaviorExecutor(
            runtime_factory=object(),
            runner_factory=lambda runtime: object(),
            wall_clock=lambda: 101.0,
            monotonic_clock=lambda: 7.0,
            sender_constructed=True,
        )
        coordinator = EdgeCycleCoordinator(executor)
        events = []

        coordinator.handle(
            {
                "schema_version": "orin_behavior_rpc.v1",
                "type": "start_cycle",
                "session_id": "pc-cycle-session",
                "seq": 0,
                "request_id": "dig-leg-request",
                "cycle_id": "cycle-1",
                "dig_trajectory": trajectory_snapshot(),
            },
            events.append,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "rejected")
        self.assertEqual(events[0]["cycle_id"], "cycle-1")
        self.assertEqual(events[0]["reason_code"], "MOTION_NOT_READY")
        self.assertEqual(events[0]["message"], "state_unavailable")
        self.assertEqual(coordinator.status.stage, "FAILED")


if __name__ == "__main__":
    unittest.main()
