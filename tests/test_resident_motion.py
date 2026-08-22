import math
import unittest

from edge_runtime.resident_motion import (
    ControlMode,
    HandoffPhase,
    MotionCandidate,
    PolicyBinding,
    ResidentMotionAuthority,
)


ZERO = (0.0, 0.0, 0.0, 0.0)


def candidate(
    *,
    source: str,
    generation: int,
    mode: ControlMode,
    action=(0.1, -0.2, 0.3, -0.4),
    created_ns: int = 1_000_000_000,
    valid_until_ns: int = 1_100_000_000,
) -> MotionCandidate:
    return MotionCandidate(
        source=source,
        generation=generation,
        mode=mode,
        action=action,
        created_monotonic_ns=created_ns,
        valid_until_monotonic_ns=valid_until_ns,
    )


class ResidentMotionAuthorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.authority = ResidentMotionAuthority(max_future_skew_ms=5.0)
        self.rl = PolicyBinding("rl_follow", ControlMode.VELOCITY_REFERENCE)
        self.act = PolicyBinding("act_dig", ControlMode.MANUAL_ACTION)

    def activate(self, binding: PolicyBinding, *, now_ns: int = 1_000_000_000) -> int:
        generation = self.authority.request_handoff(binding, now_monotonic_ns=now_ns)
        pending = self.authority.pending_zero()
        self.assertIsNotNone(pending)
        self.assertEqual(pending.mode, binding.mode)
        self.assertEqual(pending.purpose, "target_mode_claim")
        self.authority.acknowledge_zero(
            generation=generation,
            mode=binding.mode,
            action=ZERO,
            acknowledged_monotonic_ns=now_ns + 10_000_000,
        )
        return generation

    def test_first_policy_requires_target_mode_zero_before_nonzero(self) -> None:
        generation = self.authority.request_handoff(
            self.act, now_monotonic_ns=1_000_000_000
        )

        before_ack = self.authority.route(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
            ),
            now_monotonic_ns=1_010_000_000,
            safety_permits_motion=True,
        )
        self.assertFalse(before_ack.accepted)
        self.assertEqual(before_ack.effective_action, ZERO)
        self.assertEqual(before_ack.reason, "handoff_in_progress")

        with self.assertRaisesRegex(ValueError, "pending handoff mode"):
            self.authority.acknowledge_zero(
                generation=generation,
                mode=ControlMode.VELOCITY_REFERENCE,
                action=ZERO,
                acknowledged_monotonic_ns=1_020_000_000,
            )

        self.authority.acknowledge_zero(
            generation=generation,
            mode=ControlMode.MANUAL_ACTION,
            action=ZERO,
            acknowledged_monotonic_ns=1_020_000_000,
        )
        decision = self.authority.route(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
            ),
            now_monotonic_ns=1_030_000_000,
            safety_permits_motion=True,
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "accepted")
        self.assertEqual(decision.effective_action, (0.1, -0.2, 0.3, -0.4))

    def test_rl_to_act_requires_terminal_velocity_zero_then_manual_zero(self) -> None:
        rl_generation = self.activate(self.rl)
        act_generation = self.authority.request_handoff(
            self.act, now_monotonic_ns=1_100_000_000
        )
        self.assertGreater(act_generation, rl_generation)

        terminal = self.authority.pending_zero()
        self.assertEqual(terminal.mode, ControlMode.VELOCITY_REFERENCE)
        self.assertEqual(terminal.purpose, "terminal_source_zero")

        stale_rl = self.authority.route(
            candidate(
                source="rl_follow",
                generation=rl_generation,
                mode=ControlMode.VELOCITY_REFERENCE,
                created_ns=1_100_000_000,
                valid_until_ns=1_200_000_000,
            ),
            now_monotonic_ns=1_110_000_000,
            safety_permits_motion=True,
        )
        self.assertFalse(stale_rl.accepted)
        self.assertEqual(stale_rl.reason, "handoff_in_progress")

        self.authority.acknowledge_zero(
            generation=act_generation,
            mode=ControlMode.VELOCITY_REFERENCE,
            action=ZERO,
            acknowledged_monotonic_ns=1_120_000_000,
        )
        claim = self.authority.pending_zero()
        self.assertEqual(claim.mode, ControlMode.MANUAL_ACTION)
        self.assertEqual(claim.purpose, "target_mode_claim")

        self.authority.acknowledge_zero(
            generation=act_generation,
            mode=ControlMode.MANUAL_ACTION,
            action=ZERO,
            acknowledged_monotonic_ns=1_140_000_000,
        )
        snapshot = self.authority.snapshot()
        self.assertEqual(snapshot.phase, HandoffPhase.ACTIVE)
        self.assertEqual(snapshot.active_binding, self.act)
        self.assertIsNone(snapshot.handoff_requested_monotonic_ns)
        self.assertIsNone(snapshot.last_handoff_latency_ms)

        self.authority.record_handoff_latency(
            generation=act_generation,
            terminal_zero_acknowledged_monotonic_ns=1_120_000_000,
            first_nonzero_acknowledged_monotonic_ns=1_170_000_000,
        )
        self.assertAlmostEqual(
            self.authority.snapshot().last_handoff_latency_ms,
            50.0,
        )

    def test_act_to_rl_uses_the_same_two_zero_transition(self) -> None:
        self.activate(self.act)
        generation = self.authority.request_handoff(
            self.rl, now_monotonic_ns=1_100_000_000
        )
        self.assertEqual(
            self.authority.pending_zero().mode, ControlMode.MANUAL_ACTION
        )
        self.authority.acknowledge_zero(
            generation=generation,
            mode=ControlMode.MANUAL_ACTION,
            action=ZERO,
            acknowledged_monotonic_ns=1_110_000_000,
        )
        self.assertEqual(
            self.authority.pending_zero().mode, ControlMode.VELOCITY_REFERENCE
        )
        self.authority.acknowledge_zero(
            generation=generation,
            mode=ControlMode.VELOCITY_REFERENCE,
            action=ZERO,
            acknowledged_monotonic_ns=1_120_000_000,
        )
        self.assertEqual(self.authority.snapshot().active_binding, self.rl)

    def test_only_current_source_generation_and_mode_can_move(self) -> None:
        generation = self.activate(self.act)
        cases = (
            ("other", generation, ControlMode.MANUAL_ACTION, "wrong_source"),
            ("act_dig", generation - 1, ControlMode.MANUAL_ACTION, "stale_generation"),
            ("act_dig", generation, ControlMode.VELOCITY_REFERENCE, "wrong_mode"),
        )
        for source, value, mode, reason in cases:
            with self.subTest(reason=reason):
                decision = self.authority.route(
                    candidate(source=source, generation=value, mode=mode),
                    now_monotonic_ns=1_030_000_000,
                    safety_permits_motion=True,
                )
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.effective_action, ZERO)
                self.assertEqual(decision.reason, reason)

    def test_manual_actions_are_normalized_but_velocity_is_not_clamped(self) -> None:
        act_generation = self.activate(self.act)
        rejected = self.authority.route(
            candidate(
                source="act_dig",
                generation=act_generation,
                mode=ControlMode.MANUAL_ACTION,
                action=(1.01, 0.0, 0.0, 0.0),
            ),
            now_monotonic_ns=1_030_000_000,
            safety_permits_motion=True,
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "invalid_manual_action")

        rl_generation = self.authority.request_handoff(
            self.rl, now_monotonic_ns=1_100_000_000
        )
        self.authority.acknowledge_zero(
            generation=rl_generation,
            mode=ControlMode.MANUAL_ACTION,
            action=ZERO,
            acknowledged_monotonic_ns=1_110_000_000,
        )
        self.authority.acknowledge_zero(
            generation=rl_generation,
            mode=ControlMode.VELOCITY_REFERENCE,
            action=ZERO,
            acknowledged_monotonic_ns=1_120_000_000,
        )
        physical = (0.25, -0.30, 0.40, -0.50)
        accepted = self.authority.route(
            candidate(
                source="rl_follow",
                generation=rl_generation,
                mode=ControlMode.VELOCITY_REFERENCE,
                action=physical,
                created_ns=1_120_000_000,
                valid_until_ns=1_220_000_000,
            ),
            now_monotonic_ns=1_130_000_000,
            safety_permits_motion=True,
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.effective_action, physical)

    def test_stale_future_nonfinite_and_unsafe_candidates_fail_closed(self) -> None:
        generation = self.activate(self.rl)
        cases = (
            (
                candidate(
                    source="rl_follow",
                    generation=generation,
                    mode=ControlMode.VELOCITY_REFERENCE,
                    valid_until_ns=1_020_000_000,
                ),
                1_030_000_000,
                True,
                "action_expired",
            ),
            (
                candidate(
                    source="rl_follow",
                    generation=generation,
                    mode=ControlMode.VELOCITY_REFERENCE,
                    created_ns=1_040_000_000,
                    valid_until_ns=1_100_000_000,
                ),
                1_030_000_000,
                True,
                "action_from_future",
            ),
            (
                candidate(
                    source="rl_follow",
                    generation=generation,
                    mode=ControlMode.VELOCITY_REFERENCE,
                    action=(math.nan, 0.0, 0.0, 0.0),
                ),
                1_030_000_000,
                True,
                "invalid_action",
            ),
            (
                candidate(
                    source="rl_follow",
                    generation=generation,
                    mode=ControlMode.VELOCITY_REFERENCE,
                ),
                1_030_000_000,
                False,
                "safety_rejected",
            ),
        )
        for action, now_ns, safe, reason in cases:
            with self.subTest(reason=reason):
                decision = self.authority.route(
                    action,
                    now_monotonic_ns=now_ns,
                    safety_permits_motion=safe,
                )
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.effective_action, ZERO)
                self.assertEqual(decision.reason, reason)

    def test_stop_invalidates_active_generation_and_requires_terminal_zero(self) -> None:
        active_generation = self.activate(self.act)
        stop_generation = self.authority.request_stop(
            now_monotonic_ns=1_100_000_000
        )
        self.assertGreater(stop_generation, active_generation)
        pending = self.authority.pending_zero()
        self.assertEqual(pending.mode, ControlMode.MANUAL_ACTION)
        self.assertEqual(pending.purpose, "terminal_source_zero")

        self.authority.acknowledge_zero(
            generation=stop_generation,
            mode=ControlMode.MANUAL_ACTION,
            action=ZERO,
            acknowledged_monotonic_ns=1_120_000_000,
        )
        snapshot = self.authority.snapshot()
        self.assertEqual(snapshot.phase, HandoffPhase.IDLE)
        self.assertIsNone(snapshot.active_binding)
        self.assertIsNone(snapshot.handoff_requested_monotonic_ns)

        decision = self.authority.route(
            candidate(
                source="act_dig",
                generation=active_generation,
                mode=ControlMode.MANUAL_ACTION,
            ),
            now_monotonic_ns=1_130_000_000,
            safety_permits_motion=True,
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "no_active_policy")

    def test_zero_ack_must_be_exactly_zero_and_current_generation(self) -> None:
        generation = self.authority.request_handoff(
            self.rl, now_monotonic_ns=1_000_000_000
        )
        with self.assertRaisesRegex(ValueError, "current handoff generation"):
            self.authority.acknowledge_zero(
                generation=generation + 1,
                mode=ControlMode.VELOCITY_REFERENCE,
                action=ZERO,
                acknowledged_monotonic_ns=1_010_000_000,
            )
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            self.authority.acknowledge_zero(
                generation=generation,
                mode=ControlMode.VELOCITY_REFERENCE,
                action=(0.0, 0.0, 0.0, 1e-6),
                acknowledged_monotonic_ns=1_010_000_000,
            )


if __name__ == "__main__":
    unittest.main()
