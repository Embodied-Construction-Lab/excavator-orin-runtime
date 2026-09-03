import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_resident_owner_launcher_has_valid_bash_syntax():
    script = ROOT / "scripts" / "run_resident_mission_runtime.sh"

    subprocess.run(["bash", "-n", str(script)], check=True)


def test_resident_owner_launcher_reports_missing_fixed_cycle_plan(tmp_path):
    script = ROOT / "scripts" / "run_resident_mission_runtime.sh"
    missing_plan = tmp_path / "missing-field-plan.json"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--authorization",
            "ALLOW_HYBRID_MACHINE_MOTION",
            "--serial-port",
            "/dev/null",
            "--fixed-cycle-plan",
            str(missing_plan),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stderr.strip() == f"固定循环 plan 不存在：{missing_plan}"


def test_resident_owner_launcher_owns_serial_and_wires_resident_motion_core():
    script = (
        ROOT / "scripts" / "run_resident_mission_runtime.sh"
    ).read_text(encoding="utf-8")

    assert "--resident-motion-core" in script
    assert "--resident-act-socket" in script
    assert "--resident-control-socket" in script
    assert "--edge-config" in script
    assert "--control-enabled" in script
    assert "--input-format csv" in script
    assert "--edge-motion-authorization ALLOW_EDGE_MACHINE_MOTION" in script
    assert '"--authorization"' in script
    assert '[[ "${authorization}" != "ALLOW_HYBRID_MACHINE_MOTION" ]]' in script
    assert "/dev/ttyTHS1" in script
    assert "run_act_resident.sh" not in script
    assert "pgrep -f" in script
    assert "fuser" in script
    assert "/dev/ttyTHS1" in script
    # The PC lifecycle records this shell PID before starting the launcher.
    # A trailing pipeline would keep bash as that PID while Python becomes a
    # child, defeating the exact process-identity check during shutdown.
    assert "| tee" not in script
    assert 'resident_python="${RESIDENT_PYTHON:-python3}"' in script
    assert 'exec "${resident_python}" -u' in script
    assert '[[ -e "${resident_control_socket}"' not in script
    assert '[[ -e "${resident_act_socket}"' not in script
    assert '"--commissioning-authorization"' in script
    assert "ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING" in script
    assert "--resident-fixed-cycle-commissioning-authorization" in script
    assert '"--trajectory-controller-commissioning-authorization"' in script
    assert "ALLOW_CARTESIAN_P_MACHINE_MOTION" in script
    assert "--trajectory-controller-commissioning-authorization" in script
    assert (
        'resident_fixed_cycle_control_socket="${RESIDENT_FIXED_CYCLE_CONTROL_SOCKET:-'
        '${resident_runtime_root}/fixed-cycle.sock}"'
    ) in script
    assert '--resident-fixed-cycle-control-socket' in script
    assert '"${resident_fixed_cycle_control_socket}"' in script
    assert 'resident_action_audit_path="${RESIDENT_ACTION_AUDIT_PATH:-' in script
    assert '--resident-action-audit-path' in script
    assert '"${resident_action_audit_path}"' in script


def test_resident_owner_example_config_uses_resident_sink_remote_control():
    config_path = ROOT / "deploy" / "edge_runtime.resident.remote.example.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["schema_version"] == "orin_edge_runtime.v1"
    assert config["mode"] == "remote_control"
    assert config["action_transport"] == "resident_sink"
    assert "trajectory_path" not in config
    assert config["remote_behavior"]["bind_port"] == 18083
    assert config["fixed_action_profile_path"].endswith("fixed_actions.json")
    assert config["follow_action_startup_slew_rate_per_s"] == 4.0
    assert config["follow_action_slew_rate_per_s"] == 3.0
