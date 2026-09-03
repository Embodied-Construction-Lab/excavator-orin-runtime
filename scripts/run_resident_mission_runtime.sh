#!/usr/bin/env bash
set -euo pipefail

pc_host="${PC_STATE_HOST:-192.168.50.1}"
serial_port="${STM32_SERIAL_PORT:-/dev/ttyTHS1}"
print_every="${PRINT_EVERY:-100}"
edge_config="${EDGE_CONFIG:-deploy/edge_runtime.resident.remote.json}"
resident_runtime_root="${RESIDENT_RUNTIME_ROOT:-${HOME}/.local/run/excavator-resident}"
resident_act_socket="${RESIDENT_ACT_SOCKET:-${resident_runtime_root}/act.sock}"
resident_control_socket="${RESIDENT_CONTROL_SOCKET:-${resident_runtime_root}/control.sock}"
resident_fixed_cycle_control_socket="${RESIDENT_FIXED_CYCLE_CONTROL_SOCKET:-${resident_runtime_root}/fixed-cycle.sock}"
resident_python="${RESIDENT_PYTHON:-python3}"
fixed_cycle_plan=""
commissioning_authorization=""
trajectory_controller_commissioning_authorization=""
expected_dig_catalog_sha256=""
authorization=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    "--authorization")
      [[ $# -ge 2 ]] || { echo "--authorization 缺少值" >&2; exit 2; }
      authorization="$2"
      shift 2
      ;;
    "--pc-host")
      [[ $# -ge 2 ]] || { echo "--pc-host 缺少值" >&2; exit 2; }
      pc_host="$2"
      shift 2
      ;;
    "--serial-port")
      [[ $# -ge 2 ]] || { echo "--serial-port 缺少值" >&2; exit 2; }
      serial_port="$2"
      shift 2
      ;;
    "--edge-config")
      [[ $# -ge 2 ]] || { echo "--edge-config 缺少值" >&2; exit 2; }
      edge_config="$2"
      shift 2
      ;;
    "--print-every")
      [[ $# -ge 2 ]] || { echo "--print-every 缺少值" >&2; exit 2; }
      print_every="$2"
      shift 2
      ;;
    "--fixed-cycle-plan")
      [[ $# -ge 2 ]] || { echo "--fixed-cycle-plan 缺少值" >&2; exit 2; }
      fixed_cycle_plan="$2"
      shift 2
      ;;
    "--commissioning-authorization")
      [[ $# -ge 2 ]] || { echo "--commissioning-authorization 缺少值" >&2; exit 2; }
      commissioning_authorization="$2"
      shift 2
      ;;
    "--trajectory-controller-commissioning-authorization")
      [[ $# -ge 2 ]] || { echo "--trajectory-controller-commissioning-authorization 缺少值" >&2; exit 2; }
      trajectory_controller_commissioning_authorization="$2"
      shift 2
      ;;
    "--expected-dig-catalog-sha256")
      [[ $# -ge 2 ]] || { echo "--expected-dig-catalog-sha256 缺少值" >&2; exit 2; }
      expected_dig_catalog_sha256="$2"
      shift 2
      ;;
    *)
      echo "未知参数：$1" >&2
      exit 2
      ;;
  esac
done

if [[ "${authorization}" != "ALLOW_HYBRID_MACHINE_MOTION" ]]; then
  echo "resident Mission owner 需要精确授权：ALLOW_HYBRID_MACHINE_MOTION" >&2
  exit 1
fi
if [[ ! "${print_every}" =~ ^[0-9]+$ ]]; then
  echo "--print-every 必须是非负整数。" >&2
  exit 2
fi
if [[ ! "${pc_host}" =~ ^[0-9A-Za-z._:-]+$ ]]; then
  echo "--pc-host 必须是安全的主机名或 IPv4 文本。" >&2
  exit 2
fi
if [[ -n "${commissioning_authorization}" && "${commissioning_authorization}" != "ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING" ]]; then
  echo "V3-A 候选轨迹需要精确授权：ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING" >&2
  exit 1
fi
if [[ -n "${commissioning_authorization}" && -z "${fixed_cycle_plan}" ]]; then
  echo "候选轨迹授权只能与 --fixed-cycle-plan 一起使用。" >&2
  exit 2
fi
if [[ -n "${trajectory_controller_commissioning_authorization}" ]] \
  && [[ "${trajectory_controller_commissioning_authorization}" != "ALLOW_CARTESIAN_P_MACHINE_MOTION" ]]; then
  echo "Cartesian-P 真机调试需要精确授权：ALLOW_CARTESIAN_P_MACHINE_MOTION" >&2
  exit 1
fi
if [[ -n "${expected_dig_catalog_sha256}" && ! "${expected_dig_catalog_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "--expected-dig-catalog-sha256 必须是小写 SHA-256。" >&2
  exit 2
fi
if [[ -n "${expected_dig_catalog_sha256}" && -z "${fixed_cycle_plan}" ]]; then
  echo "目录摘要只能与 --fixed-cycle-plan 一起使用。" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
resident_action_audit_path="${RESIDENT_ACTION_AUDIT_PATH:-${repo_dir}/logs/resident_action_audit.jsonl}"
example_edge_config="${repo_dir}/deploy/edge_runtime.resident.remote.example.json"
resolved_edge_config="${edge_config}"
if [[ "${resolved_edge_config}" != /* ]]; then
  resolved_edge_config="${repo_dir}/${resolved_edge_config}"
fi
if [[ ! -f "${resolved_edge_config}" ]]; then
  if [[ "${resolved_edge_config}" == "${repo_dir}/deploy/edge_runtime.resident.remote.json" ]]; then
    cp "${example_edge_config}" "${resolved_edge_config}"
  else
    echo "edge config 不存在：${resolved_edge_config}" >&2
    exit 1
  fi
fi

if [[ -n "${fixed_cycle_plan}" ]]; then
  if [[ "${fixed_cycle_plan}" != /* ]]; then
    fixed_cycle_plan="${repo_dir}/${fixed_cycle_plan}"
  fi
  if [[ ! -f "${fixed_cycle_plan}" ]]; then
    echo "固定循环 plan 不存在：${fixed_cycle_plan}" >&2
    exit 1
  fi
fi

test -c "${serial_port}"
mkdir -p "${resident_runtime_root}"
test -w "${resident_runtime_root}"
if fuser "${serial_port}" >/dev/null 2>&1; then
  echo "拒绝启动：${serial_port} 已被其他进程占用。" >&2
  exit 1
fi
if pgrep -f 'excavator-il (collect|act-runtime)|orin_state_sender.py' >/dev/null; then
  echo "拒绝启动：检测到竞争的 Collector、ACT Runtime 或其他 orin_state_sender.py。" >&2
  exit 1
fi
python3 -m json.tool "${resolved_edge_config}" >/dev/null
python3 - "${resolved_edge_config}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("schema_version") != "orin_edge_runtime.v1":
    raise SystemExit("edge config schema_version 错误")
if value.get("mode") != "remote_control":
    raise SystemExit("resident owner requires mode=remote_control")
if value.get("action_transport") != "resident_sink":
    raise SystemExit("resident owner requires action_transport=resident_sink")
PY

fixed_cycle_args=()
if [[ -n "${fixed_cycle_plan}" ]]; then
  fixed_cycle_args=(--resident-fixed-cycle-plan "${fixed_cycle_plan}")
  if [[ -n "${expected_dig_catalog_sha256}" ]]; then
    fixed_cycle_args+=(
      --resident-fixed-cycle-expected-dig-catalog-sha256
      "${expected_dig_catalog_sha256}"
    )
  fi
  if [[ -n "${commissioning_authorization}" ]]; then
    fixed_cycle_args+=(
      --resident-fixed-cycle-commissioning-authorization
      "${commissioning_authorization}"
    )
  fi
fi

trajectory_controller_args=()
if [[ -n "${trajectory_controller_commissioning_authorization}" ]]; then
  trajectory_controller_args=(
    --trajectory-controller-commissioning-authorization
    "${trajectory_controller_commissioning_authorization}"
  )
fi

echo "启动 resident Mission owner：${serial_port} 将保持为唯一 STM32 owner。"
echo "ACT worker 通过 ${resident_act_socket} 发送候选动作；低频控制命令走 ${resident_control_socket}。"

exec "${resident_python}" -u "${repo_dir}/orin_state_sender.py" \
  --serial-port "${serial_port}" \
  --control-enabled \
  --input-format csv \
  --pc-host "${pc_host}" \
  --edge-config "${resolved_edge_config}" \
  --edge-motion-authorization ALLOW_EDGE_MACHINE_MOTION \
  "${trajectory_controller_args[@]}" \
  --resident-motion-core \
  --resident-act-socket "${resident_act_socket}" \
  --resident-control-socket "${resident_control_socket}" \
  --resident-action-audit-path "${resident_action_audit_path}" \
  --resident-fixed-cycle-control-socket "${resident_fixed_cycle_control_socket}" \
  "${fixed_cycle_args[@]}" \
  --print-every "${print_every}"
