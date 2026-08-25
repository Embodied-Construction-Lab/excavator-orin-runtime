#!/usr/bin/env python3
"""Build an uncommissioned V3-A fixed-cycle deployment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_runtime.fixed_cycle_deployment import build_candidate_deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission-config", required=True)
    parser.add_argument("--demo-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deployed-root", required=True)
    parser.add_argument("--act-max-steps", type=int, default=130)
    args = parser.parse_args()
    result = build_candidate_deployment(
        mission_path=args.mission_config,
        demo_path=args.demo_config,
        output_dir=args.output_dir,
        deployed_root=args.deployed_root,
        act_max_steps=args.act_max_steps,
    )
    print(result)


if __name__ == "__main__":
    main()
