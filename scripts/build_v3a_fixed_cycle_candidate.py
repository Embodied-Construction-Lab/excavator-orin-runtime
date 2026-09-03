#!/usr/bin/env python3
"""Build an uncommissioned catalog-driven fixed-cycle deployment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_runtime.fixed_cycle_deployment import build_candidate_deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission-config", required=True)
    parser.add_argument("--mission-definition", required=True)
    parser.add_argument("--dig-point-catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deployed-root", required=True)
    parser.add_argument(
        "--intermediate-waypoint-tolerance-m",
        type=float,
        default=0.40,
    )
    args = parser.parse_args()
    result = build_candidate_deployment(
        mission_path=args.mission_config,
        mission_definition_path=args.mission_definition,
        dig_point_catalog_path=args.dig_point_catalog,
        output_dir=args.output_dir,
        deployed_root=args.deployed_root,
        intermediate_waypoint_tolerance_m=(
            args.intermediate_waypoint_tolerance_m
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
