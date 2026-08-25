#!/usr/bin/env python3
"""Promote a passed V3-A candidate to field-validated deployment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_runtime.fixed_cycle_deployment import promote_candidate_deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--validation-record", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deployed-root", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    result = promote_candidate_deployment(
        candidate_plan_path=args.candidate_plan,
        validation_record_path=args.validation_record,
        output_dir=args.output_dir,
        deployed_root=args.deployed_root,
        authorization=args.authorization,
    )
    print(result)


if __name__ == "__main__":
    main()
