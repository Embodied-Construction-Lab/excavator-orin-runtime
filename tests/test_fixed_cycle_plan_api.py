import pytest

import edge_runtime.resident_fixed_cycle as resident_fixed_cycle
from fixed_cycle_v5_support import plan_document
from edge_runtime.resident_fixed_cycle import (
    FixedCyclePlan,
    FixedTargetCatalogArtifact,
    FixedTrajectoryArtifact,
)


def _plan_document() -> dict[str, object]:
    return plan_document()


def test_plan_api_remains_available_from_resident_fixed_cycle() -> None:
    from edge_runtime import _fixed_cycle_plan

    public_plan_api = (
        "SCHEMA_VERSION",
        "CATALOG_SCHEMA_VERSION",
        "MAX_REQUESTED_CYCLES",
        "FixedTrajectoryArtifact",
        "FixedTargetCatalogArtifact",
        "FixedTrajectoryTemplate",
        "FixedCyclePlan",
        "load_fixed_cycle_plan",
        "verify_fixed_cycle_artifacts",
        "load_fixed_cycle_registry",
    )

    assert all(
        getattr(resident_fixed_cycle, name) is getattr(_fixed_cycle_plan, name)
        for name in public_plan_api
    )


def test_plan_and_artifact_contracts_reject_malformed_values() -> None:
    with pytest.raises(ValueError, match="catalog artifact fields"):
        FixedTargetCatalogArtifact.from_mapping({})

    catalog = {
        "catalog_id": "field-catalog",
        "path": "catalog.json",
        "sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="path must be absolute"):
        FixedTargetCatalogArtifact.from_mapping(catalog)

    catalog["path"] = "/opt/excavator-trajectories/catalog.json"
    catalog["sha256"] = "invalid"
    with pytest.raises(ValueError, match="sha256"):
        FixedTargetCatalogArtifact.from_mapping(catalog)

    with pytest.raises(ValueError, match="fields"):
        FixedCyclePlan.from_mapping([])
    with pytest.raises(ValueError, match="unsupported.*schema"):
        FixedCyclePlan.from_mapping({"schema_version": "unknown"})

    for legacy_schema in (
        "resident_fixed_cycle_plan.v1",
        "resident_fixed_cycle_plan.v2",
        "resident_fixed_cycle_plan.v3",
        "resident_fixed_cycle_plan.v4",
    ):
        legacy = _plan_document()
        legacy["schema_version"] = legacy_schema
        with pytest.raises(ValueError, match="unsupported.*schema"):
            FixedCyclePlan.from_mapping(legacy)

    for sequence in ([], ["dig_01", "dig_01"]):
        invalid_sequence = _plan_document()
        invalid_sequence["dig_sequence"] = sequence
        with pytest.raises(ValueError, match="dig_sequence"):
            FixedCyclePlan.from_mapping(invalid_sequence)


def test_grouped_plan_rejects_invalid_group_membership() -> None:
    def grouped_document() -> dict[str, object]:
        return _plan_document()

    invalid_groups = (
        ({}, "non-empty object"),
        ({"all": []}, "non-empty list"),
        ({"all": ["dig_01", "dig_01"]}, "must be unique"),
        ({"all": ["dig_01", "dig_02", "unknown"]}, "unknown point"),
        ({"all": ["dig_01", "dig_02"]}, "exactly match"),
    )
    for groups, message in invalid_groups:
        document = grouped_document()
        document["dig_groups"] = groups
        with pytest.raises(ValueError, match=message):
            FixedCyclePlan.from_mapping(document)

    missing_default = grouped_document()
    missing_default["default_dig_group"] = "near"
    with pytest.raises(ValueError, match="default_dig_group"):
        FixedCyclePlan.from_mapping(missing_default)
