import pandas as pd
import pytest

from Scripts.V2.locked_candidates import (
    HISTORY_GAP_PPG_FEATURES,
    LOCKED_BLEND_WEIGHTS,
    LOCKED_FEATURE_SETS,
    LOG_ADP_LASSO_FEATURES,
    PARTICIPATION_CANDIDATE_FEATURES,
    PARTICIPATION_FEATURES,
    POSITION_FEATURES,
    PRIMARY_PPG_FEATURES,
    PROJECTION_TRAJECTORY_FEATURES,
    RESIDUAL_CANDIDATE_FEATURES,
    lock_version_for_scoring,
    locked_metadata,
    specification_table,
    validate_feature_lock,
)


def _manifests() -> pd.DataFrame:
    rows = []
    for name, features in (
        ("residual_candidate_v1", RESIDUAL_CANDIDATE_FEATURES),
        (
            "residual_projection_trajectory_challenger_v1",
            PROJECTION_TRAJECTORY_FEATURES,
        ),
        ("participation_candidate_v1", PARTICIPATION_CANDIDATE_FEATURES),
    ):
        rows.extend(
            {"manifest_name": name, "feature_name": feature}
            for feature in features
        )
    return pd.DataFrame(rows)


def test_locked_feature_sets_have_expected_shape_and_adp_semantics():
    assert len(PRIMARY_PPG_FEATURES) == 40
    assert len(PARTICIPATION_FEATURES) == 23
    assert len(HISTORY_GAP_PPG_FEATURES) == 45
    assert set(POSITION_FEATURES).issubset(PRIMARY_PPG_FEATURES)
    assert "adp_median" in PRIMARY_PPG_FEATURES
    assert "adp_log" not in PRIMARY_PPG_FEATURES
    assert "adp_log" in LOG_ADP_LASSO_FEATURES
    assert "adp_median" not in LOG_ADP_LASSO_FEATURES
    assert sum(LOCKED_BLEND_WEIGHTS.values()) == pytest.approx(1.0)


def test_feature_lock_matches_reviewed_manifests_and_columns():
    columns = sorted(
        {
            feature
            for feature_set in LOCKED_FEATURE_SETS.values()
            for feature in feature_set
        }
    )
    features = pd.DataFrame(columns=columns)
    validate_feature_lock(features, _manifests())


def test_feature_lock_rejects_manifest_drift():
    manifests = _manifests()
    manifests = manifests[
        ~(
            manifests["manifest_name"].eq("residual_candidate_v1")
            & manifests["feature_name"].eq("adp_median")
        )
    ]
    columns = sorted(
        {
            feature
            for feature_set in LOCKED_FEATURE_SETS.values()
            for feature in feature_set
        }
    )
    with pytest.raises(ValueError, match="Feature lock mismatch"):
        validate_feature_lock(pd.DataFrame(columns=columns), manifests)


def test_specification_table_carries_hashes_and_fixed_blend():
    specs = specification_table()
    feature_specs = specs[specs["record_type"].eq("feature_set")]
    assert len(feature_specs) == len(LOCKED_FEATURE_SETS)
    assert feature_specs["feature_hash"].str.len().eq(64).all()
    assert (
        specs["specification_name"]
        .eq("conditional_ppg_equal_thirds")
        .sum()
        == 1
    )


def test_scoring_specific_lock_keeps_features_but_changes_provenance():
    beta_lock = lock_version_for_scoring("beta")
    assert beta_lock == "v2_conditional_ppg_2026_candidate_beta_v1"
    assert specification_table(beta_lock)["lock_version"].eq(beta_lock).all()
    metadata = locked_metadata("beta", beta_lock)
    assert metadata["scoring_objective"] == "beta"
    assert metadata["lock_version"] == beta_lock
