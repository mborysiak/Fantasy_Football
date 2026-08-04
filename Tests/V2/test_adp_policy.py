from __future__ import annotations

import math

import pandas as pd
import pytest

from Scripts.V2.adp_policy import (
    ADP_POLICY_VERSION,
    NFFC_AGGREGATION_POLICY,
    NFFC_BOUNDS_POLICY,
    NFFC_STD_DEV_POLICY,
    build_nffc_two_feed_aggregate,
    canonical_adp_family_values,
    validate_nffc_pair_agreement,
)
from Scripts.V2.build_feature_mart import build_market_consensus


def test_canonical_family_policy_excludes_recent_mfl_and_deduplicates_dk():
    values = pd.DataFrame(
        [
            {"player_key": "p1", "season": 2024, "source": "adp_mfl", "adp": 10},
            {"player_key": "p1", "season": 2024, "source": "adp_fpros", "adp": 20},
            {"player_key": "p1", "season": 2024, "source": "fantasypros_best_ball_adp", "adp": 30},
            {"player_key": "p1", "season": 2024, "source": "adp_average_nffc", "adp": 40},
            {"player_key": "p1", "season": 2024, "source": "adp_average_dk", "adp": 50},
            {"player_key": "p1", "season": 2024, "source": "draftkings_adp", "adp": 500},
            {"player_key": "p1", "season": 2024, "source": "ffa_projection", "adp": 1},
            {"player_key": "p1", "season": 2025, "source": "adp_mfl", "adp": 2},
            {"player_key": "p1", "season": 2025, "source": "adp_fpros", "adp": 20},
            {"player_key": "p1", "season": 2025, "source": "fantasypros_best_ball_adp", "adp": 30},
            {"player_key": "p1", "season": 2025, "source": "adp_average_nffc", "adp": 40},
            {"player_key": "p1", "season": 2025, "source": "adp_average_dk", "adp": 50},
            {"player_key": "p1", "season": 2025, "source": "draftkings_adp", "adp": 500},
        ]
    )
    values["expert_rank"] = float("nan")
    values["source_position_rank"] = float("nan")

    family = canonical_adp_family_values(values)
    counts = family.groupby("season")["source"].nunique().to_dict()
    consensus = build_market_consensus(values).set_index("season")

    assert counts == {2024: 5, 2025: 4}
    assert consensus.loc[2024, "adp_median"] == 30
    assert consensus.loc[2024, "adp_source_count"] == 5
    assert consensus.loc[2025, "adp_median"] == 35
    assert consensus.loc[2025, "adp_source_count"] == 4
    assert 500 not in family["adp"].tolist()


def test_nffc_aggregate_uses_equal_centers_bounds_and_pooled_variance():
    raw = pd.DataFrame(
        [
            {
                "player": "Josh Allen",
                "pos": "QB",
                "year": 2026,
                "source": "nffc_best_ball_overall",
                "pick_nffc": 24.0,
                "min_pick": 10.0,
                "max_pick": 40.0,
            },
            {
                "player": "Josh Allen",
                "pos": "QB",
                "year": 2026,
                "source": "nffc_best_ball_25s50s",
                "pick_nffc": 30.0,
                "min_pick": 12.0,
                "max_pick": 48.0,
            },
        ]
    )

    row = build_nffc_two_feed_aggregate(raw).iloc[0]
    expected_sd = math.sqrt((((40 - 10) / 5) ** 2 + 3**2 + ((48 - 12) / 5) ** 2 + 3**2) / 2)

    assert row["avg_pick"] == 27
    assert row["min_pick"] == 11
    assert row["max_pick"] == 44
    assert row["std_dev"] == pytest.approx(expected_sd)
    assert row["source_count"] == 2
    assert row["feed_gap"] == 6
    assert row["aggregation_policy"] == NFFC_AGGREGATION_POLICY
    assert row["bounds_policy"] == NFFC_BOUNDS_POLICY
    assert row["std_dev_policy"] == NFFC_STD_DEV_POLICY
    assert row["adp_policy_version"] == ADP_POLICY_VERSION


def test_nffc_single_feed_row_discloses_missing_pair():
    raw = pd.DataFrame(
        [
            {
                "player": "Fringe Player",
                "pos": "WR",
                "year": 2026,
                "source": "nffc_best_ball_overall",
                "pick_nffc": 275.0,
                "min_pick": 240.0,
                "max_pick": 355.0,
            }
        ]
    )

    row = build_nffc_two_feed_aggregate(raw).iloc[0]

    assert row["source_count"] == 1
    assert pd.isna(row["feed_gap"])
    assert row["avg_pick"] == 275
    assert row["std_dev"] == 23


def test_nffc_pair_agreement_reports_rank_and_center_alignment():
    rows = []
    for pick in range(1, 201):
        for source, offset in (
            ("nffc_best_ball_overall", 0.0),
            ("nffc_best_ball_25s50s", 1.0),
        ):
            rows.append(
                {
                    "player": f"Player {pick}",
                    "pos": "WR",
                    "source": source,
                    "pick_nffc": pick + offset,
                }
            )

    result = validate_nffc_pair_agreement(pd.DataFrame(rows))

    assert result["common_top_pick_rows"] == 200
    assert result["spearman"] == pytest.approx(1.0)
    assert result["median_abs_gap"] == 1.0
