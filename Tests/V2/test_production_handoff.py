import pandas as pd
import pytest

from Scripts.V2.contracts import scoring_hash
from Scripts.V2.production_handoff import (
    CURRENT_RESIDUAL_COLUMNS,
    NEXT_RESIDUAL_SOURCE_COLUMNS,
    build_production_projection_slice,
)


def fixture_frames():
    legacy = pd.DataFrame(
        {
            "player": ["Rookie Runner"],
            "pos": ["RB"],
            "year": [2026],
            "version": ["dk"],
            "dataset": ["final_ensemble"],
            "pred_fp_per_game": [8.0],
            "pred_fp_per_game_ny": [9.0],
            **{column: [float(index)] for index, column in enumerate(
                CURRENT_RESIDUAL_COLUMNS
            )},
            **{
                destination: [float(index)]
                for index, destination in enumerate(
                    NEXT_RESIDUAL_SOURCE_COLUMNS.values()
                )
            },
        }
    )
    player_map = legacy[
        ["player", "pos", "year", "version", "dataset"]
    ].copy()
    player_map["player_key"] = "player-key"
    current = pd.DataFrame(
        {
            "player_key": ["player-key"],
            "conditional_ppg_shadow": [8.5],
            "participation_probability": [0.94],
            "lock_version": ["current-lock"],
        }
    )
    next_year = pd.DataFrame(
        {
            "player_key": ["player-key"],
            "predicted_next_year_conditional_ppg": [10.5],
            "predicted_next_year_appearance_probability": [0.60],
            "target_version": ["next-lock"],
            "scoring_hash": [scoring_hash("dk")],
            **{
                source: [value]
                for source, value in zip(
                    NEXT_RESIDUAL_SOURCE_COLUMNS,
                    (-5.0, -4.0, -2.0, 2.0, 4.0, 6.0),
                )
            },
        }
    )
    return legacy, player_map, current, next_year


def test_production_handoff_uses_one_current_template_residual():
    legacy, player_map, current, next_year = fixture_frames()
    output, audit = build_production_projection_slice(
        legacy,
        player_map,
        current,
        next_year,
        league="dk",
    )

    row = output.iloc[0]
    assert row["player_key"] == "player-key"
    assert row["pred_fp_per_game"] == 8.5
    assert row["pred_fp_per_game_ny"] == 10.5
    assert row["pred_appear_ny"] == 0.60
    assert row["current_uncertainty_source"] == "joint_weekly_template_only"
    assert row["independent_current_residual_draw_allowed"] == 0
    assert output[list(CURRENT_RESIDUAL_COLUMNS)].eq(0).all().all()
    assert output[
        list(NEXT_RESIDUAL_SOURCE_COLUMNS.values())
    ].iloc[0].tolist() == [-5.0, -4.0, -2.0, 2.0, 4.0, 6.0]
    assert audit.iloc[0]["current_ppg_delta"] == 0.5
    assert audit.iloc[0]["next_ppg_delta"] == 1.5


def test_production_handoff_can_refresh_an_existing_v2_publish():
    legacy, player_map, current, next_year = fixture_frames()
    first, _ = build_production_projection_slice(
        legacy,
        player_map,
        current,
        next_year,
        league="dk",
    )

    refreshed, audit = build_production_projection_slice(
        first,
        player_map,
        current,
        next_year,
        league="dk",
    )

    assert refreshed.columns.is_unique
    assert not any(
        column.endswith(("_x", "_y")) for column in refreshed.columns
    )
    assert refreshed.loc[0, "player_key"] == "player-key"
    assert refreshed.loc[0, "current_projection_model_version"] == (
        "current-lock"
    )
    assert refreshed.loc[0, "next_projection_model_version"] == "next-lock"
    assert audit.loc[0, "current_ppg_delta"] == 0
    assert audit.loc[0, "next_ppg_delta"] == 0


def test_production_handoff_fails_closed_on_missing_next_appearance():
    legacy, player_map, current, next_year = fixture_frames()
    next_year.loc[0, "predicted_next_year_appearance_probability"] = None

    with pytest.raises(ValueError, match="handoff is incomplete"):
        build_production_projection_slice(
            legacy,
            player_map,
            current,
            next_year,
            league="dk",
        )


def test_production_handoff_rejects_nonmonotone_next_quantiles():
    legacy, player_map, current, next_year = fixture_frames()
    next_year.loc[0, "pred_resid_25_ny_shadow"] = -10.0

    with pytest.raises(ValueError, match="not monotone"):
        build_production_projection_slice(
            legacy,
            player_map,
            current,
            next_year,
            league="dk",
        )
