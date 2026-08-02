import pandas as pd
import pytest

from Scripts.Modeling import s5_Auction_Selection_Premium as premium


def _historical_seeds(
    salary_method=premium.HISTORICAL_SALARY_METHOD_VERSION,
    seed_method=premium.HISTORICAL_SEED_METHOD,
):
    rows = []
    for year in (2023, 2024, 2025):
        for index, pos in enumerate(("QB", "RB", "WR", "TE")):
            point_salary = 8.0 + index * 3 + (year - 2023)
            rows.append(
                {
                    "year": year,
                    "league": "beta",
                    "player": f"{year} {pos}",
                    "player_key": f"{year}-{pos}",
                    "pos": pos,
                    "point_salary": point_salary,
                    "selection_rate": 0.15 + index * 0.1,
                    "actual_salary": point_salary + index - 1,
                    "actual_salary_recorded": 1,
                    "salary_residual": float(index - 1),
                    "salary_method_version": salary_method,
                    "seed_method_version": seed_method,
                }
            )
    return pd.DataFrame(rows)


def _current_surface():
    return pd.DataFrame(
        [
            {
                "player": "Current QB",
                "player_key": "current-qb",
                "pos": "QB",
                "point_salary": 20.0,
                "selection_rate": 0.5,
                "selection_slots": 50,
            },
            {
                "player": "Current WR",
                "player_key": "current-wr",
                "pos": "WR",
                "point_salary": 14.0,
                "selection_rate": 0.3,
                "selection_slots": 30,
            },
        ]
    )


def test_calibrator_records_governed_v5_to_v6_transfer():
    calibrated, coefficients, metadata = premium.fit_calibrator(
        _historical_seeds(),
        _current_surface(),
        2026,
    )

    assert len(calibrated) == 2
    assert metadata["training_salary_method_versions"] == [
        premium.HISTORICAL_SALARY_METHOD_VERSION
    ]
    assert metadata["training_seed_method_versions"] == [
        premium.HISTORICAL_SEED_METHOD
    ]
    assert (
        metadata["calibration_transfer_policy"]
        == premium.CALIBRATION_TRANSFER_POLICY
    )
    assert set(coefficients.calibration_transfer_policy) == {
        premium.CALIBRATION_TRANSFER_POLICY
    }

    published = premium.build_premium_rows(
        calibrated,
        2026,
        "beta",
        0.5,
        100,
        100,
        metadata,
    )
    assert set(published.salary_method_version) == {
        premium.SALARY_METHOD_VERSION
    }
    assert set(published.calibration_transfer_policy) == {
        premium.CALIBRATION_TRANSFER_POLICY
    }
    assert set(published.training_salary_method_versions) == {
        premium.HISTORICAL_SALARY_METHOD_VERSION
    }


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (
            "salary_method_version",
            premium.SALARY_METHOD_VERSION,
            "historical v5 salary surface",
        ),
        (
            "seed_method_version",
            premium.CURRENT_SEED_METHOD,
            "historical Target seed method",
        ),
    ],
)
def test_calibrator_rejects_unapproved_training_surface(
    column,
    value,
    message,
):
    seeds = _historical_seeds()
    seeds[column] = value

    with pytest.raises(ValueError, match=message):
        premium.fit_calibrator(seeds, _current_surface(), 2026)
