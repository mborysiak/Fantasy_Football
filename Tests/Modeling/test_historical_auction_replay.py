from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Scripts.Modeling.build_historical_auction_replay import (
    additive_floor_normalize_market,
    get_replay_contract,
)


def test_2025_beta_replay_contract_separates_primary_and_fallback_rows():
    contract = get_replay_contract(2025, "BETA")

    assert contract.current_method_projection_rows == 305
    assert contract.current_method_salary_rows == 305
    assert contract.legacy_fallback_rows == 4
    assert contract.projection_rows == contract.salary_rows == 309
    assert contract.offensive_actual_rows == 156
    assert contract.raw_actual_rows == 179
    assert contract.historical_etr_rows == 238
    assert contract.projection_training_through_year == 2024
    assert contract.salary_training_through_year == 2024


@pytest.mark.parametrize(
    (
        "year",
        "projection_rows",
        "primary_projection_rows",
        "salary_rows",
        "fallback_rows",
        "offensive_actual_rows",
        "keeper_count",
        "keeper_spend",
    ),
    [
        (2022, 308, 307, 299, 1, 149, 20, 690.0),
        (2023, 315, 313, 311, 2, 155, 21, 871.0),
        (2024, 316, 314, 304, 2, 155, 18, 563.0),
    ],
)
def test_older_beta_replay_contracts_are_explicit_and_prior_year_bounded(
    year,
    projection_rows,
    primary_projection_rows,
    salary_rows,
    fallback_rows,
    offensive_actual_rows,
    keeper_count,
    keeper_spend,
):
    contract = get_replay_contract(year, "beta")

    assert contract.projection_rows == projection_rows
    assert contract.current_method_projection_rows == primary_projection_rows
    assert contract.salary_rows == contract.current_method_salary_rows == salary_rows
    assert contract.legacy_fallback_rows == fallback_rows
    assert contract.offensive_actual_rows == offensive_actual_rows
    assert contract.keeper_count == keeper_count
    assert contract.keeper_spend == keeper_spend
    assert contract.historical_etr_rows == projection_rows
    assert contract.projection_training_through_year == year - 1
    assert contract.salary_training_through_year == year - 1


def test_unregistered_replay_context_fails_closed():
    with pytest.raises(ValueError, match="No reviewed Auction historical replay"):
        get_replay_contract(2025, "nv")


def test_additive_floor_normalization_hits_budget_without_changing_order():
    values = pd.Series([40.0, 20.0, 5.0, 2.0, 1.0])
    adjusted, shift, pre_total, post_total = additive_floor_normalize_market(
        values,
        slots=3,
        budget=60.0,
    )

    assert pre_total == 65.0
    assert shift < 0
    assert np.isclose(post_total, 60.0)
    assert np.isclose(adjusted.nlargest(3).sum(), 60.0)
    assert adjusted.ge(1.0).all()
    assert adjusted.sort_values(ascending=False).index.tolist() == [0, 1, 2, 3, 4]


def test_additive_floor_normalization_rejects_impossible_market():
    with pytest.raises(ValueError, match="Invalid replay salary market"):
        additive_floor_normalize_market(
            pd.Series([2.0, 1.0]),
            slots=3,
            budget=3.0,
        )
