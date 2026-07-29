import pandas as pd
import pytest

from Scripts.V2.modeling import (
    CONDITIONAL_PPG_TARGET,
    PARTICIPATION_TARGET,
    POSITION_FEATURES,
    ModelSpec,
    _build_pipeline,
    _load_scikit_model,
    build_feature_sets,
    initial_model_specs,
    make_fold_assignments,
    rolling_position_rate,
)


def _manifest(name: str, features: tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {
            "manifest_name": name,
            "feature_name": feature,
        }
        for feature in features
    ]


def test_compact_feature_sets_stay_inside_reviewed_manifests():
    from Scripts.V2.modeling import (
        PARTICIPATION_COMPACT_FEATURES,
        RESIDUAL_COMPACT_FEATURES,
    )

    manifests = pd.DataFrame(
        _manifest(
            "residual_candidate_v1",
            RESIDUAL_COMPACT_FEATURES + ("residual_extra",),
        )
        + _manifest(
            "participation_candidate_v1",
            PARTICIPATION_COMPACT_FEATURES + ("participation_extra",),
        )
    )
    feature_sets = build_feature_sets(manifests)

    residual = feature_sets[CONDITIONAL_PPG_TARGET]
    participation = feature_sets[PARTICIPATION_TARGET]
    assert residual["compact"] == RESIDUAL_COMPACT_FEATURES + POSITION_FEATURES
    assert set(RESIDUAL_COMPACT_FEATURES).issubset(
        residual["full_manifest"]
    )
    assert participation["compact"] == (
        PARTICIPATION_COMPACT_FEATURES + POSITION_FEATURES
    )
    assert set(PARTICIPATION_COMPACT_FEATURES).issubset(
        participation["full_manifest"]
    )


def test_fold_assignments_are_deterministic_and_strictly_prior():
    rows = []
    for season in (2017, 2018, 2019):
        for player in range(15):
            rows.append(
                {
                    "player_key": f"{season}-{player}",
                    "season": season,
                    "position": ("QB", "RB", "WR")[player % 3],
                }
            )
    frame = pd.DataFrame(rows)
    first = make_fold_assignments(
        frame,
        target_name=CONDITIONAL_PPG_TARGET,
        run_id="run",
        validation_start_season=2017,
        n_splits=5,
        random_seed=1234,
    )
    second = make_fold_assignments(
        frame,
        target_name=CONDITIONAL_PPG_TARGET,
        run_id="run",
        validation_start_season=2017,
        n_splits=5,
        random_seed=1234,
    )

    pd.testing.assert_frame_equal(first, second)
    assert (first["training_through_season"] == first["season"] - 1).all()
    assert first.groupby("season")["fold"].nunique().eq(5).all()
    assert not first.duplicated(["player_key", "season"]).any()


def test_prior_position_rate_does_not_use_current_or_future_outcomes():
    frame = pd.DataFrame(
        [
            {"season": 2018, "position": "RB", "appeared": 1},
            {"season": 2018, "position": "RB", "appeared": 0},
            {"season": 2019, "position": "RB", "appeared": 1},
            {"season": 2020, "position": "RB", "appeared": 0},
        ]
    )
    first = rolling_position_rate(frame, prior_strength=0)
    changed = frame.copy()
    changed.loc[changed["season"].ge(2019), "appeared"] = [0, 1]
    second = rolling_position_rate(changed, prior_strength=0)

    assert first.loc[frame["season"].eq(2019)].iloc[0] == pytest.approx(0.5)
    assert second.loc[frame["season"].eq(2019)].iloc[0] == pytest.approx(0.5)
    assert first.loc[frame["season"].eq(2020)].iloc[0] == pytest.approx(2 / 3)
    assert second.loc[frame["season"].eq(2020)].iloc[0] == pytest.approx(1 / 3)


def test_initial_model_surface_is_small_and_separates_transform_challengers():
    specs = initial_model_specs(search_iterations=4)
    assert len(specs) == 18
    assert {spec.model_family for spec in specs} == {
        "baseline",
        "ridge",
        "logistic",
        "lightgbm",
    }
    assert not any(
        spec.model_family in {"lasso", "elastic_net", "random_forest"}
        for spec in specs
    )
    transformed = {
        spec.pipeline_variant
        for spec in specs
        if spec.pipeline_variant not in {"none", "raw"}
    }
    assert transformed == {"kbest", "pca", "agglomeration"}
    assert all(
        spec.search_iterations <= 4
        for spec in specs
        if spec.model_family != "baseline"
    )


@pytest.mark.parametrize(
    ("model_family", "model_piece"),
    (("lasso", "lasso"), ("elastic_net", "enet")),
)
def test_sparse_linear_challengers_are_scaled_and_convergence_safe(
    model_family: str,
    model_piece: str,
):
    data = pd.DataFrame(
        {
            "player": ["player"],
            "team": ["team"],
            "week": [1],
            "year": [2025],
            "y_act": [1.0],
        }
    )
    SciKitModel = _load_scikit_model()
    skm = SciKitModel(data, model_obj="reg", set_seed=1234)
    spec = ModelSpec(
        CONDITIONAL_PPG_TARGET,
        f"direct_{model_family}_test",
        model_family,
        "direct",
        "test",
        "raw",
        model_piece,
        {},
        1,
    )

    pipeline = _build_pipeline(skm, spec, ("feature",))

    assert "std_scale" in pipeline.named_steps
    assert pipeline.named_steps[model_piece].max_iter == 20_000
    assert pipeline.named_steps[model_piece].tol == pytest.approx(1e-6)


def test_knn_challenger_is_scaled():
    data = pd.DataFrame(
        {
            "player": ["player"],
            "team": ["team"],
            "week": [1],
            "year": [2025],
            "y_act": [1.0],
        }
    )
    SciKitModel = _load_scikit_model()
    skm = SciKitModel(data, model_obj="reg", set_seed=1234)
    spec = ModelSpec(
        CONDITIONAL_PPG_TARGET,
        "direct_knn_test",
        "knn",
        "direct",
        "test",
        "raw",
        "knn",
        {},
        1,
    )

    pipeline = _build_pipeline(skm, spec, ("feature",))

    assert "std_scale" in pipeline.named_steps
