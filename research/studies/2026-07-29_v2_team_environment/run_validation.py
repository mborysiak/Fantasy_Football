"""Test compact preseason team-environment and QB-style feature families."""

from __future__ import annotations

import importlib.util
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
BASE_STUDY_DIR = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-28_v2_projection_trajectory_adp"
)
BASE_RESULTS = BASE_STUDY_DIR / "results"
SPEC = importlib.util.spec_from_file_location(
    "projection_trajectory_validation",
    BASE_STUDY_DIR / "run_validation.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("Unable to load trajectory validation module")
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)

QB_YARDAGE = (
    "team_qb1_passing_yards",
    "team_qb1_rushing_yards",
)
QB_TDS = (
    "team_qb1_passing_tds",
    "team_qb1_rushing_tds",
)
QB_STYLE = ("team_qb1_rush_point_share",)
TEAM_SUPPORT = (
    "team_core_skill_points",
    "team_core_skill_projection_percentile",
    "team_supporting_cast_points",
)
TEAM_RUSH_SCORING = (
    "team_projected_rushing_yards",
    "team_projected_rushing_tds",
    "team_projected_offensive_tds",
)
TEAM_ENVIRONMENT_FEATURES = (
    *QB_YARDAGE,
    *QB_TDS,
    *QB_STYLE,
    *TEAM_SUPPORT,
    *TEAM_RUSH_SCORING,
)
VARIANTS = (
    "trajectory_base",
    "qb_yardage",
    "qb_tds",
    "qb_style",
    "team_support",
    "team_rush_scoring",
    "all_environment",
)


def _feature_variants(manifests):
    incumbent = validation._manifest(manifests, "residual_candidate_v1")
    trajectory = validation._manifest(
        manifests,
        "residual_projection_trajectory_challenger_v1",
    )
    team_environment = set(
        validation._manifest(
            manifests,
            "residual_team_environment_challenger_v1",
        )
    )
    if team_environment != set(TEAM_ENVIRONMENT_FEATURES):
        raise ValueError("Team-environment manifest is unexpected")
    base = tuple(
        dict.fromkeys(
            (
                *incumbent,
                *trajectory,
                *validation.POSITION_FEATURES,
            )
        )
    )

    def expanded(features):
        return tuple(dict.fromkeys((*base, *features)))

    return {
        "incumbent": base,
        "trajectory_base": base,
        "qb_yardage": expanded(QB_YARDAGE),
        "qb_tds": expanded(QB_TDS),
        "qb_style": expanded(QB_STYLE),
        "team_support": expanded(TEAM_SUPPORT),
        "team_rush_scoring": expanded(TEAM_RUSH_SCORING),
        "all_environment": expanded(TEAM_ENVIRONMENT_FEATURES),
    }


def _load_trajectory_oof(
    target,
    assignments,
    reference_features,
):
    source_names = tuple(
        f"trajectory_{family}" for family in validation.FAMILIES
    )
    source = pd.read_csv(BASE_RESULTS / "oof_predictions.csv")
    source = source[source["model_name"].isin(source_names)].copy()
    specifications = pd.read_csv(
        BASE_RESULTS / "model_specifications.csv"
    )
    specifications = specifications[
        specifications["model_name"].isin(source_names)
    ].copy()
    expected_features = set(reference_features)
    for row in specifications.itertuples(index=False):
        if set(json.loads(row.feature_names_json)) != expected_features:
            raise ValueError(
                f"Trajectory feature bridge failed for {row.model_name}"
            )

    keys = ["player_key", "season"]
    current_target = target[
        target["season"].between(
            validation.VALIDATION_START,
            validation.VALIDATION_END,
        )
    ][[*keys, "actual_target", "baseline_prediction"]]
    source_target = source[
        [*keys, "actual", "baseline_prediction"]
    ].drop_duplicates(keys)
    check = current_target.merge(
        source_target,
        on=keys,
        how="outer",
        suffixes=("_current", "_source"),
        indicator=True,
        validate="one_to_one",
    )
    if not check["_merge"].eq("both").all():
        raise ValueError("Trajectory OOF target keys changed")
    if not np.allclose(
        check["actual_target"],
        check["actual"],
        equal_nan=True,
    ):
        raise ValueError("Trajectory OOF actuals changed")
    if not np.allclose(
        check["baseline_prediction_current"],
        check["baseline_prediction_source"],
        equal_nan=True,
    ):
        raise ValueError("Trajectory OOF baselines changed")

    old_assignments = pd.read_csv(BASE_RESULTS / "fold_assignments.csv")
    fold_check = assignments[
        [*keys, "fold", "training_through_season"]
    ].merge(
        old_assignments[
            [*keys, "fold", "training_through_season"]
        ],
        on=keys,
        how="outer",
        suffixes=("_current", "_source"),
        indicator=True,
        validate="one_to_one",
    )
    if not fold_check["_merge"].eq("both").all():
        raise ValueError("Trajectory OOF fold keys changed")
    if not (
        fold_check["fold_current"].eq(fold_check["fold_source"])
        & fold_check["training_through_season_current"].eq(
            fold_check["training_through_season_source"]
        )
    ).all():
        raise ValueError("Trajectory OOF fold assignments changed")

    rename = {
        f"trajectory_{family}": f"trajectory_base_{family}"
        for family in validation.FAMILIES
    }
    source["model_name"] = source["model_name"].replace(rename)
    source["feature_set"] = "trajectory_base"
    specifications["model_name"] = specifications["model_name"].replace(
        rename
    )
    specifications["feature_variant"] = "trajectory_base"
    oof_frames = [
        group.copy()
        for _, group in source.groupby("model_name", sort=True)
    ]
    bridge = pd.DataFrame(
        [
            {
                "source_study": str(BASE_STUDY_DIR),
                "source_feature_run_id": source["feature_run_id"].iloc[0],
                "current_feature_run_id": target["run_id"].iloc[0],
                "target_rows": len(current_target),
                "target_keys_equal": 1,
                "actuals_equal": 1,
                "baselines_equal": 1,
                "folds_equal": 1,
                "reference_feature_names_equal": 1,
                "bridge_reason": (
                    "additive team-environment mart columns; trajectory "
                    "reference features and targets unchanged"
                ),
            }
        ]
    )
    return oof_frames, specifications, bridge


def _comparison_pairs():
    rows = []
    for variant in VARIANTS[1:]:
        for family in validation.FAMILIES:
            rows.append(
                (
                    f"{variant}_{family}",
                    f"trajectory_base_{family}",
                    f"{variant}_vs_trajectory_base_{family}",
                )
            )
        for suffix in ("tree_average", "equal_thirds"):
            rows.append(
                (
                    f"{variant}_{suffix}",
                    f"trajectory_base_{suffix}",
                    f"{variant}_vs_trajectory_base_{suffix}",
                )
            )
        rows.append(
            (
                f"{variant}_equal_thirds",
                f"{variant}_tree_average",
                f"{variant}_equal_thirds_vs_tree",
            )
        )
    return rows


_base_comparisons = validation._comparisons


def _comparisons(scores):
    output = _base_comparisons(scores)
    primary = (
        output["challenger"].str.endswith("_equal_thirds")
        & output["reference"].eq("trajectory_base_equal_thirds")
    )
    pvalues = output.loc[primary, "sign_flip_pvalue"].to_numpy()
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 1.0
    for reverse_index in range(len(order) - 1, -1, -1):
        rank = reverse_index + 1
        source_index = order[reverse_index]
        running = min(running, pvalues[source_index] * len(order) / rank)
        adjusted[source_index] = running
    output["primary_bh_qvalue"] = np.nan
    output.loc[primary, "primary_bh_qvalue"] = adjusted
    return output


def _feature_audit(target):
    rows = []
    for column in TEAM_ENVIRONMENT_FEATURES:
        values = pd.to_numeric(target[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "non_null_rows": int(values.notna().sum()),
                "missing_rows": int(values.isna().sum()),
                "zero_rows": int(values.fillna(np.inf).eq(0).sum()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def _role_route_diagnostics(comparison):
    base_name = "trajectory_base_equal_thirds"
    actual = pd.to_numeric(comparison["actual"], errors="coerce")
    base = pd.to_numeric(comparison[base_name], errors="coerce")

    def rmse(observed, predicted):
        return float(np.sqrt(np.square(observed - predicted).mean()))

    route_masks = {
        "pass_catcher_only": comparison["position"].isin(["WR", "TE"]),
        "rb_only": comparison["position"].eq("RB"),
        "all_skill": comparison["position"].isin(["RB", "WR", "TE"]),
    }
    rows = []
    for variant in VARIANTS[1:]:
        variant_name = f"{variant}_equal_thirds"
        variant_prediction = pd.to_numeric(
            comparison[variant_name],
            errors="coerce",
        )
        for route_name, route_mask in route_masks.items():
            routed = base.copy()
            routed.loc[route_mask] = variant_prediction.loc[route_mask]
            season_delta = []
            for _, group in comparison.assign(
                _base=base,
                _routed=routed,
            ).groupby("season"):
                season_delta.append(
                    rmse(group["actual"], group["_routed"])
                    - rmse(group["actual"], group["_base"])
                )
            delta = np.asarray(season_delta, dtype=float)
            rng = np.random.default_rng(
                validation.RANDOM_SEED
                + len(rows)
            )
            draws = np.asarray(
                [
                    rng.choice(delta, len(delta), replace=True).mean()
                    for _ in range(20_000)
                ]
            )
            rows.append(
                {
                    "variant": variant,
                    "route": route_name,
                    "challenger_rmse": rmse(actual, routed),
                    "reference_rmse": rmse(actual, base),
                    "pooled_delta": (
                        rmse(actual, routed) - rmse(actual, base)
                    ),
                    "mean_season_delta": float(delta.mean()),
                    "recent_mean_delta": float(delta[-3:].mean()),
                    "season_wins": int((delta < 0).sum()),
                    "season_count": len(delta),
                    "sign_flip_pvalue": (
                        validation._sign_flip_pvalue(delta)
                    ),
                    "bootstrap_95_low": float(
                        np.quantile(draws, 0.025)
                    ),
                    "bootstrap_95_high": float(
                        np.quantile(draws, 0.975)
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values("pooled_delta")


_base_score_predictions = validation._score_predictions


def _score_predictions(frame, methods):
    scored = _base_score_predictions(frame, methods)
    diagnostics = frame.copy()
    diagnostics["role_group"] = diagnostics["position"].replace(
        {"WR": "PASS_CATCHER", "TE": "PASS_CATCHER"}
    )
    diagnostics["qb_style_band"] = pd.cut(
        pd.to_numeric(
            diagnostics["team_qb1_rush_point_share"],
            errors="coerce",
        ),
        bins=(-np.inf, 0.08, 0.16, np.inf),
        labels=("low_rush", "balanced", "high_rush"),
    ).astype(object).fillna("missing")
    diagnostics["team_strength_band"] = pd.cut(
        pd.to_numeric(
            diagnostics["team_core_skill_projection_percentile"],
            errors="coerce",
        ),
        bins=(-np.inf, 0.25, 0.75, np.inf),
        labels=("bottom_quartile", "middle_half", "top_quartile"),
    ).astype(object).fillna("missing")
    rows = []
    for slice_type, column in (
        ("role_group", "role_group"),
        ("qb_style_band", "qb_style_band"),
        ("team_strength_band", "team_strength_band"),
    ):
        rows.extend(
            validation._score_rows(
                diagnostics,
                methods,
                slice_type,
                column,
            )
        )
    return pd.concat((scored, pd.DataFrame(rows)), ignore_index=True)


_base_summary = validation._summary_markdown


def _summary_markdown(scores, comparisons):
    return _base_summary(scores, comparisons).replace(
        "# Projection Trajectory and Logged-ADP Results",
        "# Team Environment and QB Style Results",
        1,
    ).replace("Incumbent", "Trajectory reference")


_base_create_run_id = validation.create_run_id
validation.RESULTS_DIR = STUDY_DIR / "results"
validation.VARIANTS = VARIANTS
validation.TRAJECTORY_FEATURES = (
    *validation.TRAJECTORY_FEATURES,
    *TEAM_ENVIRONMENT_FEATURES,
)
validation._feature_variants = _feature_variants
validation._load_incumbent_oof = _load_trajectory_oof
validation._comparison_pairs = _comparison_pairs
validation._comparisons = _comparisons
validation._feature_audit = _feature_audit
validation._score_predictions = _score_predictions
validation._summary_markdown = _summary_markdown
validation.create_run_id = lambda _: _base_create_run_id(
    "m4a_team_environment"
)


def _study_context():
    features, manifests, feature_run_id = validation._load_inputs()
    target = validation.build_target_frames(
        features,
        validation.VALIDATION_END,
    )[validation.CONDITIONAL_PPG_TARGET]
    variants = _feature_variants(manifests)
    run_id = validation.create_run_id("m4a_team_environment")
    assignments = validation.make_fold_assignments(
        target,
        validation.CONDITIONAL_PPG_TARGET,
        run_id,
        validation.VALIDATION_START,
        validation.N_SPLITS,
        validation.RANDOM_SEED,
    )
    reference_oof, reference_specs, bridge = _load_trajectory_oof(
        target,
        assignments,
        variants["incumbent"],
    )
    return (
        target,
        variants,
        feature_run_id,
        run_id,
        assignments,
        reference_oof,
        reference_specs,
        bridge,
    )


def _run_variant(variant: str) -> None:
    if variant not in VARIANTS[1:]:
        raise ValueError(f"Unsupported challenger variant: {variant}")
    (
        target,
        variants,
        feature_run_id,
        run_id,
        assignments,
        _,
        _,
        _,
    ) = _study_context()
    batch_dir = STUDY_DIR / "results" / "batches" / variant
    batch_dir.mkdir(parents=True, exist_ok=True)
    oof_frames = []
    parameter_frames = []
    specification_rows = []
    feature_columns = variants[variant]
    for family in validation.FAMILIES:
        spec = validation._model_spec(family, variant)
        print(f"Fitting {spec.model_name}", flush=True)
        oof, parameters = validation.run_model_spec(
            target,
            assignments,
            spec,
            feature_columns,
            run_id,
            feature_run_id,
            validation.VALIDATION_START,
            validation.N_SPLITS,
            validation.RANDOM_SEED,
        )
        oof_frames.append(oof)
        parameter_frames.append(parameters)
        specification_rows.append(
            {
                "run_id": run_id,
                "feature_run_id": feature_run_id,
                "model_name": spec.model_name,
                "model_family": family,
                "feature_variant": variant,
                "feature_count": len(feature_columns),
                "feature_names_json": json.dumps(feature_columns),
                "hyperparameters_json": json.dumps(
                    spec.parameters,
                    default=list,
                    sort_keys=True,
                ),
            }
        )
        pd.concat(oof_frames, ignore_index=True).to_csv(
            batch_dir / "oof_predictions.csv",
            index=False,
        )
        pd.concat(parameter_frames, ignore_index=True).to_csv(
            batch_dir / "hyperparameters.csv",
            index=False,
        )
        pd.DataFrame(specification_rows).to_csv(
            batch_dir / "model_specifications.csv",
            index=False,
        )
    assignments.to_csv(batch_dir / "fold_assignments.csv", index=False)
    print(f"Completed {variant}", flush=True)


def _compile_batches() -> None:
    (
        target,
        variants,
        feature_run_id,
        _,
        assignments,
        reference_oof,
        reference_specs,
        bridge,
    ) = _study_context()
    oof_frames = list(reference_oof)
    parameter_frames = []
    specification_frames = []
    expected_rows = int(
        target["season"].between(
            validation.VALIDATION_START,
            validation.VALIDATION_END,
        ).sum()
    )
    for variant in VARIANTS[1:]:
        batch_dir = STUDY_DIR / "results" / "batches" / variant
        oof = pd.read_csv(batch_dir / "oof_predictions.csv")
        parameters = pd.read_csv(batch_dir / "hyperparameters.csv")
        specifications = pd.read_csv(
            batch_dir / "model_specifications.csv"
        )
        expected_models = {
            f"{variant}_{family}" for family in validation.FAMILIES
        }
        if set(oof["model_name"].unique()) != expected_models:
            raise ValueError(f"Incomplete batch models for {variant}")
        if not oof.groupby("model_name").size().eq(expected_rows).all():
            raise ValueError(f"Incomplete batch rows for {variant}")
        if set(oof["feature_run_id"].unique()) != {feature_run_id}:
            raise ValueError(f"Stale feature lineage for {variant}")
        oof_frames.append(oof)
        parameter_frames.append(parameters)
        specification_frames.append(specifications)

    oof_all = pd.concat(oof_frames, ignore_index=True)
    counts = oof_all.groupby("model_name").size()
    if len(counts) != len(VARIANTS) * len(validation.FAMILIES):
        raise ValueError(f"Unexpected model count: {counts.to_dict()}")
    if not counts.eq(expected_rows).all():
        raise ValueError(f"Incomplete OOF models: {counts.to_dict()}")

    keys = ["player_key", "season"]
    metadata = target.loc[
        target["season"].between(
            validation.VALIDATION_START,
            validation.VALIDATION_END,
        ),
        [
            *keys,
            "position",
            "baseline_prediction",
            "year_exp",
            "is_rookie",
            "has_prior_outcome",
            "adp_median",
            "actual_target",
            *validation.TRAJECTORY_FEATURES,
        ],
    ].rename(columns={"actual_target": "actual"})
    wide = oof_all.pivot(
        index=keys,
        columns="model_name",
        values="final_prediction",
    ).reset_index()
    comparison = metadata.merge(
        wide,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} comparison rows, found "
            f"{len(comparison)}"
        )
    blend_methods = validation._add_blends(comparison)
    comparison["history_depth"] = validation._history_depth(comparison)
    comparison["projection_history"] = (
        validation._projection_history_depth(comparison)
    )
    comparison["adp_band"] = validation._adp_band(
        comparison["adp_median"]
    )
    model_methods = sorted(oof_all["model_name"].unique())
    methods = [*model_methods, *blend_methods]
    scores = _score_predictions(comparison, methods)
    paired = _comparisons(scores)
    audit = _feature_audit(
        target[
            target["season"].between(
                validation.VALIDATION_START,
                validation.VALIDATION_END,
            )
        ]
    )
    role_routes = _role_route_diagnostics(comparison)

    results_dir = STUDY_DIR / "results"
    oof_all.to_csv(results_dir / "oof_predictions.csv", index=False)
    pd.concat(parameter_frames, ignore_index=True).to_csv(
        results_dir / "hyperparameters.csv",
        index=False,
    )
    assignments.to_csv(results_dir / "fold_assignments.csv", index=False)
    reference_specs.to_csv(
        results_dir / "source_reference_specifications.csv",
        index=False,
    )
    pd.concat(specification_frames, ignore_index=True).to_csv(
        results_dir / "model_specifications.csv",
        index=False,
    )
    bridge.to_csv(results_dir / "lineage_bridge.csv", index=False)
    comparison.to_csv(
        results_dir / "comparison_predictions.csv",
        index=False,
    )
    scores.to_csv(results_dir / "model_scores.csv", index=False)
    paired.to_csv(results_dir / "model_comparisons.csv", index=False)
    audit.to_csv(results_dir / "feature_audit.csv", index=False)
    role_routes.to_csv(
        results_dir / "role_route_diagnostics.csv",
        index=False,
    )
    (results_dir / "summary.md").write_text(
        _summary_markdown(scores, paired),
        encoding="utf-8",
    )
    print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS[1:])
    parser.add_argument("--compile", action="store_true")
    arguments = parser.parse_args()
    if arguments.variant:
        _run_variant(arguments.variant)
    elif arguments.compile:
        _compile_batches()
    else:
        parser.error("Pass --variant or --compile")
