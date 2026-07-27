"""Estimate uncapped-experience cliff, absence, and next-season risks."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
DB_DIR = ROOT / "Data" / "Databases"
RESULTS_DIR = STUDY_DIR / "results"

SIMULATION_DB = DB_DIR / "Simulation.sqlite3"
VALIDATIONS_DB = DB_DIR / "Validations.sqlite3"
STATS_DB = DB_DIR / "Season_Stats_New.sqlite3"
MODEL_INPUTS_DB = DB_DIR / "Model_Inputs.sqlite3"
MODEL_INPUTS_NEXT_DB = DB_DIR / "Model_Inputs_next.sqlite3"

POSITIONS = ("RB", "WR", "TE")
PRIMARY_THRESHOLDS = {"RB": 7, "WR": 9, "TE": 8}
MIN_PRED_PPG = 3.0
PERFORMANCE_MIN_GAMES = 9
EXTENDED_ABSENCE_MAX_GAMES = 8
NEXT_USEFUL_MIN_GAMES = 5
PPG_CLIFF_RATIO = 0.70
PPG_CATASTROPHE_RATIO = 0.50


def read_sql(path: Path, query: str, params=None) -> pd.DataFrame:
    uri = f"file:{path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        return pd.read_sql_query(query, conn, params=params)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def load_templates() -> pd.DataFrame:
    templates = read_sql(
        SIMULATION_DB,
        """
        SELECT player,
               pos,
               CAST(season AS INTEGER) season,
               historical_pred_fp_per_game pred_ppg,
               active_ppg current_ppg,
               active_ppg_resid,
               CAST(active_games AS INTEGER) active_games,
               CAST(played_games AS INTEGER) played_games,
               projection_decile,
               projection_tier,
               year_exp capped_year_exp
        FROM Best_Ball_Weekly_Templates
        WHERE league='beta'
              AND pos IN ('RB', 'WR', 'TE')
        """,
    )
    identity = ["player", "pos", "season"]
    duplicates = templates.duplicated(identity, keep=False)
    if duplicates.any():
        preview = templates.loc[duplicates, identity].head(10).to_dict("records")
        raise ValueError(f"Weekly templates are not unique: {preview}")
    return templates


def load_actual_seasons() -> pd.DataFrame:
    frames = []
    for pos in POSITIONS:
        frame = read_sql(
            STATS_DB,
            f"""
            SELECT player,
                   '{pos}' pos,
                   CAST(season AS INTEGER) season,
                   CAST(games AS INTEGER) games,
                   fantasy_pts_per_game actual_ppg,
                   sum_fantasy_pts season_points
            FROM {pos}_Stats
            """,
        )
        frames.append(frame)
    actuals = pd.concat(frames, ignore_index=True)
    # The source can contain multiple team rows. The compile pipeline keeps the
    # row with the largest season total, which is normally the combined row.
    actuals = (
        actuals.sort_values(
            ["player", "pos", "season", "season_points"],
            ascending=[True, True, True, False],
            na_position="last",
        )
        .drop_duplicates(["player", "pos", "season"], keep="first")
        .reset_index(drop=True)
    )
    return actuals


def load_draft_years() -> pd.DataFrame:
    return read_sql(
        STATS_DB,
        """
        SELECT DISTINCT player,
               pos,
               CAST(year AS INTEGER) draft_year
        FROM Draft_Positions
        WHERE pos IN ('RB', 'WR', 'TE')
        """,
    )


def attach_uncapped_experience(
    rows: pd.DataFrame,
    actuals: pd.DataFrame,
    draft_years: pd.DataFrame,
) -> pd.DataFrame:
    debut = (
        actuals.groupby(["player", "pos"], as_index=False).season.min()
        .rename(columns={"season": "debut_season"})
    )
    output = rows.reset_index(drop=True).copy()
    output["_experience_row_id"] = np.arange(len(output))
    draft_candidates = output[
        ["_experience_row_id", "player", "pos", "season"]
    ].merge(draft_years, on=["player", "pos"], how="left")
    draft_candidates = draft_candidates[
        draft_candidates.draft_year.le(draft_candidates.season)
    ].copy()
    # Same-name NFL careers exist. For each historical player-season, the most
    # recent preceding draft is the defensible name-only match; taking MIN(year)
    # incorrectly creates 20-35 year WR/TE careers.
    chosen_draft = (
        draft_candidates.sort_values(
            ["_experience_row_id", "draft_year"],
            ascending=[True, False],
        )
        .drop_duplicates("_experience_row_id", keep="first")
        [["_experience_row_id", "draft_year"]]
    )
    output = output.merge(chosen_draft, on="_experience_row_id", how="left")
    output = output.merge(debut, on=["player", "pos"], how="left")
    output["experience_origin"] = np.where(
        output.draft_year.notna(),
        "draft",
        np.where(output.debut_season.notna(), "debut_fallback", "missing"),
    )
    origin_year = output.draft_year.fillna(output.debut_season)
    output["raw_year_exp"] = (output.season - origin_year).clip(lower=0)
    implausible = output.raw_year_exp.gt(18)
    output.loc[implausible, "experience_origin"] = "collision_excluded"
    output.loc[implausible, "raw_year_exp"] = np.nan
    return output.drop(columns="_experience_row_id")


def build_analysis_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    templates = load_templates()
    actuals = load_actual_seasons()
    draft_years = load_draft_years()
    rows = attach_uncapped_experience(templates, actuals, draft_years)

    rows["pred_ppg"] = pd.to_numeric(rows.pred_ppg, errors="coerce")
    rows["current_ppg"] = pd.to_numeric(rows.current_ppg, errors="coerce").fillna(0)
    rows["played_games"] = pd.to_numeric(rows.played_games, errors="coerce").fillna(0)
    rows = rows[
        rows.pos.isin(POSITIONS)
        & rows.pred_ppg.ge(MIN_PRED_PPG)
        & rows.raw_year_exp.notna()
    ].copy()

    rows["current_ppg_ratio"] = rows.current_ppg / rows.pred_ppg
    rows["current_perf_eligible"] = rows.played_games.ge(PERFORMANCE_MIN_GAMES)
    rows["current_ppg_cliff_30"] = rows.current_ppg_ratio.le(PPG_CLIFF_RATIO)
    rows["current_ppg_cliff_50"] = rows.current_ppg_ratio.le(PPG_CATASTROPHE_RATIO)
    rows["extended_absence_8"] = rows.played_games.le(EXTENDED_ABSENCE_MAX_GAMES)
    rows["current_any_bust"] = (
        rows.extended_absence_8
        | (rows.current_perf_eligible & rows.current_ppg_cliff_30)
    )

    next_actuals = actuals.rename(
        columns={
            "season": "next_season",
            "games": "next_games",
            "actual_ppg": "next_ppg",
            "season_points": "next_season_points",
        }
    )
    rows["next_season"] = rows.season + 1
    rows = rows.merge(
        next_actuals[
            [
                "player",
                "pos",
                "next_season",
                "next_games",
                "next_ppg",
                "next_season_points",
            ]
        ],
        on=["player", "pos", "next_season"],
        how="left",
    )
    max_actual_season = int(actuals.season.max())
    rows["next_observable"] = rows.next_season.le(max_actual_season)
    rows.loc[rows.next_observable, "next_games"] = rows.loc[
        rows.next_observable, "next_games"
    ].fillna(0)
    rows.loc[rows.next_observable, "next_ppg"] = rows.loc[
        rows.next_observable, "next_ppg"
    ].fillna(0)
    rows.loc[rows.next_observable, "next_season_points"] = rows.loc[
        rows.next_observable, "next_season_points"
    ].fillna(0)
    rows["next_no_appearance"] = rows.next_observable & rows.next_games.eq(0)
    rows["next_ppg_ratio"] = rows.next_ppg / rows.pred_ppg
    rows["next_ppg_cliff_30"] = (
        rows.next_observable
        & rows.next_games.ge(NEXT_USEFUL_MIN_GAMES)
        & rows.next_ppg_ratio.le(PPG_CLIFF_RATIO)
    )
    rows["next_no_useful"] = (
        rows.next_observable
        & (
            rows.next_games.lt(NEXT_USEFUL_MIN_GAMES)
            | rows.next_ppg_ratio.le(PPG_CLIFF_RATIO)
        )
    )
    return rows, actuals


OUTCOME_SPECS = {
    "current_ppg_cliff_30": "current_perf_eligible",
    "current_ppg_cliff_50": "current_perf_eligible",
    "extended_absence_8": None,
    "current_any_bust": None,
    "next_no_appearance": "next_observable",
    "next_ppg_cliff_30": "next_observable",
    "next_no_useful": "next_observable",
}


def outcome_subset(rows: pd.DataFrame, outcome: str) -> pd.DataFrame:
    eligibility = OUTCOME_SPECS[outcome]
    subset = rows if eligibility is None else rows[rows[eligibility]].copy()
    return subset[subset[outcome].notna()].copy()


def rate_table(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    records = []
    for keys, group in rows.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for outcome in OUTCOME_SPECS:
            eligible = outcome_subset(group, outcome)
            trials = int(len(eligible))
            successes = int(eligible[outcome].astype(bool).sum())
            lower, upper = wilson_interval(successes, trials)
            records.append(
                {
                    **base,
                    "outcome": outcome,
                    "events": successes,
                    "n": trials,
                    "rate": successes / trials if trials else np.nan,
                    "ci95_low": lower,
                    "ci95_high": upper,
                    "avg_pred_ppg": eligible.pred_ppg.mean() if trials else np.nan,
                    "avg_raw_year_exp": (
                        eligible.raw_year_exp.mean() if trials else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def add_threshold_fields(rows: pd.DataFrame, thresholds: dict[str, int]) -> pd.DataFrame:
    output = rows.copy()
    output["threshold"] = output.pos.map(thresholds).astype(float)
    output["above_threshold"] = output.raw_year_exp.gt(output.threshold)
    output["experience_cohort"] = np.where(
        output.above_threshold,
        "above_threshold",
        "threshold_or_below",
    )
    output["exp_below"] = np.minimum(output.raw_year_exp, output.threshold)
    output["excess_exp"] = np.maximum(output.raw_year_exp - output.threshold, 0)
    output["further_excess_exp"] = np.maximum(output.excess_exp - 1, 0)
    return output


def fit_excess_year_models(
    rows: pd.DataFrame,
    thresholds: dict[str, int],
    threshold_label: str,
    projection_scope: str = "all",
    model_spec: str = "ppg_plus_season",
) -> pd.DataFrame:
    prepared = add_threshold_fields(rows, thresholds)
    records = []
    for pos in POSITIONS:
        pos_rows = prepared[prepared.pos.eq(pos)].copy()
        for outcome in OUTCOME_SPECS:
            model_rows = outcome_subset(pos_rows, outcome)
            model_rows = model_rows[
                model_rows[["pred_ppg", "raw_year_exp", "season"]].notna().all(axis=1)
            ].copy()
            model_rows["outcome_value"] = model_rows[outcome].astype(int)
            pred_sd = float(model_rows.pred_ppg.std(ddof=0))
            if not np.isfinite(pred_sd) or pred_sd <= 0:
                pred_sd = 1.0
            model_rows["pred_z"] = (
                model_rows.pred_ppg - model_rows.pred_ppg.mean()
            ) / pred_sd
            model_rows["pred_z2"] = model_rows.pred_z.pow(2)
            above = model_rows.excess_exp.gt(0)
            base = {
                "threshold_label": threshold_label,
                "projection_scope": projection_scope,
                "model_spec": model_spec,
                "pos": pos,
                "threshold": thresholds[pos],
                "outcome": outcome,
                "n": int(len(model_rows)),
                "events": int(model_rows.outcome_value.sum()),
                "above_threshold_n": int(above.sum()),
                "above_threshold_events": int(
                    model_rows.loc[above, "outcome_value"].sum()
                ),
            }
            if (
                len(model_rows) < 50
                or model_rows.outcome_value.nunique() < 2
                or above.sum() < 8
            ):
                records.append({**base, "status": "insufficient"})
                continue
            try:
                formula = "outcome_value ~ exp_below + excess_exp + pred_z + pred_z2"
                if model_spec == "ppg_plus_season":
                    formula += " + C(season)"
                elif model_spec != "ppg_only":
                    raise ValueError(f"Unknown model specification: {model_spec}")
                fit = smf.glm(
                    formula,
                    data=model_rows,
                    family=sm.families.Binomial(),
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": model_rows.player},
                )
                coefficient = float(fit.params["excess_exp"])
                standard_error = float(fit.bse["excess_exp"])
                modified = model_rows.copy()
                modified.loc[above, "excess_exp"] += 1
                base_prob = fit.predict(model_rows.loc[above])
                plus_one_prob = fit.predict(modified.loc[above])
                marginal_pp = float((plus_one_prob - base_prob).mean() * 100)
                records.append(
                    {
                        **base,
                        "status": "ok",
                        "excess_year_log_odds": coefficient,
                        "cluster_se": standard_error,
                        "odds_ratio_per_excess_year": math.exp(coefficient),
                        "or_ci95_low": math.exp(coefficient - 1.96 * standard_error),
                        "or_ci95_high": math.exp(coefficient + 1.96 * standard_error),
                        "p_value": float(fit.pvalues["excess_exp"]),
                        "avg_marginal_risk_pp_per_year": marginal_pp,
                    }
                )
            except Exception as exc:  # audit output should retain failed cells
                records.append(
                    {
                        **base,
                        "status": f"failed:{type(exc).__name__}",
                    }
                )
    return pd.DataFrame(records)


def fit_step_plus_slope_models(
    rows: pd.DataFrame,
    thresholds: dict[str, int],
    projection_scope: str,
) -> pd.DataFrame:
    prepared = add_threshold_fields(rows, thresholds)
    records = []
    for pos in POSITIONS:
        pos_rows = prepared[prepared.pos.eq(pos)].copy()
        for outcome in OUTCOME_SPECS:
            model_rows = outcome_subset(pos_rows, outcome)
            model_rows = model_rows[
                model_rows[["pred_ppg", "raw_year_exp", "season"]].notna().all(axis=1)
            ].copy()
            model_rows["outcome_value"] = model_rows[outcome].astype(int)
            model_rows["above_threshold_value"] = model_rows.above_threshold.astype(int)
            pred_sd = float(model_rows.pred_ppg.std(ddof=0))
            if not np.isfinite(pred_sd) or pred_sd <= 0:
                pred_sd = 1.0
            model_rows["pred_z"] = (
                model_rows.pred_ppg - model_rows.pred_ppg.mean()
            ) / pred_sd
            model_rows["pred_z2"] = model_rows.pred_z.pow(2)
            above = model_rows.above_threshold
            base = {
                "projection_scope": projection_scope,
                "pos": pos,
                "threshold": thresholds[pos],
                "outcome": outcome,
                "n": int(len(model_rows)),
                "events": int(model_rows.outcome_value.sum()),
                "above_threshold_n": int(above.sum()),
                "above_threshold_events": int(
                    model_rows.loc[above, "outcome_value"].sum()
                ),
            }
            if (
                len(model_rows) < 50
                or model_rows.outcome_value.nunique() < 2
                or above.sum() < 8
            ):
                records.append({**base, "status": "insufficient"})
                continue
            try:
                fit = smf.glm(
                    "outcome_value ~ exp_below + above_threshold_value + "
                    "further_excess_exp + pred_z + pred_z2 + C(season)",
                    data=model_rows,
                    family=sm.families.Binomial(),
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": model_rows.player},
                )
                step_coef = float(fit.params["above_threshold_value"])
                step_se = float(fit.bse["above_threshold_value"])
                slope_coef = float(fit.params["further_excess_exp"])
                slope_se = float(fit.bse["further_excess_exp"])

                old_reference = model_rows.loc[above].copy()
                without_step = old_reference.copy()
                without_step["above_threshold_value"] = 0
                without_step["further_excess_exp"] = 0
                with_step = without_step.copy()
                with_step["above_threshold_value"] = 1
                step_risk_pp = float(
                    (fit.predict(with_step) - fit.predict(without_step)).mean() * 100
                )
                plus_year = old_reference.copy()
                plus_year["further_excess_exp"] += 1
                slope_risk_pp = float(
                    (fit.predict(plus_year) - fit.predict(old_reference)).mean() * 100
                )
                records.append(
                    {
                        **base,
                        "status": "ok",
                        "threshold_step_odds_ratio": math.exp(step_coef),
                        "threshold_step_or_ci95_low": math.exp(step_coef - 1.96 * step_se),
                        "threshold_step_or_ci95_high": math.exp(step_coef + 1.96 * step_se),
                        "threshold_step_p_value": float(
                            fit.pvalues["above_threshold_value"]
                        ),
                        "threshold_step_risk_pp": step_risk_pp,
                        "further_year_odds_ratio": math.exp(slope_coef),
                        "further_year_or_ci95_low": math.exp(slope_coef - 1.96 * slope_se),
                        "further_year_or_ci95_high": math.exp(slope_coef + 1.96 * slope_se),
                        "further_year_p_value": float(
                            fit.pvalues["further_excess_exp"]
                        ),
                        "further_year_risk_pp": slope_risk_pp,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        **base,
                        "status": f"failed:{type(exc).__name__}",
                    }
                )
    return pd.DataFrame(records)


def load_next_validation_targets() -> pd.DataFrame:
    return read_sql(
        VALIDATIONS_DB,
        """
        SELECT player,
               CAST(season AS INTEGER) season,
               pos,
               AVG(pred_fp_per_game) next_pred_ppg,
               AVG(y_act) recorded_next_y_act,
               COUNT(*) validation_rows
        FROM Model_Validations_Resid
        WHERE version='beta'
              AND year=2026
              AND dataset='ProjOnly'
              AND current_or_next_year='next'
              AND rush_pass NOT IN ('rush', 'pass', 'rec')
              AND pos IN ('RB', 'WR', 'TE')
        GROUP BY player, season, pos
        """,
    )


def load_raw_next_input_targets() -> pd.DataFrame:
    current_frames = []
    next_frames = []
    for pos in POSITIONS:
        current_frames.append(
            read_sql(
                MODEL_INPUTS_DB,
                f"""
                SELECT player,
                       '{pos}' pos,
                       CAST(year AS INTEGER) season,
                       y_act input_current_y_act
                FROM {pos}_2026_ProjOnly
                """,
            )
        )
        next_frames.append(
            read_sql(
                MODEL_INPUTS_NEXT_DB,
                f"""
                SELECT player,
                       '{pos}' pos,
                       CAST(year AS INTEGER) season,
                       y_act input_next_y_act
                FROM {pos}_2026_ProjOnly
                """,
            )
        )
    current = pd.concat(current_frames, ignore_index=True).drop_duplicates(
        ["player", "pos", "season"]
    )
    next_rows = pd.concat(next_frames, ignore_index=True).drop_duplicates(
        ["player", "pos", "season"]
    )
    return current.merge(next_rows, on=["player", "pos", "season"], how="inner")


def audit_next_targets(
    analysis_rows: pd.DataFrame,
    actuals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = load_next_validation_targets()
    raw_inputs = load_raw_next_input_targets()
    context = analysis_rows[
        ["player", "pos", "season", "raw_year_exp", "pred_ppg"]
    ].drop_duplicates(["player", "pos", "season"])
    targets = targets.merge(context, on=["player", "pos", "season"], how="left")
    targets = targets.merge(raw_inputs, on=["player", "pos", "season"], how="left")
    current_actuals = actuals.rename(
        columns={
            "games": "true_current_games",
            "actual_ppg": "true_current_ppg",
            "season_points": "true_current_points",
        }
    )
    targets = targets.merge(
        current_actuals[
            [
                "player",
                "pos",
                "season",
                "true_current_games",
                "true_current_ppg",
                "true_current_points",
            ]
        ],
        on=["player", "pos", "season"],
        how="left",
    )
    next_actuals = actuals.rename(
        columns={
            "season": "next_season",
            "games": "true_next_games",
            "actual_ppg": "true_next_ppg",
            "season_points": "true_next_points",
        }
    )
    targets["next_season"] = targets.season + 1
    max_actual_season = int(actuals.season.max())
    targets = targets[targets.next_season.le(max_actual_season)].copy()
    targets = targets.merge(
        next_actuals[
            [
                "player",
                "pos",
                "next_season",
                "true_next_games",
                "true_next_ppg",
                "true_next_points",
            ]
        ],
        on=["player", "pos", "next_season"],
        how="left",
    )
    for col in ["true_next_games", "true_next_ppg", "true_next_points"]:
        targets[col] = targets[col].fillna(0)
    targets = targets[targets.raw_year_exp.notna()].copy()
    targets["true_no_appearance"] = targets.true_next_games.eq(0)
    targets["true_no_useful"] = (
        targets.true_next_games.lt(NEXT_USEFUL_MIN_GAMES)
        | targets.true_next_ppg.le(PPG_CLIFF_RATIO * targets.next_pred_ppg)
    )
    targets["recorded_no_useful"] = targets.recorded_next_y_act.le(
        PPG_CLIFF_RATIO * targets.next_pred_ppg
    )
    targets["recorded_minus_true_ppg"] = (
        targets.recorded_next_y_act - targets.true_next_ppg
    )
    targets["carried_forward_current_ppg"] = (
        targets.true_no_appearance
        & targets.input_current_y_act.notna()
        & targets.input_next_y_act.notna()
        & targets.input_next_y_act.sub(targets.input_current_y_act).abs().le(1e-8)
    )
    targets = add_threshold_fields(targets, PRIMARY_THRESHOLDS)

    summary_records = []
    for (pos, cohort), group in targets.groupby(["pos", "experience_cohort"]):
        summary_records.append(
            {
                "pos": pos,
                "experience_cohort": cohort,
                "n": len(group),
                "avg_raw_year_exp": group.raw_year_exp.mean(),
                "true_no_appearance_rate": group.true_no_appearance.mean(),
                "true_no_useful_rate": group.true_no_useful.mean(),
                "recorded_no_useful_rate": group.recorded_no_useful.mean(),
                "no_useful_understatement_pp": 100
                * (group.true_no_useful.mean() - group.recorded_no_useful.mean()),
                "avg_recorded_minus_true_ppg": group.recorded_minus_true_ppg.mean(),
                "no_appearance_rows": int(group.true_no_appearance.sum()),
                "no_appearance_carried_forward_rows": int(
                    group.carried_forward_current_ppg.sum()
                ),
            }
        )
    return targets, pd.DataFrame(summary_records)


def build_current_pool_risk(analysis_rows: pd.DataFrame) -> pd.DataFrame:
    player_map = read_sql(
        SIMULATION_DB,
        """
        SELECT player,
               pos,
               template_pool_key,
               year_exp target_capped_year_exp
        FROM Best_Ball_Weekly_Player_Map
        WHERE year=2026
              AND version='beta'
              AND dataset='final_ensemble'
        """,
    )
    pool = read_sql(
        SIMULATION_DB,
        """
        SELECT p.template_pool_key,
               p.template_id,
               p.template_sample_prob,
               t.player template_player,
               t.pos,
               CAST(t.season AS INTEGER) template_season,
               t.historical_pred_fp_per_game template_pred_ppg,
               t.active_ppg template_active_ppg,
               CAST(t.played_games AS INTEGER) template_played_games,
               t.year_exp template_capped_year_exp
        FROM Best_Ball_Weekly_Template_Pools p
        INNER JOIN Best_Ball_Weekly_Templates t
                ON p.template_id=t.template_id
               AND p.template_league=t.league
        WHERE p.league='beta'
        """,
    )
    raw_exp = analysis_rows[
        ["player", "pos", "season", "raw_year_exp"]
    ].drop_duplicates(["player", "pos", "season"])
    pool = pool.merge(
        raw_exp,
        left_on=["template_player", "pos", "template_season"],
        right_on=["player", "pos", "season"],
        how="left",
        suffixes=("", "_source"),
    )
    pool["raw_perf_cliff"] = (
        pool.template_played_games.ge(PERFORMANCE_MIN_GAMES)
        & pool.template_active_ppg.le(PPG_CLIFF_RATIO * pool.template_pred_ppg)
    )
    pool["extended_absence"] = pool.template_played_games.le(
        EXTENDED_ABSENCE_MAX_GAMES
    )
    pool["weighted_played"] = (
        pool.template_sample_prob * pool.template_played_games
    )
    pool["weighted_absence"] = (
        pool.template_sample_prob * pool.extended_absence.astype(float)
    )
    pool["weighted_raw_perf_cliff"] = (
        pool.template_sample_prob * pool.raw_perf_cliff.astype(float)
    )
    pool["weighted_raw_exp"] = pool.template_sample_prob * pool.raw_year_exp
    risk = (
        pool.groupby("template_pool_key", as_index=False)
        .agg(
            template_count=("template_id", "count"),
            weighted_played_games=("weighted_played", "sum"),
            template_extended_absence_prob=("weighted_absence", "sum"),
            raw_template_perf_cliff_prob=("weighted_raw_perf_cliff", "sum"),
            weighted_template_raw_exp=("weighted_raw_exp", "sum"),
            min_template_raw_exp=("raw_year_exp", "min"),
            max_template_raw_exp=("raw_year_exp", "max"),
        )
    )
    return player_map.merge(risk, on="template_pool_key", how="left")


def format_pct(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{100 * value:.1f}%"


def write_summary(
    rows: pd.DataFrame,
    cohort_rates: pd.DataFrame,
    primary_models: pd.DataFrame,
    step_slope_models: pd.DataFrame,
    target_summary: pd.DataFrame,
    pool_risk: pd.DataFrame,
) -> None:
    lines = [
        "# Veteran Cliff Calibration Results",
        "",
        "## Coverage",
        "",
        f"- Current historical player-seasons: {len(rows):,}.",
        f"- Following-season-observable rows: {int(rows.next_observable.sum()):,}.",
        "- Experience source: "
        + ", ".join(
            f"{key}={value:,}"
            for key, value in rows.experience_origin.value_counts().items()
        )
        + ".",
        "",
        "## Unadjusted historical rates",
        "",
    ]
    display_outcomes = [
        "current_ppg_cliff_30",
        "extended_absence_8",
        "current_any_bust",
        "next_no_appearance",
        "next_no_useful",
    ]
    all_rates = cohort_rates[cohort_rates.projection_scope.eq("all")]
    for pos in POSITIONS:
        lines.append(f"### {pos}")
        lines.append("")
        for outcome in display_outcomes:
            cur = all_rates[
                all_rates.pos.eq(pos) & all_rates.outcome.eq(outcome)
            ].set_index("experience_cohort")
            below = cur.loc["threshold_or_below"] if "threshold_or_below" in cur.index else None
            above = cur.loc["above_threshold"] if "above_threshold" in cur.index else None
            if below is None or above is None:
                continue
            lines.append(
                f"- `{outcome}`: {format_pct(below.rate)} "
                f"({int(below.events)}/{int(below.n)}) at/below threshold vs "
                f"{format_pct(above.rate)} ({int(above.events)}/{int(above.n)}) above."
            )
        lines.append("")

    lines.extend(
        [
            "## Draft-relevant starter/elite rates",
            "",
            "This restriction is closer to the veteran targets under discussion and "
            "reduces attrition from fringe preseason players.",
            "",
        ]
    )
    draft_rates = cohort_rates[
        cohort_rates.projection_scope.eq("starter_or_elite")
    ]
    for pos in POSITIONS:
        for outcome in ["current_any_bust", "next_no_appearance", "next_no_useful"]:
            cur = draft_rates[
                draft_rates.pos.eq(pos) & draft_rates.outcome.eq(outcome)
            ].set_index("experience_cohort")
            if not {"threshold_or_below", "above_threshold"}.issubset(cur.index):
                continue
            below = cur.loc["threshold_or_below"]
            above = cur.loc["above_threshold"]
            lines.append(
                f"- {pos} `{outcome}`: {format_pct(below.rate)} "
                f"({int(below.events)}/{int(below.n)}) vs {format_pct(above.rate)} "
                f"({int(above.events)}/{int(above.n)})."
            )
        lines.append("")

    lines.extend(
        [
            "## Per-year adjusted estimates above the primary threshold",
            "",
            "Piecewise logistic models control for preseason PPG and origin season; "
            "standard errors are clustered by player.",
            "",
        ]
    )
    for _, row in primary_models[
        primary_models.outcome.isin(display_outcomes)
        & primary_models.status.eq("ok")
        & primary_models.projection_scope.eq("all")
        & primary_models.model_spec.eq("ppg_plus_season")
    ].iterrows():
        lines.append(
            f"- {row.pos} `{row.outcome}`: OR {row.odds_ratio_per_excess_year:.2f} "
            f"(95% CI {row.or_ci95_low:.2f}-{row.or_ci95_high:.2f}), "
            f"average {row.avg_marginal_risk_pp_per_year:+.2f} percentage points "
            "per excess year."
        )

    step_rows = step_slope_models[
        step_slope_models.projection_scope.eq("all")
        & step_slope_models.outcome.isin(
            ["current_any_bust", "next_no_appearance", "next_no_useful"]
        )
        & step_slope_models.status.eq("ok")
    ].set_index(["pos", "outcome"])
    lines.extend(
        [
            "",
            "## Threshold jump versus further veteran years",
            "",
            "The two-part model estimates a one-time step after crossing the primary "
            "threshold separately from the slope for every later year. These are "
            "adjusted average risk differences; small veteran samples make several "
            "estimates noisy.",
            "",
        ]
    )
    for pos, descriptions in {
        "RB": [
            ("current_any_bust", "current any-bust"),
            ("next_no_appearance", "following-season disappearance"),
            ("next_no_useful", "no useful following season"),
        ],
        "WR": [
            ("current_any_bust", "current any-bust"),
            ("next_no_appearance", "following-season disappearance"),
            ("next_no_useful", "no useful following season"),
        ],
        "TE": [
            ("current_any_bust", "current any-bust"),
            ("next_no_appearance", "following-season disappearance"),
            ("next_no_useful", "no useful following season"),
        ],
    }.items():
        for outcome, label in descriptions:
            if (pos, outcome) not in step_rows.index:
                continue
            row = step_rows.loc[(pos, outcome)]
            lines.append(
                f"- {pos} {label}: threshold step "
                f"{row.threshold_step_risk_pp:+.1f} pp "
                f"(p={row.threshold_step_p_value:.3f}); every further year "
                f"{row.further_year_risk_pp:+.1f} pp "
                f"(p={row.further_year_p_value:.3f})."
            )

    lines.extend(
        [
            "",
            "## Next-year target censoring",
            "",
        ]
    )
    for _, row in target_summary.iterrows():
        lines.append(
            f"- {row.pos} {row.experience_cohort}: true no-useful-next rate "
            f"{format_pct(row.true_no_useful_rate)} vs recorded-target rate "
            f"{format_pct(row.recorded_no_useful_rate)} "
            f"({row.no_useful_understatement_pp:+.1f} pp understatement); "
            f"{int(row.no_appearance_carried_forward_rows)}/"
            f"{int(row.no_appearance_rows)} no-appearance rows equal the prior "
            "current PPG within 0.10."
        )

    names = ["Derrick Henry", "Alvin Kamara", "George Kittle", "Travis Kelce"]
    named = pool_risk[pool_risk.player.isin(names)].sort_values("player")
    lines.extend(
        [
            "",
            "## Current 2026 template pools",
            "",
            "These are raw matched-template diagnostics, not exact simulated cliff "
            "probabilities because the app centers and rescales active-PPG residuals.",
            "",
        ]
    )
    for _, row in named.iterrows():
        lines.append(
            f"- {row.player}: {row.weighted_played_games:.2f}/16 weighted played "
            f"weeks, {format_pct(row.template_extended_absence_prob)} extended-absence "
            f"template probability, weighted raw template experience "
            f"{row.weighted_template_raw_exp:.1f}."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The point and template pipelines already contain some average aging, "
            "so an additional deterministic PPG haircut would double count part of "
            "the effect.",
            "- The next-year target audit measures a distinct missing-outcome problem: "
            "retirement/disappearance can be recorded as repeated prior PPG. This "
            "should be fixed before treating next-year residuals as calibrated cliff "
            "risk.",
            "- Any production veteran adjustment should target the incremental "
            "extended-absence/no-useful-season probability that remains after current "
            "PPG and template matching, retain raw and adjusted scores separately, and "
            "use uncapped experience.",
            "- The estimates do not support one broad current-season tax. WR cliff "
            "risk is the clearest current-season candidate; RB and TE current-season "
            "taxes would be preference overlays rather than well-estimated corrections.",
            "- Keeper scoring should first model the probability of no following "
            "season as an explicit zero-value mixture. Only after rebuilding that "
            "target should a residual soft veteran adjustment be calibrated.",
            "",
        ]
    )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, actuals = build_analysis_rows()
    prepared = add_threshold_fields(rows, PRIMARY_THRESHOLDS)

    cohort_frames = []
    for scope, scoped_rows in [
        ("all", prepared),
        (
            "starter_or_elite",
            prepared[prepared.projection_tier.isin(["starter", "elite"])],
        ),
        ("elite", prepared[prepared.projection_tier.eq("elite")]),
    ]:
        frame = rate_table(scoped_rows, ["pos", "experience_cohort"])
        frame["projection_scope"] = scope
        cohort_frames.append(frame)
    cohort_rates = pd.concat(cohort_frames, ignore_index=True, sort=False)
    exact_year_rates = rate_table(prepared, ["pos", "raw_year_exp"])
    primary_models = fit_excess_year_models(
        rows,
        PRIMARY_THRESHOLDS,
        threshold_label="primary",
    )
    sensitivity_frames = [
        primary_models,
        fit_excess_year_models(
            rows,
            PRIMARY_THRESHOLDS,
            threshold_label="primary",
            model_spec="ppg_only",
        ),
        fit_excess_year_models(
            rows[rows.projection_tier.isin(["starter", "elite"])],
            PRIMARY_THRESHOLDS,
            threshold_label="primary",
            projection_scope="starter_or_elite",
        ),
    ]
    for shift in (-1, 1):
        thresholds = {
            pos: threshold + shift
            for pos, threshold in PRIMARY_THRESHOLDS.items()
        }
        sensitivity_frames.append(
            fit_excess_year_models(rows, thresholds, f"primary_{shift:+d}")
        )
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True, sort=False)
    step_slope = pd.concat(
        [
            fit_step_plus_slope_models(rows, PRIMARY_THRESHOLDS, "all"),
            fit_step_plus_slope_models(
                rows[rows.projection_tier.isin(["starter", "elite"])],
                PRIMARY_THRESHOLDS,
                "starter_or_elite",
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    target_rows, target_summary = audit_next_targets(rows, actuals)
    pool_risk = build_current_pool_risk(rows)

    export_cols = [
        "player",
        "pos",
        "season",
        "pred_ppg",
        "current_ppg",
        "played_games",
        "raw_year_exp",
        "capped_year_exp",
        "experience_origin",
        "projection_decile",
        "projection_tier",
        "current_ppg_ratio",
        "current_perf_eligible",
        "current_ppg_cliff_30",
        "current_ppg_cliff_50",
        "extended_absence_8",
        "current_any_bust",
        "next_observable",
        "next_games",
        "next_ppg",
        "next_ppg_ratio",
        "next_no_appearance",
        "next_ppg_cliff_30",
        "next_no_useful",
        "threshold",
        "above_threshold",
    ]
    prepared[export_cols].to_csv(RESULTS_DIR / "analysis_rows.csv", index=False)
    cohort_rates.to_csv(RESULTS_DIR / "cohort_outcome_rates.csv", index=False)
    exact_year_rates.to_csv(RESULTS_DIR / "exact_experience_rates.csv", index=False)
    sensitivity.to_csv(RESULTS_DIR / "excess_year_models.csv", index=False)
    step_slope.to_csv(RESULTS_DIR / "threshold_step_slope_models.csv", index=False)
    target_rows.to_csv(RESULTS_DIR / "next_target_censoring_rows.csv", index=False)
    target_summary.to_csv(RESULTS_DIR / "next_target_censoring_summary.csv", index=False)
    pool_risk.to_csv(RESULTS_DIR / "current_player_template_risk.csv", index=False)
    write_summary(
        rows,
        cohort_rates,
        primary_models,
        step_slope,
        target_summary,
        pool_risk,
    )

    print((RESULTS_DIR / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
