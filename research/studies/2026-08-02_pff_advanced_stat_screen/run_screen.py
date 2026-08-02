"""Leakage-safe predictive and redundancy screen for prior-season PFF stats."""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.locked_candidates import PRIMARY_PPG_FEATURES


STUDY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STUDY_DIR / "results"
RAW_DB = REPO_ROOT / "Data" / "Databases" / "Season_Stats_New.sqlite3"
LEAGUE_DBS = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
}
VALIDATION_START = 2017
VALIDATION_END = 2025
MODEL_START = 2018
DEVELOPMENT_END = 2022
REC_OPPORTUNITY_THRESHOLD = 100.0
RUSH_OPPORTUNITY_THRESHOLD = 50.0
RIDGE_ALPHA = 10.0
LOGISTIC_C = 0.20
BOOTSTRAP_DRAWS = 20_000
RANDOM_SEED = 1234


@dataclass(frozen=True)
class Candidate:
    name: str
    domain: str
    family: str
    denominator: str
    prior_strength: float
    positions: tuple[str, ...]


REC_CANDIDATES = (
    Candidate("rec_yprr", "receiving", "earning", "routes", 100.0, ("RB", "WR", "TE")),
    Candidate("rec_first_downs_per_route", "receiving", "earning", "routes", 100.0, ("RB", "WR", "TE")),
    Candidate("rec_targets_per_route", "receiving", "earning", "routes", 100.0, ("RB", "WR", "TE")),
    Candidate("rec_touchdowns_per_route", "receiving", "scoring", "routes", 100.0, ("RB", "WR", "TE")),
    Candidate("rec_yards_per_target", "receiving", "efficiency", "targets", 25.0, ("RB", "WR", "TE")),
    Candidate("rec_adot", "receiving", "role_style", "targets", 25.0, ("RB", "WR", "TE")),
    Candidate("rec_yac_per_route", "receiving", "role_style", "routes", 100.0, ("RB", "WR", "TE")),
    Candidate("rec_mtf_per_reception", "receiving", "tackle_breaking", "receptions", 20.0, ("RB", "WR", "TE")),
    Candidate("rec_route_rate", "receiving", "participation", "routes", 100.0, ("RB", "WR", "TE")),
    Candidate("rec_slot_rate", "receiving", "alignment", "routes", 100.0, ("RB", "WR", "TE")),
    Candidate("rec_wide_rate", "receiving", "alignment", "routes", 100.0, ("RB", "WR", "TE")),
    Candidate("rec_route_grade", "receiving", "grade", "routes", 100.0, ("RB", "WR", "TE")),
    Candidate("rec_targeted_qb_rating", "receiving", "efficiency", "targets", 25.0, ("RB", "WR", "TE")),
)

RUSH_CANDIDATES = (
    Candidate("rush_ypa", "rushing", "efficiency", "attempts", 50.0, ("RB", "QB")),
    Candidate("rush_first_downs_per_attempt", "rushing", "earning", "attempts", 50.0, ("RB", "QB")),
    Candidate("rush_mtf_per_attempt", "rushing", "tackle_breaking", "attempts", 50.0, ("RB", "QB")),
    Candidate("rush_yco_per_attempt", "rushing", "tackle_breaking", "attempts", 50.0, ("RB", "QB")),
    Candidate("rush_explosive_per_attempt", "rushing", "explosiveness", "attempts", 50.0, ("RB", "QB")),
    Candidate("rush_breakaway_percent", "rushing", "explosiveness", "attempts", 50.0, ("RB", "QB")),
    Candidate("rush_breakaway_yards_per_attempt", "rushing", "explosiveness", "attempts", 50.0, ("RB", "QB")),
    Candidate("rush_gap_share", "rushing", "scheme", "attempts", 50.0, ("RB", "QB")),
    Candidate("rush_zone_share", "rushing", "scheme", "attempts", 50.0, ("RB", "QB")),
    Candidate("rush_run_grade", "rushing", "grade", "attempts", 50.0, ("RB", "QB")),
    Candidate("rush_elusive_rating", "rushing", "tackle_breaking", "attempts", 50.0, ("RB",)),
    Candidate("rush_designed_yards_per_attempt", "rushing", "qb_style", "attempts", 50.0, ("QB",)),
    Candidate("rush_scramble_share", "rushing", "qb_style", "attempts", 50.0, ("QB",)),
)

CANDIDATES = (*REC_CANDIDATES, *RUSH_CANDIDATES)
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}


def _read_sql(database: Path, query: str) -> pd.DataFrame:
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return pd.read_sql_query(query, connection)


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.where(denominator.gt(0))
    return result.replace([np.inf, -np.inf], np.nan)


def _canonical_position(values: pd.Series) -> pd.Series:
    return values.replace({"HB": "RB", "FB": "RB"})


def _normalize_pff_id(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").round().astype("Int64")


def _load_pff() -> tuple[pd.DataFrame, pd.DataFrame]:
    rec = _read_sql(RAW_DB, "SELECT * FROM PFF_Rec_Stats")
    rush = _read_sql(RAW_DB, "SELECT * FROM PFF_Rush_Stats")
    rec["pff_id_num"] = _normalize_pff_id(rec["player_id"])
    rush["pff_id_num"] = _normalize_pff_id(rush["player_id"])
    rec["source_position"] = _canonical_position(rec["position"])
    rush["source_position"] = _canonical_position(rush["position"])
    rec = rec.rename(columns={"year": "source_year"})
    rush = rush.rename(columns={"year": "source_year"})

    rec_numeric = [
        "routes", "targets", "receptions", "yards", "touchdowns",
        "first_downs", "yards_after_catch", "avoided_tackles",
        "avg_depth_of_target", "route_rate", "slot_rate", "wide_rate",
        "grades_pass_route", "targeted_qb_rating",
    ]
    rush_numeric = [
        "attempts", "yards", "first_downs", "avoided_tackles",
        "yards_after_contact", "explosive", "breakaway_percent",
        "breakaway_yards", "gap_attempts", "zone_attempts", "grades_run",
        "elusive_rating", "designed_yards", "scrambles",
    ]
    _numeric(rec, rec_numeric)
    _numeric(rush, rush_numeric)

    rec["rec_yprr"] = _safe_divide(rec["yards"], rec["routes"])
    rec["rec_first_downs_per_route"] = _safe_divide(rec["first_downs"], rec["routes"])
    rec["rec_targets_per_route"] = _safe_divide(rec["targets"], rec["routes"])
    rec["rec_touchdowns_per_route"] = _safe_divide(rec["touchdowns"], rec["routes"])
    rec["rec_yards_per_target"] = _safe_divide(rec["yards"], rec["targets"])
    rec["rec_adot"] = rec["avg_depth_of_target"]
    rec["rec_yac_per_route"] = _safe_divide(rec["yards_after_catch"], rec["routes"])
    rec["rec_mtf_per_reception"] = _safe_divide(rec["avoided_tackles"], rec["receptions"])
    rec["rec_route_rate"] = rec["route_rate"] / 100.0
    rec["rec_slot_rate"] = rec["slot_rate"] / 100.0
    rec["rec_wide_rate"] = rec["wide_rate"] / 100.0
    rec["rec_route_grade"] = rec["grades_pass_route"]
    rec["rec_targeted_qb_rating"] = rec["targeted_qb_rating"]

    rush["rush_ypa"] = _safe_divide(rush["yards"], rush["attempts"])
    rush["rush_first_downs_per_attempt"] = _safe_divide(rush["first_downs"], rush["attempts"])
    rush["rush_mtf_per_attempt"] = _safe_divide(rush["avoided_tackles"], rush["attempts"])
    rush["rush_yco_per_attempt"] = _safe_divide(rush["yards_after_contact"], rush["attempts"])
    rush["rush_explosive_per_attempt"] = _safe_divide(rush["explosive"], rush["attempts"])
    rush["rush_breakaway_percent"] = rush["breakaway_percent"] / 100.0
    rush["rush_breakaway_yards_per_attempt"] = _safe_divide(rush["breakaway_yards"], rush["attempts"])
    rush["rush_gap_share"] = _safe_divide(rush["gap_attempts"], rush["attempts"])
    rush["rush_zone_share"] = _safe_divide(rush["zone_attempts"], rush["attempts"])
    rush["rush_run_grade"] = rush["grades_run"]
    rush["rush_elusive_rating"] = rush["elusive_rating"]
    rush["rush_designed_yards_per_attempt"] = _safe_divide(rush["designed_yards"], rush["attempts"])
    rush["rush_scramble_share"] = _safe_divide(rush["scrambles"], rush["attempts"])

    if rec.duplicated(["pff_id_num", "source_year"]).any():
        raise ValueError("PFF receiving rows are not unique by ID-season")
    if rush.duplicated(["pff_id_num", "source_year"]).any():
        raise ValueError("PFF rushing rows are not unique by ID-season")
    return rec, rush


def _load_league(league: str, database: Path) -> pd.DataFrame:
    features = _read_sql(database, "SELECT * FROM player_season_features")
    identity = _read_sql(
        database,
        "SELECT player_key, pff_id FROM player_identity",
    )
    predictions = _read_sql(
        database,
        """
        SELECT player_key, season, position, prediction AS base_prediction,
               actual
        FROM locked_whole_season_predictions
        WHERE method = 'conditional_ppg_primary_blend'
          AND target_name = 'conditional_ppg'
        """,
    )
    identity["pff_id_num"] = _normalize_pff_id(identity["pff_id"])
    if identity["player_key"].duplicated().any():
        raise ValueError(f"{league}: duplicate player_key in identity")
    frame = predictions.merge(
        features,
        on=["player_key", "season", "position"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    ).merge(
        identity[["player_key", "pff_id_num"]],
        on="player_key",
        how="left",
        validate="many_to_one",
    )
    frame = frame[
        frame["season"].between(VALIDATION_START, VALIDATION_END)
        & frame["position"].isin(["QB", "RB", "WR", "TE"])
        & frame["conditional_ppg_training_eligible"].eq(1)
    ].copy()
    frame["league"] = league
    frame["source_year"] = frame["season"] - 1
    frame["production_residual"] = frame["actual"] - frame["base_prediction"]
    frame["adp_log"] = np.log1p(pd.to_numeric(frame["adp_median"], errors="coerce"))
    frame["q90_threshold"] = frame.groupby(["season", "position"])["actual"].transform(
        lambda values: values.quantile(0.90)
    )
    frame["q90_event"] = frame["actual"].ge(frame["q90_threshold"]).astype(int)
    missing_primary = [column for column in PRIMARY_PPG_FEATURES if column not in frame]
    position_only = [column for column in missing_primary if column.startswith("position_")]
    if set(missing_primary) != set(position_only):
        raise ValueError(f"{league}: missing locked features: {missing_primary}")
    return frame


def _weighted_group_summary(
    source: pd.DataFrame,
    candidate: Candidate,
) -> pd.DataFrame:
    work = source[
        source["source_position"].isin(candidate.positions)
        & source[candidate.name].notna()
        & source[candidate.denominator].gt(0)
    ][
        ["source_year", "source_position", candidate.name, candidate.denominator]
    ].copy()

    def summarize(group: pd.DataFrame) -> pd.Series:
        raw = group[candidate.name].astype(float)
        weight = group[candidate.denominator].astype(float)
        low = float(raw.quantile(0.01))
        high = float(raw.quantile(0.99))
        clipped = raw.clip(low, high)
        neutral = float(np.average(clipped, weights=weight))
        return pd.Series({"neutral": neutral, "clip_low": low, "clip_high": high})

    return (
        work.groupby(["source_year", "source_position"], as_index=False)
        .apply(summarize, include_groups=False)
        .reset_index(drop=True)
        .rename(columns={"source_position": "position"})
    )


def _attach_pff(
    frame: pd.DataFrame,
    rec: pd.DataFrame,
    rush: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    for domain, source, candidates, opportunity in (
        ("receiving", rec, REC_CANDIDATES, "routes"),
        ("rushing", rush, RUSH_CANDIDATES, "attempts"),
    ):
        selected = ["pff_id_num", "source_year", opportunity]
        for candidate in candidates:
            selected.extend([candidate.name, candidate.denominator])
        selected = list(dict.fromkeys(selected))
        renamed = {
            opportunity: f"{domain}__opportunity",
            **{
                column: f"{domain}__source__{column}"
                for column in selected
                if column not in {"pff_id_num", "source_year", opportunity}
            },
        }
        result = result.merge(
            source[selected].rename(columns=renamed),
            on=["pff_id_num", "source_year"],
            how="left",
            validate="many_to_one",
        )
        result[f"{domain}__available"] = result[f"{domain}__opportunity"].gt(0).astype(float)
        result[f"{domain}__log_opportunity"] = np.log1p(
            result[f"{domain}__opportunity"].fillna(0).clip(lower=0)
        )
        for candidate in candidates:
            summary = _weighted_group_summary(source, candidate).rename(
                columns={
                    "neutral": f"{candidate.name}__neutral",
                    "clip_low": f"{candidate.name}__clip_low",
                    "clip_high": f"{candidate.name}__clip_high",
                }
            )
            result = result.merge(
                summary,
                on=["source_year", "position"],
                how="left",
                validate="many_to_one",
            )
            raw_column = f"{domain}__source__{candidate.name}"
            denominator_column = (
                f"{domain}__opportunity"
                if candidate.denominator == opportunity
                else f"{domain}__source__{candidate.denominator}"
            )
            raw = pd.to_numeric(result[raw_column], errors="coerce")
            sample_size = pd.to_numeric(result[denominator_column], errors="coerce").fillna(0).clip(lower=0)
            neutral = result[f"{candidate.name}__neutral"]
            clipped = raw.clip(
                lower=result[f"{candidate.name}__clip_low"],
                upper=result[f"{candidate.name}__clip_high"],
            )
            reliability = sample_size / (sample_size + candidate.prior_strength)
            result[f"{candidate.name}__raw"] = raw
            result[f"{candidate.name}__sample_size"] = sample_size
            result[f"{candidate.name}__log_sample_size"] = np.log1p(sample_size)
            result[f"{candidate.name}__reliability"] = reliability
            result[f"{candidate.name}__value"] = neutral + reliability * (clipped - neutral)
            result[f"{candidate.name}__value"] = result[f"{candidate.name}__value"].fillna(neutral)
            if result.loc[result["position"].isin(candidate.positions), f"{candidate.name}__value"].isna().any():
                raise ValueError(f"Missing neutral value for {candidate.name}")
    return result


def _coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain, positions, thresholds in (
        ("receiving", ("RB", "WR", "TE"), (1, 50, 100, 200)),
        ("rushing", ("RB", "QB"), (1, 25, 50, 100)),
    ):
        opportunity = frame[f"{domain}__opportunity"]
        for position in positions:
            selected = frame["position"].eq(position)
            record: dict[str, object] = {
                "domain": domain,
                "position": position,
                "eligible_rows": int(selected.sum()),
                "identity_coverage": float(frame.loc[selected, "pff_id_num"].notna().mean()),
                "source_row_coverage": float(opportunity[selected].notna().mean()),
                "positive_opportunity_coverage": float(opportunity[selected].gt(0).mean()),
                "median_positive_opportunity": float(opportunity[selected & opportunity.gt(0)].median()),
            }
            for threshold in thresholds:
                record[f"coverage_ge_{threshold}"] = float(opportunity[selected].ge(threshold).mean())
            rows.append(record)
    return pd.DataFrame(rows)


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    valid = x.notna() & y.notna()
    n = int(valid.sum())
    if n < 10 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan, np.nan, n
    result = stats.spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue), n


def _partial_rank_correlation(
    x: pd.Series,
    y: pd.Series,
    controls: pd.DataFrame,
) -> tuple[float, float, int]:
    joined = pd.concat([x.rename("x"), y.rename("y"), controls], axis=1).dropna()
    n = len(joined)
    if n < 15 or joined["x"].nunique() < 2 or joined["y"].nunique() < 2:
        return np.nan, np.nan, n
    ranked = joined.rank(method="average")
    design = np.column_stack([np.ones(n), ranked[controls.columns].to_numpy(float)])
    x_residual = ranked["x"].to_numpy(float) - design @ np.linalg.lstsq(design, ranked["x"].to_numpy(float), rcond=None)[0]
    y_residual = ranked["y"].to_numpy(float) - design @ np.linalg.lstsq(design, ranked["y"].to_numpy(float), rcond=None)[0]
    result = stats.pearsonr(x_residual, y_residual)
    return float(result.statistic), float(result.pvalue), n


def _bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    count = len(valid)
    if count == 0:
        return result
    adjusted = valid.to_numpy(float) * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1.0)
    result.loc[valid.index] = adjusted
    return result


def _residual_associations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        domain = candidate.domain
        threshold = REC_OPPORTUNITY_THRESHOLD if domain == "receiving" else RUSH_OPPORTUNITY_THRESHOLD
        for position in candidate.positions:
            selected = frame[
                frame["position"].eq(position)
                & frame[f"{domain}__opportunity"].ge(threshold)
            ]
            corr, pvalue, n = _spearman(
                selected[f"{candidate.name}__raw"],
                selected["production_residual"],
            )
            partial, partial_p, partial_n = _partial_rank_correlation(
                selected[f"{candidate.name}__raw"],
                selected["production_residual"],
                selected[[f"{domain}__log_opportunity", f"{candidate.name}__log_sample_size"]],
            )
            rows.append(
                {
                    "league": selected["league"].iloc[0] if not selected.empty else frame["league"].iloc[0],
                    "domain": domain,
                    "family": candidate.family,
                    "position": position,
                    "candidate": candidate.name,
                    "rows": n,
                    "residual_spearman": corr,
                    "residual_pvalue": pvalue,
                    "volume_adjusted_rank_corr": partial,
                    "volume_adjusted_pvalue": partial_p,
                    "volume_adjusted_rows": partial_n,
                }
            )
    output = pd.DataFrame(rows)
    output["residual_qvalue"] = output.groupby(["league", "position"])["residual_pvalue"].transform(_bh_adjust)
    output["volume_adjusted_qvalue"] = output.groupby(["league", "position"])["volume_adjusted_pvalue"].transform(_bh_adjust)
    return output


def _persistence(
    rec: pd.DataFrame,
    rush: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        source = rec if candidate.domain == "receiving" else rush
        threshold = REC_OPPORTUNITY_THRESHOLD if candidate.domain == "receiving" else RUSH_OPPORTUNITY_THRESHOLD
        base = source[
            source["source_position"].isin(candidate.positions)
            & source[candidate.name].notna()
            & source[("routes" if candidate.domain == "receiving" else "attempts")].ge(threshold)
        ][["pff_id_num", "source_year", "source_position", candidate.name]].copy()
        future = base.rename(
            columns={
                "source_year": "future_year",
                "source_position": "future_position",
                candidate.name: "future_value",
            }
        )
        joined = base.merge(future, on="pff_id_num", how="inner")
        joined = joined[
            joined["future_year"].eq(joined["source_year"] + 1)
            & joined["future_position"].eq(joined["source_position"])
        ]
        for position in candidate.positions:
            selected = joined[joined["source_position"].eq(position)]
            corr, pvalue, n = _spearman(selected[candidate.name], selected["future_value"])
            rows.append(
                {
                    "domain": candidate.domain,
                    "family": candidate.family,
                    "position": position,
                    "candidate": candidate.name,
                    "opportunity_threshold_both_years": threshold,
                    "consecutive_pairs": n,
                    "year_to_year_spearman": corr,
                    "pvalue": pvalue,
                }
            )
    return pd.DataFrame(rows)


def _redundancy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing = [
        "base_prediction", "expert_ppg_team_game_median", "adp_log",
        "prior_year_ppg", "prior_3year_weighted_ppg", "career_weighted_ppg",
        "proj_targets", "projected_receiving_point_share",
        "proj_rush_attempts", "projected_rush_point_share",
    ]
    baseline_rows = []
    pair_rows = []
    for domain, candidates in (("receiving", REC_CANDIDATES), ("rushing", RUSH_CANDIDATES)):
        threshold = REC_OPPORTUNITY_THRESHOLD if domain == "receiving" else RUSH_OPPORTUNITY_THRESHOLD
        positions = sorted({position for candidate in candidates for position in candidate.positions})
        for position in positions:
            selected = frame[
                frame["position"].eq(position)
                & frame[f"{domain}__opportunity"].ge(threshold)
            ]
            applicable = [candidate for candidate in candidates if position in candidate.positions]
            for candidate in applicable:
                for feature in existing:
                    corr, _, n = _spearman(selected[f"{candidate.name}__raw"], selected[feature])
                    baseline_rows.append(
                        {
                            "domain": domain,
                            "position": position,
                            "candidate": candidate.name,
                            "existing_feature": feature,
                            "rows": n,
                            "spearman": corr,
                        }
                    )
            for index, left in enumerate(applicable):
                for right in applicable[index + 1 :]:
                    corr, _, n = _spearman(selected[f"{left.name}__raw"], selected[f"{right.name}__raw"])
                    pair_rows.append(
                        {
                            "domain": domain,
                            "position": position,
                            "left": left.name,
                            "right": right.name,
                            "rows": n,
                            "spearman": corr,
                        }
                    )
    return pd.DataFrame(baseline_rows), pd.DataFrame(pair_rows)


def _fit_ridge(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )
    model.fit(train[columns], train["production_residual"])
    return model.predict(test[columns])


def _fit_logistic(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=LOGISTIC_C, max_iter=2_000, random_state=RANDOM_SEED),
    )
    model.fit(train[columns], train["q90_event"])
    return model.predict_proba(test[columns])[:, 1]


def _rolling_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ppg_outputs = []
    upside_outputs = []
    for candidate in CANDIDATES:
        domain = candidate.domain
        for position in candidate.positions:
            subset = frame[frame["position"].eq(position)].copy()
            control_columns = [
                f"{domain}__available",
                f"{domain}__log_opportunity",
                f"{candidate.name}__log_sample_size",
            ]
            challenger_columns = [*control_columns, f"{candidate.name}__value"]
            for season in range(MODEL_START, VALIDATION_END + 1):
                train = subset[subset["season"].lt(season)]
                test = subset[subset["season"].eq(season)]
                if test.empty or len(train) < 25:
                    continue
                control_correction = _fit_ridge(train, test, control_columns)
                candidate_correction = _fit_ridge(train, test, challenger_columns)
                ppg = test[[
                    "league", "player_key", "season", "position", "actual",
                    "base_prediction", "q90_event",
                ]].copy()
                ppg["domain"] = domain
                ppg["candidate"] = candidate.name
                ppg["control_prediction"] = ppg["base_prediction"] + control_correction
                ppg["candidate_prediction"] = ppg["base_prediction"] + candidate_correction
                ppg_outputs.append(ppg)

                base_columns = ["base_prediction"]
                control_upside_columns = [*base_columns, *control_columns]
                candidate_upside_columns = [*base_columns, *challenger_columns]
                if train["q90_event"].nunique() == 2:
                    upside = test[[
                        "league", "player_key", "season", "position",
                        "q90_event", "actual", "base_prediction",
                    ]].copy()
                    upside["domain"] = domain
                    upside["candidate"] = candidate.name
                    upside["base_probability"] = _fit_logistic(train, test, base_columns)
                    upside["control_probability"] = _fit_logistic(train, test, control_upside_columns)
                    upside["candidate_probability"] = _fit_logistic(train, test, candidate_upside_columns)
                    upside_outputs.append(upside)
    return pd.concat(ppg_outputs, ignore_index=True), pd.concat(upside_outputs, ignore_index=True)


def _rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def _period_label(season: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(season.le(DEVELOPMENT_END), "development_2018_2022", "temporal_2023_2025"),
        index=season.index,
    )


def _bootstrap_season_mean(values: pd.DataFrame, value_column: str, seed: int) -> tuple[float, float]:
    seasons = values["season"].drop_duplicates().to_numpy()
    per_season = values.groupby("season")[value_column].mean()
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for index in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(seasons, size=len(seasons), replace=True)
        draws[index] = float(per_season.loc[sampled].mean())
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _summarize_ppg(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["period"] = _period_label(predictions["season"])
    rows = []
    keys = ["league", "domain", "position", "candidate"]
    for key, group in predictions.groupby(keys, sort=True):
        production_rmse = _rmse(group["actual"], group["base_prediction"])
        control_rmse = _rmse(group["actual"], group["control_prediction"])
        candidate_rmse = _rmse(group["actual"], group["candidate_prediction"])
        season_delta = (
            group.assign(
                control_sq=np.square(group["actual"] - group["control_prediction"]),
                candidate_sq=np.square(group["actual"] - group["candidate_prediction"]),
            )
            .groupby("season")
            .apply(
                lambda values: pd.Series(
                    {
                        "delta": _rmse(values["actual"], values["candidate_prediction"])
                        - _rmse(values["actual"], values["control_prediction"])
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        low, high = _bootstrap_season_mean(season_delta, "delta", RANDOM_SEED + len(rows))
        record = dict(zip(keys, key))
        record.update(
            {
                "rows": len(group),
                "seasons": group["season"].nunique(),
                "production_rmse": production_rmse,
                "opportunity_control_rmse": control_rmse,
                "candidate_rmse": candidate_rmse,
                "candidate_delta_vs_production": candidate_rmse - production_rmse,
                "candidate_delta_vs_opportunity_control": candidate_rmse - control_rmse,
                "mean_season_delta_vs_control": float(season_delta["delta"].mean()),
                "season_wins_vs_control": int(season_delta["delta"].lt(0).sum()),
                "season_bootstrap_low": low,
                "season_bootstrap_high": high,
            }
        )
        for period, selected in group.groupby("period"):
            record[f"{period}_delta_vs_control"] = _rmse(selected["actual"], selected["candidate_prediction"]) - _rmse(selected["actual"], selected["control_prediction"])
        tail = group[group["q90_event"].eq(1)]
        record["q90_rows"] = len(tail)
        record["q90_rmse_delta_vs_control"] = _rmse(tail["actual"], tail["candidate_prediction"]) - _rmse(tail["actual"], tail["control_prediction"])
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["league", "position", "candidate_delta_vs_opportunity_control"])


def _summarize_upside(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["period"] = _period_label(predictions["season"])
    rows = []
    keys = ["league", "domain", "position", "candidate"]
    for key, group in predictions.groupby(keys, sort=True):
        event = group["q90_event"].astype(int)
        base_brier = float(np.mean(np.square(event - group["base_probability"])))
        control_brier = float(np.mean(np.square(event - group["control_probability"])))
        candidate_brier = float(np.mean(np.square(event - group["candidate_probability"])))
        working = group.assign(
            brier_delta=np.square(event - group["candidate_probability"])
            - np.square(event - group["control_probability"])
        )
        low, high = _bootstrap_season_mean(working, "brier_delta", RANDOM_SEED + 10_000 + len(rows))
        record = dict(zip(keys, key))
        record.update(
            {
                "rows": len(group),
                "events": int(event.sum()),
                "base_brier": base_brier,
                "opportunity_control_brier": control_brier,
                "candidate_brier": candidate_brier,
                "candidate_brier_delta_vs_base": candidate_brier - base_brier,
                "candidate_brier_delta_vs_opportunity_control": candidate_brier - control_brier,
                "candidate_brier_bootstrap_low": low,
                "candidate_brier_bootstrap_high": high,
                "base_average_precision": float(average_precision_score(event, group["base_probability"])),
                "control_average_precision": float(average_precision_score(event, group["control_probability"])),
                "candidate_average_precision": float(average_precision_score(event, group["candidate_probability"])),
            }
        )
        record["candidate_ap_delta_vs_control"] = record["candidate_average_precision"] - record["control_average_precision"]
        for period, selected in group.groupby("period"):
            selected_event = selected["q90_event"].astype(int)
            record[f"{period}_brier_delta_vs_control"] = float(
                np.mean(np.square(selected_event - selected["candidate_probability"]))
                - np.mean(np.square(selected_event - selected["control_probability"]))
            )
            record[f"{period}_ap_delta_vs_control"] = float(
                average_precision_score(selected_event, selected["candidate_probability"])
                - average_precision_score(selected_event, selected["control_probability"])
            )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["league", "position", "candidate_brier_delta_vs_opportunity_control"])


def _season_diagnostics(
    ppg_predictions: pd.DataFrame,
    upside_predictions: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["league", "domain", "position", "candidate", "season"]
    ppg_rows = []
    for key, group in ppg_predictions.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        record.update(
            {
                "rows": len(group),
                "production_rmse": _rmse(group["actual"], group["base_prediction"]),
                "opportunity_control_rmse": _rmse(group["actual"], group["control_prediction"]),
                "candidate_rmse": _rmse(group["actual"], group["candidate_prediction"]),
            }
        )
        record["ppg_delta_vs_control"] = (
            record["candidate_rmse"] - record["opportunity_control_rmse"]
        )
        ppg_rows.append(record)
    output = pd.DataFrame(ppg_rows)

    upside_rows = []
    for key, group in upside_predictions.groupby(keys, sort=True):
        event = group["q90_event"].astype(int)
        control_brier = float(
            np.mean(np.square(event - group["control_probability"]))
        )
        candidate_brier = float(
            np.mean(np.square(event - group["candidate_probability"]))
        )
        record = dict(zip(keys, key))
        record.update(
            {
                "q90_events": int(event.sum()),
                "opportunity_control_brier": control_brier,
                "candidate_brier": candidate_brier,
                "q90_brier_delta_vs_control": candidate_brier - control_brier,
                "control_average_precision": float(
                    average_precision_score(event, group["control_probability"])
                ),
                "candidate_average_precision": float(
                    average_precision_score(event, group["candidate_probability"])
                ),
            }
        )
        record["q90_ap_delta_vs_control"] = (
            record["candidate_average_precision"]
            - record["control_average_precision"]
        )
        upside_rows.append(record)
    return output.merge(
        pd.DataFrame(upside_rows),
        on=keys,
        how="outer",
        validate="one_to_one",
    ).sort_values(keys)


def _candidate_summary(
    persistence: pd.DataFrame,
    redundancy: pd.DataFrame,
    associations: pd.DataFrame,
    ppg: pd.DataFrame,
    upside: pd.DataFrame,
) -> pd.DataFrame:
    persist = persistence[["domain", "position", "candidate", "consecutive_pairs", "year_to_year_spearman"]]
    redundant = (
        redundancy.assign(abs_spearman=redundancy["spearman"].abs())
        .sort_values("abs_spearman", ascending=False)
        .groupby(["domain", "position", "candidate"], as_index=False)
        .first()[["domain", "position", "candidate", "existing_feature", "spearman"]]
        .rename(columns={"existing_feature": "most_redundant_existing_feature", "spearman": "max_existing_spearman"})
    )
    output = ppg.merge(upside, on=["league", "domain", "position", "candidate"], how="left", suffixes=("_ppg", "_upside"))
    output = output.merge(persist, on=["domain", "position", "candidate"], how="left")
    output = output.merge(redundant, on=["domain", "position", "candidate"], how="left")
    output = output.merge(
        associations[[
            "league", "domain", "position", "candidate", "residual_spearman",
            "residual_qvalue", "volume_adjusted_rank_corr", "volume_adjusted_qvalue",
        ]],
        on=["league", "domain", "position", "candidate"],
        how="left",
    )
    return output.sort_values(["league", "candidate_delta_vs_opportunity_control"])


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 12) -> list[str]:
    shown = frame.head(limit)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(shown[column]) else "---" for column in columns) + "|"
    lines = [header, separator]
    for row in shown[columns].itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def _write_findings(
    coverage: pd.DataFrame,
    pairs: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    dk = summary[summary["league"].eq("dk")].copy()
    ppg_top = dk.sort_values("candidate_delta_vs_opportunity_control")
    upside_top = dk.sort_values("candidate_brier_delta_vs_opportunity_control")
    significant_ppg = dk[dk["season_bootstrap_high"].lt(0)]
    significant_upside = dk[dk["candidate_brier_bootstrap_high"].lt(0)]
    joint_clear = dk[
        dk["season_bootstrap_high"].lt(0)
        & dk["candidate_brier_bootstrap_high"].lt(0)
    ].sort_values("candidate_delta_vs_opportunity_control")
    replicated = dk.merge(
        summary[summary["league"].eq("beta")][[
            "domain", "position", "candidate", "candidate_delta_vs_opportunity_control",
            "candidate_brier_delta_vs_opportunity_control",
        ]],
        on=["domain", "position", "candidate"],
        suffixes=("_dk", "_beta"),
    )
    replicated = replicated[
        replicated["candidate_delta_vs_opportunity_control_dk"].lt(0)
        & replicated["candidate_delta_vs_opportunity_control_beta"].lt(0)
        & replicated["candidate_brier_delta_vs_opportunity_control_dk"].lt(0)
        & replicated["candidate_brier_delta_vs_opportunity_control_beta"].lt(0)
    ].sort_values("candidate_delta_vs_opportunity_control_dk")

    high_pairs = pairs[pairs["spearman"].abs().ge(0.75)].copy()
    if joint_clear.empty:
        decision_lines = [
            "No candidate cleared both DK season-bootstrap screens. Retain all "
            "PFF metrics as research-only diagnostics.",
        ]
    else:
        leader = joint_clear.iloc[0]
        beta_leader = summary[
            summary["league"].eq("beta")
            & summary["position"].eq(leader["position"])
            & summary["candidate"].eq(leader["candidate"])
        ].iloc[0]
        decision_lines = [
            f"Advance `{leader['position']} {leader['candidate']}` to a full locked-model "
            "and template-matcher test, but do not promote it from this screen. It is the "
            "only DK row whose season-bootstrap intervals clear zero for both point PPG "
            "and q90 Brier.",
            f"Its DK RMSE delta versus the opportunity control is "
            f"{leader['candidate_delta_vs_opportunity_control']:+.4f}, including "
            f"{leader['temporal_2023_2025_delta_vs_control']:+.4f} in 2023-2025; "
            f"its q90 Brier delta is "
            f"{leader['candidate_brier_delta_vs_opportunity_control']:+.4f}, including "
            f"{leader['temporal_2023_2025_brier_delta_vs_control']:+.4f} recently. "
            f"Beta agrees directionally (PPG "
            f"{beta_leader['candidate_delta_vs_opportunity_control']:+.4f}; q90 Brier "
            f"{beta_leader['candidate_brier_delta_vs_opportunity_control']:+.4f}).",
            "Because this was a broad screen, the interval does not correct for selecting "
            "the best of many candidates. Treat it as a prespecified follow-up candidate, "
            "not confirmatory evidence.",
        ]
    lines = [
        "# PFF advanced-stat screen findings",
        "",
        "## Read this first",
        "",
        "This is a strictly prior-season predictive screen, not a causal-effect estimate. "
        "The baseline is the locked production PPG forecast, which already contains expert "
        "projection, ADP, experience, historical production, projected role, room context, "
        "and projection trajectory. Negative deltas favor the PFF rate challenger over a "
        "prior-PFF-opportunity-only control.",
        "",
        "## Screening decision",
        "",
        *decision_lines,
        "",
        f"DK rows with a season-bootstrap PPG interval entirely below zero: **{len(significant_ppg)}**.",
        f"DK rows with a season-bootstrap q90 Brier interval entirely below zero: **{len(significant_upside)}**.",
        f"Rows directionally improving both PPG and q90 Brier in both DK and beta: **{len(replicated)}**.",
        "",
        "## Best DK PPG screens",
        "",
        *_markdown_table(
            ppg_top,
            [
                "position", "candidate", "candidate_delta_vs_opportunity_control",
                "temporal_2023_2025_delta_vs_control", "season_bootstrap_low",
                "season_bootstrap_high", "year_to_year_spearman",
                "max_existing_spearman",
            ],
        ),
        "",
        "## Best DK q90-upside screens",
        "",
        *_markdown_table(
            upside_top,
            [
                "position", "candidate", "candidate_brier_delta_vs_opportunity_control",
                "temporal_2023_2025_brier_delta_vs_control",
                "candidate_ap_delta_vs_control", "candidate_brier_bootstrap_low",
                "candidate_brier_bootstrap_high",
            ],
        ),
        "",
        "## Cross-scoring directional replication",
        "",
    ]
    if replicated.empty:
        lines.append("No position-metric row improved both point PPG and q90 Brier in both scoring systems.")
    else:
        lines.extend(
            _markdown_table(
                replicated,
                [
                    "position", "candidate", "candidate_delta_vs_opportunity_control_dk",
                    "candidate_delta_vs_opportunity_control_beta",
                    "candidate_brier_delta_vs_opportunity_control_dk",
                    "candidate_brier_delta_vs_opportunity_control_beta",
                ],
            )
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            *_markdown_table(
                coverage,
                [
                    "domain", "position", "eligible_rows", "identity_coverage",
                    "positive_opportunity_coverage", "median_positive_opportunity",
                ],
                limit=len(coverage),
            ),
            "",
            "## High within-PFF redundancy",
            "",
            f"There are {len(high_pairs)} position-specific candidate pairs with |Spearman| >= 0.75. "
            "See `pff_pair_correlations.csv` for the full matrix. Highly correlated rates "
            "should enter later model/template tests as alternative representatives, not as a bundle.",
            "",
            "## Promotion rule",
            "",
            "This screen does not change production. A candidate should advance only if it has "
            "reasonable coverage and persistence, adds directionally consistent PPG or upside "
            "value beyond the opportunity control, survives the 2023-2025 slice, and is not "
            "merely a duplicate of an existing projection/history feature. Advanced candidates "
            "then need the full locked model or template validation rather than direct promotion.",
            "",
        ]
    )
    (RESULTS_DIR / "findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rec, rush = _load_pff()
    league_frames = {
        league: _attach_pff(_load_league(league, database), rec, rush)
        for league, database in LEAGUE_DBS.items()
    }
    primary = league_frames["dk"]
    coverage = _coverage(primary)
    persistence = _persistence(rec, rush)
    redundancy, pairs = _redundancy(primary)
    associations = pd.concat(
        [_residual_associations(frame) for frame in league_frames.values()],
        ignore_index=True,
    )
    ppg_predictions = []
    upside_predictions = []
    for frame in league_frames.values():
        ppg, upside = _rolling_predictions(frame)
        ppg_predictions.append(ppg)
        upside_predictions.append(upside)
    ppg_predictions = pd.concat(ppg_predictions, ignore_index=True)
    upside_predictions = pd.concat(upside_predictions, ignore_index=True)
    ppg_results = _summarize_ppg(ppg_predictions)
    upside_results = _summarize_upside(upside_predictions)
    season_diagnostics = _season_diagnostics(
        ppg_predictions,
        upside_predictions,
    )
    summary = _candidate_summary(
        persistence,
        redundancy,
        associations,
        ppg_results,
        upside_results,
    )

    definitions = pd.DataFrame([asdict(candidate) for candidate in CANDIDATES])
    definitions["positions"] = definitions["positions"].map("/".join)
    definitions.to_csv(RESULTS_DIR / "candidate_definitions.csv", index=False)
    coverage.to_csv(RESULTS_DIR / "coverage.csv", index=False)
    persistence.to_csv(RESULTS_DIR / "persistence.csv", index=False)
    redundancy.to_csv(RESULTS_DIR / "baseline_redundancy.csv", index=False)
    pairs.to_csv(RESULTS_DIR / "pff_pair_correlations.csv", index=False)
    associations.to_csv(RESULTS_DIR / "residual_associations.csv", index=False)
    ppg_results.to_csv(RESULTS_DIR / "ppg_model_results.csv", index=False)
    upside_results.to_csv(RESULTS_DIR / "upside_model_results.csv", index=False)
    season_diagnostics.to_csv(RESULTS_DIR / "season_diagnostics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "candidate_summary.csv", index=False)
    _write_findings(coverage, pairs, summary)

    manifest = {
        "validation_seasons": [VALIDATION_START, VALIDATION_END],
        "rolling_test_seasons": [MODEL_START, VALIDATION_END],
        "development_end": DEVELOPMENT_END,
        "primary_league": "dk",
        "replication_league": "beta",
        "baseline": "locked conditional_ppg_primary_blend",
        "baseline_feature_count": len(PRIMARY_PPG_FEATURES),
        "baseline_includes_expert_projection": "expert_ppg_team_game_median" in PRIMARY_PPG_FEATURES,
        "baseline_includes_adp": "adp_median" in PRIMARY_PPG_FEATURES,
        "ridge_alpha": RIDGE_ALPHA,
        "logistic_c": LOGISTIC_C,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "candidate_count": len(CANDIDATES),
        "target_rows": {league: len(frame) for league, frame in league_frames.items()},
    }
    (RESULTS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print((RESULTS_DIR / "findings.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
