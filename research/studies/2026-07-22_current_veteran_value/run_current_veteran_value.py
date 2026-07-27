"""Evaluate current-season veteran value conditional on projection and market."""

from __future__ import annotations

import importlib.util
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
RESULTS_DIR = STUDY_DIR / "results"
SIMULATION_DB = ROOT / "Data" / "Databases" / "Simulation.sqlite3"
VALIDATIONS_DB = ROOT / "Data" / "Databases" / "Validations.sqlite3"
VETERAN_STUDY = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-21_veteran_cliff_calibration"
    / "run_veteran_cliff_calibration.py"
)

POSITIONS = ("RB", "WR", "TE")
THRESHOLDS = {"RB": 7, "WR": 9, "TE": 8}
WAIVER_PPG = {"RB": 7.0, "WR": 7.0, "TE": 5.0}
WEEKS = tuple(range(1, 17))
MATCH_K = 5
MATCH_MAX_DISTANCE = 1.5
BOOTSTRAPS = 2000
RANDOM_SEED = 20260722

BINARY_OUTCOMES = {
    "managed_miss_30": "managed miss rate",
    "managed_upside_20": "managed upside-hit rate",
}
CONTINUOUS_OUTCOMES = {
    "managed_points_above_waiver": "managed points above waiver",
    "managed_residual_points": "managed points versus forecast",
    "boom_weeks": "1.5x-projection boom weeks",
}


def read_sql(path: Path, query: str) -> pd.DataFrame:
    uri = f"file:{path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        return pd.read_sql_query(query, conn)


def load_veteran_module():
    spec = importlib.util.spec_from_file_location("veteran_cliff_study", VETERAN_STUDY)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {VETERAN_STUDY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_weekly_details() -> pd.DataFrame:
    weekly_cols = ",\n               ".join(
        f"managed_week_{week}" for week in WEEKS
    )
    return read_sql(
        SIMULATION_DB,
        f"""
        SELECT player,
               pos,
               CAST(season AS INTEGER) season,
               avg_pick,
               active_ppg,
               season_points,
               {weekly_cols}
        FROM Best_Ball_Weekly_Templates
        WHERE league='beta'
              AND pos IN ('RB', 'WR', 'TE')
        """,
    )


def load_auction_market() -> pd.DataFrame:
    return read_sql(
        VALIDATIONS_DB,
        """
        SELECT player,
               pos,
               CAST(year AS INTEGER) season,
               pred_salary,
               pred_salary_raw,
               actual_salary,
               actual_salary_observed,
               training_through_year,
               data_rolling_origin,
               normalization_uses_target_actuals,
               method_version
        FROM Salary_Backtest_Predictions
        WHERE league='beta'
              AND method_version='current_locked_spec_v5_compact_salary_features'
        """,
    )


def build_analysis_rows() -> tuple[pd.DataFrame, object]:
    veteran = load_veteran_module()
    base, actuals = veteran.build_analysis_rows()
    base = veteran.add_threshold_fields(base, THRESHOLDS)
    weekly = load_weekly_details()
    rows = base.merge(
        weekly,
        on=["player", "pos", "season"],
        how="inner",
        validate="one_to_one",
    )
    weekly_columns = [f"managed_week_{week}" for week in WEEKS]
    normalized = rows[weekly_columns].to_numpy(dtype=float)
    actual_weekly = normalized * rows.active_ppg.to_numpy(dtype=float)[:, None]
    baselines = rows.pos.map(WAIVER_PPG).to_numpy(dtype=float)
    rows["managed_points_above_waiver"] = np.maximum(
        actual_weekly - baselines[:, None], 0.0
    ).sum(axis=1)
    rows["forecast_managed_points"] = 16.0 * np.maximum(
        rows.pred_ppg.to_numpy(dtype=float) - baselines, 0.0
    )
    rows["managed_residual_points"] = (
        rows.managed_points_above_waiver - rows.forecast_managed_points
    )
    rows["managed_miss_30"] = (
        rows.managed_points_above_waiver
        <= 0.70 * rows.forecast_managed_points
    )
    rows["managed_upside_20"] = (
        rows.managed_points_above_waiver
        >= 1.20 * rows.forecast_managed_points
    )
    rows["boom_weeks"] = (
        actual_weekly >= 1.50 * rows.pred_ppg.to_numpy(dtype=float)[:, None]
    ).sum(axis=1)
    rows["above_threshold_value"] = rows.above_threshold.astype(int)

    auction = load_auction_market()
    auction_rows = rows.merge(
        auction,
        on=["player", "pos", "season"],
        how="inner",
        validate="one_to_one",
    )
    if auction_rows.data_rolling_origin.ne(1).any():
        raise ValueError("Auction market contains a non-rolling row")
    if auction_rows.normalization_uses_target_actuals.ne(0).any():
        raise ValueError("Auction market normalization uses target actuals")
    return rows, auction_rows, veteran, actuals


def dataset_views(
    long_rows: pd.DataFrame,
    auction_rows: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, str, str]]:
    return {
        "adp_top_100": (
            long_rows[long_rows.avg_pick.le(100)].copy(),
            "avg_pick",
            "adp",
        ),
        "adp_top_200": (
            long_rows[long_rows.avg_pick.le(200)].copy(),
            "avg_pick",
            "adp",
        ),
        "auction_all": (auction_rows.copy(), "pred_salary", "salary"),
        "auction_salary_5plus": (
            auction_rows[auction_rows.pred_salary.ge(5)].copy(),
            "pred_salary",
            "salary",
        ),
    }


def add_market_strength(
    rows: pd.DataFrame,
    market_column: str,
    market_kind: str,
) -> pd.DataFrame:
    output = rows.copy()
    if market_kind == "adp":
        output["market_strength"] = -np.log(output[market_column].clip(lower=1e-6))
    else:
        output["market_strength"] = np.log1p(output[market_column].clip(lower=0))
    return output


def cohort_summary(
    views: dict[str, tuple[pd.DataFrame, str, str]],
) -> pd.DataFrame:
    frames = []
    for dataset, (rows, market_column, market_kind) in views.items():
        rows = add_market_strength(rows, market_column, market_kind)
        summary = (
            rows.groupby(["pos", "experience_cohort"], as_index=False)
            .agg(
                player_seasons=("player", "size"),
                unique_players=("player", "nunique"),
                seasons=("season", "nunique"),
                avg_raw_year_exp=("raw_year_exp", "mean"),
                avg_pred_ppg=("pred_ppg", "mean"),
                avg_market=(market_column, "mean"),
                avg_managed_points=("managed_points_above_waiver", "mean"),
                avg_managed_residual=("managed_residual_points", "mean"),
                managed_miss_rate=("managed_miss_30", "mean"),
                managed_upside_rate=("managed_upside_20", "mean"),
                avg_boom_weeks=("boom_weeks", "mean"),
            )
        )
        summary.insert(0, "dataset", dataset)
        summary.insert(1, "market_column", market_column)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True, sort=False)


def match_veterans(
    rows: pd.DataFrame,
    dataset: str,
    market_column: str,
    market_kind: str,
    match_mode: str,
) -> pd.DataFrame:
    data = add_market_strength(rows, market_column, market_kind)
    feature_map = {
        "projection": ["pred_ppg"],
        "market": ["market_strength"],
        "projection_and_market": ["pred_ppg", "market_strength"],
    }
    features = feature_map[match_mode]
    records = []
    for _, veteran in data[data.above_threshold].iterrows():
        cell = data[
            data.pos.eq(veteran.pos) & data.season.eq(veteran.season)
        ]
        peers = cell[~cell.above_threshold].copy()
        if len(peers) < 3:
            continue
        squared_distance = np.zeros(len(peers), dtype=float)
        for feature in features:
            feature_sd = float(cell[feature].std(ddof=0))
            if not np.isfinite(feature_sd) or feature_sd <= 0:
                feature_sd = 1.0
            squared_distance += (
                (peers[feature].to_numpy(dtype=float) - float(veteran[feature]))
                / feature_sd
            ) ** 2
        peers["match_distance"] = np.sqrt(squared_distance)
        peers = peers.nsmallest(min(MATCH_K, len(peers)), "match_distance")
        if float(peers.match_distance.max()) > MATCH_MAX_DISTANCE:
            continue
        record = {
            "dataset": dataset,
            "match_mode": match_mode,
            "market_column": market_column,
            "player": veteran.player,
            "pos": veteran.pos,
            "season": int(veteran.season),
            "raw_year_exp": float(veteran.raw_year_exp),
            "threshold": float(veteran.threshold),
            "peer_count": int(len(peers)),
            "mean_match_distance": float(peers.match_distance.mean()),
            "peer_players": " | ".join(peers.player.astype(str)),
        }
        comparison_columns = [
            "pred_ppg",
            market_column,
            *CONTINUOUS_OUTCOMES,
            *BINARY_OUTCOMES,
        ]
        for column in dict.fromkeys(comparison_columns):
            record[f"veteran_{column}"] = float(veteran[column])
            record[f"peer_{column}"] = float(peers[column].mean())
            record[f"delta_{column}"] = (
                record[f"veteran_{column}"] - record[f"peer_{column}"]
            )
        records.append(record)
    return pd.DataFrame(records)


def clustered_bootstrap_mean(
    rows: pd.DataFrame,
    column: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    players = rows.player.drop_duplicates().to_numpy()
    if len(players) < 4:
        return np.nan, np.nan
    groups = {player: rows[rows.player.eq(player)] for player in players}
    estimates = np.empty(BOOTSTRAPS, dtype=float)
    for idx in range(BOOTSTRAPS):
        sampled = rng.choice(players, size=len(players), replace=True)
        values = np.concatenate(
            [groups[player][column].to_numpy(dtype=float) for player in sampled]
        )
        estimates[idx] = float(np.mean(values))
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def summarize_matches(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(RANDOM_SEED)
    records = []
    outcome_columns = [*CONTINUOUS_OUTCOMES, *BINARY_OUTCOMES]
    for keys, group in matches.groupby(["dataset", "match_mode", "pos"]):
        dataset, match_mode, pos = keys
        base = {
            "dataset": dataset,
            "match_mode": match_mode,
            "pos": pos,
            "matched_veteran_seasons": int(len(group)),
            "matched_unique_veterans": int(group.player.nunique()),
            "matched_seasons": int(group.season.nunique()),
            "mean_match_distance": float(group.mean_match_distance.mean()),
            "mean_pred_ppg_delta": float(group.delta_pred_ppg.mean()),
        }
        market_column = str(group.market_column.iloc[0])
        base["mean_market_delta"] = float(
            group[f"delta_{market_column}"].mean()
        )
        seasons = sorted(group.season.unique())
        for outcome in outcome_columns:
            column = f"delta_{outcome}"
            mean_delta = float(group[column].mean())
            ci_low, ci_high = clustered_bootstrap_mean(group, column, rng)
            leave_one_out = [
                float(group.loc[~group.season.eq(season), column].mean())
                for season in seasons
                if (~group.season.eq(season)).any()
            ]
            scale = 100.0 if outcome in BINARY_OUTCOMES else 1.0
            records.append(
                {
                    **base,
                    "outcome": outcome,
                    "delta_unit": (
                        "percentage_points"
                        if outcome in BINARY_OUTCOMES
                        else "points"
                    ),
                    "mean_delta": mean_delta * scale,
                    "cluster_boot_ci95_low": ci_low * scale,
                    "cluster_boot_ci95_high": ci_high * scale,
                    "leave_one_season_out_min": (
                        min(leave_one_out) * scale if leave_one_out else np.nan
                    ),
                    "leave_one_season_out_max": (
                        max(leave_one_out) * scale if leave_one_out else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def standardize_controls(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    for source, target in [
        ("pred_ppg", "pred_z"),
        ("market_strength", "market_z"),
    ]:
        sd = float(output[source].std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        output[target] = (output[source] - output[source].mean()) / sd
        output[f"{target}2"] = output[target] ** 2
    return output


def fit_regression_models(
    views: dict[str, tuple[pd.DataFrame, str, str]],
) -> pd.DataFrame:
    records = []
    outcomes = {**CONTINUOUS_OUTCOMES, **BINARY_OUTCOMES}
    for dataset, (rows, market_column, market_kind) in views.items():
        data = add_market_strength(rows, market_column, market_kind)
        for pos in POSITIONS:
            pos_rows = standardize_controls(data[data.pos.eq(pos)].copy())
            old_n = int(pos_rows.above_threshold.sum())
            for spec_name, controls in {
                "market_only": "market_z + market_z2",
                "projection_and_market": (
                    "pred_z + pred_z2 + market_z + market_z2"
                ),
            }.items():
                formula_tail = (
                    "exp_below + excess_exp + " + controls + " + C(season)"
                )
                for outcome in outcomes:
                    base = {
                        "dataset": dataset,
                        "pos": pos,
                        "model_spec": spec_name,
                        "outcome": outcome,
                        "n": int(len(pos_rows)),
                        "above_threshold_n": old_n,
                        "unique_players": int(pos_rows.player.nunique()),
                    }
                    if len(pos_rows) < 50 or old_n < 12:
                        records.append({**base, "status": "insufficient"})
                        continue
                    try:
                        formula = f"{outcome} ~ {formula_tail}"
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            if outcome in BINARY_OUTCOMES:
                                fit = smf.glm(
                                    formula,
                                    data=pos_rows,
                                    family=sm.families.Binomial(),
                                ).fit(
                                    cov_type="cluster",
                                    cov_kwds={"groups": pos_rows.player},
                                )
                            else:
                                fit = smf.ols(formula, data=pos_rows).fit(
                                    cov_type="cluster",
                                    cov_kwds={"groups": pos_rows.player},
                                )
                        if outcome in BINARY_OUTCOMES:
                            old = pos_rows.above_threshold
                            plus_year = pos_rows.loc[old].copy()
                            plus_year["excess_exp"] += 1
                            marginal = float(
                                (
                                    fit.predict(plus_year)
                                    - fit.predict(pos_rows.loc[old])
                                ).mean()
                                * 100
                            )
                            unit = "percentage_points"
                        else:
                            marginal = float(fit.params["excess_exp"])
                            unit = "points"
                        coefficient = float(fit.params["excess_exp"])
                        standard_error = float(fit.bse["excess_exp"])
                        records.append(
                            {
                                **base,
                                "status": "ok",
                                "coefficient": coefficient,
                                "cluster_se": standard_error,
                                "coefficient_ci95_low": coefficient
                                - 1.96 * standard_error,
                                "coefficient_ci95_high": coefficient
                                + 1.96 * standard_error,
                                "p_value": float(fit.pvalues["excess_exp"]),
                                "marginal_delta_per_excess_year": marginal,
                                "marginal_unit": unit,
                            }
                        )
                    except Exception as exc:
                        records.append(
                            {**base, "status": f"failed:{type(exc).__name__}"}
                        )
    return pd.DataFrame(records)


def fit_quantile_models(
    views: dict[str, tuple[pd.DataFrame, str, str]],
) -> pd.DataFrame:
    records = []
    for dataset in ("adp_top_100", "adp_top_200"):
        rows, market_column, market_kind = views[dataset]
        data = add_market_strength(rows, market_column, market_kind)
        for pos in POSITIONS:
            pos_rows = standardize_controls(data[data.pos.eq(pos)].copy())
            old_n = int(pos_rows.above_threshold.sum())
            for quantile in (0.10, 0.50, 0.90):
                base = {
                    "dataset": dataset,
                    "pos": pos,
                    "quantile": quantile,
                    "n": int(len(pos_rows)),
                    "above_threshold_n": old_n,
                }
                if len(pos_rows) < 100 or old_n < 15:
                    records.append({**base, "status": "insufficient"})
                    continue
                try:
                    fit = smf.quantreg(
                        "managed_points_above_waiver ~ exp_below + excess_exp + "
                        "pred_z + pred_z2 + market_z + market_z2 + C(season)",
                        pos_rows,
                    ).fit(q=quantile, max_iter=5000)
                    coefficient = float(fit.params["excess_exp"])
                    standard_error = float(fit.bse["excess_exp"])
                    records.append(
                        {
                            **base,
                            "status": "ok",
                            "points_per_excess_year": coefficient,
                            "standard_error": standard_error,
                            "ci95_low": coefficient - 1.96 * standard_error,
                            "ci95_high": coefficient + 1.96 * standard_error,
                            "p_value": float(fit.pvalues["excess_exp"]),
                            "uncertainty_note": "asymptotic_nonclustered",
                        }
                    )
                except Exception as exc:
                    records.append(
                        {**base, "status": f"failed:{type(exc).__name__}"}
                    )
    return pd.DataFrame(records)


def current_named_context(veteran, actuals: pd.DataFrame) -> pd.DataFrame:
    current = read_sql(
        SIMULATION_DB,
        """
        SELECT m.player,
               m.pos,
               CAST(m.year AS INTEGER) season,
               m.pred_fp_per_game pred_ppg,
               m.year_exp capped_year_exp,
               s.salary market_salary
        FROM Best_Ball_Weekly_Player_Map m
        INNER JOIN Salaries_Pred s
                ON m.player=s.player
               AND m.year=s.year
               AND s.league='betapred'
        WHERE m.year=2026
              AND m.version='beta'
              AND m.pos IN ('RB', 'WR', 'TE')
        """,
    )
    current = veteran.attach_uncapped_experience(
        current,
        actuals,
        veteran.load_draft_years(),
    )
    current = veteran.add_threshold_fields(current, THRESHOLDS)
    names = ["Derrick Henry", "Alvin Kamara", "George Kittle", "Travis Kelce"]
    records = []
    for _, player in current[current.player.isin(names)].iterrows():
        peers = current[
            current.pos.eq(player.pos) & ~current.above_threshold
        ].copy()
        peers["ppg_gap"] = (peers.pred_ppg - player.pred_ppg).abs()
        peers = peers.nsmallest(MATCH_K, "ppg_gap")
        records.append(
            {
                "player": player.player,
                "pos": player.pos,
                "raw_year_exp": player.raw_year_exp,
                "capped_year_exp": player.capped_year_exp,
                "threshold": player.threshold,
                "pred_ppg": player.pred_ppg,
                "market_salary": player.market_salary,
                "projection_peer_salary": peers.market_salary.mean(),
                "salary_vs_projection_peers": (
                    player.market_salary - peers.market_salary.mean()
                ),
                "mean_peer_ppg_gap": peers.ppg_gap.mean(),
                "projection_peers": " | ".join(peers.player.astype(str)),
            }
        )
    return pd.DataFrame(records).sort_values(["pos", "player"])


def fmt_interval(row: pd.Series) -> str:
    return (
        f"{row.mean_delta:+.1f} "
        f"[{row.cluster_boot_ci95_low:+.1f}, "
        f"{row.cluster_boot_ci95_high:+.1f}]"
    )


def write_summary(
    long_rows: pd.DataFrame,
    auction_rows: pd.DataFrame,
    match_summary: pd.DataFrame,
    named: pd.DataFrame,
) -> None:
    lines = [
        "# Current-Season Veteran Value Results",
        "",
        "## Coverage and estimand",
        "",
        f"- Long ADP history: {len(long_rows):,} player-seasons from "
        f"{int(long_rows.season.min())}-{int(long_rows.season.max())}.",
        f"- Current-method auction history: {len(auction_rows):,} player-seasons "
        f"from {int(auction_rows.season.min())}-{int(auction_rows.season.max())}.",
        "- Outcomes are current-season only. Managed points sum positive weekly "
        "points above the position waiver baseline.",
        "- Bracketed ranges below are 95% player-cluster bootstrap intervals. "
        "They describe historical association, not a causal effect of age.",
        "",
        "## Long-history matched results",
        "",
        "Each veteran is compared with up to five same-position, same-season "
        "younger peers matched jointly on preseason PPG and market ADP.",
        "",
    ]
    for dataset, label in [
        ("adp_top_100", "Top-100 ADP"),
        ("adp_top_200", "Top-200 ADP"),
    ]:
        lines.append(f"### {label}")
        lines.append("")
        subset = match_summary[
            match_summary.dataset.eq(dataset)
            & match_summary.match_mode.eq("projection_and_market")
        ]
        for pos in POSITIONS:
            cur = subset[subset.pos.eq(pos)].set_index("outcome")
            if cur.empty:
                continue
            n = int(cur.matched_veteran_seasons.iloc[0])
            players = int(cur.matched_unique_veterans.iloc[0])
            if n < 12:
                lines.append(
                    f"- {pos}: only {n} adequately matched veteran-seasons/"
                    f"{players} players; insufficient for interpretation."
                )
                continue
            managed = fmt_interval(cur.loc["managed_points_above_waiver"])
            miss = fmt_interval(cur.loc["managed_miss_30"])
            upside = fmt_interval(cur.loc["managed_upside_20"])
            boom = fmt_interval(cur.loc["boom_weeks"])
            lines.append(
                f"- {pos}: {n} veteran-seasons/{players} players; managed points "
                f"{managed}, miss rate {miss} pp, upside-hit rate {upside} pp, "
                f"boom weeks {boom}."
            )
        lines.append("")

    lines.extend(
        [
            "## Exact recent auction evidence",
            "",
            "These matches use rolling v5 predicted auction salary rather than ADP. "
            "The full pool is usable as a direction check; the `$5+` veteran cells "
            "are too small for a stable production penalty.",
            "",
        ]
    )
    for dataset, label in [
        ("auction_all", "All modeled salaries"),
        ("auction_salary_5plus", "Predicted salary at least $5"),
    ]:
        lines.append(f"### {label}")
        lines.append("")
        subset = match_summary[
            match_summary.dataset.eq(dataset)
            & match_summary.match_mode.eq("projection_and_market")
        ]
        for pos in POSITIONS:
            cur = subset[subset.pos.eq(pos)].set_index("outcome")
            if cur.empty:
                lines.append(f"- {pos}: no adequate matched veteran cell.")
                continue
            n = int(cur.matched_veteran_seasons.iloc[0])
            players = int(cur.matched_unique_veterans.iloc[0])
            if n < 12:
                lines.append(
                    f"- {pos}: only {n} adequately matched veteran-seasons/"
                    f"{players} players; insufficient for interpretation."
                )
                continue
            managed = fmt_interval(cur.loc["managed_points_above_waiver"])
            miss = fmt_interval(cur.loc["managed_miss_30"])
            upside = fmt_interval(cur.loc["managed_upside_20"])
            lines.append(
                f"- {pos}: {n} veteran-seasons/{players} players; managed points "
                f"{managed}, miss rate {miss} pp, upside-hit rate {upside} pp."
            )
        lines.append("")

    lines.extend(["## Current 2026 named-player market context", ""])
    for _, row in named.iterrows():
        lines.append(
            f"- {row.player}: raw experience {row.raw_year_exp:.0f}, "
            f"modeled experience {row.capped_year_exp:.0f}, {row.pred_ppg:.2f} PPG, "
            f"market ${row.market_salary:.1f}; salary is "
            f"{row.salary_vs_projection_peers:+.1f} versus the mean of five younger "
            "same-position projection peers."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A blanket RB/WR/TE current-season age tax is not supported. Long-run "
            "managed value is close to younger matched peers for RB, modestly lower "
            "for WR, and not reliably estimable for premium TE.",
            "- Premium veteran RBs look more compressed than worse: mean value is "
            "neutral, with both miss and upside-hit rates modestly lower than matched "
            "peers. That can justify an explicit ceiling preference, not an expected-"
            "value haircut.",
            "- Premium veteran WRs provide the only recurring warning: fewer managed "
            "points and a higher miss rate in every leave-one-season-out Top-100 "
            "match. Player-cluster intervals remain wide, so this is a candidate for "
            "prospective template testing rather than a calibrated penalty.",
            "- The market does not apply one uniform veteran discount. Projection-"
            "matched historical RB/WR veterans were drafted at nearly the same Top-"
            "100 ADP as younger peers; individual 2026 TE prices are discounted, "
            "while Derrick Henry is not cheaper than his projection peers.",
            "- The recent `$5+` auction sample is too small and directionally "
            "unstable to estimate a dollar penalty: it contains only a handful of "
            "above-threshold RB/TE seasons.",
            "- Do not alter current point forecasts from this evidence. If production "
            "changes, first validate a premium-WR current-outcome mixture. Express "
            "any broader veteran fade as an explicit ceiling/risk utility preference "
            "using uncapped experience, not as a claimed forecast correction.",
            "",
        ]
    )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    long_rows, auction_rows, veteran, actuals = build_analysis_rows()
    views = dataset_views(long_rows, auction_rows)
    cohorts = cohort_summary(views)

    match_frames = []
    for dataset, (rows, market_column, market_kind) in views.items():
        for match_mode in ("projection", "market", "projection_and_market"):
            match_frames.append(
                match_veterans(
                    rows,
                    dataset,
                    market_column,
                    market_kind,
                    match_mode,
                )
            )
    matches = pd.concat(match_frames, ignore_index=True, sort=False)
    match_summary = summarize_matches(matches)
    regression = fit_regression_models(views)
    quantiles = fit_quantile_models(views)
    named = current_named_context(veteran, actuals)

    export_columns = [
        "player",
        "pos",
        "season",
        "pred_ppg",
        "avg_pick",
        "raw_year_exp",
        "capped_year_exp",
        "threshold",
        "experience_cohort",
        "projection_tier",
        "played_games",
        "active_ppg",
        "season_points",
        "managed_points_above_waiver",
        "forecast_managed_points",
        "managed_residual_points",
        "managed_miss_30",
        "managed_upside_20",
        "boom_weeks",
    ]
    long_rows[export_columns].to_csv(
        RESULTS_DIR / "analysis_rows.csv", index=False
    )
    auction_rows[
        export_columns
        + [
            "pred_salary",
            "pred_salary_raw",
            "actual_salary",
            "actual_salary_observed",
            "training_through_year",
            "data_rolling_origin",
            "normalization_uses_target_actuals",
            "method_version",
        ]
    ].to_csv(RESULTS_DIR / "auction_analysis_rows.csv", index=False)
    cohorts.to_csv(RESULTS_DIR / "cohort_summary.csv", index=False)
    matches.to_csv(RESULTS_DIR / "matched_veteran_rows.csv", index=False)
    match_summary.to_csv(RESULTS_DIR / "matched_summary.csv", index=False)
    regression.to_csv(RESULTS_DIR / "regression_models.csv", index=False)
    quantiles.to_csv(RESULTS_DIR / "quantile_models.csv", index=False)
    named.to_csv(RESULTS_DIR / "current_named_market_context.csv", index=False)
    write_summary(long_rows, auction_rows, match_summary, named)
    print((RESULTS_DIR / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
