"""Audit the production Bijan Buy-vs-Pass decision in the current beta state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
APP_DIR = ROOT.parent / "Fantasy_Football_App" / "app"
APP_DB = APP_DIR / "Simulation.sqlite3"
RESULTS_DIR = STUDY_DIR / "results"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import zSequential_Target as sequential  # noqa: E402
from zSim_Helper import FootballSimulation  # noqa: E402


YEAR = 2026
LEAGUE = "beta"
PRED_VERSION = "final_ensemble"
SALARY_SOURCE = "pred"
SALARY_CAP = 298
NUM_TEAMS = 12
ROSTER_SIZE = 13
LINEUP_REQUIRE = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}
POS_MIN = {"QB": 1, "RB": 4, "WR": 4, "TE": 1}
RELAXED_RB_POS_MIN = {"QB": 1, "RB": 2, "WR": 4, "TE": 1}
POS_MAX = {"QB": 1, "RB": 6, "WR": 6, "TE": 2}
COMPUTE_BUDGET = 320
REQUIRE_TOP_N = 12
CANDIDATE = "Bijan Robinson"
FIXED_SALARIES = {
    "Jahmyr Gibbs": 110.0,
    "Chase Brown": 34.0,
    "Bhayshul Tuten": 11.0,
}


def evidence_seed(
    sim: FootballSimulation,
    waiver_baselines: dict[str, float],
    *,
    enforce_top_n: bool,
    use_selection_premium: bool,
    variation: int,
    pos_min_counts: dict[str, int] | None = None,
) -> int:
    pos_min_counts = dict(pos_min_counts or POS_MIN)
    components: tuple[object, ...] = (
        int(sim.set_year),
        str(sim.league),
        str(sim.pred_vers),
        str(sim.sal_pred_actual),
        int(COMPUTE_BUDGET),
        bool(enforce_top_n),
        int(REQUIRE_TOP_N),
        tuple(sorted(LINEUP_REQUIRE.items())),
        int(ROSTER_SIZE),
        tuple(sorted(pos_min_counts.items())),
        tuple(sorted(POS_MAX.items())),
        tuple(sorted(waiver_baselines.items())),
        bool(use_selection_premium),
    )
    if variation > 0:
        components += ("user_variation", int(variation))
    return int(sequential.stable_sequential_evidence_seed(*components))


def json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if pd.isna(value) if not isinstance(value, (str, bool, pd.DataFrame)) else False:
        return None
    return value


def run_case(
    sim: FootballSimulation,
    *,
    to_add: dict[str, list],
    to_drop: list[str],
    remaining_market_budget: float,
    remaining_market_slots: int,
    waiver_baselines: dict[str, float],
    candidate_price: int,
    label: str,
    variation: int,
    enforce_top_n: bool,
    use_selection_premium: bool,
    profile_bid: bool,
    capture_paths: bool = False,
    pos_min_counts: dict[str, int] | None = None,
    seed_override: int | None = None,
) -> dict[str, object]:
    pos_min_counts = dict(pos_min_counts or POS_MIN)
    seed = (
        int(seed_override)
        if seed_override is not None
        else evidence_seed(
            sim,
            waiver_baselines,
            enforce_top_n=enforce_top_n,
            use_selection_premium=use_selection_premium,
            variation=variation,
            pos_min_counts=pos_min_counts,
        )
    )
    started = time.perf_counter()
    captured_paths: list[dict[str, object]] = []
    original_branch_simulator = sequential.simulate_history_only_branch

    def capture_branch(*args, **kwargs):
        branch = original_branch_simulator(*args, **kwargs)
        if (
            capture_paths
            and kwargs.get("candidate") == CANDIDATE
            and int(kwargs.get("candidate_price")) == int(candidate_price)
        ):
            captured_paths.append(
                {
                    "branch": "buy" if kwargs.get("force_buy") else "pass",
                    "complete": bool(branch.get("complete")),
                    "failure_reason": branch.get("failure_reason"),
                    "roster": sorted(branch.get("roster", [])),
                    "salary_map": dict(branch.get("salary_map", {})),
                }
            )
        return branch

    if capture_paths:
        sequential.simulate_history_only_branch = capture_branch
    try:
        result = sequential.run_sequential_nomination_analysis(
            sim,
            to_add,
            to_drop,
            CANDIDATE,
            candidate_price,
            compute_budget=COMPUTE_BUDGET,
            require_top_n=REQUIRE_TOP_N,
            enforce_top_n=enforce_top_n,
            roster_size=ROSTER_SIZE,
            lineup_require=LINEUP_REQUIRE,
            pos_min_counts=pos_min_counts,
            pos_max_counts=POS_MAX,
            waiver_baselines=waiver_baselines,
            remaining_market_budget=remaining_market_budget,
            remaining_market_slots=remaining_market_slots,
            use_selection_premium=use_selection_premium,
            random_seed=seed,
            profile_bid=profile_bid,
            full_curve=False,
        )
    finally:
        sequential.simulate_history_only_branch = original_branch_simulator

    if capture_paths:
        if len(captured_paths) != 2 * int(result.get("requested_paths", 0)):
            raise AssertionError(
                "Primary path capture did not retain exactly one Buy and Pass "
                f"branch per requested path: {len(captured_paths)}"
            )
        ppg_map = sim.player_data.set_index("player").pred_fp_per_game.to_dict()
        pos_map = sim.player_data.set_index("player").pos.to_dict()

        def nominal_lineup(roster: list[str]) -> tuple[list[str], float]:
            available = set(roster)
            selected: list[str] = []
            for pos in ("QB", "RB", "WR", "TE"):
                candidates = sorted(
                    (player for player in available if pos_map[player] == pos),
                    key=lambda player: (-float(ppg_map[player]), player),
                )
                chosen = candidates[: int(LINEUP_REQUIRE.get(pos, 0))]
                selected.extend(chosen)
                available.difference_update(chosen)
            flex_candidates = sorted(
                (
                    player
                    for player in available
                    if pos_map[player] in {"RB", "WR", "TE"}
                ),
                key=lambda player: (-float(ppg_map[player]), player),
            )
            selected.extend(
                flex_candidates[: int(LINEUP_REQUIRE.get("FLEX", 0))]
            )
            return selected, float(sum(ppg_map[player] for player in selected))

        path_rows = []
        for path_index, branch in enumerate(captured_paths):
            roster = list(branch["roster"])
            starters, nominal_ppg = nominal_lineup(roster)
            salary_map = branch["salary_map"]
            counts = pd.Series([pos_map[player] for player in roster]).value_counts()
            path_rows.append(
                {
                    "capture_order": path_index,
                    "pair_index": path_index // 2,
                    "branch": branch["branch"],
                    "complete": branch["complete"],
                    "failure_reason": branch["failure_reason"],
                    "roster_spend": float(sum(salary_map.values())),
                    "nominal_lineup_ppg": nominal_ppg,
                    "qb_count": int(counts.get("QB", 0)),
                    "rb_count": int(counts.get("RB", 0)),
                    "wr_count": int(counts.get("WR", 0)),
                    "te_count": int(counts.get("TE", 0)),
                    "starters": " | ".join(starters),
                    "roster": " | ".join(roster),
                    "salary_map": json.dumps(salary_map, sort_keys=True),
                }
            )
        path_frame = pd.DataFrame(path_rows)
        path_frame.to_csv(RESULTS_DIR / "primary_path_rosters.csv", index=False)

        frequency_rows = []
        for branch_name in ("buy", "pass"):
            branch_paths = [
                branch
                for branch in captured_paths
                if branch["branch"] == branch_name and branch["complete"]
            ]
            for player in sorted(
                {player for branch in branch_paths for player in branch["roster"]}
            ):
                salaries = [
                    float(branch["salary_map"][player])
                    for branch in branch_paths
                    if player in branch["roster"]
                ]
                frequency_rows.append(
                    {
                        "branch": branch_name,
                        "player": player,
                        "pos": pos_map[player],
                        "paths": len(salaries),
                        "path_rate": len(salaries) / len(branch_paths),
                        "mean_acquisition_salary": float(np.mean(salaries)),
                        "pred_ppg": float(ppg_map[player]),
                    }
                )
        pd.DataFrame(frequency_rows).to_csv(
            RESULTS_DIR / "primary_roster_player_frequencies.csv",
            index=False,
        )
    curve = result.pop("price_curve")
    curve.to_csv(RESULTS_DIR / f"price_curve_{label}.csv", index=False)
    result.update(
        {
            "case": label,
            "variation": variation,
            "enforce_top_n": enforce_top_n,
            "use_selection_premium": use_selection_premium,
            "position_minimums": pos_min_counts,
            "evidence_seed_reconstructed": seed,
            "seed_override": seed_override,
            "runtime_seconds_external": time.perf_counter() - started,
        }
    )
    with (RESULTS_DIR / f"decision_{label}.json").open("w", encoding="utf-8") as handle:
        json.dump(json_value(result), handle, indent=2, sort_keys=True)
    return result


def run_manual_completion_counterfactuals(
    sim: FootballSimulation,
    *,
    to_drop: list[str],
    remaining_market_budget: float,
    remaining_market_slots: int,
    waiver_baselines: dict[str, float],
    candidate_price: int,
) -> pd.DataFrame:
    """Score central completion plans, including the user's two-WR alternative."""
    seed = evidence_seed(
        sim,
        waiver_baselines,
        enforce_top_n=True,
        use_selection_premium=False,
        variation=14,
    )
    seed_values = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(seed).spawn(8)
    ]
    wr_pair = {"Tee Higgins": 34.0, "Emeka Egbuka": 37.0}
    required_players = {CANDIDATE, *FIXED_SALARIES, *wr_pair}
    with sim.temp_seed(seed_values[0]):
        canonical_predictions = sim.get_predictions(
            "pred_fp_per_game",
            num_options=512,
        )
    predictions = sim.drop_players(canonical_predictions, to_drop)
    predictions, _ = sequential.apply_sequential_draft_pool_filter(
        predictions,
        sequential._sequential_draft_pool_metadata(sim),
        sim.league,
        required_players=required_players,
    )
    missing = sorted(required_players - set(predictions.player))
    if missing:
        raise ValueError(
            "Manual counterfactual players missing from sequential pool: "
            + ", ".join(missing)
        )
    state_indices = sequential._canonical_state_indices(
        canonical_predictions,
        predictions,
    )
    canonical_aligned = sequential._aligned_player_frame(
        sim,
        canonical_predictions,
    )
    aligned = sequential._aligned_player_frame(sim, predictions)
    canonical_market_prices = canonical_aligned.salary.to_numpy(dtype=np.float64)
    market_prices = aligned.salary.to_numpy(dtype=np.float64)
    canonical_available_mask = (
        np.isin(
            canonical_predictions.player.to_numpy(),
            predictions.player.to_numpy(),
        )
        & ~canonical_predictions.player.isin(FIXED_SALARIES).to_numpy()
    )
    canonical_base_prices = sim.normalize_salary_market_values(
        canonical_market_prices,
        canonical_available_mask,
        remaining_market_budget=remaining_market_budget,
        remaining_market_slots=remaining_market_slots,
    )
    base_prices = canonical_base_prices[state_indices]
    predictions["salary"] = market_prices
    premiums = np.zeros(len(predictions), dtype=np.float64)
    managed_blocks = sequential._sample_construction_value_blocks(
        sim,
        canonical_predictions,
        predictions,
        list(FIXED_SALARIES),
        block_count=4,
        contexts_per_block=32,
        num_weeks=16,
        waiver_baselines=waiver_baselines,
        lineup_require=LINEUP_REQUIRE,
        learn_weeks=6,
        max_learn_weight=0.65,
        random_seed=seed_values[1],
    )
    validation_seeds = np.random.SeedSequence(seed_values[6]).spawn(4)
    base_price_map = dict(zip(predictions.player, base_prices))
    scenarios = {
        "open_central_choice": {
            "owned": dict(FIXED_SALARIES),
            "excluded": set(),
            "observed": [],
        },
        "buy_bijan_105": {
            "owned": {**FIXED_SALARIES, CANDIDATE: float(candidate_price)},
            "excluded": set(),
            "observed": [(float(candidate_price), base_price_map[CANDIDATE])],
        },
        "pass_bijan": {
            "owned": dict(FIXED_SALARIES),
            "excluded": {CANDIDATE},
            "observed": [(float(candidate_price - 1), base_price_map[CANDIDATE])],
        },
        "pass_bijan_force_two_mid_wrs": {
            "owned": {**FIXED_SALARIES, **wr_pair},
            "excluded": {CANDIDATE},
            "observed": [(float(candidate_price - 1), base_price_map[CANDIDATE])],
        },
    }
    rows = []
    score_rows = []
    all_players = set(predictions.player)
    static_cache = {}
    for block_index, managed_values in enumerate(managed_blocks):
        validation_seed = int(
            validation_seeds[block_index].generate_state(
                1,
                dtype=np.uint32,
            )[0]
        )
        validation_bank = sequential._sample_validation_bank(
            sim,
            predictions,
            64,
            16,
            6,
            0.65,
            validation_seed,
            canonical_predictions=canonical_predictions,
        )
        for scenario_name, scenario in scenarios.items():
            owned = scenario["owned"]
            unresolved = all_players - set(owned) - scenario["excluded"]
            plan = sequential.solve_history_only_plan(
                sim,
                predictions,
                managed_values,
                base_prices,
                premiums,
                owned,
                unresolved,
                ROSTER_SIZE,
                POS_MIN,
                POS_MAX,
                REQUIRE_TOP_N,
                True,
                observed_sales=scenario["observed"],
                static_matrix_cache=static_cache,
            )
            if plan is None:
                raise RuntimeError(
                    f"No manual completion for {scenario_name} block {block_index}."
                )
            roster = sorted(plan["selected"])
            scores = sequential._score_roster_bank(
                sim,
                predictions,
                roster,
                *validation_bank,
                LINEUP_REQUIRE,
                waiver_baselines,
                {},
            )
            score_rows.extend(
                {
                    "scenario": scenario_name,
                    "block": block_index,
                    "context": context_index,
                    "season_score": float(score),
                }
                for context_index, score in enumerate(scores)
            )
            rows.append(
                {
                    "scenario": scenario_name,
                    "block": block_index,
                    "score_mean": float(np.mean(scores)),
                    "score_p10": float(np.percentile(scores, 10)),
                    "forecast_spend": float(
                        sum(plan["forecast_cost"][player] for player in roster)
                    ),
                    "roster": " | ".join(roster),
                    "contains_bijan": CANDIDATE in roster,
                    "contains_tee_higgins": "Tee Higgins" in roster,
                    "contains_emeka_egbuka": "Emeka Egbuka" in roster,
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(
        RESULTS_DIR / "manual_completion_counterfactuals_by_block.csv",
        index=False,
    )
    context_scores = pd.DataFrame(score_rows)
    context_scores.to_csv(
        RESULTS_DIR / "manual_completion_counterfactual_scores.csv",
        index=False,
    )
    score_summary = (
        context_scores.groupby("scenario").season_score
        .agg(
            score_mean="mean",
            score_p10=lambda values: float(np.percentile(values, 10)),
        )
        .reset_index()
    )
    plan_summary = (
        output.groupby("scenario", as_index=False)
        .agg(
            forecast_spend=("forecast_spend", "mean"),
            contains_bijan=("contains_bijan", "mean"),
            contains_tee_higgins=("contains_tee_higgins", "mean"),
            contains_emeka_egbuka=("contains_emeka_egbuka", "mean"),
        )
    )
    summary = score_summary.merge(plan_summary, on="scenario", validate="one_to_one")
    summary.to_csv(
        RESULTS_DIR / "manual_completion_counterfactuals_summary.csv",
        index=False,
    )
    return summary


def write_standard_variation_summary() -> None:
    rows = []
    for path in sorted(RESULTS_DIR.glob("decision_variation*.json")):
        with path.open(encoding="utf-8") as handle:
            decision = json.load(handle)
        label = str(decision.get("case", ""))
        standard_label = (
            label in {
                "variation0_top_on_reserve_off",
                "variation14_top_on_reserve_off",
            }
            or label.endswith("_top_on_reserve_off_extra")
        )
        if not standard_label:
            continue
        if not bool(decision.get("enforce_top_n")):
            continue
        if bool(decision.get("use_selection_premium")):
            continue
        position_minimums = decision.get("position_minimums", POS_MIN)
        if position_minimums != POS_MIN:
            continue
        rows.append(
            {
                "case": label,
                "variation": int(decision["variation"]),
                "gain": float(decision["SequentialGain"]),
                "se": float(decision["SequentialSE"]),
                "lcb80": float(decision["SequentialLCB80"]),
                "recommendation": decision["recommendation"],
                "block_positive_rate": float(decision["BlockPositiveRate"]),
                "block_gain_min": float(decision["BlockGainMin"]),
                "block_gain_max": float(decision["BlockGainMax"]),
                "buy_completion": float(decision["BuyCompletion"]),
                "pass_completion": float(decision["PassCompletion"]),
                "paired_rate": float(decision["PairedRate"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values("variation").reset_index(drop=True)
    frame.to_csv(RESULTS_DIR / "variation_stability.csv", index=False)
    summary = {
        "variation_count": int(len(frame)),
        "target_count": int(frame.recommendation.eq("TARGET").sum()),
        "gain_mean": float(frame.gain.mean()),
        "gain_sd": float(frame.gain.std(ddof=1)),
        "gain_min": float(frame.gain.min()),
        "gain_max": float(frame.gain.max()),
        "lcb80_min": float(frame.lcb80.min()),
        "lcb80_max": float(frame.lcb80.max()),
    }
    with (RESULTS_DIR / "variation_stability_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main(
    *,
    manual_only: bool = False,
    case_label: str | None = None,
    variation: int | None = None,
    skip_manual: bool = False,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(APP_DB)
    try:
        sim = FootballSimulation(
            conn,
            YEAR,
            LINEUP_REQUIRE,
            SALARY_CAP,
            PRED_VERSION,
            LEAGUE,
            sal_pred_actual=SALARY_SOURCE,
        )
        sim.load_weekly_template_profiles()
        keepers = pd.read_sql_query(
            """
            SELECT player AS source_player, player_key, keeper_salary
            FROM League_Keepers
            WHERE year = :year AND league = :league
            ORDER BY source_player
            """,
            conn,
            params={"year": YEAR, "league": LEAGUE},
        )
        canonical_by_key = sim.player_data.set_index("player_key").player
        keepers["player"] = keepers.player_key.map(canonical_by_key)
        if keepers.player.isna().any():
            missing_keys = keepers.loc[
                keepers.player.isna(), "player_key"
            ].astype(str).tolist()
            raise ValueError(
                "League keepers missing from the canonical simulation pool: "
                + ", ".join(missing_keys)
            )
        keeper_salary_map = dict(
            zip(keepers.player, keepers.keeper_salary.astype(float))
        )
        owned_keeper_players = set(FIXED_SALARIES) & set(keeper_salary_map)
        for player in owned_keeper_players:
            if not np.isclose(
                float(FIXED_SALARIES[player]),
                float(keeper_salary_map[player]),
            ):
                raise ValueError(
                    f"Owned keeper salary mismatch for {player}: "
                    f"state={FIXED_SALARIES[player]} "
                    f"table={keeper_salary_map[player]}"
                )
        required = {CANDIDATE, *FIXED_SALARIES, *keeper_salary_map}
        missing = sorted(required - set(sim.player_data.player))
        if missing:
            raise ValueError("Required players missing from app database: " + ", ".join(missing))

        waiver_baselines = sim.estimate_waiver_baselines(
            num_teams=NUM_TEAMS,
            roster_size=ROSTER_SIZE,
        )
        candidate_row = sim.player_data.loc[
            sim.player_data.player.eq(CANDIDATE)
        ].iloc[0]
        candidate_price = int(round(float(candidate_row.salary)))
        to_add = {
            "players": list(FIXED_SALARIES),
            "salaries": list(FIXED_SALARIES.values()),
        }
        to_drop = sorted(set(keeper_salary_map) - set(FIXED_SALARIES))
        nonkeeper_fixed_salary = sum(
            salary
            for player, salary in FIXED_SALARIES.items()
            if player not in keeper_salary_map
        )
        remaining_market_budget = float(
            NUM_TEAMS * SALARY_CAP
            - sum(keeper_salary_map.values())
            - nonkeeper_fixed_salary
        )
        remaining_market_slots = int(
            NUM_TEAMS * ROSTER_SIZE
            - len(keeper_salary_map)
            - sum(player not in keeper_salary_map for player in FIXED_SALARIES)
        )

        player_inputs = sim.player_data.loc[
            sim.player_data.player.isin(
                [
                    CANDIDATE,
                    *FIXED_SALARIES,
                    "Brock Purdy",
                    "Makai Lemon",
                    "KC Concepcion",
                ]
            ),
            [
                "player",
                "pos",
                "pred_fp_per_game",
                "salary",
                "selection_premium",
            ],
        ].sort_values(["pos", "salary"], ascending=[True, False])
        player_inputs.to_csv(RESULTS_DIR / "player_inputs.csv", index=False)
        keepers.to_csv(RESULTS_DIR / "league_keepers.csv", index=False)

        top_players = (
            sim.player_data.loc[~sim.player_data.player.isin(to_drop)]
            .sort_values(["salary", "player"], ascending=[False, True])
            .head(REQUIRE_TOP_N)
            [["player", "pos", "salary"]]
        )
        top_players["owned_before_bijan"] = top_players.player.isin(FIXED_SALARIES)
        top_players.to_csv(RESULTS_DIR / "top_n_available_branch.csv", index=False)
        if CANDIDATE not in set(sim.player_data.player):
            raise AssertionError("Bijan is not in the candidate pool.")
        if "Jahmyr Gibbs" not in set(top_players.player):
            raise AssertionError("Gibbs does not satisfy the reconstructed Top-N constraint.")

        state = {
            "year": YEAR,
            "league": LEAGUE,
            "database": str(APP_DB),
            "fixed_salaries": FIXED_SALARIES,
            "fixed_spend": sum(FIXED_SALARIES.values()),
            "candidate": CANDIDATE,
            "candidate_price": candidate_price,
            "personal_salary_after_candidate": (
                SALARY_CAP - sum(FIXED_SALARIES.values()) - candidate_price
            ),
            "personal_open_slots_after_candidate": (
                ROSTER_SIZE - len(FIXED_SALARIES) - 1
            ),
            "keeper_count": len(keeper_salary_map),
            "keeper_spend": sum(keeper_salary_map.values()),
            "owned_keeper_players": sorted(owned_keeper_players),
            "nonkeeper_fixed_spend": nonkeeper_fixed_salary,
            "remaining_market_budget": remaining_market_budget,
            "remaining_market_slots": remaining_market_slots,
            "waiver_baselines": waiver_baselines,
            "lineup_require": LINEUP_REQUIRE,
            "position_minimums": POS_MIN,
            "position_maximums": POS_MAX,
            "compute_budget": COMPUTE_BUDGET,
            "top_n": REQUIRE_TOP_N,
            "gibbs_satisfies_top_n": True,
        }
        with (RESULTS_DIR / "state.json").open("w", encoding="utf-8") as handle:
            json.dump(json_value(state), handle, indent=2, sort_keys=True)

        primary_variation14_seed = evidence_seed(
            sim,
            waiver_baselines,
            enforce_top_n=True,
            use_selection_premium=False,
            variation=14,
            pos_min_counts=POS_MIN,
        )
        cases = [
            {
                "label": "variation14_top_on_reserve_off",
                "variation": 14,
                "enforce_top_n": True,
                "use_selection_premium": False,
                "profile_bid": True,
                "capture_paths": True,
            },
            {
                "label": "variation14_top_off_reserve_off",
                "variation": 14,
                "enforce_top_n": False,
                "use_selection_premium": False,
                "profile_bid": False,
            },
            {
                "label": "variation0_top_on_reserve_off",
                "variation": 0,
                "enforce_top_n": True,
                "use_selection_premium": False,
                "profile_bid": False,
            },
            {
                "label": "variation14_top_on_reserve_on",
                "variation": 14,
                "enforce_top_n": True,
                "use_selection_premium": True,
                "profile_bid": False,
            },
            {
                "label": "variation14_relaxed_rb_min",
                "variation": 14,
                "enforce_top_n": True,
                "use_selection_premium": False,
                "profile_bid": False,
                "pos_min_counts": RELAXED_RB_POS_MIN,
            },
            {
                "label": "variation14_relaxed_rb_min_same_bank",
                "variation": 14,
                "enforce_top_n": True,
                "use_selection_premium": False,
                "profile_bid": False,
                "pos_min_counts": RELAXED_RB_POS_MIN,
                "seed_override": primary_variation14_seed,
            },
        ]
        if variation is not None:
            dynamic_label = f"variation{int(variation)}_top_on_reserve_off_extra"
            cases = [
                {
                    "label": dynamic_label,
                    "variation": int(variation),
                    "enforce_top_n": True,
                    "use_selection_premium": False,
                    "profile_bid": False,
                }
            ]
            case_label = dynamic_label
        elif case_label is not None:
            cases = [case for case in cases if case["label"] == case_label]
            if not cases:
                raise ValueError(f"Unknown case label: {case_label}")
        if not manual_only:
            rows = []
            for index, case in enumerate(cases, start=1):
                print(f"{index}/{len(cases)} {case['label']}", flush=True)
                result = run_case(
                    sim,
                    to_add=to_add,
                    to_drop=to_drop,
                    remaining_market_budget=remaining_market_budget,
                    remaining_market_slots=remaining_market_slots,
                    waiver_baselines=waiver_baselines,
                    candidate_price=candidate_price,
                    **case,
                )
                rows.append(
                    {
                        "case": result["case"],
                        "variation": result["variation"],
                        "enforce_top_n": result["enforce_top_n"],
                        "use_selection_premium": result["use_selection_premium"],
                        "position_minimums": json.dumps(
                            result["position_minimums"],
                            sort_keys=True,
                        ),
                        "price": result.get("Price"),
                        "recommendation": result.get("recommendation"),
                        "sequential_gain": result.get("SequentialGain"),
                        "sequential_se": result.get("SequentialSE"),
                        "lcb80": result.get("SequentialLCB80"),
                        "buy_ev": result.get("BuyEV"),
                        "pass_ev": result.get("PassEV"),
                        "buy_completion": result.get("BuyCompletion"),
                        "pass_completion": result.get("PassCompletion"),
                        "paired_rate": result.get("PairedRate"),
                        "block_positive_rate": result.get("BlockPositiveRate"),
                        "block_gain_min": result.get("BlockGainMin"),
                        "block_gain_max": result.get("BlockGainMax"),
                        "common_fallback": result.get("CommonFallback"),
                        "top_alternatives": " | ".join(
                            result.get("top_alternatives", [])
                        ),
                        "buy_completion_core": result.get("BuyCompletionCore"),
                        "policy_max_bid": result.get("policy_max_bid"),
                        "risk_neutral_bid": result.get("risk_neutral_bid"),
                        "runtime_seconds": result.get("runtime_seconds_external"),
                    }
                )
                print(
                    f"  gain={result.get('SequentialGain'):+.2f} "
                    f"LCB80={result.get('SequentialLCB80'):+.2f} "
                    f"decision={result.get('recommendation')}",
                    flush=True,
                )
            summary_name = (
                "decision_summary.csv"
                if case_label is None
                else f"decision_summary_{case_label}.csv"
            )
            pd.DataFrame(rows).to_csv(
                RESULTS_DIR / summary_name,
                index=False,
            )
        write_standard_variation_summary()
        if not skip_manual:
            manual_summary = run_manual_completion_counterfactuals(
                sim,
                to_drop=to_drop,
                remaining_market_budget=remaining_market_budget,
                remaining_market_slots=remaining_market_slots,
                waiver_baselines=waiver_baselines,
                candidate_price=candidate_price,
            )
            print("\nManual central-completion comparison", flush=True)
            print(manual_summary.to_string(index=False), flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--variation", type=int)
    parser.add_argument("--skip-manual", action="store_true")
    arguments = parser.parse_args()
    main(
        manual_only=arguments.manual_only,
        case_label=arguments.case,
        variation=arguments.variation,
        skip_manual=arguments.skip_manual,
    )
