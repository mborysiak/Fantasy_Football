"""Approved season-specific entry points for the production refresh.

The generic data and publication code can follow ``current_season`` from the
environment, but the accepted validation runners are annual, frozen research
artifacts.  Registering a cycle here makes that distinction explicit: changing
``--year`` selects a reviewed set of runners and table contracts instead of
silently reusing the prior season's validation evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_LEAGUES = ("dk", "nffc", "beta")


@dataclass(frozen=True)
class ProductionCycle:
    """Immutable contract for one current/next-season production release."""

    current_season: int
    status: str
    locked_runner: Path
    next_year_runner: Path
    template_audit_runner: Path
    current_shadow_table: str
    next_shadow_table: str
    locked_versions: Mapping[str, str]
    next_target_version: str
    source_market_minimums: Mapping[str, int]
    nffc_source_feed_minimums: Mapping[str, int]
    nffc_source_feed_pick_boundaries: Mapping[str, int]
    model_input_position_minimums: Mapping[str, int]
    production_population_minimums: Mapping[str, int]
    production_position_minimums: Mapping[str, Mapping[str, int]]
    weekly_horizons: Mapping[str, int]
    template_min_seasons: Mapping[str, int]
    template_center_policies: Mapping[str, tuple[str, ...]]
    template_context_sources: Mapping[str, str]
    leagues: tuple[str, ...] = PRODUCTION_LEAGUES

    @property
    def next_season(self) -> int:
        return self.current_season + 1

    def receipt(self) -> dict[str, Any]:
        return {
            "current_season": self.current_season,
            "status": self.status,
            "next_season": self.next_season,
            "leagues": list(self.leagues),
            "locked_runner": str(self.locked_runner.resolve()),
            "next_year_runner": str(self.next_year_runner.resolve()),
            "template_audit_runner": str(
                self.template_audit_runner.resolve()
            ),
            "current_shadow_table": self.current_shadow_table,
            "next_shadow_table": self.next_shadow_table,
            "locked_versions": dict(self.locked_versions),
            "next_target_version": self.next_target_version,
            "source_market_minimums": dict(self.source_market_minimums),
            "nffc_source_feed_minimums": dict(
                self.nffc_source_feed_minimums
            ),
            "nffc_source_feed_pick_boundaries": dict(
                self.nffc_source_feed_pick_boundaries
            ),
            "model_input_position_minimums": dict(
                self.model_input_position_minimums
            ),
            "production_population_minimums": dict(
                self.production_population_minimums
            ),
            "production_position_minimums": dict(
                (
                    league,
                    dict(minimums),
                )
                for league, minimums in (
                    self.production_position_minimums.items()
                )
            ),
            "weekly_horizons": dict(self.weekly_horizons),
            "template_min_seasons": dict(self.template_min_seasons),
            "template_center_policies": {
                league: list(policies)
                for league, policies in self.template_center_policies.items()
            },
            "template_context_sources": dict(
                self.template_context_sources
            ),
        }

    def contract_sha256(self) -> str:
        payload = json.dumps(
            self.receipt(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


_LOCKED_2026_STUDY = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-29_v2_locked_final_validation"
)
_NEXT_2027_STUDY = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-29_v2_next_year_residual"
)

APPROVED_PRODUCTION_CYCLES: Mapping[int, ProductionCycle] = {
    2026: ProductionCycle(
        current_season=2026,
        status="approved",
        locked_runner=_LOCKED_2026_STUDY / "run_validation.py",
        next_year_runner=_NEXT_2027_STUDY / "run_validation.py",
        template_audit_runner=_LOCKED_2026_STUDY
        / "audit_template_handoff.py",
        current_shadow_table="locked_2026_shadow_predictions",
        next_shadow_table="next_year_2027_shadow_predictions",
        locked_versions={
            "dk": "v2_conditional_ppg_2026_candidate_v1",
            "nffc": "v2_conditional_ppg_2026_candidate_nffc_v1",
            "beta": "v2_conditional_ppg_2026_candidate_beta_v1",
        },
        next_target_version="v2_next_year_expert_residual_v1",
        source_market_minimums={
            "dk": 300,
            "nffc": 360,
            "etr": 180,
        },
        nffc_source_feed_minimums={
            "nffc_best_ball_overall": 400,
            "nffc_best_ball_25s50s": 400,
        },
        nffc_source_feed_pick_boundaries={
            "nffc_best_ball_overall": 360,
            "nffc_best_ball_25s50s": 360,
        },
        model_input_position_minimums={
            "QB": 40,
            "RB": 75,
            "WR": 100,
            "TE": 35,
        },
        production_population_minimums={
            "dk": 300,
            "nffc": 360,
            "beta": 300,
        },
        production_position_minimums={
            "dk": {
                "QB": 40,
                "RB": 75,
                "WR": 105,
                "TE": 38,
            },
            "nffc": {
                "QB": 50,
                "RB": 100,
                "WR": 130,
                "TE": 50,
            },
            "beta": {
                "QB": 40,
                "RB": 75,
                "WR": 105,
                "TE": 38,
            },
        },
        weekly_horizons={
            "dk": 16,
            "nffc": 17,
            "beta": 16,
        },
        template_min_seasons={
            "dk": 2008,
            "nffc": 2021,
            "beta": 2008,
        },
        template_center_policies={
            "dk": (
                "legacy_validated_oos",
                "preseason_projection_fallback",
            ),
            "nffc": ("nffc_scored_expert_consensus",),
            "beta": (
                "legacy_validated_oos",
                "beta_scored_expert_fallback",
            ),
        },
        template_context_sources={
            "nffc": "v2_nffc_scoring_matched_preseason",
            "beta": "v2_beta_scoring_matched_preseason",
        },
    ),
}
DEFAULT_PRODUCTION_YEAR = max(APPROVED_PRODUCTION_CYCLES)


def get_production_cycle(year: int) -> ProductionCycle:
    """Return the reviewed cycle or fail closed before any build starts."""

    try:
        cycle = APPROVED_PRODUCTION_CYCLES[int(year)]
    except KeyError as error:
        registered = ", ".join(
            str(value) for value in sorted(APPROVED_PRODUCTION_CYCLES)
        )
        raise ValueError(
            f"Season {year} is not an approved production cycle. "
            f"Registered seasons: {registered}. Add the annual locked/current "
            "and next-year validation runners to "
            "Scripts/V2/production_cycle.py before running it."
        ) from error
    for runner in (
        cycle.locked_runner,
        cycle.next_year_runner,
        cycle.template_audit_runner,
    ):
        if not runner.is_file():
            raise FileNotFoundError(
                f"Approved season {year} runner is missing: {runner}"
            )
    if cycle.status != "approved":
        raise ValueError(
            f"Season {year} is registered with status={cycle.status!r}, "
            "not approved for production"
        )
    expected_current = f"locked_{cycle.current_season}_shadow_predictions"
    expected_next = (
        f"next_year_{cycle.next_season}_shadow_predictions"
    )
    if cycle.current_shadow_table != expected_current:
        raise ValueError(
            f"Season {year} current shadow contract is inconsistent: "
            f"{cycle.current_shadow_table!r} != {expected_current!r}"
        )
    if cycle.next_shadow_table != expected_next:
        raise ValueError(
            f"Season {year} next shadow contract is inconsistent: "
            f"{cycle.next_shadow_table!r} != {expected_next!r}"
        )
    if set(cycle.locked_versions) != set(cycle.leagues):
        raise ValueError(
            f"Season {year} lock-version leagues do not match its production "
            "leagues"
        )
    if set(cycle.production_position_minimums) != set(cycle.leagues):
        raise ValueError(
            f"Season {year} production-floor leagues do not match its "
            "production leagues"
        )
    if set(cycle.production_population_minimums) != set(cycle.leagues):
        raise ValueError(
            f"Season {year} production-population-floor leagues do not match "
            "its production leagues"
        )
    if (
        "nffc" in cycle.leagues
        and not cycle.nffc_source_feed_minimums
    ):
        raise ValueError(
            f"Season {year} lacks an NFFC source-feed contract"
        )
    invalid_nffc_feed_floors = {
        str(label): floor
        for label, floor in cycle.nffc_source_feed_minimums.items()
        if not isinstance(label, str)
        or not label.strip()
        or isinstance(floor, bool)
        or not isinstance(floor, int)
        or floor <= 0
    }
    if invalid_nffc_feed_floors:
        raise ValueError(
            f"Season {year} has invalid NFFC source-feed floors: "
            f"{invalid_nffc_feed_floors}"
        )
    if set(cycle.nffc_source_feed_pick_boundaries) != set(
        cycle.nffc_source_feed_minimums
    ):
        raise ValueError(
            f"Season {year} NFFC source-feed boundary labels do not match "
            "its row-floor labels"
        )
    invalid_nffc_feed_boundaries = {
        str(label): boundary
        for label, boundary in (
            cycle.nffc_source_feed_pick_boundaries.items()
        )
        if not isinstance(label, str)
        or not label.strip()
        or isinstance(boundary, bool)
        or not isinstance(boundary, int)
        or boundary <= 0
    }
    if invalid_nffc_feed_boundaries:
        raise ValueError(
            f"Season {year} has invalid NFFC source-feed pick boundaries: "
            f"{invalid_nffc_feed_boundaries}"
        )
    for contract_name, contract in (
        ("weekly horizons", cycle.weekly_horizons),
        ("template minimum seasons", cycle.template_min_seasons),
        ("template center policies", cycle.template_center_policies),
    ):
        if set(contract) != set(cycle.leagues):
            raise ValueError(
                f"Season {year} {contract_name} do not match its production "
                "leagues"
            )
    if not set(cycle.template_context_sources).issubset(
        set(cycle.leagues)
    ):
        raise ValueError(
            f"Season {year} template-context-source leagues do not match its "
            "production leagues"
        )
    if any(
        not policies
        for policies in cycle.template_center_policies.values()
    ):
        raise ValueError(
            f"Season {year} has an empty template center policy contract"
        )
    return cycle
