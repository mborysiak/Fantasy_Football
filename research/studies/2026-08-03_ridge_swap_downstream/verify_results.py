"""Verify Ridge swap artifacts, pairing, provenance, and non-mutation receipts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
LEAGUES = ("dk", "beta")
EXPECTED_RUNS = {
    "dk": "v2_locked_final_dk_20260803T041942Z_43a6ddee",
    "beta": "v2_locked_final_beta_20260803T042536Z_b4f0f20f",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    for league in LEAGUES:
        projection_dir = STUDY_DIR / f"results_projection_{league}"
        template_dir = STUDY_DIR / f"results_template_{league}"
        roster_dir = STUDY_DIR / f"results_roster_{league}"
        projection_meta = read_json(projection_dir / "run_metadata.json")
        template_meta = read_json(template_dir / "run_metadata.json")
        roster_meta = read_json(roster_dir / "run_metadata.json")
        check(
            f"{league}_corrected_run",
            projection_meta["source_model_run_id"] == EXPECTED_RUNS[league],
            projection_meta["source_model_run_id"],
        )
        check(
            f"{league}_feature_lock",
            projection_meta["feature_count"] == 40
            and projection_meta["source_feature_run_id"]
            == projection_meta["loaded_feature_run_id"],
            projection_meta["loaded_feature_run_id"],
        )
        check(
            f"{league}_frozen_ridge",
            projection_meta["ridge_alpha"] == 10.0,
            str(projection_meta["ridge_alpha"]),
        )
        check(
            f"{league}_read_only_receipts",
            not projection_meta["production_changed"]
            and not template_meta["production_changed"]
            and not roster_meta["production_changed"],
            "all stages report production_changed=false",
        )

        point = pd.read_csv(projection_dir / "paired_point_predictions.csv")
        key = ["player_key", "season", "position"]
        baseline = point[point.method.eq("production")][key + ["actual"]]
        challenger = point[point.method.eq("ridge_swap")][key + ["actual"]]
        paired = challenger.merge(
            baseline,
            on=key,
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        check(
            f"{league}_point_pairing",
            len(paired) == len(baseline) == len(challenger)
            and np.allclose(
                paired.actual_candidate, paired.actual_baseline, atol=0, rtol=0
            ),
            f"paired_rows={len(paired)}",
        )

        calibrated = pd.read_csv(
            projection_dir / "strict_prior_residuals.csv"
        )
        available = calibrated[calibrated.resid_calibration_available.eq(1)]
        check(
            f"{league}_strict_prior_minimum",
            available.resid_calibration_rows.ge(100).all()
            and available.season.ge(2018).all(),
            f"available_rows={len(available)}",
        )

        target_frames = {
            method: pd.read_csv(template_dir / f"target_rows_{method}.csv")
            for method in ("production", "ridge_swap")
        }
        target_keys = ["player_key", "season", "pos"]
        check(
            f"{league}_target_pairing",
            target_frames["production"][target_keys].equals(
                target_frames["ridge_swap"][target_keys]
            ),
            f"target_rows={len(target_frames['production'])}",
        )

        roster = pd.read_csv(roster_dir / "roster_predictions.csv")
        roster_key = ["season", "room", "team", "roster_id"]
        roster_baseline = roster[roster.method.eq("production")][roster_key]
        roster_challenger = roster[roster.method.eq("ridge_swap")][roster_key]
        check(
            f"{league}_roster_pairing",
            roster_baseline.reset_index(drop=True).equals(
                roster_challenger.reset_index(drop=True)
            ),
            f"paired_rosters={len(roster_baseline)}",
        )
        expected_seasons = list(range(2018, 2026)) if league == "dk" else list(range(2019, 2026))
        check(
            f"{league}_roster_origins",
            sorted(roster.season.unique().tolist()) == expected_seasons,
            str(sorted(roster.season.unique().tolist())),
        )
        check(
            f"{league}_scenario_contract",
            roster_meta["scenarios"] == 384
            and roster_meta["rooms_per_origin"] == 12
            and roster_meta["teams_per_room"] == 12
            and roster_meta["roster_size"] == 20,
            "384 scenarios; 12x12 rooms; 20-player rosters",
        )

    gates = pd.read_csv(STUDY_DIR / "results" / "gate_summary.csv")
    decision = read_json(STUDY_DIR / "results" / "decision.json")
    check(
        "decision_matches_gates",
        decision["promote_ridge_swap"] == bool(gates.passed.all())
        and set(decision["failed_gates"])
        == set(gates.loc[~gates.passed, "gate"]),
        str(decision),
    )
    receipt = {
        "checks": checks,
        "check_count": len(checks),
        "all_passed": all(row["passed"] for row in checks),
        "production_changed": False,
    }
    results_dir = STUDY_DIR / "results"
    (results_dir / "verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
