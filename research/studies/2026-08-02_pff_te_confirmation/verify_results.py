"""Focused integrity verification for the PFF TE confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent


def main() -> None:
    for league in ("dk", "beta"):
        projection = pd.read_csv(
            STUDY_DIR / f"results_projection_{league}" / "ppg_summary.csv"
        )
        template_meta = json.loads(
            (
                STUDY_DIR
                / f"results_template_{league}"
                / "run_metadata.json"
            ).read_text(encoding="utf-8")
        )
        roster_meta = json.loads(
            (
                STUDY_DIR
                / f"results_roster_{league}"
                / "run_metadata.json"
            ).read_text(encoding="utf-8")
        )
        primary = projection[
            projection["scope"].eq("te")
            & projection["method"].eq("te_pff_mtf__te_route")
            & projection["period"].isin(
                ["development_2017_2022", "temporal_2023_2025"]
            )
        ]
        assert len(primary) == 2
        assert primary["rmse_delta"].lt(0).all()
        assert template_meta["prediction_rows"] == 10_588
        assert template_meta["non_te_parity_max_abs"] == 0
        assert roster_meta["prediction_rows"] == 2_592
        assert template_meta["production_changed"] is False
        assert roster_meta["production_changed"] is False

    profiles = pd.read_csv(
        STUDY_DIR / "results_projection_dk" / "pff_te_profiles.csv"
    )
    assert profiles["season"].eq(profiles["pff_te_source_season"] + 1).all()
    assert profiles.duplicated(["player_key", "season"]).sum() == 0
    manifest = json.loads(
        (STUDY_DIR / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["template_decision"] == "reject"
    assert manifest["production_changed"] is False
    print("verified projection, template, roster, identity, and lag contracts")


if __name__ == "__main__":
    main()

