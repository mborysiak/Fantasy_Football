"""Verify saved upside-objective replay artifacts."""

from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent


def main() -> None:
    for league in ("dk", "beta"):
        players = pd.read_csv(
            STUDY_DIR / f"results_player_{league}" / "target_predictions.csv"
        )
        assert len(players) == 2_647 * 6
        assert players.groupby(["player", "pos", "season"]).method.nunique().eq(6).all()
        assert (players.threshold_history_end < players.season).all()
        for column in ("prob_league_winner_q90", "prob_league_winner_q95"):
            assert players[column].between(0, 1).all()

        rosters = pd.read_csv(
            STUDY_DIR
            / f"results_roster_{league}"
            / "roster_championship_predictions.csv"
        )
        assert len(rosters) == 108 * 12 * 3
        room_groups = rosters.groupby(["season", "room", "matcher"])
        probability_error = (
            room_groups.championship_probability.sum() - 1.0
        ).abs().max()
        assert probability_error < 1e-9
        assert room_groups.actual_champion.sum().eq(1).all()
        print(
            league,
            f"player_rows={len(players)}",
            f"roster_rows={len(rosters)}",
            f"max_room_probability_error={probability_error:.3g}",
        )

    for artifact in (
        "player_bootstrap.csv",
        "roster_bootstrap.csv",
        "saved_fastr_tail_diagnostics.csv",
        "findings.md",
    ):
        assert (STUDY_DIR / "results" / artifact).is_file()
    print("validation ok")


if __name__ == "__main__":
    main()

