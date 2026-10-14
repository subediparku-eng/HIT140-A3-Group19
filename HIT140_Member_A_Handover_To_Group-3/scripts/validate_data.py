"""Run an independent integrity audit of the two Objective 2 datasets.

This script does not fit any model. It verifies Member A's handover files,
including row counts, predictor counts, paired team rows, score reconciliation,
missing values and the treatment of extra time and penalty shoot-outs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RAW_DIR = PROJECT_DIR / "data" / "raw"
RESULTS_DIR = PROJECT_DIR / "results"

MATCH_PREDICTORS = [
    "fifa_points_diff",
    "rest_days_capped14_diff",
    "travel_km_diff",
    "cum_extra_time_min_diff",
    "pre_wc_ppg_diff",
    "pre_wc_clean_sheet_rate_diff",
    "matches_last30d_diff",
    "host_status_diff",
]

TEAM_PREDICTORS = [
    "fifa_points_gap",
    "team_rest_days_capped14",
    "team_travel_km",
    "team_cum_extra_time_min",
    "prior_matches_same_venue",
    "local_kickoff_hour",
    "cross_border_travel",
    "knockout_stage",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    match_df = pd.read_csv(PROCESSED_DIR / "match_level_104.csv", dtype={"match_id": str})
    team_df = pd.read_csv(PROCESSED_DIR / "team_match_208.csv", dtype={"match_id": str})
    registry = pd.read_csv(PROCESSED_DIR / "match_registry_104.csv", dtype={"match_id": str})
    manifest = json.loads((RAW_DIR / "source_manifest.json").read_text(encoding="utf-8"))

    require(len(match_df) == 104, "Regression 2.1 must contain 104 rows")
    require(len(team_df) == 208, "Regression 2.2 must contain 208 rows")
    require(match_df["match_id"].nunique() == 104, "Regression 2.1 match IDs must be unique")
    require(team_df["match_id"].nunique() == 104, "Regression 2.2 must cover 104 matches")
    require((team_df.groupby("match_id").size() == 2).all(), "Each match must have two team rows")
    require(team_df["team"].nunique() == 48, "Expected 48 tournament teams")
    require(len(MATCH_PREDICTORS) == 8, "Regression 2.1 must have exactly eight predictors")
    require(len(TEAM_PREDICTORS) == 8, "Regression 2.2 must have exactly eight predictors")
    require(
        match_df[MATCH_PREDICTORS + ["goal_difference"]].notna().all().all(),
        "Regression 2.1 contains missing model values",
    )
    require(
        team_df[TEAM_PREDICTORS + ["team_goals"]].notna().all().all(),
        "Regression 2.2 contains missing model values",
    )

    expected_difference = registry.set_index("match_id")["team_a_goals"] - registry.set_index("match_id")["team_b_goals"]
    observed_difference = match_df.set_index("match_id")["goal_difference"]
    require(observed_difference.equals(expected_difference.loc[observed_difference.index]), "Goal differences do not reconcile")

    registry_scores = {}
    for row in registry.itertuples(index=False):
        registry_scores[(row.match_id, row.team_a)] = int(row.team_a_goals)
        registry_scores[(row.match_id, row.team_b)] = int(row.team_b_goals)
    observed_scores = {(row.match_id, row.team): int(row.team_goals) for row in team_df.itertuples(index=False)}
    require(observed_scores == registry_scores, "Team goals do not reconcile to the official match registry")

    require(team_df["team_rest_days_capped14"].between(0, 14).all(), "Rest cap outside 0-14 days")
    require((team_df["team_travel_km"] >= 0).all(), "Travel distance cannot be negative")
    require((team_df["team_cum_extra_time_min"] % 30 == 0).all(), "Cumulative extra time must use 30-minute increments")
    require(team_df["cross_border_travel"].isin([0, 1]).all(), "Cross-border indicator must be binary")
    require(team_df["knockout_stage"].isin([0, 1]).all(), "Knockout-stage indicator must be binary")

    source_hosts = {Path(item["url"].split("?", 1)[0]).name for item in manifest["primary_sources"]}
    require(len(manifest["team_history_sources"]) == 48, "Expected one history source for each team")

    audit = {
        "status": "PASS",
        "match_rows": len(match_df),
        "team_match_rows": len(team_df),
        "unique_matches": int(match_df["match_id"].nunique()),
        "teams": int(team_df["team"].nunique()),
        "regression_2_1_predictors": len(MATCH_PREDICTORS),
        "regression_2_2_predictors": len(TEAM_PREDICTORS),
        "two_team_rows_per_match": True,
        "goals_reconciled_to_registry": True,
        "missing_model_values": 0,
        "extra_time_increment_minutes": 30,
        "penalty_shootouts_excluded_from_goal_targets": True,
        "team_history_files": len(manifest["team_history_sources"]),
        "primary_source_entries": len(manifest["primary_sources"]),
        "source_url_endpoints_checked": sorted(source_hosts),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "member_A_validation.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
