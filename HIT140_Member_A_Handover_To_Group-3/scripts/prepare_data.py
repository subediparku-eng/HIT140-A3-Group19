"""Create the 104-row and 208-row regression datasets.

All explanatory variables are calculated from information available before the
relevant match. The travel variable is an explicitly documented approximation:
group-stage travel starts at the official team base-camp city; knockout travel
starts at the team's previous match venue.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

TOURNAMENT_START = datetime(2026, 6, 11, tzinfo=timezone.utc)
GROUP_STAGE_ID = "289273"
HOSTS = {"CAN", "MEX", "USA"}


# Approximate city-centre coordinates. The city assignments come from FIFA's
# official Team Base Camps article; coordinates are fixed analyst inputs used
# solely for the Haversine distance transformation.
BASE_CAMPS = {
    "ALG": ("Kansas City", "USA", 39.0997, -94.5786),
    "ARG": ("Kansas City", "USA", 39.0997, -94.5786),
    "AUS": ("San Francisco Bay Area", "USA", 37.3541, -121.9552),
    "AUT": ("Goleta", "USA", 34.4358, -119.8276),
    "BEL": ("Renton", "USA", 47.4829, -122.2171),
    "BIH": ("Sandy", "USA", 40.5649, -111.8389),
    "BRA": ("New York New Jersey", "USA", 40.7128, -74.0060),
    "CAN": ("Vancouver", "CAN", 49.2827, -123.1207),
    "CIV": ("Philadelphia", "USA", 39.9526, -75.1652),
    "COD": ("Houston", "USA", 29.7604, -95.3698),
    "COL": ("Guadalajara", "MEX", 20.6597, -103.3496),
    "CPV": ("Tampa", "USA", 27.9506, -82.4572),
    "CRO": ("Alexandria", "USA", 38.8048, -77.0469),
    "CUW": ("Boca Raton", "USA", 26.3683, -80.1289),
    "CZE": ("Dallas", "USA", 32.7767, -96.7970),
    "ECU": ("Columbus", "USA", 39.9612, -82.9988),
    "EGY": ("Spokane", "USA", 47.6588, -117.4260),
    "ENG": ("Kansas City", "USA", 39.0997, -94.5786),
    "ESP": ("Chattanooga", "USA", 35.0456, -85.3097),
    "FRA": ("Boston", "USA", 42.3601, -71.0589),
    "GER": ("Winston-Salem", "USA", 36.0999, -80.2442),
    "GHA": ("Boston", "USA", 42.3601, -71.0589),
    "HAI": ("New York New Jersey", "USA", 40.7128, -74.0060),
    "IRN": ("Tijuana", "MEX", 32.5149, -117.0382),
    "IRQ": ("Greenbrier County", "USA", 37.8018, -80.4456),
    "JOR": ("Portland", "USA", 45.5152, -122.6784),
    "JPN": ("Nashville", "USA", 36.1627, -86.7816),
    "KOR": ("Guadalajara", "MEX", 20.6597, -103.3496),
    "KSA": ("Austin", "USA", 30.2672, -97.7431),
    "MAR": ("New York New Jersey", "USA", 40.7128, -74.0060),
    "MEX": ("Mexico City", "MEX", 19.4326, -99.1332),
    "NED": ("Kansas City", "USA", 39.0997, -94.5786),
    "NOR": ("Greensboro", "USA", 36.0726, -79.7920),
    "NZL": ("San Diego", "USA", 32.7157, -117.1611),
    "PAN": ("New Tecumseth", "CAN", 44.0834, -79.7549),
    "PAR": ("San Francisco Bay Area", "USA", 37.3541, -121.9552),
    "POR": ("Palm Beach Gardens", "USA", 26.8234, -80.1387),
    "QAT": ("Santa Barbara", "USA", 34.4208, -119.6982),
    "RSA": ("Pachuca", "MEX", 20.1011, -98.7591),
    "SCO": ("Charlotte", "USA", 35.2271, -80.8431),
    "SEN": ("New York New Jersey", "USA", 40.7128, -74.0060),
    "SUI": ("San Diego", "USA", 32.7157, -117.1611),
    "SWE": ("Dallas", "USA", 32.7767, -96.7970),
    "TUN": ("Monterrey", "MEX", 25.6866, -100.3161),
    "TUR": ("Mesa", "USA", 33.4152, -111.8315),
    "URU": ("Cancun", "MEX", 21.1619, -86.8515),
    "USA": ("Irvine", "USA", 33.6846, -117.8265),
    "UZB": ("Atlanta", "USA", 33.7490, -84.3880),
}


# Approximate stadium coordinates keyed by FIFA stadium ID.
VENUE_COORDS = {
    "400098290": ("Atlanta", "USA", 33.7554, -84.4008),
    "400248623": ("Boston", "USA", 42.0909, -71.2643),
    "400257526": ("Dallas", "USA", 32.7473, -97.0945),
    "400252150": ("Guadalajara", "MEX", 20.6818, -103.4620),
    "400249385": ("Houston", "USA", 29.6847, -95.4107),
    "400254717": ("Kansas City", "USA", 39.0489, -94.4839),
    "400017978": ("Los Angeles", "USA", 33.9535, -118.3390),
    "400222084": ("Mexico City", "MEX", 19.3029, -99.1505),
    "400257525": ("Miami", "USA", 25.9580, -80.2389),
    "400238450": ("Monterrey", "MEX", 25.6694, -100.2443),
    "400257536": ("New Jersey", "USA", 40.8135, -74.0745),
    "400248622": ("Philadelphia", "USA", 39.9008, -75.1675),
    "400257521": ("San Francisco Bay Area", "USA", 37.4030, -121.9700),
    "400216606": ("Seattle", "USA", 47.5952, -122.3316),
    "400242032": ("Toronto", "CAN", 43.6332, -79.4186),
    "400248370": ("Vancouver", "CAN", 49.2767, -123.1119),
}


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def team_score(match: dict, country_code: str) -> tuple[int, int]:
    if match["Home"]["IdCountry"] == country_code:
        return int(match["HomeTeamScore"]), int(match["AwayTeamScore"])
    if match["Away"]["IdCountry"] == country_code:
        return int(match["AwayTeamScore"]), int(match["HomeTeamScore"])
    raise ValueError(f"{country_code} not found in match {match['IdMatch']}")


def valid_completed_match(match: dict) -> bool:
    return (
        match.get("Date")
        and match.get("HomeTeamScore") is not None
        and match.get("AwayTeamScore") is not None
        and match.get("Home")
        and match.get("Away")
    )


def history_for_team(country_code: str) -> list[dict]:
    payload = load_json(RAW_DIR / "team_histories" / f"{country_code}.json")
    deduplicated: dict[str, dict] = {}
    for match in payload.get("Results", []):
        if not valid_completed_match(match):
            continue
        match_date = parse_datetime(match["Date"])
        if match_date >= TOURNAMENT_START:
            continue
        if country_code not in {match["Home"]["IdCountry"], match["Away"]["IdCountry"]}:
            continue
        deduplicated[str(match["IdMatch"])] = match
    return sorted(deduplicated.values(), key=lambda item: parse_datetime(item["Date"]))


def pre_world_cup_metrics(country_code: str, history: list[dict]) -> dict[str, float]:
    recent = history[-10:]
    if len(recent) < 10:
        raise ValueError(f"{country_code} has only {len(recent)} valid pre-tournament matches; 10 required")
    points = 0
    clean_sheets = 0
    for match in recent:
        goals_for, goals_against = team_score(match, country_code)
        points += 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
        clean_sheets += int(goals_against == 0)
    return {
        "pre_wc_ppg": points / len(recent),
        "pre_wc_clean_sheet_rate": clean_sheets / len(recent),
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    world_cup = load_json(RAW_DIR / "fifa_world_cup_2026_matches.json")["Results"]
    rankings_raw = load_json(RAW_DIR / "fifa_rankings_2026-06-11.json")["Results"]
    world_cup = sorted(world_cup, key=lambda item: parse_datetime(item["Date"]))

    if len(world_cup) != 104:
        raise ValueError(f"Expected 104 World Cup matches, found {len(world_cup)}")

    teams: dict[str, dict] = {}
    for match in world_cup:
        for side in ("Home", "Away"):
            team = match[side]
            teams[team["IdCountry"]] = {
                "team_id": team["IdTeam"],
                "team_name": team["ShortClubName"],
            }
    if len(teams) != 48:
        raise ValueError(f"Expected 48 teams, found {len(teams)}")
    if set(teams) != set(BASE_CAMPS):
        raise ValueError(f"Base camp mapping mismatch: {set(teams) ^ set(BASE_CAMPS)}")

    ranking_points = {row["IdCountry"]: float(row["DecimalTotalPoints"]) for row in rankings_raw}
    ranking_positions = {row["IdCountry"]: int(row["Rank"]) for row in rankings_raw}
    missing_rankings = set(teams) - set(ranking_points)
    if missing_rankings:
        raise ValueError(f"Missing FIFA ranking points: {sorted(missing_rankings)}")

    histories = {code: history_for_team(code) for code in teams}
    pre_metrics = {code: pre_world_cup_metrics(code, history) for code, history in histories.items()}

    base_camp_df = pd.DataFrame(
        [
            {
                "country_code": code,
                "team": teams[code]["team_name"],
                "base_camp_city": values[0],
                "base_camp_country": values[1],
                "latitude_approx": values[2],
                "longitude_approx": values[3],
            }
            for code, values in sorted(BASE_CAMPS.items())
        ]
    )
    stadium_names = {
        str(match["Stadium"]["IdStadium"]): match["Stadium"]["Name"][0]["Description"]
        for match in world_cup
    }
    venue_df = pd.DataFrame(
        [
            {
                "stadium_id": stadium_id,
                "stadium": stadium_names[stadium_id],
                "host_city": values[0],
                "host_country": values[1],
                "latitude_approx": values[2],
                "longitude_approx": values[3],
            }
            for stadium_id, values in sorted(VENUE_COORDS.items())
        ]
    )
    team_metric_df = pd.DataFrame(
        [
            {
                "country_code": code,
                "team_id": teams[code]["team_id"],
                "team": teams[code]["team_name"],
                "fifa_rank_2026_06_11": ranking_positions[code],
                "fifa_points_2026_06_11": ranking_points[code],
                "pre_wc_last10_ppg": pre_metrics[code]["pre_wc_ppg"],
                "pre_wc_last10_clean_sheet_rate": pre_metrics[code]["pre_wc_clean_sheet_rate"],
                "last_pre_wc_match_utc": histories[code][-1]["Date"],
                "valid_history_records_since_2024": len(histories[code]),
            }
            for code in sorted(teams)
        ]
    )

    prior_dates = {
        code: [parse_datetime(match["Date"]) for match in history]
        for code, history in histories.items()
    }
    previous_match_date = {code: dates[-1] for code, dates in prior_dates.items()}
    cumulative_extra_time = defaultdict(int)
    previous_venue: dict[str, tuple[str, str, float, float]] = {}
    venue_counts: dict[str, defaultdict[str, int]] = {
        code: defaultdict(int) for code in teams
    }

    match_rows: list[dict] = []
    team_rows: list[dict] = []

    for match in world_cup:
        match_dt = parse_datetime(match["Date"])
        local_hour = int(match["LocalDate"][11:13]) + int(match["LocalDate"][14:16]) / 60
        home_code = match["Home"]["IdCountry"]
        away_code = match["Away"]["IdCountry"]
        stadium_id = str(match["Stadium"]["IdStadium"])
        if stadium_id not in VENUE_COORDS:
            raise ValueError(f"Missing venue coordinates for {stadium_id}")
        venue_city, venue_country, venue_lat, venue_lon = VENUE_COORDS[stadium_id]
        knockout = int(str(match["IdStage"]) != GROUP_STAGE_ID)

        feature_by_team: dict[str, dict] = {}
        for code, opponent_code in ((home_code, away_code), (away_code, home_code)):
            rest_days_raw = (match_dt - previous_match_date[code]).total_seconds() / 86400
            # Recovery effects should not be assumed to increase linearly forever.
            # Cap at 14 days to limit leverage from teams whose last friendly was
            # several weeks before the tournament.
            rest_days = min(rest_days_raw, 14.0)
            recent_count = sum(
                match_dt - timedelta(days=30) <= date < match_dt for date in prior_dates[code]
            )

            if knockout and code in previous_venue:
                origin_city, origin_country, origin_lat, origin_lon = previous_venue[code]
            else:
                origin_city, origin_country, origin_lat, origin_lon = BASE_CAMPS[code]
            travel_km = haversine_km(origin_lat, origin_lon, venue_lat, venue_lon)

            features = {
                "fifa_points": ranking_points[code],
                "rest_days": rest_days,
                "travel_km": travel_km,
                "cum_extra_time_min": cumulative_extra_time[code],
                "pre_wc_ppg": pre_metrics[code]["pre_wc_ppg"],
                "pre_wc_clean_sheet_rate": pre_metrics[code]["pre_wc_clean_sheet_rate"],
                "matches_last30d": recent_count,
                "host_status": int(code in HOSTS),
                "prior_matches_same_venue": venue_counts[code][stadium_id],
                "cross_border_travel": int(origin_country != venue_country),
                "origin_city": origin_city,
            }
            feature_by_team[code] = features

            team_goals, _ = team_score(match, code)
            team_rows.append(
                {
                    "match_id": str(match["IdMatch"]),
                    "team_id": teams[code]["team_id"],
                    "opponent_id": teams[opponent_code]["team_id"],
                    "team": teams[code]["team_name"],
                    "opponent": teams[opponent_code]["team_name"],
                    "fifa_points_gap": features["fifa_points"] - ranking_points[opponent_code],
                    "team_rest_days_capped14": features["rest_days"],
                    "team_travel_km": features["travel_km"],
                    "team_cum_extra_time_min": features["cum_extra_time_min"],
                    "prior_matches_same_venue": features["prior_matches_same_venue"],
                    "local_kickoff_hour": local_hour,
                    "cross_border_travel": features["cross_border_travel"],
                    "knockout_stage": knockout,
                    "team_goals": team_goals,
                }
            )

        home_features = feature_by_team[home_code]
        away_features = feature_by_team[away_code]
        match_rows.append(
            {
                "match_id": str(match["IdMatch"]),
                "team_a": teams[home_code]["team_name"],
                "team_b": teams[away_code]["team_name"],
                "fifa_points_diff": home_features["fifa_points"] - away_features["fifa_points"],
                "rest_days_capped14_diff": home_features["rest_days"] - away_features["rest_days"],
                "travel_km_diff": home_features["travel_km"] - away_features["travel_km"],
                "cum_extra_time_min_diff": home_features["cum_extra_time_min"]
                - away_features["cum_extra_time_min"],
                "pre_wc_ppg_diff": home_features["pre_wc_ppg"] - away_features["pre_wc_ppg"],
                "pre_wc_clean_sheet_rate_diff": home_features["pre_wc_clean_sheet_rate"]
                - away_features["pre_wc_clean_sheet_rate"],
                "matches_last30d_diff": home_features["matches_last30d"]
                - away_features["matches_last30d"],
                "host_status_diff": home_features["host_status"] - away_features["host_status"],
                "goal_difference": int(match["HomeTeamScore"]) - int(match["AwayTeamScore"]),
            }
        )

        went_to_extra_time = int(match.get("ResultType") in (2, 3))
        for code in (home_code, away_code):
            cumulative_extra_time[code] += 30 * went_to_extra_time
            previous_match_date[code] = match_dt
            prior_dates[code].append(match_dt)
            previous_venue[code] = (venue_city, venue_country, venue_lat, venue_lon)
            venue_counts[code][stadium_id] += 1

    match_df = pd.DataFrame(match_rows)
    team_df = pd.DataFrame(team_rows)
    match_registry_df = pd.DataFrame(
        [
            {
                "match_id": str(match["IdMatch"]),
                "match_number": int(match["MatchNumber"]),
                "date_utc": match["Date"],
                "local_date_time": match["LocalDate"],
                "stage": match["StageName"][0]["Description"] if match["StageName"] else "",
                "group": match["GroupName"][0]["Description"] if match["GroupName"] else "",
                "stadium_id": str(match["Stadium"]["IdStadium"]),
                "stadium": match["Stadium"]["Name"][0]["Description"],
                "host_city": match["Stadium"]["CityName"][0]["Description"],
                "host_country": match["Stadium"]["IdCountry"],
                "team_a_code": match["Home"]["IdCountry"],
                "team_a": match["Home"]["ShortClubName"],
                "team_b_code": match["Away"]["IdCountry"],
                "team_b": match["Away"]["ShortClubName"],
                "team_a_goals": int(match["HomeTeamScore"]),
                "team_b_goals": int(match["AwayTeamScore"]),
                "team_a_penalties": match.get("HomeTeamPenaltyScore"),
                "team_b_penalties": match.get("AwayTeamPenaltyScore"),
                "result_type": int(match["ResultType"]),
                "went_to_extra_time": int(match.get("ResultType") in (2, 3)),
                "attendance": int(match["Attendance"]),
            }
            for match in world_cup
        ]
    ).sort_values("match_number")

    match_df[MATCH_PREDICTORS] = match_df[MATCH_PREDICTORS].round(4)
    team_df[TEAM_PREDICTORS] = team_df[TEAM_PREDICTORS].round(4)

    checks = {
        "match_rows": len(match_df),
        "team_match_rows": len(team_df),
        "unique_matches_match_level": int(match_df["match_id"].nunique()),
        "unique_matches_team_level": int(team_df["match_id"].nunique()),
        "unique_teams": int(team_df["team"].nunique()),
        "match_predictor_count": len(MATCH_PREDICTORS),
        "team_predictor_count": len(TEAM_PREDICTORS),
        "match_missing_cells": int(match_df[MATCH_PREDICTORS + ["goal_difference"]].isna().sum().sum()),
        "team_missing_cells": int(team_df[TEAM_PREDICTORS + ["team_goals"]].isna().sum().sum()),
        "penalty_shootout_matches": int(sum(match.get("HomeTeamPenaltyScore") is not None for match in world_cup)),
        "extra_time_matches": int(sum(match.get("ResultType") in (2, 3) for match in world_cup)),
        "validation_status": "PASS",
    }

    required = {
        "match_rows": 104,
        "team_match_rows": 208,
        "unique_matches_match_level": 104,
        "unique_matches_team_level": 104,
        "unique_teams": 48,
        "match_predictor_count": 8,
        "team_predictor_count": 8,
        "match_missing_cells": 0,
        "team_missing_cells": 0,
    }
    for key, expected in required.items():
        if checks[key] != expected:
            checks["validation_status"] = "FAIL"
            raise ValueError(f"Validation failed for {key}: {checks[key]} != {expected}")

    match_df.to_csv(PROCESSED_DIR / "match_level_104.csv", index=False)
    team_df.to_csv(PROCESSED_DIR / "team_match_208.csv", index=False)
    match_registry_df.to_csv(PROCESSED_DIR / "match_registry_104.csv", index=False)
    base_camp_df.to_csv(PROCESSED_DIR / "base_camps_48.csv", index=False)
    venue_df.to_csv(PROCESSED_DIR / "venues_16.csv", index=False)
    team_metric_df.to_csv(PROCESSED_DIR / "pre_world_cup_team_metrics_48.csv", index=False)
    (RESULTS_DIR / "data_quality.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "predictor_lists.json").write_text(
        json.dumps(
            {"linear_regression_2_1": MATCH_PREDICTORS, "linear_regression_2_2": TEAM_PREDICTORS},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
