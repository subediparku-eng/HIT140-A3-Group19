"""Download FIFA World Cup 2026 source data from FIFA's public API.

The script deliberately stores unmodified JSON responses under data/raw so the
processed datasets can be rebuilt and audited. FBref is retained as a validation
source in the source manifest; its Cloudflare protection prevents reliable
automated collection in this environment.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
HISTORY_DIR = RAW_DIR / "team_histories"

FIFA_API = "https://api.fifa.com/api/v3"
SEASON_ID = "285023"
RANKING_SCHEDULE_ID = "id15136"  # Official ranking published 11 June 2026.
TOURNAMENT_START = "2026-06-11T00:00:00Z"


def get_json(url: str, retries: int = 3) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; HIT140 student project; +https://www.fifa.com/)",
            "Accept": "application/json",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network failure path
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Unable to download {url}: {last_error}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    matches_url = f"{FIFA_API}/calendar/matches?language=en&count=500&idSeason={SEASON_ID}"
    rankings_url = (
        f"{FIFA_API}/rankingsbyschedule?rankingScheduleId={RANKING_SCHEDULE_ID}"
        "&language=en&count=500"
    )
    schedules_url = (
        f"{FIFA_API}/rankingschedules/all?type=0&gender=1"
        "&idClient=64e9afa8-c5c0-413d-882b-bc9e6a81e264&language=en&count=500"
    )

    matches = get_json(matches_url)
    rankings = get_json(rankings_url)
    schedules = get_json(schedules_url)
    save_json(RAW_DIR / "fifa_world_cup_2026_matches.json", matches)
    save_json(RAW_DIR / "fifa_rankings_2026-06-11.json", rankings)
    save_json(RAW_DIR / "fifa_ranking_schedules.json", schedules)

    teams: dict[str, tuple[str, str]] = {}
    for match in matches["Results"]:
        for side in ("Home", "Away"):
            team = match[side]
            teams[team["IdCountry"]] = (team["IdTeam"], team["ShortClubName"])

    history_sources = []
    for index, (country_code, (team_id, team_name)) in enumerate(sorted(teams.items()), start=1):
        query = urllib.parse.urlencode(
            {
                "idTeam": team_id,
                "language": "en",
                "count": 500,
                "from": "2024-01-01T00:00:00Z",
                "to": "2026-06-10T23:59:59Z",
            }
        )
        url = f"{FIFA_API}/calendar/matches?{query}"
        payload = get_json(url)
        save_json(HISTORY_DIR / f"{country_code}.json", payload)
        history_sources.append(
            {
                "country_code": country_code,
                "team_id": team_id,
                "team_name": team_name,
                "url": url,
                "records": len(payload.get("Results", [])),
            }
        )
        if index < len(teams):
            time.sleep(0.08)

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_sources": [
            {
                "name": "FIFA World Cup 2026 matches",
                "url": matches_url,
                "purpose": "Official 104-match schedule, scores, stages, venues and result types",
            },
            {
                "name": "FIFA Men's Ranking, 11 June 2026",
                "url": rankings_url,
                "purpose": "Pre-tournament FIFA ranking points",
            },
            {
                "name": "FIFA World Cup 2026 Team Base Camps finalised",
                "url": "https://www.fifa.com/en/articles/team-base-camps-finalised",
                "purpose": "Official team base-camp cities and countries",
            },
            {
                "name": "FBref 2026 World Cup scores and fixtures",
                "url": "https://fbref.com/en/comps/1/2026/schedule/2026-World-Cup-Scores-and-Fixtures",
                "purpose": "Independent validation of dates, teams and final scores",
            },
        ],
        "team_history_sources": history_sources,
        "notes": [
            "FIFA match scores include extra-time goals; penalty shoot-out scores are separate fields.",
            "FBref may require interactive Cloudflare verification and is not used as an automated fallback.",
        ],
    }
    save_json(RAW_DIR / "source_manifest.json", manifest)

    print(f"Downloaded {len(matches['Results'])} World Cup matches and {len(teams)} team histories.")


if __name__ == "__main__":
    main()
