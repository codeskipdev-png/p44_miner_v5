"""Fetch new daily benchmark releases from the public Poker44 API.

Saves files in the exact format of 03_data/benchmark/raw (the raw
{success, data:{...}} payload), with cursor pagination merged into one file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import requests

API = "https://api.poker44.net/api/v1/benchmark"
BENCH_DIR = Path("/root/Skip/poker/SN126/03_data/benchmark/raw")


def _get(url: str, params: dict | None = None, tries: int = 3) -> dict:
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"fetch failed for {url}: {last}")


def latest_source_date() -> Optional[str]:
    payload = _get(API)
    data = payload.get("data", payload) or {}
    return data.get("latestSourceDate")


def fetch_date(source_date: str, *, limit: int = 24) -> dict:
    """Download every chunk for one release date (cursor-paginated)."""
    all_chunks: List[dict] = []
    cursor = None
    base = None
    while True:
        params = {"sourceDate": source_date, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        payload = _get(f"{API}/chunks", params)
        data = payload.get("data", payload) or {}
        if base is None:
            base = {k: v for k, v in data.items() if k != "chunks"}
        all_chunks.extend(data.get("chunks", []) or [])
        cursor = data.get("nextCursor")
        if not cursor:
            break
    merged = dict(base or {})
    merged["chunks"] = all_chunks
    merged["nextCursor"] = None
    return {"success": True, "data": merged}


def sync(*, log=print) -> List[str]:
    """Download any release dates newer than what we have. Returns new dates."""
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    have = {p.stem.replace("benchmark_", "") for p in BENCH_DIR.glob("benchmark_*.json")}
    latest = latest_source_date()
    if not latest:
        log("benchmark status endpoint returned no latestSourceDate")
        return []

    import datetime as dt

    new_dates: List[str] = []
    day = max(have) if have else "2026-05-26"
    cur = dt.date.fromisoformat(day)
    end = dt.date.fromisoformat(latest)
    while cur < end:
        cur += dt.timedelta(days=1)
        date = cur.isoformat()
        if date in have:
            continue
        try:
            payload = fetch_date(date)
        except RuntimeError as e:
            log(f"  {date}: fetch failed ({e}); skipping")
            continue
        n = len(payload["data"].get("chunks", []))
        if n == 0:
            log(f"  {date}: no chunks published; skipping")
            continue
        out = BENCH_DIR / f"benchmark_{date}.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False))
        tmp.replace(out)
        new_dates.append(date)
        log(f"  {date}: saved {n} release chunks ({out.stat().st_size/1e6:.1f} MB)")
    return new_dates


if __name__ == "__main__":
    print("new dates:", sync())
