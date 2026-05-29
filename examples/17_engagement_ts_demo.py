"""
Demo 17: Time-Series DB — User Engagement Analytics
====================================================
Tracks user watch sessions and interaction events over 14 days using
SQLite as a lightweight engagement time-series store.

Demonstrates rolling-window aggregations, peak-hour detection, daily
active user (DAU) trends, content dropout rates, and cohort retention
analysis — standard product analytics queries for any streaming service.

Real-world parallel: Netflix / Disney+ engagement pipeline — every play,
pause, and session-complete event lands in a Kafka topic then sinks to
ClickHouse for near-real-time OLAP dashboards. This demo reproduces the
same schema, indexes, and analytics patterns without any infrastructure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.engagement_ts import EngagementStore
from databaseai.seed_data import MOVIES, USERS

console = Console()

# ── Seed parameters ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

BASE = datetime(2024, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
DAYS = 14
START_TS = "2024-03-01 00:00:00"
END_TS   = "2024-03-14 23:59:59"

# Ten movies with approximate runtime in minutes
_RUNTIME_MIN = {
    "m01": 148,   # Inception
    "m02": 152,   # The Dark Knight
    "m03": 169,   # Interstellar
    "m05": 132,   # Parasite
    "m07": 142,   # The Shawshank Redemption
    "m08": 125,   # Spirited Away
    "m11": 116,   # Arrival
    "m13": 105,   # Coco
    "m15": 175,   # The Godfather
    "m16": 139,   # Everything Everywhere All at Once
}

_TITLE = {m["id"]: m["title"] for m in MOVIES}

# Calibrated completion probabilities — vary from 85% (family animation) to 54% (long epic)
_COMPLETION_PROB = {
    "m08": 0.85,  # Spirited Away
    "m13": 0.82,  # Coco
    "m07": 0.78,  # The Shawshank Redemption
    "m05": 0.74,  # Parasite
    "m15": 0.71,  # The Godfather
    "m02": 0.69,  # The Dark Knight
    "m16": 0.67,  # Everything Everywhere All at Once
    "m01": 0.64,  # Inception
    "m11": 0.61,  # Arrival
    "m03": 0.54,  # Interstellar (longest, most dropout)
}

# Hour-of-day weights: evening prime time (19-22) has 5-7x weekday weight
_HOUR_WEIGHTS = [
    0.3, 0.2, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5,   # 00-07
    0.7, 0.8, 0.8, 0.9, 1.4, 1.4, 1.4, 1.2,   # 08-15
    1.0, 1.4, 2.5, 4.0, 4.5, 5.0, 4.0, 2.0,   # 16-23
]
_HOUR_TOTAL = sum(_HOUR_WEIGHTS)

# Cohort entry days: users join the platform on different days
_ENTRY_DAY = {"u01": 0, "u02": 0, "u03": 2, "u04": 4, "u05": 7}


def _ts(day: int, hour: int, minute: int = 0) -> str:
    return (BASE + timedelta(days=day, hours=hour, minutes=minute)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _pick_hour() -> int:
    r = random.random() * _HOUR_TOTAL
    cumulative = 0.0
    for h, w in enumerate(_HOUR_WEIGHTS):
        cumulative += w
        if r <= cumulative:
            return h
    return 21


def _seed_engagement(store: EngagementStore) -> tuple:
    movies  = list(_RUNTIME_MIN.keys())
    session_rows: list = []
    event_rows:   list = []

    for user in USERS:
        uid       = user["id"]
        entry_day = _ENTRY_DAY.get(uid, 0)

        for day_offset in range(entry_day, DAYS):
            if random.random() < 0.15:   # ~15% daily skip probability
                continue

            for _ in range(random.randint(1, 4)):
                mid          = random.choice(movies)
                runtime_sec  = _RUNTIME_MIN[mid] * 60
                hour         = _pick_hour()
                minute       = random.randint(0, 59)
                started_at   = _ts(day_offset, hour, minute)
                completed    = random.random() < _COMPLETION_PROB[mid]
                duration_sec = int(
                    runtime_sec * (random.uniform(0.90, 1.05) if completed
                                   else random.uniform(0.12, 0.76))
                )
                session_rows.append((uid, mid, started_at, duration_sec, completed))

                # Companion interaction events for this session
                event_rows.append((uid, "play", started_at, {"movie_id": mid}))
                if random.random() < 0.45:
                    offset_min = random.randint(5, max(6, duration_sec // 60))
                    pause_ts   = (
                        BASE + timedelta(days=day_offset, hours=hour, minutes=minute + offset_min)
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    event_rows.append((uid, "pause", pause_ts, {"movie_id": mid}))
                if random.random() < 0.20:
                    event_rows.append((uid, "search", started_at, {"query": "recommendations"}))
                if random.random() < 0.30:
                    event_rows.append((uid, "click", started_at, {"target": "thumbnail"}))
                if random.random() < 0.12:
                    event_rows.append((uid, "add_to_list", started_at, {"movie_id": mid}))

    store.bulk_record_sessions(session_rows)
    store.bulk_record_events(event_rows)
    return len(session_rows), len(event_rows)


def _truncate(s: str, n: int = 32) -> str:
    return s if len(s) <= n else s[: n - 1] + "..."


def main():
    console.rule("[bold cyan]Time-Series DB Demo 17 - User Engagement Analytics[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: Netflix / Disney+ engagement pipeline - "
        "Kafka -> Flink -> ClickHouse for sub-second OLAP on watch sessions[/dim]\n"
    )

    store = EngagementStore()

    # -- Section 1: Data Ingestion
    console.print(Panel(
        "[bold]1. Data Ingestion - 14 days of watch sessions and interaction events[/bold]",
        box=box.ROUNDED,
    ))
    n_sessions, n_events = _seed_engagement(store)
    console.print(
        f"  [green]v[/green] {n_sessions:,} watch sessions  "
        f"| {n_events:,} interaction events"
    )
    console.print(
        f"  [dim]5 users  .  10 movies  .  {DAYS} days  .  "
        f"~15 % daily skip rate  .  evening-skewed session times[/dim]"
    )
    console.print(
        "  [dim]Schema: watch_sessions(user_id, movie_id, started_at*, duration_sec, completed) "
        "+ engagement_events(user_id, event_type, occurred_at*)[/dim]"
    )
    console.print(
        "  [dim]* indexed columns - BETWEEN scans are O(log n + k) not O(n)[/dim]"
    )

    # -- Section 2: Rolling Window Stats
    console.print()
    console.print(Panel(
        "[bold]2. Rolling Window Stats - Aggregate Metrics at Three Horizons[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]BETWEEN on an indexed (started_at) column is the fundamental time-series "
        "query primitive; shorter windows answer faster as n shrinks.[/dim]"
    )

    day12_start = (BASE + timedelta(days=11)).strftime("%Y-%m-%d %H:%M:%S")
    day7_start  = (BASE + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")

    windows = [
        ("Last 48 h",   day12_start,  END_TS),
        ("Last 7 days", day7_start,   END_TS),
        ("Full 14 days", START_TS,    END_TS),
    ]

    t = Table(
        "Window", "Sessions", "Unique Users",
        "Avg Watch (min)", "Max (min)", "Completion %",
        box=box.SIMPLE_HEAD,
    )
    for label, s, e in windows:
        st = store.rolling_window_stats(s, e)
        t.add_row(
            label,
            str(st["session_count"]),
            str(st["unique_users"]),
            str(st["avg_watch_min"]),
            str(st["max_watch_min"]),
            f"{st['completion_pct']} %",
        )
    console.print(t)

    # -- Section 3: Hourly Activity
    console.print()
    console.print(Panel(
        "[bold]3. Hourly Activity - Session Buckets for Two Sample Days[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]strftime('%Y-%m-%d %H:00', started_at) GROUP BY is equivalent to "
        "TimescaleDB time_bucket('1 hour', ...) - evaluates after the index scan.[/dim]"
    )

    sample_start = "2024-03-03 00:00:00"
    sample_end   = "2024-03-04 23:59:59"
    hourly = store.hourly_activity(sample_start, sample_end)
    t = Table(
        "Hour Bucket", "Sessions", "Unique Users",
        "Avg Watch (min)", "Completion %",
        box=box.SIMPLE_HEAD,
    )
    for row in hourly:
        t.add_row(
            row["hour_bucket"],
            str(row["sessions"]),
            str(row["unique_users"]),
            str(row["avg_watch_min"]),
            f"{row['completion_pct']} %",
        )
    if not hourly:
        console.print("  [dim](no sessions in this 48-h sample window)[/dim]")
    else:
        console.print(t)

    # -- Section 4: Peak Hours
    console.print()
    console.print(Panel(
        "[bold]4. Peak Hours - Top-5 Busiest Hours of Day (Full 14 Days)[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Aggregating CAST(strftime('%H', ...) AS INTEGER) across 14 days reveals "
        "the platform's prime-time window - key for CDN pre-warming and capacity planning.[/dim]"
    )
    peaks = store.peak_hours(START_TS, END_TS, top_n=5)
    t = Table(
        "Rank", "Hour of Day", "Sessions", "Unique Users", "Avg Watch (min)",
        box=box.SIMPLE_HEAD,
    )
    for i, row in enumerate(peaks, 1):
        label = f"{row['hour_of_day']:02d}:00-{row['hour_of_day']:02d}:59"
        prime = " [yellow]* prime time[/yellow]" if i == 1 else ""
        t.add_row(
            str(i),
            label + prime,
            str(row["session_count"]),
            str(row["unique_users"]),
            str(row["avg_watch_min"]),
        )
    console.print(t)

    # -- Section 5: Daily Active Users
    console.print()
    console.print(Panel(
        "[bold]5. Daily Active Users (DAU) - 14-Day Trend[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]COUNT(DISTINCT user_id) GROUP BY DATE(started_at) is the canonical DAU query. "
        "Expect DAU to rise as users from later cohorts join (days 3, 5, 8).[/dim]"
    )
    dau_rows = store.daily_active_users(START_TS, END_TS)
    t = Table(
        "Date", "DAU", "Sessions", "Avg Watch (min)", "Activity",
        box=box.SIMPLE_HEAD,
    )
    max_dau = max(r["dau"] for r in dau_rows) if dau_rows else 1
    for row in dau_rows:
        bar = "|" * row["dau"] + "." * (max_dau - row["dau"])
        t.add_row(
            row["day"],
            str(row["dau"]),
            str(row["sessions"]),
            str(row["avg_watch_min"]),
            f"[cyan]{bar}[/cyan]",
        )
    console.print(t)

    # -- Section 6: Content Completion Rates
    console.print()
    console.print(Panel(
        "[bold]6. Content Dropout Analysis - Completion Rate by Movie[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]SUM(completed)/COUNT(*) per movie_id identifies high-dropout titles. "
        "Low completion drives thumbnail / trailer A/B tests and re-cut decisions.[/dim]"
    )
    movie_list = list(_RUNTIME_MIN.keys())
    rates = store.completion_rate_by_movie(movie_list)
    t = Table(
        "Rank", "Movie", "Genre", "Plays", "Completions",
        "Completion %", "Avg Watch (min)", "Signal",
        box=box.SIMPLE_HEAD,
    )
    movie_genre = {m["id"]: m["genre"] for m in MOVIES}
    for i, row in enumerate(rates, 1):
        mid     = row["movie_id"]
        pct     = row["completion_pct"]
        signal  = (
            "[green]strong[/green]"      if pct >= 75
            else "[yellow]moderate[/yellow]" if pct >= 60
            else "[red]high dropout[/red]"
        )
        t.add_row(
            str(i),
            _truncate(_TITLE.get(mid, mid)),
            movie_genre.get(mid, "-"),
            str(row["total_plays"]),
            str(row["completions"]),
            f"{pct} %",
            str(row["avg_watch_min"]),
            signal,
        )
    console.print(t)

    # -- Section 7: Cohort Retention
    console.print()
    console.print(Panel(
        "[bold]7. Cohort Retention - Day-0 / Day-1 / Day-3 / Day-7[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Users grouped by first-watch day (JOIN cohort). "
        "Retention = users still watching on day N / cohort_size. "
        "Day-7 < Day-1 signals a normal engagement decay curve.[/dim]"
    )
    cohorts = store.cohort_retention(START_TS, END_TS)
    t = Table(
        "Cohort Day", "Size", "Day-0", "Day-1", "Day-3", "Day-7",
        "D7 Retention %",
        box=box.SIMPLE_HEAD,
    )
    for row in cohorts:
        d7_pct = (
            f"{row['day_7'] / row['cohort_size'] * 100:.0f} %"
            if row["cohort_size"] > 0 else "-"
        )
        t.add_row(
            row["cohort_day"],
            str(row["cohort_size"]),
            str(row["day_0"]),
            str(row["day_1"]),
            str(row["day_3"]),
            str(row["day_7"]),
            d7_pct,
        )
    console.print(t)
    console.print(
        "  [dim]Cohort 2024-03-08 (eva_t) may show 0 at day-7 - the window ends "
        "before 2024-03-15, so the milestone is outside the data range.[/dim]"
    )

    # -- Section 8: Interaction Event Breakdown
    console.print()
    console.print(Panel(
        "[bold]8. Interaction Event Breakdown - Full 14-Day Window[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Events are stored in the same append-only style as sessions. "
        "play:pause ratio > 1 is expected; high 'add_to_list' rates signal "
        "strong intent that converts to sessions later.[/dim]"
    )
    breakdown = store.event_type_breakdown(START_TS, END_TS)
    t = Table(
        "Event Type", "Count", "Unique Users", "Share %",
        box=box.SIMPLE_HEAD,
    )
    total_events = sum(r["event_count"] for r in breakdown)
    for row in breakdown:
        share = row["event_count"] / total_events * 100 if total_events else 0
        t.add_row(
            row["event_type"],
            str(row["event_count"]),
            str(row["unique_users"]),
            f"{share:.1f} %",
        )
    console.print(t)
    console.print(
        f"  [green]v[/green] {total_events:,} total events  "
        f"| {store.event_count():,} stored in DB"
    )

    # -- Takeaways
    console.print()
    console.print("[bold green]Key Engagement Time-Series Takeaways:[/bold green]")
    console.print(
        "  - [cyan]Append-only writes[/cyan]         - no UPDATEs; "
        "matches Kafka WAL semantics and avoids write lock contention"
    )
    console.print(
        "  - [cyan]strftime() GROUP BY[/cyan]         - SQLite's time_bucket(); "
        "index filters rows first, aggregation runs on the subset"
    )
    console.print(
        "  - [cyan]Peak-hour detection[/cyan]         - cross-day hour aggregation "
        "drives CDN pre-warming and encoder capacity scheduling"
    )
    console.print(
        "  - [cyan]DAU trend[/cyan]                   - COUNT(DISTINCT user_id) per day; "
        "step increases reveal cohort join dates"
    )
    console.print(
        "  - [cyan]Completion-rate dropout[/cyan]     - SUM(completed)/COUNT(*) per movie; "
        "low completion triggers thumbnail and re-cut A/B tests"
    )
    console.print(
        "  - [cyan]Cohort retention CTE[/cyan]        - two CTEs + one LEFT JOIN fold four "
        "day-N milestones into a single aggregation scan"
    )
    console.print(
        "  [dim]Production stacks: TimescaleDB hypertables, ClickHouse MergeTree + "
        "Materialized Views, Apache Flink streaming aggregations[/dim]"
    )


if __name__ == "__main__":
    main()
