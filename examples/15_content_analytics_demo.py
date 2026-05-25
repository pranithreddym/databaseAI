"""
Demo 15: Time-Series DB — Content Performance Analytics
========================================================
Tracks daily engagement metrics for movie launches on a streaming platform.
Demonstrates time-windowed queries, rolling averages, retention curves,
peak detection, and genre-level trend aggregation using SQLite.

Real-world parallel: Netflix Content Intelligence tracking 30-day post-launch
performance (views, completions, ratings) to drive renewal decisions and
recommendation weight adjustments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
from datetime import date, timedelta

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.content_analytics import ContentMetricsStore
from databaseai.seed_data import MOVIES

console = Console()

# ── Simulation parameters ────────────────────────────────────────────────────
SEED = 777
random.seed(SEED)

DAYS = 30
LAUNCH = date(2024, 11, 1)

# Six spotlight movies — one per genre represented in seed data
SPOTLIGHT = {
    "m01": "Inception",
    "m02": "The Dark Knight",
    "m05": "Parasite",
    "m07": "The Shawshank Redemption",
    "m08": "Spirited Away",
    "m09": "Get Out",
}

# Per-movie viewership profiles: base launch views, daily decay multiplier,
# noise level, and optional "second wind" day (award, press, algorithm boost)
_PROFILES = {
    "m01": {"base": 120_000, "decay": 0.92, "noise": 0.12, "sw_day": 14, "sw_boost": 1.30},
    "m02": {"base": 150_000, "decay": 0.90, "noise": 0.15, "sw_day": None, "sw_boost": 1.0},
    "m05": {"base":  80_000, "decay": 0.94, "noise": 0.10, "sw_day":  7, "sw_boost": 1.50},
    "m07": {"base":  60_000, "decay": 0.97, "noise": 0.08, "sw_day": 20, "sw_boost": 1.20},
    "m08": {"base":  50_000, "decay": 0.96, "noise": 0.09, "sw_day": None, "sw_boost": 1.0},
    "m09": {"base":  70_000, "decay": 0.89, "noise": 0.18, "sw_day": 10, "sw_boost": 1.60},
}


def _date(offset: int) -> str:
    return (LAUNCH + timedelta(days=offset)).strftime("%Y-%m-%d")


def _seed_metrics(store: ContentMetricsStore) -> None:
    rows = []
    for mid, p in _PROFILES.items():
        current = float(p["base"])
        for d in range(DAYS):
            if p["sw_day"] and d == p["sw_day"]:
                current *= p["sw_boost"]
            multiplier = 1.0 + random.gauss(0, p["noise"])
            views = max(1, int(current * multiplier))
            completions = int(views * random.uniform(0.45, 0.75))
            ratings = int(views * random.uniform(0.01, 0.05))
            watch_pct = round(random.uniform(55.0, 92.0), 1)
            rows.append((mid, _date(d), views, completions, ratings, watch_pct))
            current *= p["decay"]
    store.bulk_record_metrics(rows)

    events = [
        ("m01", "launch",      "Inception added to catalog",
         {"platform": "global"},                          _date(0)),
        ("m05", "award",       "Parasite wins Academy Award",
         {"category": "Best Picture"},                    _date(7)),
        ("m09", "press",       "Get Out featured in 'Best of Horror' list",
         {"source": "Rotten Tomatoes"},                   _date(10)),
        (None,  "algorithm",   "Recommendation model v2 deployed",
         {"boost_genres": ["sci-fi"]},                    _date(14)),
        ("m01", "algorithm",   "Inception boosted by sci-fi model update",
         {"rank_delta": "+12"},                           _date(14)),
        ("m07", "anniversary", "Shawshank 30th anniversary campaign",
         {"channels": ["email", "banner"]},               _date(20)),
    ]
    for movie_id, etype, desc, meta, ts in events:
        store.record_event(movie_id, etype, desc, meta, ts)


def main():
    console.rule("[bold cyan]Time-Series DB Demo 15 — Content Performance Analytics[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: Netflix Content Intelligence tracking 30-day "
        "post-launch engagement to drive renewal and recommendation decisions[/dim]\n"
    )

    store = ContentMetricsStore()

    # ── Section 1: Data ingestion ─────────────────────────────────────────────
    console.print(Panel(
        "[bold]1. Data Ingestion — 30 Days of Daily Engagement Metrics[/bold]",
        box=box.ROUNDED,
    ))
    _seed_metrics(store)
    console.print(
        f"  [green]✓[/green] Inserted {store.metric_count():,} metric rows  "
        f"({len(_PROFILES)} movies × {DAYS} days)"
    )
    console.print(f"  [green]✓[/green] Inserted {store.event_count()} content events")
    console.print(
        "  [dim]Metrics per row: views · completions · ratings_given · avg_watch_pct[/dim]"
    )

    # ── Section 2: Time-windowed aggregation ──────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]2. Time-Windowed Queries — Week-by-Week Aggregation[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Composite PK (movie_id, event_date) makes each BETWEEN scan "
        "O(log n + k) — no separate index needed[/dim]"
    )
    weeks = [
        ("Week 1 (D0–D6)",   _date(0),  _date(6)),
        ("Week 2 (D7–D13)",  _date(7),  _date(13)),
        ("Week 3 (D14–D20)", _date(14), _date(20)),
        ("Week 4 (D21–D27)", _date(21), _date(27)),
    ]
    t = Table("Week", "Movie", "Total Views", "Total Completions", "Avg Watch %",
              box=box.SIMPLE_HEAD)
    for label, start, end in weeks:
        first = True
        for mid, title in SPOTLIGHT.items():
            agg = store.aggregate_window(mid, start, end)
            t.add_row(
                label if first else "",
                title[:24],
                f"{agg['total_views']:,}"       if agg["total_views"]       else "—",
                f"{agg['total_completions']:,}"  if agg["total_completions"]  else "—",
                f"{agg['avg_watch_pct']:.1f}%"   if agg["avg_watch_pct"]      else "—",
            )
            first = False
    console.print(t)

    # ── Section 3: Rolling averages ───────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]3. Rolling 7-Day Average — Smoothing Daily Noise[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Sliding-window deque is O(n) time, O(w) space — equivalent to SQL "
        "AVG(...) OVER (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)[/dim]"
    )
    inc_roll = store.rolling_average("m01", "views", 7)
    par_roll = store.rolling_average("m05", "views", 7)
    t = Table(
        "Day", "Inception Raw", "7-Day Avg", "Parasite Raw", "7-Day Avg",
        box=box.SIMPLE_HEAD,
    )
    for i, (inc, par) in enumerate(zip(inc_roll, par_roll)):
        if i % 5 == 0 or i == DAYS - 1:
            t.add_row(
                f"Day {i:02d}",
                f"{inc['raw_value']:>10,}", f"{inc['rolling_avg']:>10,.0f}",
                f"{par['raw_value']:>10,}", f"{par['rolling_avg']:>10,.0f}",
            )
    console.print(t)

    # ── Section 4: Retention curves ───────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]4. Retention Curves — Normalised Day-Over-Day Decay[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Normalising to launch-day = 1.0 makes curves scale-invariant — "
        "a blockbuster and an indie become directly comparable[/dim]"
    )
    curves = {
        mid: {r["days_since_launch"]: r["relative_views"]
              for r in store.retention_curve(mid)}
        for mid in ("m01", "m05", "m09", "m07")
    }
    t = Table("Day", "Inception", "Parasite", "Get Out", "Shawshank", box=box.SIMPLE_HEAD)
    for day in [0, 5, 7, 10, 14, 20, 25, 29]:
        t.add_row(
            f"Day {day:02d}",
            f"{curves['m01'].get(day, 0):.3f}",
            f"{curves['m05'].get(day, 0):.3f}",
            f"{curves['m09'].get(day, 0):.3f}",
            f"{curves['m07'].get(day, 0):.3f}",
        )
    console.print(t)

    # ── Section 5: Peak detection ─────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]5. Peak Detection — When Did Each Title Spike?[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]ORDER BY metric DESC LIMIT 1 — in production this feeds anomaly "
        "alerting rules that fire on unusual spikes or drops[/dim]"
    )
    t = Table("Movie", "Peak Date", "Peak Views", "Days After Launch", box=box.SIMPLE_HEAD)
    for mid, title in SPOTLIGHT.items():
        peak = store.peak_day(mid, "views")
        if peak:
            days_after = (date.fromisoformat(peak["event_date"]) - LAUNCH).days
            style = "[green]" if days_after > 0 else ""
            end_style = "[/green]" if days_after > 0 else ""
            t.add_row(
                title[:24],
                peak["event_date"],
                f"{peak['peak_value']:,}",
                f"{style}{days_after}{end_style}",
            )
    console.print(t)

    # ── Section 6: Genre trend aggregation ───────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]6. Genre Trend — Aggregated Views Across Titles[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Summing across a movie cohort produces a 'genre health' signal "
        "used to tune recommendation diversity carousels[/dim]"
    )
    movie_genre_map = {m["id"]: m["genre"] for m in MOVIES}
    t = Table("Genre", "Week 1", "Week 2", "Week 3", "Week 4", box=box.SIMPLE_HEAD)
    for genre in ("sci-fi", "action", "thriller", "drama", "animation", "horror"):
        week_totals = []
        for _, start, end in weeks:
            trend = store.genre_trend(genre, "views", start, end, movie_genre_map)
            total = sum(r["total_value"] for r in trend)
            week_totals.append(f"{total:,}" if total else "—")
        t.add_row(genre, *week_totals)
    console.print(t)

    # ── Section 7: Top movies per week ───────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]7. Top-3 Movies by Views Each Week[/bold]",
        box=box.ROUNDED,
    ))
    title_map = {m["id"]: m["title"] for m in MOVIES}
    t = Table("Week", "Rank", "Movie", "Total Views", box=box.SIMPLE_HEAD)
    for label, start, end in weeks:
        top = store.top_movies_in_window(start, end, "views", 3)
        for rank, row in enumerate(top, 1):
            t.add_row(
                label if rank == 1 else "",
                str(rank),
                title_map.get(row["movie_id"], row["movie_id"])[:24],
                f"{row['total_value']:,}",
            )
    console.print(t)

    # ── Section 8: Events timeline ────────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]8. Content Events Timeline[/bold]", box=box.ROUNDED))
    events = store.query_events(_date(0), _date(DAYS))
    t = Table("Date", "Movie", "Event Type", "Description", box=box.SIMPLE_HEAD)
    for ev in events:
        t.add_row(
            ev["occurred_at"],
            ev["movie_id"] or "(platform)",
            ev["event_type"],
            ev["description"],
        )
    console.print(t)

    # ── Takeaways ─────────────────────────────────────────────────────────────
    console.print()
    console.print("[bold green]Key Takeaways — Content Performance Time-Series:[/bold green]")
    console.print(
        "  • [cyan]Daily grain[/cyan]              — pre-aggregating to day-level "
        "reduces row count 86,400x vs. per-second; same as BigQuery date partitions"
    )
    console.print(
        "  • [cyan]Composite PK as index[/cyan]    — (movie_id, event_date) covers "
        "the dominant query pattern at zero extra storage cost"
    )
    console.print(
        "  • [cyan]Rolling averages[/cyan]          — deque sliding window is O(n) "
        "time and O(w) space; suppresses daily noise for trend visibility"
    )
    console.print(
        "  • [cyan]Retention curves[/cyan]          — normalised to launch-day = 1.0; "
        "scale-invariant shape is the renewal signal used by content teams"
    )
    console.print(
        "  • [cyan]Genre aggregation[/cyan]         — cohort rollup reveals carousel "
        "health and drives diversity tuning in recommendation models"
    )
    console.print(
        "  [dim]Production stacks: Netflix Druid + Iceberg, "
        "Spotify BigQuery daily partitions, TikTok ClickHouse MergeTree[/dim]"
    )


if __name__ == "__main__":
    main()
