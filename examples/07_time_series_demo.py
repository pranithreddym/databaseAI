"""
Demo 7: Time-Series Database
==============================
Tracks recommendation model A/B test metrics and system events over time.
Demonstrates time-windowed queries, downsampling, and trend detection
using SQLite as a lightweight time-series store.

Real-world parallel: Netflix Atlas / Prometheus monitoring precision@10 and
NDCG@10 for two model variants during a live A/B test rollout.
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

from databaseai.time_series_db import MetricsStore

console = Console()

# ── Synthetic data parameters ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

BASE = datetime(2024, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
HOURS = 7 * 24  # 7 days of hourly data

CONTROL_PREC_BASE   = 0.361
TREATMENT_PREC_BASE = 0.381
CONTROL_NDCG_BASE   = 0.472
TREATMENT_NDCG_BASE = 0.502
NOISE_STDDEV        = 0.010


def _ts(offset_hours: int) -> str:
    return (BASE + timedelta(hours=offset_hours)).strftime("%Y-%m-%d %H:%M:%S")


def _seed_metrics(store: MetricsStore) -> None:
    rows = []
    for h in range(HOURS):
        ts = _ts(h)
        trend = h / HOURS * 0.012          # treatment improves ~1.2 pp over the week
        noise = lambda: random.gauss(0, NOISE_STDDEV)

        rows.extend([
            ("v1.0", "control",   "precision_at_10", round(CONTROL_PREC_BASE   + noise(), 4), ts),
            ("v1.1", "treatment", "precision_at_10", round(TREATMENT_PREC_BASE + trend + noise(), 4), ts),
            ("v1.0", "control",   "ndcg_at_10",      round(CONTROL_NDCG_BASE   + noise(), 4), ts),
            ("v1.1", "treatment", "ndcg_at_10",       round(TREATMENT_NDCG_BASE + trend + noise(), 4), ts),
        ])

    store.bulk_record_metrics(rows)

    # System events marking key moments in the A/B rollout
    events = [
        ("deployment", "Model v1.0 deployed to 100% production",
         {"version": "v1.0", "traffic_pct": 100},       _ts(0)),
        ("ab_start",   "A/B test started: v1.1 challenger at 10% traffic",
         {"treatment_pct": 10, "metric": "precision@10"}, _ts(24)),
        ("scale_up",   "Treatment traffic ramped to 50%",
         {"treatment_pct": 50},                           _ts(72)),
        ("alert",      "Control precision dipped below warning threshold",
         {"threshold": 0.340, "observed": 0.337},         _ts(96)),
        ("scale_up",   "Treatment traffic ramped to 90%",
         {"treatment_pct": 90},                           _ts(144)),
    ]
    for event_type, desc, meta, ts in events:
        store.record_event(event_type, desc, meta, ts)


def main():
    console.rule("[bold cyan]Time-Series Database Demo[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: Netflix Atlas monitoring recommendation "
        "model A/B tests over time[/dim]\n"
    )

    store = MetricsStore()

    # ── 1. Data Ingestion ──────────────────────────────────────────────────
    console.print(Panel(
        "[bold]1. Data Ingestion — 7 days of hourly A/B test metrics[/bold]",
        box=box.ROUNDED,
    ))
    _seed_metrics(store)
    console.print(f"  [green]✓[/green] Inserted {store.metric_count():,} metric rows  "
                  f"({HOURS} hours × 2 variants × 2 metrics)")
    console.print(f"  [green]✓[/green] Inserted {store.event_count()} system events")
    console.print("  [dim]Metrics: precision@10 and NDCG@10  |  "
                  "Variants: control (v1.0) vs treatment (v1.1)[/dim]")

    # ── 2. Time-Windowed Queries ───────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]2. Time-Windowed Queries — Rolling Windows on precision@10[/bold]",
        box=box.ROUNDED,
    ))
    console.print("  [dim]BETWEEN on an indexed timestamp column is O(log n + k) — "
                  "the core time-series query primitive[/dim]")

    windows = [
        ("Last 24 h",  _ts(HOURS - 24), _ts(HOURS - 1)),
        ("Last 48 h",  _ts(HOURS - 48), _ts(HOURS - 1)),
        ("All 7 days", _ts(0),          _ts(HOURS - 1)),
    ]

    t = Table("Window", "Rows", "Avg precision@10", "Min", "Max", box=box.SIMPLE_HEAD)
    for label, start, end in windows:
        rows = store.query_window("precision_at_10", start, end)
        if rows:
            vals = [r["metric_value"] for r in rows]
            t.add_row(
                label,
                str(len(rows)),
                f"{sum(vals)/len(vals):.4f}",
                f"{min(vals):.4f}",
                f"{max(vals):.4f}",
            )
    console.print(t)

    # ── 3. Downsampling ────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]3. Downsampling — Hourly Data → Daily Averages[/bold]",
        box=box.ROUNDED,
    ))
    console.print("  [dim]7 days × 2 variants = 336 raw rows → 14 daily buckets  "
                  "(24× storage reduction)[/dim]")

    start_all = _ts(0)
    end_all   = _ts(HOURS - 1)

    daily = store.downsample("precision_at_10", "%Y-%m-%d", start_all, end_all)

    t = Table("Date", "Variant", "Avg prec@10", "Min", "Max", "Samples",
              box=box.SIMPLE_HEAD)
    for row in daily:
        t.add_row(
            row["bucket"],
            row["variant"],
            f"{row['avg_value']:.4f}",
            f"{row['min_value']:.4f}",
            f"{row['max_value']:.4f}",
            str(row["sample_count"]),
        )
    console.print(t)

    raw_per_metric = store.metric_count() // 2
    console.print(f"  Raw rows per metric: {raw_per_metric} → "
                  f"Downsampled rows: {len(daily)}  "
                  f"({raw_per_metric // max(len(daily), 1)}× reduction)")

    # ── 4. A/B Test Summary ────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]4. A/B Test Summary — Which Variant Wins?[/bold]",
        box=box.ROUNDED,
    ))

    for metric in ("precision_at_10", "ndcg_at_10"):
        summary = store.ab_test_summary(metric)
        t = Table(
            "Variant", "Samples", "Avg", "Min", "Max",
            title=f"[bold]{metric}[/bold]",
            box=box.SIMPLE_HEAD,
        )
        for i, row in enumerate(summary):
            tag = " [green]← winner[/green]" if i == 0 else ""
            t.add_row(
                row["variant"] + tag,
                str(row["sample_count"]),
                f"{row['avg_value']:.4f}",
                f"{row['min_value']:.4f}",
                f"{row['max_value']:.4f}",
            )
        console.print(t)

    prec = {r["variant"]: r["avg_value"] for r in store.ab_test_summary("precision_at_10")}
    if "treatment" in prec and "control" in prec and prec["control"] > 0:
        lift = (prec["treatment"] - prec["control"]) / prec["control"] * 100
        console.print(
            f"  [green]Treatment lift over control (precision@10): "
            f"+{lift:.1f}%[/green]"
        )

    # ── 5. Trend Detection ─────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]5. Trend Detection — Slope of Last N Observations[/bold]",
        box=box.ROUNDED,
    ))
    console.print("  [dim]Positive slope → improving; negative → degrading.  "
                  "Slope < −0.001 triggers an auto-rollback alert in production.[/dim]")

    t = Table("Variant", "Metric", "Window (n)", "Slope", "Direction",
              box=box.SIMPLE_HEAD)
    direction_style = {
        "improving": "[green]improving ↑[/green]",
        "degrading":  "[red]degrading ↓[/red]",
        "stable":     "[yellow]stable →[/yellow]",
    }
    for variant in ("control", "treatment"):
        for metric in ("precision_at_10", "ndcg_at_10"):
            for window in (10, 30):
                trend = store.detect_trend(metric, variant, window_size=window)
                t.add_row(
                    variant,
                    metric,
                    str(window),
                    f"{trend['slope']:+.6f}",
                    direction_style[trend["direction"]],
                )
    console.print(t)

    # ── 6. System Events Timeline ──────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]6. System Events Timeline — A/B Rollout Chronicle[/bold]",
        box=box.ROUNDED,
    ))

    events = store.query_events(_ts(0), _ts(HOURS))
    t = Table("Timestamp", "Event Type", "Description", box=box.SIMPLE_HEAD)
    for ev in events:
        t.add_row(ev["occurred_at"], ev["event_type"], ev["description"])
    console.print(t)

    # ── Key Takeaways ──────────────────────────────────────────────────────
    console.print()
    console.print("[bold green]Key Time-Series DB Takeaways:[/bold green]")
    console.print("  • [cyan]Time-windowed queries[/cyan]    — "
                  "BETWEEN on (metric_name, recorded_at) index is O(log n + k)")
    console.print("  • [cyan]Downsampling[/cyan]            — "
                  "strftime() bucketing reduces 336 rows to 14 with one GROUP BY")
    console.print("  • [cyan]Trend detection[/cyan]         — "
                  "OLS slope on a rolling window drives auto-alerts and rollbacks")
    console.print("  • [cyan]A/B test metrics[/cyan]        — "
                  "per-variant aggregation reveals the winning model variant")
    console.print("  [dim]Production: TimescaleDB hypertables, InfluxDB TSM engine, "
                  "Prometheus + Thanos[/dim]")


if __name__ == "__main__":
    main()
