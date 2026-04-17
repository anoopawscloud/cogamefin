"""Experiment tracker: TSV logging for autoresearch loop."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from autoresearch.runner import EvalResult

RESULTS_DIR = Path(__file__).parent.parent / "results"
EXPERIMENTS_FILE = RESULTS_DIR / "experiments.tsv"

COLUMNS = [
    "timestamp",
    "experiment_id",
    "policy_class",
    "dota_hero",
    "variant_params",
    "episodes",
    "avg_reward",
    "junction_aligned",
    "junction_held",
    "heart_gained",
    "aligner_gained",
    "change_vibe",
    "tournament_name",
    "tournament_score",
    "status",
    "notes",
]


def _ensure_file() -> None:
    """Create results file with header if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not EXPERIMENTS_FILE.exists():
        with open(EXPERIMENTS_FILE, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(COLUMNS)


def log_experiment(
    experiment_id: str,
    policy_class: str,
    dota_hero: str,
    result: EvalResult,
    variant_params: str = "",
    tournament_name: str = "",
    tournament_score: str = "",
    status: str = "evaluated",
    notes: str = "",
) -> None:
    """Append one experiment row to the TSV log."""
    _ensure_file()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = [
        now,
        experiment_id,
        policy_class,
        dota_hero,
        variant_params,
        str(result.episodes),
        f"{result.avg_reward:.4f}",
        f"{result.junction_aligned:.3f}",
        f"{result.junction_held:.1f}",
        f"{result.heart_gained:.3f}",
        f"{result.aligner_gained:.3f}",
        f"{result.change_vibe_success:.3f}",
        tournament_name,
        tournament_score,
        status,
        notes,
    ]
    with open(EXPERIMENTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(row)


def read_experiments() -> list[dict[str, str]]:
    """Read all experiments from the TSV log."""
    _ensure_file()
    with open(EXPERIMENTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def best_experiment(metric: str = "avg_reward") -> dict[str, str] | None:
    """Return the experiment with the highest value for the given metric."""
    experiments = read_experiments()
    if not experiments:
        return None
    return max(experiments, key=lambda e: float(e.get(metric, "0") or "0"))


def latest_experiments(n: int = 10) -> list[dict[str, str]]:
    """Return the N most recent experiments."""
    experiments = read_experiments()
    return experiments[-n:]
