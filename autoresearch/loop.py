"""Autoresearch loop orchestrator for cogames policy iteration.

Designed to be run via Claude Code /loop or as a standalone script.
Each iteration:
1. Reads experiment history
2. Generates next variant to test
3. Runs scrimmage evaluation
4. Logs results
5. Uploads to tournament if improved
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from autoresearch.runner import EvalResult, print_result, run_scrimmage
from autoresearch.tracker import (
    best_experiment,
    latest_experiments,
    log_experiment,
    read_experiments,
)

# DOTA 2 hero names for policy variants, in order of use
DOTA_HEROES = [
    "invoker",     # base FSM aligner
    "rubick",      # copy best + tweak
    "chen",        # aligner-heavy role mix
    "enigma",      # converge on junctions
    "oracle",      # parameter sweep winner
    "phoenix",     # recovery variant
    "io",          # coordination variant
    "morphling",   # adaptive strategy
    "tinker",      # rapid iteration
    "meepo",       # parallel coordination
    "dazzle",      # support-heavy
    "pugna",       # aggressive expansion
    "warlock",     # area control
    "treant",      # defensive hold
    "visage",      # scout-heavy
]

# Role mix variants to sweep
ROLE_MIXES = [
    # (name, role_cycle)
    ("default", "aligner,miner,aligner,scout,aligner,miner,aligner,scrambler"),
    ("aligner_heavy", "aligner,aligner,aligner,aligner,aligner,miner,aligner,scout"),
    ("pure_aligner", "aligner,aligner,aligner,aligner,aligner,aligner,aligner,aligner"),
    ("balanced", "aligner,miner,scrambler,scout,aligner,miner,scrambler,scout"),
    ("miner_heavy", "aligner,miner,miner,miner,aligner,miner,miner,scout"),
    ("dual_scrambler", "aligner,scrambler,aligner,scrambler,aligner,miner,aligner,scout"),
]

COGAMEFIN_ROOT = Path(__file__).parent.parent


def next_experiment_id(experiments: list[dict]) -> str:
    """Generate next experiment ID."""
    return f"exp{len(experiments) + 1:03d}"


def next_hero(experiments: list[dict]) -> str:
    """Pick the next unused DOTA hero name."""
    used = {e.get("dota_hero", "") for e in experiments}
    for hero in DOTA_HEROES:
        if hero not in used:
            return hero
    # Fallback: append number to heroes
    return f"{DOTA_HEROES[len(experiments) % len(DOTA_HEROES)]}{len(experiments)}"


def upload_policy(policy_spec: str, name: str) -> bool:
    """Upload policy to tournament. Returns True on success."""
    cogames_bin = str(COGAMEFIN_ROOT / ".venv" / "bin" / "cogames")
    cmd = [
        cogames_bin, "upload",
        "-p", policy_spec,
        "-n", name,
        "-f", "policies",
        "--setup-script", "setup_policy.py",
        "--skip-validation",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(COGAMEFIN_ROOT),
        )
        if proc.returncode == 0:
            print(f"Uploaded: {name}")
            return True
        else:
            print(f"Upload failed: {proc.stderr[-300:]}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"Upload error: {e}", file=sys.stderr)
        return False


def check_tournament(policy_name: str) -> str | None:
    """Check tournament score for a policy. Returns score string or None."""
    cogames_bin = str(COGAMEFIN_ROOT / ".venv" / "bin" / "cogames")
    cmd = [
        cogames_bin, "leaderboard", "beta-cvc",
        "--policy", policy_name,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=str(COGAMEFIN_ROOT),
        )
        # Parse leaderboard output for score
        for line in proc.stdout.split("\n"):
            if policy_name in line:
                parts = line.split()
                for p in parts:
                    try:
                        score = float(p)
                        return str(score)
                    except ValueError:
                        continue
    except Exception:
        pass
    return None


def run_iteration() -> None:
    """Run one iteration of the autoresearch loop."""
    experiments = read_experiments()
    exp_id = next_experiment_id(experiments)

    print(f"\n{'='*60}")
    print(f"Autoresearch iteration: {exp_id}")
    print(f"Previous experiments: {len(experiments)}")

    # Determine what to try next
    if len(experiments) == 0:
        # First run: baseline with default roles
        policy_spec = "class=policies.base_aligner.InvokerPolicy"
        hero = "invoker"
        variant = "default"
        notes = "Baseline InvokerPolicy with default role cycle"
    elif len(experiments) < len(ROLE_MIXES):
        # Sweep role mixes
        idx = len(experiments)
        variant_name, role_cycle = ROLE_MIXES[idx]
        policy_spec = f"class=policies.base_aligner.InvokerPolicy,role_cycle={role_cycle}"
        hero = next_hero(experiments)
        variant = f"role_mix={variant_name}"
        notes = f"Role mix sweep: {variant_name}"
    else:
        # After role mix sweep: refine best variant
        best = best_experiment("junction_aligned")
        if best:
            best_params = best.get("variant_params", "")
            hero = next_hero(experiments)
            variant = f"refine_{best.get('dota_hero', 'unknown')}"
            policy_spec = f"class=policies.base_aligner.InvokerPolicy"
            if "role_cycle=" in best_params:
                role_cycle = best_params.split("role_cycle=")[1].split(",")[0:8]
                policy_spec += f",role_cycle={','.join(role_cycle)}"
            notes = f"Refining best variant: {best.get('dota_hero')}"
        else:
            policy_spec = "class=policies.base_aligner.InvokerPolicy"
            hero = next_hero(experiments)
            variant = "default"
            notes = "Fallback to default"

    print(f"Hero: {hero}")
    print(f"Variant: {variant}")
    print(f"Policy: {policy_spec}")

    # Run scrimmage
    result = run_scrimmage(policy_spec, episodes=5)
    print_result(result)

    if result.error:
        log_experiment(
            experiment_id=exp_id,
            policy_class=policy_spec,
            dota_hero=hero,
            result=result,
            variant_params=variant,
            status="error",
            notes=f"Error: {result.error}",
        )
        return

    # Check if this is the best result
    best = best_experiment("avg_reward")
    best_reward = float(best["avg_reward"]) if best else 0.0
    is_best = result.avg_reward > best_reward * 1.05  # 5% improvement threshold

    # Upload if best
    tournament_name = ""
    if is_best and result.junction_aligned > 0:
        tournament_name = f"anoop.{hero}"
        success = upload_policy(policy_spec, tournament_name)
        if not success:
            tournament_name = ""

    status = "uploaded" if tournament_name else "evaluated"
    log_experiment(
        experiment_id=exp_id,
        policy_class=policy_spec,
        dota_hero=hero,
        result=result,
        variant_params=variant,
        tournament_name=tournament_name,
        status=status,
        notes=notes,
    )

    print(f"\nResult: reward={result.avg_reward:.3f}, aligned={result.junction_aligned:.3f}")
    if is_best:
        print(f"NEW BEST! (previous: {best_reward:.3f})")
    if tournament_name:
        print(f"Uploaded as: {tournament_name}")


if __name__ == "__main__":
    run_iteration()
