"""Evaluation harness: run cogames scrimmage and parse JSON results."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalResult:
    """Parsed scrimmage results."""
    policy_name: str = ""
    mission: str = "machina_1"
    episodes: int = 0
    avg_reward: float = 0.0
    per_episode_rewards: list[float] = field(default_factory=list)
    junction_aligned: float = 0.0
    junction_held: float = 0.0
    heart_gained: float = 0.0
    aligner_gained: float = 0.0
    change_vibe_success: float = 0.0
    move_success: float = 0.0
    move_failed: float = 0.0
    noop_success: float = 0.0
    raw_json: dict | None = None
    error: str | None = None


def run_scrimmage(
    policy_spec: str,
    mission: str = "machina_1",
    episodes: int = 5,
    cogames_bin: str | None = None,
    cwd: str | None = None,
) -> EvalResult:
    """Run cogames scrimmage and parse JSON output.

    Args:
        policy_spec: Policy specification (e.g., "class=policies.base_aligner.InvokerPolicy")
        mission: Mission name
        episodes: Number of episodes to run
        cogames_bin: Path to cogames binary (default: find in PATH or .venv)
        cwd: Working directory for cogames command
    """
    if cogames_bin is None:
        venv_bin = Path(__file__).parent.parent / ".venv" / "bin" / "cogames"
        if venv_bin.exists():
            cogames_bin = str(venv_bin)
        else:
            cogames_bin = "cogames"

    if cwd is None:
        cwd = str(Path(__file__).parent.parent)

    cmd = [
        cogames_bin, "scrimmage",
        "-m", mission,
        "-p", policy_spec,
        "-e", str(episodes),
        "--format", "json",
    ]

    result = EvalResult(policy_name=policy_spec, mission=mission, episodes=episodes)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        result.error = "Scrimmage timed out after 600s"
        return result
    except FileNotFoundError:
        result.error = f"cogames binary not found: {cogames_bin}"
        return result

    if proc.returncode != 0:
        result.error = f"Exit {proc.returncode}: {proc.stderr[-500:]}"
        return result

    stdout = proc.stdout.strip()
    if not stdout:
        result.error = "Empty JSON output"
        return result

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        result.error = f"JSON parse error: {e}"
        return result

    result.raw_json = data
    _parse_results(data, result)
    return result


def _parse_results(data: dict, result: EvalResult) -> None:
    """Extract key metrics from scrimmage JSON output."""
    missions = data.get("missions", [])
    if not missions:
        return

    ms = missions[0].get("mission_summary", {})

    # Game stats
    gs = ms.get("avg_game_stats", {})
    result.junction_aligned = gs.get("cogs/aligned.junction.gained", 0.0)
    result.junction_held = gs.get("cogs/aligned.junction.held", 0.0)

    # Per-episode rewards
    rewards = ms.get("per_episode_per_policy_avg_rewards", {})
    all_rewards = []
    for episode_rewards in rewards.values():
        all_rewards.extend(episode_rewards)
    if all_rewards:
        result.avg_reward = sum(all_rewards) / len(all_rewards)
        result.per_episode_rewards = all_rewards

    # Per-agent metrics
    for ps in ms.get("policy_summaries", []):
        am = ps.get("avg_agent_metrics", {})
        result.junction_aligned = max(
            result.junction_aligned,
            am.get("junction.aligned_by_agent", 0.0),
        )
        result.heart_gained = am.get("heart.gained", 0.0)
        result.aligner_gained = am.get("aligner.gained", 0.0)
        result.change_vibe_success = am.get("action.change_vibe.success", 0.0)
        result.move_success = am.get("action.move.success", 0.0)
        result.move_failed = am.get("action.move.failed", 0.0)
        result.noop_success = am.get("action.noop.success", 0.0)


def print_result(result: EvalResult) -> None:
    """Print evaluation result summary."""
    if result.error:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return

    print(f"Policy: {result.policy_name}")
    print(f"Mission: {result.mission} ({result.episodes} episodes)")
    print(f"Avg Reward: {result.avg_reward:.3f}")
    print(f"Junction Aligned: {result.junction_aligned:.3f}")
    print(f"Junction Held: {result.junction_held:.1f}")
    print(f"Heart Gained: {result.heart_gained:.3f}")
    print(f"Aligner Gained: {result.aligner_gained:.3f}")
    print(f"Change Vibe Success: {result.change_vibe_success:.3f}")
    print(f"Move Success: {result.move_success:.1f}")
    print(f"Move Failed: {result.move_failed:.1f}")
    print(f"Per-episode rewards: {[f'{r:.3f}' for r in result.per_episode_rewards]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", help="Policy spec")
    parser.add_argument("-m", "--mission", default="machina_1")
    parser.add_argument("-e", "--episodes", type=int, default=5)
    args = parser.parse_args()

    result = run_scrimmage(args.policy, args.mission, args.episodes)
    print_result(result)
