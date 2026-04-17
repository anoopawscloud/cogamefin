# cogamefin — Scripted Policy Autoresearch for CoGames Tournament

## Project

Competing in the Metta-AI cogames "Cogs vs Clips" tournament (beta-cvc season).
We use **scripted heuristic policies** (no GPU, no neural network) that align
junctions via FSM-based agent behavior.

All policies MUST be named `anoop.<dota_hero>` (e.g., anoop.invoker, anoop.rubick).

## Key Commands

```bash
# Activate venv
source .venv/bin/activate

# Run scrimmage evaluation (5 episodes, JSON output)
cogames scrimmage -m machina_1 -p "class=policies.base_aligner.InvokerPolicy" -e 5 --format json

# Upload to tournament
cogames upload -p "class=policies.base_aligner.InvokerPolicy" -n "anoop.invoker" -f policies --setup-script setup_policy.py --skip-validation

# Check tournament results
cogames leaderboard beta-cvc --policy anoop.invoker
cogames submissions --season beta-cvc
cogames matches --policy anoop.invoker

# Run one autoresearch iteration
python -m autoresearch.loop

# Run evaluation harness standalone
python -m autoresearch.runner "class=policies.base_aligner.InvokerPolicy" -e 5
```

## Architecture

- `policies/base_aligner.py` — InvokerPolicy: FSM-based agent (SEEK_GEAR → SEEK_HEARTS → SEEK_JUNCTION → align)
- `autoresearch/runner.py` — Scrimmage execution + JSON result parsing
- `autoresearch/tracker.py` — TSV experiment logging
- `autoresearch/loop.py` — Autoresearch orchestrator
- `results/experiments.tsv` — Experiment history

## Policy Design

Policies subclass `AgentPolicy` directly (NOT `StatefulAgentPolicy`) to avoid
torch dependency. The game's built-in StarterPolicy uses `StatefulAgentPolicy`
which imports torch — our design avoids this.

Key insight: when the agent stands on a target (delta=0), we cycle through
`change_vibe_*` actions to trigger interactions (gear pickup, heart collection,
junction alignment). The StarterPolicy just returns noop in this case.

## Autoresearch Loop

When running `/loop`, each iteration:
1. Read `results/experiments.tsv` for history
2. Pick next variant to test (role mix sweep, then refinement)
3. Run `cogames scrimmage` with 5 episodes
4. Log results to TSV
5. Upload to tournament if score improved >5%
6. Commit and push changes

## Variant Axes

- **Role mix**: How many of each role (aligner, miner, scout, scrambler)
- **Behavioral params**: heart_threshold, wander_radius, gear_priority
- **Strategy**: junction targeting strategy, exploration pattern

## Tournament Upload

```bash
cogames upload -p "class=policies.<module>.<Class>" -n "anoop.<hero>" -f policies --setup-script setup_policy.py --skip-validation
```

The `-f policies` flag bundles the `policies/` directory. The server already has
cogames installed; setup_policy.py just ensures version compatibility.
