"""FSM-based aligner policy for cogames Cogs vs Clips tournament.

This policy uses pure heuristics (no neural network, no GPU) to:
1. Assign agents to roles (aligner, miner, scout)
2. Navigate using Manhattan distance + BFS pathfinding
3. Execute the alignment chain: gear -> hearts -> junction -> align

Subclasses AgentPolicy directly (not StatefulAgentPolicy) to avoid
the torch dependency that StatefulAgentPolicy imports unconditionally.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation


TEAM_TAG_PREFIX = "team:"
ELEMENTS = ("carbon", "oxygen", "germanium", "silicon")
MOVE_DELTAS = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
}
WANDER_DIRECTIONS = ("east", "south", "west", "north")

# Role assignment for 8 agents by agent_id
DEFAULT_ROLE_CYCLE = ("aligner", "miner", "aligner", "scout",
                      "aligner", "miner", "aligner", "scrambler")


class AgentPhase(Enum):
    """FSM states for agent behavior."""
    SEEK_GEAR = auto()
    SEEK_HEARTS = auto()
    SEEK_JUNCTION = auto()
    SEEK_EXTRACTOR = auto()
    DEPOSIT = auto()
    EXPLORE = auto()


@dataclass
class AgentState:
    """Per-agent mutable state."""
    role: str = "aligner"
    phase: AgentPhase = AgentPhase.SEEK_GEAR
    wander_direction_idx: int = 0
    steps_in_phase: int = 0
    last_location: tuple[int, int] | None = None
    stuck_counter: int = 0


class InvokerAgentPolicy(AgentPolicy):
    """Per-agent policy implementing FSM-based behavior.

    Avoids StatefulAgentPolicy (which imports torch) by subclassing
    AgentPolicy directly and managing state internally.
    """

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int,
                 role: str = "aligner"):
        super().__init__(policy_env_info)
        self._agent_id = agent_id
        self._state = AgentState(
            role=role,
            wander_direction_idx=agent_id % len(WANDER_DIRECTIONS),
        )
        self._center = (policy_env_info.obs_height // 2,
                        policy_env_info.obs_width // 2)

        # Build tag lookups
        self._tag_name_to_id = {name: idx for idx, name in enumerate(policy_env_info.tags)}
        self._team_tag_ids = {idx for idx, name in enumerate(policy_env_info.tags)
                              if name.startswith(TEAM_TAG_PREFIX)}

        self._agent_tags = self._resolve_tags(["agent"])
        self._junction_tags = self._resolve_tags(["junction"])
        self._heart_source_tags = self._resolve_tags(["hub", "chest"])
        self._deposit_tags = self._resolve_tags(["hub", "junction"])
        self._extractor_tags = self._resolve_tags(
            [f"{e}_extractor" for e in ELEMENTS])

        # Tags that actually block movement (walls, agents, ships)
        self._blocking_tags = self._resolve_tags(["wall", "agent", "ship"])

        self._role_station_tags = {}
        for r in ("miner", "aligner", "scrambler", "scout"):
            self._role_station_tags[r] = self._resolve_tags([r, f"c:{r}"])

        # Build action lookup — includes both primary and vibe actions
        all_actions = policy_env_info.all_action_names
        self._action_set = set(all_actions)
        self._fallback = "noop" if "noop" in self._action_set else all_actions[0]

        # Identify vibe action names for cycling
        self._vibe_actions = policy_env_info.vibe_action_names
        self._vibe_cycle_idx = 0

    def _resolve_tags(self, names: list[str]) -> set[int]:
        tag_ids: set[int] = set()
        for name in names:
            if name in self._tag_name_to_id:
                tag_ids.add(self._tag_name_to_id[name])
            type_name = f"type:{name}"
            if type_name in self._tag_name_to_id:
                tag_ids.add(self._tag_name_to_id[type_name])
        return tag_ids

    def _inventory(self, obs: AgentObservation) -> dict[str, int]:
        """Parse inventory from observation tokens at center location."""
        items: dict[str, int] = {}
        for token in obs.tokens:
            if token.location != self._center:
                continue
            name = token.feature.name
            if not name.startswith("inv:"):
                continue
            suffix = name[4:]
            if not suffix:
                continue
            item_name, sep, power_str = suffix.rpartition(":p")
            if not sep or not item_name or not power_str.isdigit():
                item_name = suffix
                scale = 1
            else:
                scale = max(int(token.feature.normalization), 1) ** int(power_str)
            value = int(token.value)
            if value <= 0:
                continue
            items[item_name] = items.get(item_name, 0) + value * scale
        return items

    def _tags_by_location(self, obs: AgentObservation) -> dict[tuple[int, int], set[int]]:
        """Bucket visible tags by map cell."""
        result: dict[tuple[int, int], set[int]] = {}
        for token in obs.tokens:
            if token.feature.name != "tag" or token.location is None:
                continue
            result.setdefault(token.location, set()).add(token.value)
        return result

    def _closest(self, tags_by_loc: dict[tuple[int, int], set[int]],
                 include: set[int],
                 require: set[int] | None = None,
                 exclude: set[int] | None = None) -> tuple[int, int] | None:
        """Find closest location matching tag criteria."""
        return min(
            (loc for loc, loc_tags in tags_by_loc.items()
             if loc_tags & include
             and (require is None or loc_tags & require)
             and (exclude is None or not (loc_tags & exclude))),
            key=lambda loc: (abs(loc[0] - self._center[0]) +
                             abs(loc[1] - self._center[1]),
                             loc[0], loc[1]),
            default=None,
        )

    def _action(self, name: str) -> Action:
        return Action(name=name if name in self._action_set else self._fallback)

    def _move_toward(self, target: tuple[int, int],
                     tags_by_loc: dict[tuple[int, int], set[int]]) -> Action:
        """Move toward target using BFS pathfinding (same as StarterPolicy)."""
        dr = target[0] - self._center[0]
        dc = target[1] - self._center[1]

        if dr == 0 and dc == 0:
            # On target — try a vibe action to interact
            return self._cycle_vibe()

        # Adjacent — move directly
        if abs(dr) + abs(dc) == 1:
            if abs(dr) >= abs(dc):
                direction = "south" if dr > 0 else "north"
            else:
                direction = "east" if dc > 0 else "west"
            return self._action(f"move_{direction}")

        # BFS pathfinding toward target
        blocked = set(tags_by_loc)
        blocked.discard(self._center)
        if not (tags_by_loc.get(target, set()) & self._agent_tags):
            blocked.discard(target)

        queue = deque([self._center])
        visited = {self._center}
        first_dir: dict[tuple[int, int], str] = {}

        while queue:
            current = queue.popleft()
            drow = target[0] - current[0]
            dcol = target[1] - current[1]
            candidates: list[str] = []
            if abs(drow) >= abs(dcol):
                if drow != 0:
                    candidates.append("south" if drow > 0 else "north")
                if dcol != 0:
                    candidates.append("east" if dcol > 0 else "west")
            else:
                if dcol != 0:
                    candidates.append("east" if dcol > 0 else "west")
                if drow != 0:
                    candidates.append("south" if drow > 0 else "north")

            for d in candidates:
                delta = MOVE_DELTAS[d]
                nxt = (current[0] + delta[0], current[1] + delta[1])
                if (nxt in visited or nxt in blocked
                        or not (0 <= nxt[0] < self._policy_env_info.obs_height)
                        or not (0 <= nxt[1] < self._policy_env_info.obs_width)):
                    continue
                visited.add(nxt)
                first_dir[nxt] = first_dir.get(current, d)
                if nxt == target or abs(nxt[0] - target[0]) + abs(nxt[1] - target[1]) <= 1:
                    return self._action(f"move_{first_dir[nxt]}")
                queue.append(nxt)

        return self._explore(tags_by_loc)

    def _cycle_vibe(self) -> Action:
        """Cycle through vibe actions when standing on a target."""
        if not self._vibe_actions:
            return self._action(self._fallback)
        action_name = self._vibe_actions[self._vibe_cycle_idx % len(self._vibe_actions)]
        self._vibe_cycle_idx += 1
        return self._action(action_name)

    def _explore(self, tags_by_loc: dict[tuple[int, int], set[int]]) -> Action:
        """Explore in a fixed direction, rotating when blocked."""
        blocked = set(tags_by_loc)
        blocked.discard(self._center)

        for offset in range(len(WANDER_DIRECTIONS)):
            idx = (self._state.wander_direction_idx + offset) % len(WANDER_DIRECTIONS)
            direction = WANDER_DIRECTIONS[idx]
            delta = MOVE_DELTAS[direction]
            nxt = (self._center[0] + delta[0], self._center[1] + delta[1])
            if (nxt in blocked
                    or not (0 <= nxt[0] < self._policy_env_info.obs_height)
                    or not (0 <= nxt[1] < self._policy_env_info.obs_width)):
                continue
            return self._action(f"move_{direction}")

        # All directions blocked — try a vibe action
        return self._cycle_vibe()

    def step(self, obs: AgentObservation) -> Action:
        """Main decision logic implementing the FSM."""
        items = self._inventory(obs)
        tags_by_loc = self._tags_by_location(obs)

        own_team = tags_by_loc.get(self._center, set()) & self._team_tag_ids
        enemy_team = (self._team_tag_ids - own_team) or self._team_tag_ids

        has_role_gear = items.get(self._state.role, 0) > 0
        has_heart = items.get("heart", 0) > 0
        cargo = sum(items.get(e, 0) for e in ELEMENTS)
        role = self._state.role

        self._state.steps_in_phase += 1

        # Phase timeout — if stuck in a phase too long, rotate wander direction
        if self._state.steps_in_phase > 100:
            self._state.wander_direction_idx = (
                self._state.wander_direction_idx + 1) % len(WANDER_DIRECTIONS)
            self._state.steps_in_phase = 0

        # --- Miner role ---
        if role == "miner":
            if cargo > 0:
                target = self._closest(tags_by_loc, self._deposit_tags,
                                       require=own_team)
                if target:
                    return self._move_toward(target, tags_by_loc)
            if not has_role_gear:
                target = self._closest(tags_by_loc,
                                       self._role_station_tags["miner"],
                                       require=own_team)
                if target:
                    self._state.phase = AgentPhase.SEEK_GEAR
                    return self._move_toward(target, tags_by_loc)
            else:
                target = self._closest(tags_by_loc, self._extractor_tags)
                if target:
                    self._state.phase = AgentPhase.SEEK_EXTRACTOR
                    return self._move_toward(target, tags_by_loc)
            return self._explore(tags_by_loc)

        # --- Scout role ---
        if role == "scout":
            if not has_role_gear:
                target = self._closest(tags_by_loc,
                                       self._role_station_tags["scout"],
                                       require=own_team)
                if target:
                    return self._move_toward(target, tags_by_loc)
            return self._explore(tags_by_loc)

        # --- Aligner / Scrambler role ---
        # Step 1: Get role gear
        if not has_role_gear:
            target = self._closest(tags_by_loc,
                                   self._role_station_tags[role],
                                   require=own_team)
            if target:
                self._state.phase = AgentPhase.SEEK_GEAR
                return self._move_toward(target, tags_by_loc)
            return self._explore(tags_by_loc)

        # Step 2: Get hearts
        if not has_heart:
            target = self._closest(tags_by_loc, self._heart_source_tags,
                                   require=own_team)
            if target:
                self._state.phase = AgentPhase.SEEK_HEARTS
                return self._move_toward(target, tags_by_loc)
            return self._explore(tags_by_loc)

        # Step 3: Go to junction and align
        if role == "scrambler":
            target = self._closest(tags_by_loc, self._junction_tags,
                                   require=enemy_team)
        else:
            # Aligner: target neutral or enemy junctions (not own team)
            target = self._closest(tags_by_loc, self._junction_tags,
                                   exclude=own_team)

        if target:
            self._state.phase = AgentPhase.SEEK_JUNCTION
            return self._move_toward(target, tags_by_loc)

        return self._explore(tags_by_loc)

    def reset(self, simulation=None) -> None:
        """Reset agent state for new episode."""
        self._state.phase = AgentPhase.SEEK_GEAR
        self._state.steps_in_phase = 0
        self._state.stuck_counter = 0
        self._vibe_cycle_idx = 0


class FastExplorePolicy(MultiAgentPolicy):
    """Multi-agent policy assigning roles and creating per-agent FSM policies.

    Named after DOTA 2 hero Invoker — the spellcaster who combines elements
    to cast powerful spells. Our agents combine gear + hearts + junctions
    to align territory.
    """

    short_names = ["anoop_fast"]

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu",
                 role_cycle: str | None = None, **kwargs: Any):
        super().__init__(policy_env_info, device=device)
        self._agents: dict[int, InvokerAgentPolicy] = {}
        if role_cycle:
            # Accept either "/" or "," as separator for CLI compatibility
            sep = "/" if "/" in role_cycle else ","
            self._roles = tuple(role_cycle.split(sep))
        else:
            self._roles = DEFAULT_ROLE_CYCLE

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        if agent_id not in self._agents:
            role = self._roles[agent_id % len(self._roles)]
            self._agents[agent_id] = InvokerAgentPolicy(
                self._policy_env_info, agent_id, role=role)
        return self._agents[agent_id]
