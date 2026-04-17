"""Coordinated multi-agent policy with shared map memory and junction claiming.

Three coordination mechanisms (all via shared state in MultiAgentPolicy):
1. SharedMap — agents pool discovered locations into a global map
2. ClaimRegistry — agents claim junctions to avoid duplicating effort
3. Zone assignment — agents explore different parts of the 88x88 map

No GPU, no LLM, no RL — pure Python heuristics with coordination.
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

# Map move action names to deltas for position tracking
MOVE_ACTION_DELTAS = {
    "move_north": (-1, 0),
    "move_south": (1, 0),
    "move_west": (0, -1),
    "move_east": (0, 1),
}


class AgentPhase(Enum):
    SEEK_GEAR = auto()
    SEEK_HEARTS = auto()
    SEEK_JUNCTION = auto()
    SEEK_EXTRACTOR = auto()
    EXPLORE = auto()


# ---------------------------------------------------------------------------
# Shared coordination structures (owned by CoordinatedPolicy, shared by agents)
# ---------------------------------------------------------------------------

class SharedMap:
    """Global map built from all agents' observations.

    Stores tag sets keyed by estimated global (row, col). Agents report
    what they see each step, converting egocentric coords to global.
    """

    def __init__(self):
        self.tags: dict[tuple[int, int], set[int]] = {}
        self.junction_locs: set[tuple[int, int]] = set()
        self.hub_locs: set[tuple[int, int]] = set()
        self.gear_station_locs: dict[str, set[tuple[int, int]]] = {
            "aligner": set(), "miner": set(), "scrambler": set(), "scout": set(),
        }

    def update(self, agent_global_pos: tuple[int, int],
               egocentric_tags: dict[tuple[int, int], set[int]],
               center: tuple[int, int],
               junction_tag_ids: set[int],
               hub_tag_ids: set[int],
               gear_tag_ids: dict[str, set[int]]) -> None:
        """Merge an agent's observation into the global map."""
        gr, gc = agent_global_pos
        cr, cc = center
        for (er, ec), tag_set in egocentric_tags.items():
            grow = gr + (er - cr)
            gcol = gc + (ec - cc)
            gpos = (grow, gcol)
            existing = self.tags.get(gpos)
            if existing is None:
                self.tags[gpos] = set(tag_set)
            else:
                existing.update(tag_set)

            if tag_set & junction_tag_ids:
                self.junction_locs.add(gpos)
            if tag_set & hub_tag_ids:
                self.hub_locs.add(gpos)
            for role, role_tags in gear_tag_ids.items():
                if tag_set & role_tags:
                    self.gear_station_locs[role].add(gpos)

    def reset(self) -> None:
        self.tags.clear()
        self.junction_locs.clear()
        self.hub_locs.clear()
        for s in self.gear_station_locs.values():
            s.clear()


class ClaimRegistry:
    """Tracks which agent is targeting which junction."""

    def __init__(self):
        # junction_pos -> (agent_id, claim_step)
        self._claims: dict[tuple[int, int], tuple[int, int]] = {}

    def claim(self, pos: tuple[int, int], agent_id: int, step: int) -> None:
        self._claims[pos] = (agent_id, step)

    def release(self, agent_id: int) -> None:
        to_remove = [p for p, (aid, _) in self._claims.items() if aid == agent_id]
        for p in to_remove:
            del self._claims[p]

    def is_claimed(self, pos: tuple[int, int], agent_id: int, current_step: int) -> bool:
        """Check if pos is claimed by ANOTHER agent (not expired)."""
        claim = self._claims.get(pos)
        if claim is None:
            return False
        other_id, claim_step = claim
        if other_id == agent_id:
            return False  # own claim
        if current_step - claim_step > 500:
            return False  # expired
        return True

    def reset(self) -> None:
        self._claims.clear()


# ---------------------------------------------------------------------------
# Per-agent policy with coordination
# ---------------------------------------------------------------------------

@dataclass
class CoordAgentState:
    role: str = "aligner"
    phase: AgentPhase = AgentPhase.SEEK_GEAR
    wander_direction_idx: int = 0
    steps_in_phase: int = 0
    total_steps: int = 0
    global_row: int = 0
    global_col: int = 0
    last_move_action: str | None = None
    target_global: tuple[int, int] | None = None


class CoordinatedAgentPolicy(AgentPolicy):
    """Agent policy that uses shared map + claims for coordination."""

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int,
                 role: str, shared_map: SharedMap, claims: ClaimRegistry,
                 zone: tuple[int, int, int, int]):
        super().__init__(policy_env_info)
        self._agent_id = agent_id
        self._shared_map = shared_map
        self._claims = claims
        self._zone = zone  # (row_start, row_end, col_start, col_end)
        self._state = CoordAgentState(
            role=role,
            wander_direction_idx=agent_id % len(WANDER_DIRECTIONS),
        )
        self._center = (policy_env_info.obs_height // 2,
                        policy_env_info.obs_width // 2)

        # Tag lookups (same as clean_aligner)
        self._tag_name_to_id = {name: idx for idx, name in enumerate(policy_env_info.tags)}
        self._team_tag_ids = {idx for idx, name in enumerate(policy_env_info.tags)
                              if name.startswith(TEAM_TAG_PREFIX)}
        self._agent_tags = self._resolve_tags(["agent"])
        self._junction_tags = self._resolve_tags(["junction"])
        self._heart_source_tags = self._resolve_tags(["hub", "chest"])
        self._deposit_tags = self._resolve_tags(["hub", "junction"])
        self._extractor_tags = self._resolve_tags([f"{e}_extractor" for e in ELEMENTS])
        self._role_station_tags = {}
        self._role_station_tag_sets = {}
        for r in ("miner", "aligner", "scrambler", "scout"):
            tags = self._resolve_tags([r, f"c:{r}"])
            self._role_station_tags[r] = tags
            self._role_station_tag_sets[r] = tags

        all_actions = policy_env_info.all_action_names
        self._action_set = set(all_actions)
        self._fallback = "noop" if "noop" in self._action_set else all_actions[0]
        self._vibe_actions = policy_env_info.vibe_action_names
        self._vibe_cycle_idx = 0

        # Feature ID for last_action_move
        self._last_action_move_feature_id = None
        for f in policy_env_info.obs_features:
            if f.name == "last_action_move":
                self._last_action_move_feature_id = f.id
                break

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
        result: dict[tuple[int, int], set[int]] = {}
        for token in obs.tokens:
            if token.feature.name != "tag" or token.location is None:
                continue
            result.setdefault(token.location, set()).add(token.value)
        return result

    def _update_global_position(self, obs: AgentObservation) -> None:
        """Track global position by checking if last move succeeded."""
        if self._state.last_move_action is None:
            return
        # Check last_action_move feature at center
        move_succeeded = False
        if self._last_action_move_feature_id is not None:
            for token in obs.tokens:
                if (token.location == self._center and
                        token.feature.id == self._last_action_move_feature_id):
                    move_succeeded = int(token.value) > 0
                    break
        if move_succeeded:
            delta = MOVE_ACTION_DELTAS.get(self._state.last_move_action)
            if delta:
                self._state.global_row += delta[0]
                self._state.global_col += delta[1]
        self._state.last_move_action = None

    def _closest_visible(self, tags_by_loc: dict[tuple[int, int], set[int]],
                         include: set[int],
                         require: set[int] | None = None,
                         exclude: set[int] | None = None) -> tuple[int, int] | None:
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

    def _best_junction_from_map(self, own_team: set[int]) -> tuple[int, int] | None:
        """Find best unclaimed junction from shared map."""
        my_pos = (self._state.global_row, self._state.global_col)
        best = None
        best_dist = float("inf")
        for jpos in self._shared_map.junction_locs:
            # Skip junctions claimed by other agents
            if self._claims.is_claimed(jpos, self._agent_id, self._state.total_steps):
                continue
            # Skip junctions that appear to be on our team already
            jtags = self._shared_map.tags.get(jpos, set())
            if jtags & own_team:
                continue
            dist = abs(jpos[0] - my_pos[0]) + abs(jpos[1] - my_pos[1])
            if dist < best_dist:
                best_dist = dist
                best = jpos
        return best

    def _move_toward_global(self, global_target: tuple[int, int],
                            tags_by_loc: dict[tuple[int, int], set[int]]) -> Action:
        """Move toward a global position by converting to egocentric direction."""
        dr = global_target[0] - self._state.global_row
        dc = global_target[1] - self._state.global_col

        if dr == 0 and dc == 0:
            return self._cycle_vibe()

        # Pick the cardinal direction that reduces distance most
        blocked = set(tags_by_loc)
        blocked.discard(self._center)

        # Prefer the axis with larger distance
        if abs(dr) >= abs(dc):
            primary = "south" if dr > 0 else "north"
            secondary = "east" if dc > 0 else "west" if dc != 0 else None
        else:
            primary = "east" if dc > 0 else "west"
            secondary = "south" if dr > 0 else "north" if dr != 0 else None

        for direction in ([primary] + ([secondary] if secondary else [])):
            delta = MOVE_DELTAS[direction]
            nxt = (self._center[0] + delta[0], self._center[1] + delta[1])
            if (nxt not in blocked
                    and 0 <= nxt[0] < self._policy_env_info.obs_height
                    and 0 <= nxt[1] < self._policy_env_info.obs_width):
                action_name = f"move_{direction}"
                self._state.last_move_action = action_name
                return self._action(action_name)

        # Both preferred directions blocked — try any unblocked
        for direction in WANDER_DIRECTIONS:
            delta = MOVE_DELTAS[direction]
            nxt = (self._center[0] + delta[0], self._center[1] + delta[1])
            if (nxt not in blocked
                    and 0 <= nxt[0] < self._policy_env_info.obs_height
                    and 0 <= nxt[1] < self._policy_env_info.obs_width):
                action_name = f"move_{direction}"
                self._state.last_move_action = action_name
                return self._action(action_name)

        return self._cycle_vibe()

    def _action(self, name: str) -> Action:
        return Action(name=name if name in self._action_set else self._fallback)

    def _move_toward(self, target: tuple[int, int],
                     tags_by_loc: dict[tuple[int, int], set[int]]) -> Action:
        """Move toward a visible egocentric target using BFS."""
        dr = target[0] - self._center[0]
        dc = target[1] - self._center[1]

        if dr == 0 and dc == 0:
            return self._cycle_vibe()

        if abs(dr) + abs(dc) == 1:
            if abs(dr) >= abs(dc):
                direction = "south" if dr > 0 else "north"
            else:
                direction = "east" if dc > 0 else "west"
            action_name = f"move_{direction}"
            self._state.last_move_action = action_name
            return self._action(action_name)

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
                    action_name = f"move_{first_dir[nxt]}"
                    self._state.last_move_action = action_name
                    return self._action(action_name)
                queue.append(nxt)

        return self._explore(tags_by_loc)

    def _cycle_vibe(self) -> Action:
        if not self._vibe_actions:
            return self._action(self._fallback)
        action_name = self._vibe_actions[self._vibe_cycle_idx % len(self._vibe_actions)]
        self._vibe_cycle_idx += 1
        return self._action(action_name)

    def _explore(self, tags_by_loc: dict[tuple[int, int], set[int]]) -> Action:
        """Explore using original StarterPolicy pattern (proven to work)."""
        blocked = set(tags_by_loc)
        blocked.discard(self._center)

        for offset in range(len(WANDER_DIRECTIONS)):
            idx = (self._state.wander_direction_idx + offset) % len(WANDER_DIRECTIONS)
            direction = WANDER_DIRECTIONS[idx]
            delta = MOVE_DELTAS[direction]
            nxt = (self._center[0] + delta[0], self._center[1] + delta[1])
            if (nxt not in blocked
                    and 0 <= nxt[0] < self._policy_env_info.obs_height
                    and 0 <= nxt[1] < self._policy_env_info.obs_width):
                action_name = f"move_{direction}"
                self._state.last_move_action = action_name
                return self._action(action_name)

        return self._cycle_vibe()

    def step(self, obs: AgentObservation) -> Action:
        """Main decision logic with coordination."""
        # Track global position from last move
        self._update_global_position(obs)
        self._state.total_steps += 1

        items = self._inventory(obs)
        tags_by_loc = self._tags_by_location(obs)

        # Report observations to shared map
        self._shared_map.update(
            (self._state.global_row, self._state.global_col),
            tags_by_loc,
            self._center,
            self._junction_tags,
            self._heart_source_tags,
            self._role_station_tag_sets,
        )

        own_team = tags_by_loc.get(self._center, set()) & self._team_tag_ids
        enemy_team = (self._team_tag_ids - own_team) or self._team_tag_ids

        has_role_gear = items.get(self._state.role, 0) > 0
        has_heart = items.get("heart", 0) > 0
        cargo = sum(items.get(e, 0) for e in ELEMENTS)
        role = self._state.role

        self._state.steps_in_phase += 1
        if self._state.steps_in_phase > 200:
            self._state.wander_direction_idx = (
                self._state.wander_direction_idx + 1) % len(WANDER_DIRECTIONS)
            self._state.steps_in_phase = 0
            # Release stale claims
            self._claims.release(self._agent_id)

        # --- Miner role ---
        if role == "miner":
            if cargo > 0:
                target = self._closest_visible(tags_by_loc, self._deposit_tags,
                                               require=own_team)
                if target:
                    return self._move_toward(target, tags_by_loc)
            if not has_role_gear:
                target = self._closest_visible(tags_by_loc,
                                               self._role_station_tags["miner"],
                                               require=own_team)
                if target:
                    self._state.phase = AgentPhase.SEEK_GEAR
                    return self._move_toward(target, tags_by_loc)
            else:
                target = self._closest_visible(tags_by_loc, self._extractor_tags)
                if target:
                    self._state.phase = AgentPhase.SEEK_EXTRACTOR
                    return self._move_toward(target, tags_by_loc)
            return self._explore(tags_by_loc)

        # --- Scout role ---
        if role == "scout":
            if not has_role_gear:
                target = self._closest_visible(tags_by_loc,
                                               self._role_station_tags["scout"],
                                               require=own_team)
                if target:
                    return self._move_toward(target, tags_by_loc)
            return self._explore(tags_by_loc)

        # --- Aligner / Scrambler ---
        # Step 1: Get gear
        if not has_role_gear:
            target = self._closest_visible(tags_by_loc,
                                           self._role_station_tags[role],
                                           require=own_team)
            if target:
                self._state.phase = AgentPhase.SEEK_GEAR
                return self._move_toward(target, tags_by_loc)
            return self._explore(tags_by_loc)

        # Step 2: Get hearts
        if not has_heart:
            target = self._closest_visible(tags_by_loc, self._heart_source_tags,
                                           require=own_team)
            if target:
                self._state.phase = AgentPhase.SEEK_HEARTS
                return self._move_toward(target, tags_by_loc)
            return self._explore(tags_by_loc)

        # Step 3: Find junction — use CLAIMING to avoid duplicates
        self._state.phase = AgentPhase.SEEK_JUNCTION

        # First: try visible junctions (not claimed by others)
        if role == "scrambler":
            target = self._closest_visible(tags_by_loc, self._junction_tags,
                                           require=enemy_team)
        else:
            # Find closest visible unclaimed junction
            candidates = [
                loc for loc, loc_tags in tags_by_loc.items()
                if loc_tags & self._junction_tags
                and not (loc_tags & own_team)
            ]
            # Filter out claimed junctions
            global_pos = (self._state.global_row, self._state.global_col)
            unclaimed = []
            for loc in candidates:
                gpos = (global_pos[0] + loc[0] - self._center[0],
                        global_pos[1] + loc[1] - self._center[1])
                if not self._claims.is_claimed(gpos, self._agent_id, self._state.total_steps):
                    unclaimed.append(loc)

            if unclaimed:
                target = min(unclaimed,
                             key=lambda loc: (abs(loc[0] - self._center[0]) +
                                              abs(loc[1] - self._center[1])))
                # Claim it
                gpos = (global_pos[0] + target[0] - self._center[0],
                        global_pos[1] + target[1] - self._center[1])
                self._claims.claim(gpos, self._agent_id, self._state.total_steps)
                return self._move_toward(target, tags_by_loc)
            else:
                target = None

        if target:
            # Claim the junction
            gpos = (self._state.global_row + target[0] - self._center[0],
                    self._state.global_col + target[1] - self._center[1])
            self._claims.claim(gpos, self._agent_id, self._state.total_steps)
            return self._move_toward(target, tags_by_loc)

        # No visible junction — check shared map for known unclaimed junctions
        map_junction = self._best_junction_from_map(own_team)
        if map_junction:
            self._claims.claim(map_junction, self._agent_id, self._state.total_steps)
            self._state.target_global = map_junction
            return self._move_toward_global(map_junction, tags_by_loc)

        return self._explore(tags_by_loc)

    def reset(self, simulation=None) -> None:
        self._state.phase = AgentPhase.SEEK_GEAR
        self._state.steps_in_phase = 0
        self._state.total_steps = 0
        self._state.global_row = 0
        self._state.global_col = 0
        self._state.last_move_action = None
        self._state.target_global = None
        self._vibe_cycle_idx = 0


# ---------------------------------------------------------------------------
# Multi-agent policy with coordination
# ---------------------------------------------------------------------------

# Zone grid: 2 rows x 4 cols = 8 zones on 88x88 map
ZONE_ROWS = 2
ZONE_COLS = 4
MAP_SIZE = 88


def _zone_for_agent(agent_id: int) -> tuple[int, int, int, int]:
    """Assign agent to a map zone based on ID."""
    zone_idx = agent_id % (ZONE_ROWS * ZONE_COLS)
    zone_r = zone_idx // ZONE_COLS
    zone_c = zone_idx % ZONE_COLS
    row_size = MAP_SIZE // ZONE_ROWS
    col_size = MAP_SIZE // ZONE_COLS
    return (zone_r * row_size, (zone_r + 1) * row_size,
            zone_c * col_size, (zone_c + 1) * col_size)


class CoordinatedPolicy(MultiAgentPolicy):
    """Multi-agent policy with shared map memory and junction claiming.

    Named after DOTA 2 team coordination — agents work together
    instead of independently.
    """

    short_names = ["anoop_coord"]

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu",
                 role_cycle: str | None = None, **kwargs: Any):
        super().__init__(policy_env_info, device=device)
        self._agents: dict[int, CoordinatedAgentPolicy] = {}
        self._shared_map = SharedMap()
        self._claims = ClaimRegistry()
        if role_cycle:
            sep = "/" if "/" in role_cycle else ","
            self._roles = tuple(role_cycle.split(sep))
        else:
            self._roles = ("aligner",) * 8  # pure aligner by default

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        if agent_id not in self._agents:
            role = self._roles[agent_id % len(self._roles)]
            zone = _zone_for_agent(agent_id)
            self._agents[agent_id] = CoordinatedAgentPolicy(
                self._policy_env_info, agent_id, role,
                self._shared_map, self._claims, zone)
        return self._agents[agent_id]

    def reset(self) -> None:
        self._shared_map.reset()
        self._claims.reset()
        for agent in self._agents.values():
            agent.reset()
