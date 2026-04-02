# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Long Horizon Memory Environment Implementation.

A memory-selection environment with shaped rewards and deterministic grading.
"""
import os
import json
from pathlib import Path
from uuid import uuid4
import random
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import LongHorizonMemoryAction, LongHorizonMemoryObservation
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import LongHorizonMemoryAction, LongHorizonMemoryObservation
    except (ImportError, ModuleNotFoundError):
        from long_horizon_memory.models import LongHorizonMemoryAction, LongHorizonMemoryObservation


class LongHorizonMemoryEnvironment(Environment):
    """
    Environment where an agent decides what to keep in memory over long horizons.

    Task buckets are easy, medium, and hard and are selected via
    LONG_HORIZON_MEMORY_TASK (easy/medium/hard/all), defaulting to all.
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MEMORY_CAPACITY = 8

    def __init__(self):
        """Initialize the long_horizon_memory environment."""
        episodes_path = Path(__file__).with_name("episodes.json")
        with episodes_path.open("r", encoding="utf-8") as f:
            self.episodes = json.load(f)
        self._task_name = os.getenv("LONG_HORIZON_MEMORY_TASK", "all").strip().lower() or "all"
        seed_env = os.getenv("LONG_HORIZON_MEMORY_SEED")
        self._seed = int(seed_env) if seed_env and seed_env.lstrip("-").isdigit() else None
        self._rng = random.Random(self._seed)
        episode_id_env = os.getenv("LONG_HORIZON_MEMORY_EPISODE_ID")
        self._episode_id_override = (
            int(episode_id_env) if episode_id_env and episode_id_env.lstrip("-").isdigit() else None
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        self.episode = 0
        self.current_domain = "unknown"
        self.current_difficulty = "easy"
        self.messages: List[Dict[str, Any]] = []
        self.total_relevant_in_episode = 0
        self.total_message_number = 0
        self.memory: List[Dict[str, Any]] = []
        self.last_action_error: Optional[str] = None
        self._done = False
        self._set_random_episode()

    def _infer_difficulty(self, episode_data: Dict[str, Any], episode_index: int) -> str:
        explicit = str(episode_data.get("difficulty", "")).strip().lower()
        if explicit in {"easy", "medium", "hard"}:
            return explicit
        if episode_index <= 1:
            return "easy"
        if episode_index <= 3:
            return "medium"
        return "hard"

    def _candidate_indices_for_task(self) -> List[int]:
        if self._task_name not in {"easy", "medium", "hard", "all"}:
            self._task_name = "all"

        if self._task_name == "all":
            return list(range(len(self.episodes)))

        return [
            i
            for i, episode_data in enumerate(self.episodes)
            if self._infer_difficulty(episode_data, i) == self._task_name
        ]

    def _set_random_episode(self) -> None:
        candidates = self._candidate_indices_for_task()
        if not candidates:
            candidates = list(range(len(self.episodes)))

        chosen_episode: Optional[int] = None
        if self._episode_id_override is not None:
            for idx in candidates:
                if int(self.episodes[idx].get("episode_id", idx + 1)) == self._episode_id_override:
                    chosen_episode = idx
                    break

        self.episode = chosen_episode if chosen_episode is not None else self._rng.choice(candidates)
        episode_data = self.episodes[self.episode]
        self.current_domain = episode_data.get("conversation_domain", "unknown")
        self.current_difficulty = self._infer_difficulty(episode_data, self.episode)
        self.messages = episode_data.get("string_relevant_messages", [])
        self.total_relevant_in_episode = sum(1 for m in self.messages if m.get("isRelevant", True))

        self.total_message_number = 0
        self.memory = []
        self.last_action_error = None
        self._done = len(self.messages) == 0

    def _current_message(self) -> Optional[Dict[str, Any]]:
        if self.total_message_number >= len(self.messages):
            return None
        return self.messages[self.total_message_number]

    def _memory_stats(self) -> Dict[str, int]:
        correct = sum(1 for m in self.memory if m.get("isRelevant", False))
        incorrect = len(self.memory) - correct
        return {"correct": correct, "incorrect": incorrect}

    def _compute_quality_metrics(self) -> Dict[str, float]:
        stats = self._memory_stats()
        correct = stats["correct"]
        incorrect = stats["incorrect"]
        kept = len(self.memory)

        precision = correct / kept if kept > 0 else 0.0
        recall = correct / self.total_relevant_in_episode if self.total_relevant_in_episode > 0 else 0.0
        incorrect_rate = incorrect / kept if kept > 0 else 0.0
        overflow = max(0, kept - min(self.total_relevant_in_episode, self.MEMORY_CAPACITY))
        overflow_rate = overflow / self.MEMORY_CAPACITY if self.MEMORY_CAPACITY > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "incorrect_rate": incorrect_rate,
            "overflow_rate": overflow_rate,
        }

    def _task_score(self) -> float:
        metrics = self._compute_quality_metrics()
        score = (
            0.6 * metrics["recall"]
            + 0.4 * metrics["precision"]
            - 0.25 * metrics["incorrect_rate"]
            - 0.15 * metrics["overflow_rate"]
        )
        return max(0.0, min(1.0, score))

    def _compute_reward(self, action_penalty: float = 0.0, terminal: bool = False) -> float:
        metrics = self._compute_quality_metrics()
        shaped = (
            0.7 * metrics["recall"]
            + 0.3 * metrics["precision"]
            - 0.4 * metrics["incorrect_rate"]
            - 0.2 * metrics["overflow_rate"]
            - action_penalty
        )

        if terminal:
            shaped += 0.35 * self._task_score()

        return max(-1.0, min(1.0, shaped))

    def _observation(self, reward: float) -> LongHorizonMemoryObservation:
        current_message = self._current_message()
        new_message = "" if current_message is None else current_message.get("text", "")
        stats = self._memory_stats()

        return LongHorizonMemoryObservation(
            domain=self.current_domain,
            task_name=self.current_difficulty,
            new_message=new_message,
            memory=[m.get("text", "") for m in self.memory],
            memory_count=len(self.memory),
            reward=reward,
            done=self._done,
            metadata={
                "reset_count": self._reset_count,
                "episode_id": self.episodes[self.episode].get("episode_id", self.episode + 1),
                "task": self.current_difficulty,
                "memory_capacity": self.MEMORY_CAPACITY,
                "correct_in_memory": stats["correct"],
                "incorrect_in_memory": stats["incorrect"],
                "task_score": self._task_score(),
                "last_action_error": self.last_action_error,
            },
        )

    def reset(self) -> LongHorizonMemoryObservation:
        """
        Reset the environment.

        Returns:
            LongHorizonMemoryObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._set_random_episode()
        return self._observation(reward=0.0)

    def step(self, action: LongHorizonMemoryAction) -> LongHorizonMemoryObservation:  # type: ignore[override]
        """
        Execute one memory-management step.

        Args:
            action: operation and optional removal index

        Returns:
            LongHorizonMemoryObservation for the next timestep
        """
        self._state.step_count += 1
        self.last_action_error = None
        action_penalty = 0.0

        if self._done:
            self.last_action_error = "episode_already_done"
            return self._observation(reward=-0.25)

        operation = action.operation
        current_message = self._current_message()

        if operation == "add":
            if current_message is None:
                self.last_action_error = "no_current_message"
                action_penalty += 0.15
            elif len(self.memory) >= self.MEMORY_CAPACITY:
                self.last_action_error = "memory_capacity_reached"
                action_penalty += 0.2
            else:
                self.memory.append(
                    {
                        "text": current_message.get("text", ""),
                        "isRelevant": bool(current_message.get("isRelevant", True)),
                    }
                )
        elif operation == "remove":
            idx = action.remove_index
            if idx is None:
                self.last_action_error = "remove_index_required"
                action_penalty += 0.2
            elif idx < 0 or idx >= len(self.memory):
                self.last_action_error = "remove_index_out_of_range"
                action_penalty += 0.2
            else:
                self.memory.pop(idx)
        elif operation == "noop":
            pass
        else:
            self.last_action_error = "invalid_operation"
            action_penalty += 0.2

        self.total_message_number += 1
        if self.total_message_number >= len(self.messages):
            self._done = True

        reward = self._compute_reward(action_penalty=action_penalty, terminal=self._done)
        return self._observation(reward=reward)

    def close(self) -> None:
        """Release environment resources (no-op for local in-memory env)."""
        return None

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state


if __name__ == "__main__":
    pass
