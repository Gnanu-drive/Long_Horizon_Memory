"""Competition-compliant inference runner for long_horizon_memory."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

from openai import OpenAI

try:
    from models import LongHorizonMemoryAction, LongHorizonMemoryObservation
    from server.long_horizon_memory_environment import LongHorizonMemoryEnvironment
except (ImportError, ModuleNotFoundError):
    try:
        from .models import LongHorizonMemoryAction, LongHorizonMemoryObservation
        from .server.long_horizon_memory_environment import LongHorizonMemoryEnvironment
    except (ImportError, ModuleNotFoundError):
        from long_horizon_memory.models import LongHorizonMemoryAction, LongHorizonMemoryObservation
        from long_horizon_memory.server.long_horizon_memory_environment import LongHorizonMemoryEnvironment

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("MY_ENV_BENCHMARK", "long_horizon_memory")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.7"))
TASKS = ["easy", "medium", "hard"]
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
MAX_MODEL_RETRIES = int(os.getenv("MAX_MODEL_RETRIES", "2"))
BASELINE_SEED = int(os.getenv("BASELINE_SEED", "1337"))

SYSTEM_PROMPT = (
    "You manage a small memory under noise. Return JSON only with keys: "
    "operation (add/remove/noop) and optional remove_index integer."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    success_val = str(success).lower()
    rewards_text = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_val} steps={steps} rewards={rewards_text}", flush=True)


def _heuristic_action(observation: LongHorizonMemoryObservation) -> LongHorizonMemoryAction:
    """Safe fallback policy when LLM output is invalid."""
    text = observation.new_message.lower()
    if any(k in text for k in ["hobby", "weekend", "bought", "coffee", "keyboard", "laptop"]):
        if observation.memory_count > 0:
            return LongHorizonMemoryAction(operation="remove", remove_index=observation.memory_count - 1)
        return LongHorizonMemoryAction(operation="noop")

    if observation.memory_count < 8:
        return LongHorizonMemoryAction(operation="add")

    return LongHorizonMemoryAction(operation="remove", remove_index=observation.memory_count - 1)


def _parse_action(content: str, observation: LongHorizonMemoryObservation) -> LongHorizonMemoryAction:
    normalized = content.strip()
    if normalized.startswith("```"):
        normalized = normalized.strip("`")
        normalized = normalized.replace("json", "", 1).strip()

    try:
        payload = json.loads(normalized)
        op = payload.get("operation", "noop")
        if op == "remove":
            idx = payload.get("remove_index")
            if isinstance(idx, int):
                return LongHorizonMemoryAction(operation="remove", remove_index=idx)
            return LongHorizonMemoryAction(operation="noop")
        if op in {"add", "noop"}:
            return LongHorizonMemoryAction(operation=op)
    except Exception:
        pass
    return _heuristic_action(observation)


def choose_action(
    llm: OpenAI,
    observation: LongHorizonMemoryObservation,
    task_name: str,
) -> LongHorizonMemoryAction:
    user_prompt = {
        "task": task_name,
        "new_message": observation.new_message,
        "memory": observation.memory,
        "memory_count": observation.memory_count,
    }

    last_error: Optional[Exception] = None
    for _ in range(MAX_MODEL_RETRIES + 1):
        try:
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_prompt)},
                ],
                temperature=0.0,
                max_tokens=60,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            content = completion.choices[0].message.content or "{}"
            return _parse_action(content.strip(), observation)
        except Exception as exc:
            last_error = exc

    _ = last_error
    return _heuristic_action(observation)


def action_to_text(action: LongHorizonMemoryAction) -> str:
    if action.operation == "remove":
        return f"remove:{action.remove_index}"
    return action.operation


def run_task(task_name: str, llm: OpenAI) -> Tuple[bool, List[float]]:
    task_seed = BASELINE_SEED + TASKS.index(task_name)
    os.environ["LONG_HORIZON_MEMORY_SEED"] = str(task_seed)
    os.environ["LONG_HORIZON_MEMORY_TASK"] = task_name
    env = LongHorizonMemoryEnvironment()

    observation = env.reset()
    log_start(task_name, BENCHMARK, MODEL_NAME)

    rewards: List[float] = []
    success = False
    step_count = 0

    try:
        for step in range(1, MAX_STEPS + 1):
            step_count = step
            action = choose_action(llm, observation, task_name)
            observation = env.step(action)

            reward = float(observation.reward)
            done = bool(observation.done)
            error = observation.metadata.get("last_action_error")

            rewards.append(reward)
            log_step(step, action_to_text(action), reward, done, error)

            if done:
                score = float(observation.metadata.get("task_score", 0.0))
                success = score >= SUCCESS_SCORE_THRESHOLD
                break

        if not bool(observation.done):
            score = float(observation.metadata.get("task_score", 0.0))
            success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        log_step(step_count + 1, "noop", 0.0, True, str(exc))
        success = False
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        log_end(success, len(rewards), rewards)

    return success, rewards


def main() -> None:
    if not API_KEY:
        raise ValueError("HF_TOKEN or OPENAI_API_KEY must be set for inference.")

    llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    for task in TASKS:
        run_task(task, llm)


if __name__ == "__main__":
    main()
