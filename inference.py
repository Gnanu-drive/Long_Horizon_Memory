"""Competition-compliant inference runner for long_horizon_memory."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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

HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = os.getenv("MY_ENV_BENCHMARK", "long_horizon_memory")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.7"))
TASKS = ["easy", "medium", "hard"]
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
MAX_MODEL_RETRIES = int(os.getenv("MAX_MODEL_RETRIES", "2"))
BASELINE_SEED = int(os.getenv("BASELINE_SEED", "1337"))
SCORE_EPSILON = float(os.getenv("SCORE_EPSILON", "0.001"))
ENABLE_DEBUG_LOGS = os.getenv("ENABLE_DEBUG_LOGS", "false").lower() == "true"

SYSTEM_PROMPT = """You are an expert memory management system for long-horizon tasks. Your goal is to maintain a high-quality memory buffer by keeping relevant information and discarding noise.

SCORING FORMULA (aim for ≥0.7 for success):
task_score = 0.6×recall + 0.4×precision - 0.25×incorrect_rate - 0.15×overflow_rate

KEY INSIGHTS:
- Recall (60% weight) is MORE important than precision (40%)
- Memory capacity: 8 slots maximum
- Prioritize capturing ALL relevant info, but avoid keeping irrelevant distractors
- Better to keep relevant items even if some noise, than miss important info

RELEVANCE INDICATORS (keep these):
- Technical problems, bugs, requirements, constraints
- User goals, pain points, specific needs
- System design decisions, architecture details
- Performance issues, metrics, monitoring needs
- Domain-specific technical details

IRRELEVANCE INDICATORS (discard these):
- Personal hobbies (hobbies, sports, cooking, photography, music)
- Shopping and purchases (buying items, products)
- Lifestyle and daily activities (weekend plans, food, exercise)
- Entertainment (movies, shows, games, books unrelated to task)
- Random observations with no technical connection

STRATEGY:
1. ADD: If message contains technical content relevant to the domain
2. REMOVE: If memory contains noise AND you need space for better content
3. NOOP: If current message is noise and memory is already optimal

CRITICAL: You must output ONLY valid JSON with this exact format:
{"operation": "add"}
OR
{"operation": "remove", "remove_index": 0}
OR
{"operation": "noop"}

Think carefully about each decision based on the scoring formula."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"

    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def _strict_score(score: float) -> float:
    """Clamp score to strict open interval (0, 1) for competition compliance."""
    eps = min(max(SCORE_EPSILON, 1e-9), 0.49)
    return min(max(score, eps), 1.0 - eps)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_val = str(success).lower()
    rewards_text = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_text}",
        flush=True,
    )


def _heuristic_action(observation: LongHorizonMemoryObservation) -> LongHorizonMemoryAction:
    """Improved fallback policy with smarter pattern matching."""
    text = observation.new_message.lower()

    # Expanded irrelevance patterns
    irrelevance_keywords = [
        "hobby", "hobbies", "weekend", "bought", "buy", "purchase", "shopping",
        "coffee", "keyboard", "laptop", "gaming", "game", "movie", "show",
        "cooking", "recipe", "food", "restaurant", "pizza", "lunch", "dinner",
        "sport", "football", "basketball", "running", "jogging", "gym",
        "paint", "painting", "music", "guitar", "piano", "art",
        "vacation", "trip", "travel", "hiking", "concert", "theater",
        "photography", "camera", "book club", "reading", "novel",
        "dog", "cat", "pet", "apartment", "house", "renovating",
        "meditation", "yoga", "sleep", "tire", "bicycle", "neighbor"
    ]

    # Technical relevance patterns
    relevance_keywords = [
        "bug", "error", "issue", "problem", "fail", "crash", "slow",
        "performance", "memory", "cpu", "api", "database", "server",
        "code", "script", "function", "class", "implement", "design",
        "architecture", "system", "pipeline", "process", "monitoring",
        "test", "debug", "optimize", "scale", "deploy", "build",
        "requirement", "feature", "user", "client", "data", "model",
        "algorithm", "metric", "score", "training", "prediction"
    ]

    # Check if message is irrelevant noise
    is_irrelevant = any(kw in text for kw in irrelevance_keywords)
    is_relevant = any(kw in text for kw in relevance_keywords)

    # Decision logic
    if is_irrelevant and not is_relevant:
        # Message is noise, don't add it
        return LongHorizonMemoryAction(operation="noop")

    if is_relevant:
        # Message is relevant
        if observation.memory_count < 8:
            return LongHorizonMemoryAction(operation="add")
        else:
            # Memory full, remove oldest to make room
            return LongHorizonMemoryAction(operation="remove", remove_index=0)

    # Neutral message - conservative approach
    if observation.memory_count < 6:  # Still have room
        return LongHorizonMemoryAction(operation="add")

    return LongHorizonMemoryAction(operation="noop")


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
    # Build enhanced context-aware prompt
    memory_summary = (
        f"Currently storing {observation.memory_count}/8 items in memory.\n"
        f"Current memory contents:\n"
        + "\n".join(f"  [{i}] {msg}" for i, msg in enumerate(observation.memory))
        if observation.memory
        else "Memory is empty (0/8 items)."
    )

    metadata = observation.metadata
    current_score = metadata.get("task_score", 0.0)
    correct_count = metadata.get("correct_in_memory", 0)
    incorrect_count = metadata.get("incorrect_in_memory", 0)

    user_prompt = f"""TASK DIFFICULTY: {task_name}
DOMAIN: {observation.domain}

CURRENT STATE:
{memory_summary}

PERFORMANCE METRICS:
- Current task score: {current_score:.2f} (need ≥0.70 for success)
- Correct items in memory: {correct_count}
- Incorrect items in memory: {incorrect_count}
- Recall: {correct_count}/{metadata.get('memory_capacity', 8)} slots used

NEW INCOMING MESSAGE:
"{observation.new_message}"

DECISION REQUIRED:
Analyze if the new message is relevant to the {observation.domain} domain task.
Consider current memory state and whether you need to make room or keep current optimal state.
Output your decision as JSON only."""

    last_error: Optional[Exception] = None
    for attempt in range(MAX_MODEL_RETRIES + 1):
        try:
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Slightly higher for better reasoning
                max_tokens=150,   # More tokens for reasoning
                timeout=LLM_TIMEOUT_SECONDS,
            )
            content = completion.choices[0].message.content or "{}"
            if ENABLE_DEBUG_LOGS:
                print(f"[LLM] Using {MODEL_NAME} (attempt {attempt + 1})", flush=True)
            return _parse_action(content.strip(), observation)
        except Exception as exc:
            last_error = exc
            if ENABLE_DEBUG_LOGS:
                print(f"[LLM] API call failed (attempt {attempt + 1}): {str(exc)[:100]}", flush=True)

    if ENABLE_DEBUG_LOGS:
        print(f"[LLM] All retries exhausted, using heuristic fallback", flush=True)
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
                raw_score = float(observation.metadata.get("task_score", 0.0))
                score = _strict_score(raw_score)
                success = score >= SUCCESS_SCORE_THRESHOLD
                break

        if not bool(observation.done):
            raw_score = float(observation.metadata.get("task_score", 0.0))
            score = _strict_score(raw_score)
            success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        log_step(step_count + 1, "noop", 0.0, True, str(exc))
        success = False
    finally:
        raw_final_score = float(observation.metadata.get("task_score", 0.0)) if observation else 0.0
        final_score = _strict_score(raw_final_score)
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        log_end(success, len(rewards), final_score, rewards)

    return success, rewards


def main() -> None:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN must be set for inference.")

    llm = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    # Check if we should run all episodes
    run_all_episodes = os.getenv("RUN_ALL_EPISODES", "false").lower() == "true"

    if run_all_episodes:
        # Run each specific episode ID (will be 1-24 after we add new episodes)
        # For now start with 1-9, will expand to 24
        if ENABLE_DEBUG_LOGS:
            print(f"[INFO] Running all episodes individually", flush=True)

        total_success = 0
        all_scores = []

        # Determine max episode ID by checking episodes.json
        try:
            with open("server/episodes.json", "r", encoding="utf-8") as f:
                episodes_data = json.load(f)
                max_episode_id = max(ep["episode_id"] for ep in episodes_data)
        except Exception:
            max_episode_id = 9  # Default to 9 if can't read

        for ep_id in range(1, max_episode_id + 1):
            os.environ["LONG_HORIZON_MEMORY_EPISODE_ID"] = str(ep_id)
            os.environ["LONG_HORIZON_MEMORY_TASK"] = "all"

            # Load episode to get difficulty and domain
            try:
                with open("server/episodes.json", "r", encoding="utf-8") as f:
                    episodes_data = json.load(f)
                    episode = next((ep for ep in episodes_data if ep["episode_id"] == ep_id), None)
                    if episode:
                        difficulty = episode.get("difficulty", "unknown")
                        domain = episode.get("conversation_domain", "unknown")
                        if ENABLE_DEBUG_LOGS:
                            print(f"\n[EPISODE] id={ep_id} difficulty={difficulty} domain={domain}", flush=True)
            except Exception:
                pass

            env = LongHorizonMemoryEnvironment()
            observation = env.reset()
            log_start("all", BENCHMARK, MODEL_NAME)

            rewards: List[float] = []
            success = False

            try:
                for step in range(1, MAX_STEPS + 1):
                    action = choose_action(llm, observation, "all")
                    observation = env.step(action)

                    reward = float(observation.reward)
                    done = bool(observation.done)
                    error = observation.metadata.get("last_action_error")

                    rewards.append(reward)
                    log_step(step, action_to_text(action), reward, done, error)

                    if done:
                        raw_score = float(observation.metadata.get("task_score", 0.0))
                        score = _strict_score(raw_score)
                        success = score >= SUCCESS_SCORE_THRESHOLD
                        all_scores.append(score)
                        break

                if not bool(observation.done):
                    raw_score = float(observation.metadata.get("task_score", 0.0))
                    score = _strict_score(raw_score)
                    success = score >= SUCCESS_SCORE_THRESHOLD
                    all_scores.append(score)
            except Exception as exc:
                log_step(len(rewards) + 1, "noop", 0.0, True, str(exc))
                success = False
                all_scores.append(_strict_score(0.0))
            finally:
                raw_final_score = float(observation.metadata.get("task_score", 0.0)) if observation else 0.0
                final_score = _strict_score(raw_final_score)
                if hasattr(env, "close"):
                    try:
                        env.close()
                    except Exception:
                        pass
                log_end(success, len(rewards), final_score, rewards)

            if success:
                total_success += 1

        # Print summary
        success_rate = (total_success / max_episode_id) * 100 if max_episode_id > 0 else 0
        avg_final_reward = sum(all_scores) / len(all_scores) if all_scores else 0.0
        if ENABLE_DEBUG_LOGS:
            print(f"\n[SUMMARY] Total: {max_episode_id} episodes | Success: {total_success} ({success_rate:.1f}%) | Avg Final Reward: {avg_final_reward:.3f}", flush=True)
    else:
        # Original behavior: run easy, medium, hard tasks
        for task in TASKS:
            run_task(task, llm)


if __name__ == "__main__":
    main()
