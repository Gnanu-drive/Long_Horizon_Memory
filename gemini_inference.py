"""Gemini-powered inference runner for long_horizon_memory with performance graphs."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("google-genai not installed. Run: pip install google-genai", file=sys.stderr)
    sys.exit(1)

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

# ---------------------------------------------------------------------------
# Configuration — all secrets read from environment, never hardcoded
# ---------------------------------------------------------------------------
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
N_EPISODES: int = int(os.getenv("GEMINI_N_EPISODES", "5"))
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "20"))
SUCCESS_SCORE_THRESHOLD: float = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.7"))
SCORE_EPSILON: float = float(os.getenv("SCORE_EPSILON", "0.001"))
ENABLE_DEBUG_LOGS: bool = os.getenv("ENABLE_DEBUG_LOGS", "false").lower() == "true"
GRAPHS_DIR: str = os.getenv("GRAPHS_DIR", ".")
GEMINI_TIMEOUT: float = float(os.getenv("GEMINI_TIMEOUT", "15.0"))

SYSTEM_PROMPT = """You are an expert memory management system for long-horizon tasks.
Your goal is to maintain a high-quality memory buffer by keeping relevant information and discarding noise.

SCORING FORMULA (aim for >=0.7 for success):
task_score = 0.6*recall + 0.4*precision - 0.25*incorrect_rate - 0.15*overflow_rate

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


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _heuristic_action(observation: LongHorizonMemoryObservation) -> LongHorizonMemoryAction:
    text = observation.new_message.lower()
    irrelevance_kw = [
        "hobby", "hobbies", "weekend", "bought", "buy", "purchase", "shopping",
        "coffee", "keyboard", "laptop", "gaming", "game", "movie", "show",
        "cooking", "recipe", "food", "restaurant", "pizza", "lunch", "dinner",
        "sport", "football", "basketball", "running", "jogging", "gym",
        "paint", "painting", "music", "guitar", "piano", "art",
        "vacation", "trip", "travel", "hiking", "concert", "theater",
        "photography", "camera", "book club", "reading", "novel",
        "dog", "cat", "pet", "apartment", "house", "renovating",
        "meditation", "yoga", "sleep", "tire", "bicycle", "neighbor",
    ]
    relevance_kw = [
        "bug", "error", "issue", "problem", "fail", "crash", "slow",
        "performance", "memory", "cpu", "api", "database", "server",
        "code", "script", "function", "class", "implement", "design",
        "architecture", "system", "pipeline", "process", "monitoring",
        "test", "debug", "optimize", "scale", "deploy", "build",
        "requirement", "feature", "user", "client", "data", "model",
        "algorithm", "metric", "score", "training", "prediction",
    ]
    is_irrelevant = any(kw in text for kw in irrelevance_kw)
    is_relevant = any(kw in text for kw in relevance_kw)

    if is_irrelevant and not is_relevant:
        return LongHorizonMemoryAction(operation="noop")
    if is_relevant:
        if observation.memory_count < 8:
            return LongHorizonMemoryAction(operation="add")
        return LongHorizonMemoryAction(operation="remove", remove_index=0)
    if observation.memory_count < 6:
        return LongHorizonMemoryAction(operation="add")
    return LongHorizonMemoryAction(operation="noop")


def _parse_action(content: str, observation: LongHorizonMemoryObservation) -> LongHorizonMemoryAction:
    normalized = content.strip()
    if normalized.startswith("```"):
        normalized = normalized.strip("`").replace("json", "", 1).strip()
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


# ---------------------------------------------------------------------------
# Gemini action chooser
# ---------------------------------------------------------------------------

def _build_user_prompt(observation: LongHorizonMemoryObservation, task_name: str) -> str:
    metadata = observation.metadata
    if observation.memory:
        memory_summary = (
            f"Currently storing {observation.memory_count}/8 items in memory.\n"
            "Current memory contents:\n"
            + "\n".join(f"  [{i}] {msg}" for i, msg in enumerate(observation.memory))
        )
    else:
        memory_summary = "Memory is empty (0/8 items)."

    current_score = metadata.get("task_score", 0.0)
    correct_count = metadata.get("correct_in_memory", 0)
    incorrect_count = metadata.get("incorrect_in_memory", 0)

    return (
        f"TASK DIFFICULTY: {task_name}\n"
        f"DOMAIN: {observation.domain}\n\n"
        f"CURRENT STATE:\n{memory_summary}\n\n"
        f"PERFORMANCE METRICS:\n"
        f"- Current task score: {current_score:.2f} (need >=0.70 for success)\n"
        f"- Correct items in memory: {correct_count}\n"
        f"- Incorrect items in memory: {incorrect_count}\n"
        f"- Recall: {correct_count}/{metadata.get('memory_capacity', 8)} slots used\n\n"
        f'NEW INCOMING MESSAGE:\n"{observation.new_message}"\n\n'
        f"DECISION REQUIRED:\n"
        f"Analyze if the new message is relevant to the {observation.domain} domain task.\n"
        f"Consider current memory state and whether you need to make room or keep current optimal state.\n"
        f"Output your decision as JSON only."
    )


def choose_action_gemini(
    client: "genai.Client",
    observation: LongHorizonMemoryObservation,
    task_name: str,
) -> LongHorizonMemoryAction:
    user_prompt = _build_user_prompt(observation, task_name)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=150,
            ),
        )
        content = response.text or "{}"
        if ENABLE_DEBUG_LOGS:
            print(f"[GEMINI] Raw response: {content[:200]}", flush=True)
        return _parse_action(content.strip(), observation)
    except Exception as exc:
        if ENABLE_DEBUG_LOGS:
            print(f"[GEMINI] API call failed: {str(exc)[:120]}", flush=True)
        return _heuristic_action(observation)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _strict_score(score: float) -> float:
    eps = min(max(SCORE_EPSILON, 1e-9), 0.49)
    return min(max(score, eps), 1.0 - eps)


def run_episode(
    episode_id: int,
    client: "genai.Client",
) -> Dict:
    os.environ["LONG_HORIZON_MEMORY_EPISODE_ID"] = str(episode_id)
    os.environ["LONG_HORIZON_MEMORY_TASK"] = "all"

    env = LongHorizonMemoryEnvironment()
    observation = env.reset()

    difficulty = observation.metadata.get("task", "unknown")
    domain = observation.domain
    print(
        f"\n[EPISODE {episode_id}] difficulty={difficulty} domain={domain}",
        flush=True,
    )

    rewards: List[float] = []
    scores: List[float] = []
    actions_log: List[str] = []
    success = False

    try:
        for step in range(1, MAX_STEPS + 1):
            action = choose_action_gemini(client, observation, difficulty)
            observation = env.step(action)

            reward = float(observation.reward)
            done = bool(observation.done)
            task_score = float(observation.metadata.get("task_score", 0.0))

            rewards.append(reward)
            scores.append(task_score)
            op = action.operation
            if op == "remove":
                actions_log.append(f"remove:{action.remove_index}")
            else:
                actions_log.append(op)

            print(
                f"  step={step:02d} action={actions_log[-1]:12s} "
                f"reward={reward:+.3f} score={task_score:.3f} done={done}",
                flush=True,
            )

            if done:
                final_score = _strict_score(task_score)
                success = final_score >= SUCCESS_SCORE_THRESHOLD
                break

        if not bool(observation.done):
            raw = float(observation.metadata.get("task_score", 0.0))
            final_score = _strict_score(raw)
            success = final_score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        print(f"  [ERROR] {exc}", flush=True)
        final_score = _strict_score(0.0)
        success = False
    finally:
        raw_final = float(observation.metadata.get("task_score", 0.0)) if observation else 0.0
        final_score = _strict_score(raw_final)
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    print(
        f"[EPISODE {episode_id}] END success={success} "
        f"steps={len(rewards)} final_score={final_score:.3f}",
        flush=True,
    )

    return {
        "episode_id": episode_id,
        "difficulty": difficulty,
        "domain": domain,
        "rewards": rewards,
        "scores": scores,
        "actions": actions_log,
        "final_score": final_score,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def _episode_label(ep: Dict) -> str:
    return f"Ep{ep['episode_id']}\n({ep['difficulty'][:3]})"


def plot_results(results: List[Dict], output_dir: str = ".") -> None:
    os.makedirs(output_dir, exist_ok=True)
    n = len(results)
    colors = plt.cm.tab10(np.linspace(0, 1, n))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # Figure 1 – Rewards per step for each episode
    # ------------------------------------------------------------------ #
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for i, ep in enumerate(results):
        steps = list(range(1, len(ep["rewards"]) + 1))
        ax1.plot(steps, ep["rewards"], marker="o", markersize=4,
                 color=colors[i], label=f"Ep{ep['episode_id']} {ep['domain'][:18]}")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_title("Gemini Agent — Reward per Step per Episode", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Reward")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    p1 = os.path.join(output_dir, "gemini_rewards_per_step.png")
    fig1.savefig(p1, dpi=120)
    print(f"[GRAPH] Saved: {p1}", flush=True)
    plt.close(fig1)

    # ------------------------------------------------------------------ #
    # Figure 2 – Task score over steps for each episode
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, ep in enumerate(results):
        steps = list(range(1, len(ep["scores"]) + 1))
        ax2.plot(steps, ep["scores"], marker="s", markersize=4,
                 color=colors[i], label=f"Ep{ep['episode_id']} {ep['domain'][:18]}")
    ax2.axhline(SUCCESS_SCORE_THRESHOLD, color="red", linestyle="--",
                linewidth=1.2, label=f"Success threshold ({SUCCESS_SCORE_THRESHOLD})")
    ax2.set_title("Gemini Agent — Task Score per Step per Episode", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Task Score")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(output_dir, "gemini_scores_per_step.png")
    fig2.savefig(p2, dpi=120)
    print(f"[GRAPH] Saved: {p2}", flush=True)
    plt.close(fig2)

    # ------------------------------------------------------------------ #
    # Figure 3 – Final score bar chart + success flag
    # ------------------------------------------------------------------ #
    fig3, ax3 = plt.subplots(figsize=(max(6, n * 1.4), 5))
    labels = [_episode_label(ep) for ep in results]
    final_scores = [ep["final_score"] for ep in results]
    bar_colors = ["#4caf50" if ep["success"] else "#f44336" for ep in results]
    bars = ax3.bar(labels, final_scores, color=bar_colors, edgecolor="white", width=0.6)
    ax3.axhline(SUCCESS_SCORE_THRESHOLD, color="black", linestyle="--",
                linewidth=1.2, label=f"Success threshold ({SUCCESS_SCORE_THRESHOLD})")
    for bar, score in zip(bars, final_scores):
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4caf50", label="Success"),
        Patch(facecolor="#f44336", label="Failure"),
    ]
    ax3.legend(handles=legend_elements + [ax3.get_lines()[0] if ax3.get_lines() else
               plt.Line2D([0], [0], color="black", linestyle="--",
                          label=f"Threshold ({SUCCESS_SCORE_THRESHOLD})")],
               loc="upper right", fontsize=9)
    ax3.set_title("Gemini Agent — Final Score per Episode", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Final Task Score")
    ax3.set_ylim(0, 1.15)
    ax3.grid(True, axis="y", alpha=0.3)
    fig3.tight_layout()
    p3 = os.path.join(output_dir, "gemini_final_scores.png")
    fig3.savefig(p3, dpi=120)
    print(f"[GRAPH] Saved: {p3}", flush=True)
    plt.close(fig3)

    # ------------------------------------------------------------------ #
    # Figure 4 – Summary dashboard (2×2 grid)
    # ------------------------------------------------------------------ #
    fig4 = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig4, hspace=0.45, wspace=0.35)

    # Top-left: cumulative avg reward per episode
    ax_tl = fig4.add_subplot(gs[0, 0])
    for i, ep in enumerate(results):
        cum_avg = np.cumsum(ep["rewards"]) / np.arange(1, len(ep["rewards"]) + 1)
        ax_tl.plot(range(1, len(cum_avg) + 1), cum_avg,
                   color=colors[i], label=f"Ep{ep['episode_id']}")
    ax_tl.set_title("Cumulative Average Reward", fontweight="bold")
    ax_tl.set_xlabel("Step")
    ax_tl.set_ylabel("Avg Reward")
    ax_tl.legend(fontsize=7)
    ax_tl.grid(True, alpha=0.3)

    # Top-right: action distribution stacked bar
    ax_tr = fig4.add_subplot(gs[0, 1])
    op_counts = {ep["episode_id"]: {"add": 0, "noop": 0, "remove": 0} for ep in results}
    for ep in results:
        for act in ep["actions"]:
            if act.startswith("remove"):
                op_counts[ep["episode_id"]]["remove"] += 1
            elif act == "add":
                op_counts[ep["episode_id"]]["add"] += 1
            else:
                op_counts[ep["episode_id"]]["noop"] += 1
    ep_ids = [ep["episode_id"] for ep in results]
    adds = [op_counts[eid]["add"] for eid in ep_ids]
    noops = [op_counts[eid]["noop"] for eid in ep_ids]
    removes = [op_counts[eid]["remove"] for eid in ep_ids]
    x = np.arange(len(ep_ids))
    ax_tr.bar(x, adds, label="add", color="#2196f3")
    ax_tr.bar(x, noops, bottom=adds, label="noop", color="#ff9800")
    ax_tr.bar(x, removes, bottom=[a + n for a, n in zip(adds, noops)], label="remove", color="#e91e63")
    ax_tr.set_xticks(x)
    ax_tr.set_xticklabels([f"Ep{e}" for e in ep_ids])
    ax_tr.set_title("Action Distribution per Episode", fontweight="bold")
    ax_tr.set_ylabel("Count")
    ax_tr.legend(fontsize=8)
    ax_tr.grid(True, axis="y", alpha=0.3)

    # Bottom-left: final score comparison
    ax_bl = fig4.add_subplot(gs[1, 0])
    bc = ["#4caf50" if ep["success"] else "#f44336" for ep in results]
    ax_bl.bar([f"Ep{ep['episode_id']}" for ep in results], final_scores, color=bc)
    ax_bl.axhline(SUCCESS_SCORE_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax_bl.set_title("Final Score per Episode", fontweight="bold")
    ax_bl.set_ylabel("Final Score")
    ax_bl.set_ylim(0, 1.15)
    ax_bl.grid(True, axis="y", alpha=0.3)

    # Bottom-right: success rate pie
    ax_br = fig4.add_subplot(gs[1, 1])
    n_success = sum(1 for ep in results if ep["success"])
    n_fail = n - n_success
    if n_success + n_fail > 0:
        ax_br.pie(
            [n_success, n_fail] if n_fail > 0 else [n_success],
            labels=["Success", "Failure"] if n_fail > 0 else ["Success"],
            colors=["#4caf50", "#f44336"] if n_fail > 0 else ["#4caf50"],
            autopct="%1.0f%%",
            startangle=90,
        )
    ax_br.set_title(
        f"Success Rate  ({n_success}/{n} episodes)", fontweight="bold"
    )

    fig4.suptitle(
        f"Gemini Agent ({GEMINI_MODEL}) — Long Horizon Memory  ({n} episodes)",
        fontsize=14, fontweight="bold",
    )
    p4 = os.path.join(output_dir, "gemini_dashboard.png")
    fig4.savefig(p4, dpi=120)
    print(f"[GRAPH] Saved: {p4}", flush=True)
    plt.close(fig4)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: List[Dict]) -> None:
    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    avg_score = sum(r["final_score"] for r in results) / n if n else 0.0
    avg_steps = sum(len(r["rewards"]) for r in results) / n if n else 0.0

    print("\n" + "=" * 60, flush=True)
    print(f"  GEMINI INFERENCE SUMMARY ({GEMINI_MODEL})", flush=True)
    print("=" * 60, flush=True)
    print(f"  Episodes run   : {n}", flush=True)
    print(f"  Successes      : {n_success} ({100*n_success/n:.1f}%)" if n else "  Successes      : 0", flush=True)
    print(f"  Avg final score: {avg_score:.3f}", flush=True)
    print(f"  Avg steps/ep   : {avg_steps:.1f}", flush=True)
    print("-" * 60, flush=True)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(
            f"  {status} Ep{r['episode_id']:02d}  diff={r['difficulty']:<6}  "
            f"score={r['final_score']:.3f}  steps={len(r['rewards']):2d}  "
            f"domain={r['domain']}",
            flush=True,
        )
    print("=" * 60, flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Export it before running: export GEMINI_API_KEY=<your-key>"
        )

    genai_client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={"timeout": GEMINI_TIMEOUT},
    )

    print(f"[INFO] Gemini model  : {GEMINI_MODEL}", flush=True)
    print(f"[INFO] Episodes      : {N_EPISODES}", flush=True)
    print(f"[INFO] Max steps/ep  : {MAX_STEPS}", flush=True)

    # Determine which episode IDs to run
    episodes_path = os.path.join(os.path.dirname(__file__), "server", "episodes.json")
    try:
        with open(episodes_path, "r", encoding="utf-8") as f:
            all_episodes = json.load(f)
        all_ids = [ep["episode_id"] for ep in all_episodes]
    except Exception:
        all_ids = list(range(1, 25))

    # Pick the first N_EPISODES ids
    selected_ids = all_ids[:N_EPISODES]

    results: List[Dict] = []
    for ep_id in selected_ids:
        result = run_episode(ep_id, genai_client)
        results.append(result)

    print_summary(results)
    plot_results(results, output_dir=GRAPHS_DIR)
    print(f"\n[INFO] Graphs saved to: {os.path.abspath(GRAPHS_DIR)}", flush=True)


if __name__ == "__main__":
    main()
