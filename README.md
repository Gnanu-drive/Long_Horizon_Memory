---
title: Long Horizon Memory Environment
emoji: "🧠"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Long Horizon Memory Environment

Long Horizon Memory is a real-world inspired environment for selective context retention under noise.

The agent receives a stream of mixed relevant and irrelevant conversational messages and must manage a fixed-capacity memory buffer. The objective is to retain high-value information while deleting distractors.

## Why this environment is useful

This setup models practical assistant behavior in domains like customer support, research synthesis, incident response, and planning systems where context windows are limited.

## OpenEnv interface

### Action: `LongHorizonMemoryAction`

- `operation`: one of `add`, `remove`, `noop`
- `remove_index`: optional 0-based index, used only when `operation=remove`

### Observation: `LongHorizonMemoryObservation`

- `domain`: conversation domain
- `task_name`: `easy`, `medium`, or `hard`
- `new_message`: current message at this step
- `memory`: retained memory list
- `memory_count`: number of retained items
- `reward`: shaped step reward
- `done`: episode completion flag
- `metadata`: includes deterministic grader details such as:
  - `task_score` in `[0.0, 1.0]`
  - `correct_in_memory`, `incorrect_in_memory`
  - `memory_capacity`
  - `last_action_error`

## Tasks and graders

The environment includes 3+ deterministic task buckets:

- `easy`: lower noise, clearer relevance
- `medium`: moderate distractors and ambiguous transitions
- `hard`: long trajectories and semantically similar distractors

Task selection is controlled via environment variable:

- `LONG_HORIZON_MEMORY_TASK=easy|medium|hard|all`

The grader computes a bounded `task_score` in `[0.0, 1.0]` from precision, recall, incorrect retention, and overflow.

## Reward design

Reward is shaped per step to avoid sparse-only learning and reduce reward hacking.

Positive terms:
- recall of relevant memory
- precision of retained memory

Negative terms:
- incorrect retained memory
- memory overflow versus ideal size
- invalid/high-cost actions

A terminal bonus based on final task score is added at episode end.

## Run locally

### Start server

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Random baseline

```bash
python random_baseline.py --episodes 10 --task all --seed 42
```

### Competition inference script

The required script is at project root: `inference.py`.

Required environment variables:

- `HF_TOKEN`
- `API_BASE_URL` (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` (default: `Qwen/Qwen2.5-72B-Instruct`)

Run:

```bash
python inference.py
```

The script emits strict stdout logs:

- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`

## Docker

```bash
docker build -t long_horizon_memory-env:latest -f server/Dockerfile .
```

## Project structure

```text
long_horizon_memory/
├── __init__.py
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── random_baseline.py
├── README.md
└── server/
    ├── app.py
    ├── episodes.json
    ├── long_horizon_memory_environment.py
    └── Dockerfile
```
