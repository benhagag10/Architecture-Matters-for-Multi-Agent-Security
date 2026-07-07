#!/usr/bin/env python3
"""
BigCodeBench runner — benign baseline for the RedCode-Gen scenario.

Self-contained: dataset (fetched from HuggingFace) + tools + prompts +
13 conditions + pass@1 unit-test scorer + main.

Reuses ``conditions.py``. Uses the same tool partition as ``redcode_gen.py``
(Table 13). Scoring is via the BigCodeBench reference test cases — pass@1.

Default sample size is 50 (paper-matching). Override with ``--limit N``.

Prerequisites:
    1. ``pip install -r requirements.txt``
    2. The first run downloads ``bigcode/bigcodebench`` from HuggingFace.
    3. Test execution runs in a Docker sandbox by default (pass
       ``--sandbox local`` to run inline; use only for trusted code).

Usage:
    python bigcodebench.py --condition single_agent     --model openai/gpt-4o
    python bigcodebench.py --condition star_specialist  --model openai/gpt-4o --limit 50
    python bigcodebench.py --smoke --condition single_agent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inspect_ai import Task, eval as ai_eval, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, mean
from inspect_ai.solver import TaskState

import conditions
from conditions import ScenarioBundle, build_agent, list_conditions

# Reuse RedCode-Gen prompts + bundle: identical tool surface, only the
# benign nature of the task and the scorer differ.
import redcode_gen as RG


DEFAULT_LOG_ROOT = Path("logs/bigcodebench_icml")
DEFAULT_LIMIT = 50  # paper-matching default
SMOKE_LIMIT = 3


# ---------------------------------------------------------------------------
# Dataset loader (HuggingFace bigcode/bigcodebench)
# ---------------------------------------------------------------------------


def load_bigcodebench(limit: int = DEFAULT_LIMIT, seed: int = 42) -> list[Sample]:
    """Load `limit` BigCodeBench tasks (random sample with given seed)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Install `datasets` to load BigCodeBench: pip install datasets"
        ) from e
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
    indices = list(range(len(ds)))
    import random
    random.Random(seed).shuffle(indices)
    chosen = indices[:limit]
    samples: list[Sample] = []
    for i in chosen:
        ex = ds[int(i)]
        samples.append(
            Sample(
                id=str(ex["task_id"]),
                input=ex["complete_prompt"],
                target=ex["canonical_solution"],
                metadata={
                    "task_id": ex["task_id"],
                    "test": ex["test"],
                    "entry_point": ex["entry_point"],
                    "complete_prompt": ex["complete_prompt"],
                },
            )
        )
    return samples


# ---------------------------------------------------------------------------
# pass@1 scorer (runs the BigCodeBench unit tests against generated code)
# ---------------------------------------------------------------------------


@scorer(metrics=[accuracy(), mean()])
def bigcodebench_pass_at_1(timeout: int = 60) -> Scorer:
    """Run the reference unit tests against the agent's generated code."""
    async def score(state: TaskState, target: Target) -> Score:
        from judges import _extract_code  # reuse the code extraction helper
        code = _extract_code(state)
        if not code:
            return Score(value=0.0, explanation="No code block produced.")
        test = state.metadata.get("test", "")
        if not test:
            return Score(value=0.0, explanation="No test in sample metadata.")

        try:
            from inspect_ai.tool import bash
            tool = bash(timeout=timeout)
            program = code + "\n\n" + test + "\n\nif __name__ == '__main__': unittest.main()\n"
            result = await tool(cmd=f"python -c {program!r}")
            ok = "OK" in str(result) or "passed" in str(result).lower()
            return Score(
                value=1.0 if ok else 0.0,
                explanation=str(result)[:300],
            )
        except Exception as e:
            return Score(value=0.0, explanation=f"Sandbox error: {e}")

    return score


# ---------------------------------------------------------------------------
# Inspect Task definition
# ---------------------------------------------------------------------------


@task
def bigcodebench(
    condition: str = "single_agent",
    model: str = "openai/gpt-4o",
    limit: int = DEFAULT_LIMIT,
    seed: int = 42,
    sandbox: str = "docker",
) -> Task:
    """BigCodeBench Task. Default sandbox is ``docker``; the pass@1 scorer runs
    the reference unit tests via the sandboxed ``bash`` tool, so a sandbox is
    required. Pass ``sandbox='local'`` only when smoke-testing trusted code.
    """
    samples = load_bigcodebench(limit=limit, seed=seed)
    bundle = RG._bundle_for_condition(condition)
    agent = build_agent(condition, model=model, bundle=bundle)
    return Task(
        dataset=samples,
        solver=agent,
        scorer=bigcodebench_pass_at_1(),
        sandbox=sandbox,
        metadata={
            "scenario": "bigcodebench",
            "condition": condition,
            "n_samples": len(samples),
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="BigCodeBench runner (ICML release).")
    parser.add_argument("--condition", default="single_agent", choices=list_conditions())
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Number of samples (default: {DEFAULT_LIMIT}, paper-matching).")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-time", type=int, default=300)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--sandbox", default="docker", choices=["docker", "local"],
                        help="Sandbox for the pass@1 unit-test scorer "
                             "(default: docker). 'local' is only for smoke testing.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.limit = SMOKE_LIMIT
        args.max_time = 120
        args.max_turns = 15

    log_dir = args.log_dir or (DEFAULT_LOG_ROOT / args.condition)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"BigCodeBench (benign) | {args.condition} | {args.model} | n={args.limit}", flush=True)

    logs = ai_eval(
        bigcodebench(
            condition=args.condition,
            model=args.model,
            limit=args.limit,
            seed=args.seed,
            sandbox=args.sandbox,
        ),
        model=args.model,
        log_dir=str(log_dir),
        max_samples=args.max_samples,
        max_messages=args.max_turns * 2,
        time_limit=args.max_time,
        seed=args.seed,
    )
    statuses = [getattr(L, "status", "?") for L in logs]
    ok = all(s == "success" for s in statuses)
    print(f"  -> {len(logs)} log(s); statuses={statuses}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
