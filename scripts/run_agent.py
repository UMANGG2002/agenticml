#!/usr/bin/env python3
"""Run the LLM-driven agent (agent/llm_agent.py) against a CSV.

Unlike scripts/run_pipeline.py (the fixed profile -> target -> leakage ->
AutoML sequence), the model itself decides which tool to call, when to
state a hypothesis, and when it needs you — via the ask_user tool. When it
asks, this script prompts you on the terminal and resumes the loop with
your answer.

Needs credentials to actually talk to the LLM: OPENAI_API_KEY and
LITELLM_BASE_URL set in .env. See docs/architecture.md's "Agentic layer"
section for how the loop works — every automated test drives it against a
scripted fake client instead, so none of that needed a key.

Example:
    uv run python scripts/run_agent.py dataset/train.csv \\
        --objective "predict mobile phone price range"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, APIConnectionError, RateLimitError

from agent.dataset_registry import DatasetRegistry
from agent.llm_agent import DEFAULT_MODEL, build_resume_messages, run_agent
from agent.state import AgentState
from experiments.registry import ExperimentRegistry


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="path to the dataset CSV")
    parser.add_argument("--objective", required=True, help="plain-English description of the ML goal")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=15, help="API calls before giving up per invocation")
    parser.add_argument("--artifact-dir", default="data/artifacts")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"error: '{args.csv}' does not exist", file=sys.stderr)
        raise SystemExit(1)

    df = pd.read_csv(args.csv)
    print(f"Loaded {args.csv} — {df.shape[0]} rows x {df.shape[1]} columns")

    dataset_registry = DatasetRegistry()
    dataset_version_id = dataset_registry.register(df)
    experiment_registry = ExperimentRegistry()
    state = AgentState(project_id="cli-agent-run", dataset_version_id=dataset_version_id, objective=args.objective)

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LITELLM_BASE_URL"),
    )
    messages = None

    try:
        while True:
            result = run_agent(
                client,
                state,
                dataset_registry,
                experiment_registry,
                messages=messages,
                max_turns=args.max_turns,
                model=args.model,
                artifact_dir=args.artifact_dir,
            )

            for call in result.tool_calls:
                print(f"  [tool] {call['name']}({call['input']})")

            if result.status == "awaiting_user_input":
                print(f"\n{result.pending_question.question}")
                if result.pending_question.context:
                    print(f"  context: {json.dumps(result.pending_question.context, indent=2, default=str)}")
                answer = input("> ")
                messages = build_resume_messages(result, answer)
                continue

            if result.status == "done":
                print(f"\n{result.final_text}")
                break

            print(f"\nStopped after {args.max_turns} turns without finishing (status={result.status}).")
            break
    except AuthenticationError:
        print("\nerror: no valid LLM credentials found.", file=sys.stderr)
        print("Set OPENAI_API_KEY and LITELLM_BASE_URL in .env.", file=sys.stderr)
        raise SystemExit(1)
    except RateLimitError:
        print("\nerror: rate limited by the LLM provider — wait a moment and try again.", file=sys.stderr)
        raise SystemExit(1)
    except APIConnectionError:
        print("\nerror: could not reach the LLM provider — check your network connection.", file=sys.stderr)
        raise SystemExit(1)

    if experiment_registry.all():
        print(f"\n{len(experiment_registry.all())} experiment(s) recorded:")
        for record in experiment_registry.all():
            print(f"  {record.experiment_id}  {record.model:20s} {record.metric}={record.primary_score:.4f}")


if __name__ == "__main__":
    main()
