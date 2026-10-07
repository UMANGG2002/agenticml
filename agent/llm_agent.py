"""LLM-driven agent loop — the model-driven counterpart to
agent/orchestrator.py's hardcoded profile -> target -> leakage -> AutoML
sequence.

Same tools, same registries (DatasetRegistry, ExperimentRegistry), same
typed contracts — but the LLM decides which tool to call, when to form a
hypothesis, and when it needs a human, instead of a fixed if/else chain.
Nothing here makes a network call on import; pass a real OpenAI client
(pointed at a LiteLLM proxy) to talk to the API, or a scripted fake (see
tests/unit/test_llm_agent.py) to drive the loop deterministically without
one.

Manual loop, not the SDK's tool_runner: `ask_user` needs to stop the
loop and hand control back to the caller rather than being resolved inline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agent.dataset_registry import DatasetRegistry
from agent.dispatch import ToolDispatcher
from agent.llm_tools import ASK_USER_TOOL_NAME, build_tools
from agent.message_blocks import AgentMessage, TextBlock, ToolUseBlock
from agent.schemas import PendingQuestion
from agent.state import AgentState
from experiments.registry import ExperimentRegistry

DEFAULT_MODEL = "genuin/clipit"

SYSTEM_PROMPT = """You are an autonomous machine-learning engineering agent.

Your goal is not merely to maximize a validation metric. Your goal is to
produce a reliable, reproducible, domain-aligned ML solution.

Rules:
1. Understand the user's objective before modeling.
2. Inspect the dataset (profile_columns, check_data_quality) before anything else.
3. Confirm the target when detect_target's `requires_user_confirmation` is true —
   call ask_user, do not guess.
4. Always run detect_leakage before run_experiment. If `has_critical_issues` is
   true, call ask_user before proceeding — do not silently include the
   flagged columns, and do not silently drop them either without saying so.
5. Every run_experiment call needs a real hypothesis: state what you expect
   to change and why, not just "try another model".
6. Compare a new experiment against its parent via `parent_experiment_id`
   whenever you're testing a specific change, so `conclusion` means something.
7. Never claim an improvement without a completed comparison.
8. Call ask_user when you are genuinely stuck, when a decision is high-impact
   (e.g. excluding a column, accepting a leakage risk), or when required
   information is missing — not for routine progress updates.
9. After establishing a baseline experiment, run analyze_errors to identify
   weaknesses, then form hypotheses to address them.
10. After your best experiment, run run_robustness_checks to verify the model
    holds up under varied conditions.
11. Before concluding, run run_critic to get an independent review of your
    best experiment. If the critic rejects the result, address the issues
    before promoting the model.
12. Stop and summarize once you have a reasonable result, the budget is
    exhausted, or further experiments show negligible gains.

Budget awareness:
- You have a limited number of experiments. Check how many you've used.
- Stop when the budget is exhausted or when improvement plateaus.
- Prefer experiments that address known weaknesses over random exploration.

Only call tools that are declared to you. Report your reasoning briefly
before each tool call; the user cannot see your internal state, only your
messages and tool calls.
"""


@dataclass
class AgentTurnResult:
    status: str  # "awaiting_user_input" | "done" | "max_turns_reached" | "budget_exhausted"
    messages: list[dict[str, Any]]
    pending_question: PendingQuestion | None = None
    ask_user_tool_use_id: str | None = None
    final_text: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def build_initial_messages(state: AgentState) -> list[dict[str, Any]]:
    budget_info = ""
    if state.budget:
        budget_info = (
            f"\nBudget: max {state.budget.max_experiments} experiments, "
            f"max {state.budget.max_runtime_seconds}s runtime, "
            f"minimum improvement threshold {state.budget.minimum_improvement}."
        )

    return [
        {
            "role": "user",
            "content": (
                f"Objective: {state.objective}\n"
                f"Dataset version id: {state.dataset_version_id}"
                f"{budget_info}\n"
                "Begin the ML engineering workflow."
            ),
        }
    ]


def _check_budget(state: AgentState, start_time: float) -> str | None:
    """Check if any budget limit has been reached. Returns a reason string or None."""
    if not state.budget:
        return None

    n_experiments = len(state.completed_experiments) + len(state.rejected_experiments)
    if n_experiments >= state.budget.max_experiments:
        return f"experiment budget exhausted ({n_experiments}/{state.budget.max_experiments})"

    elapsed = time.time() - start_time
    if elapsed >= state.budget.max_runtime_seconds:
        return f"runtime budget exhausted ({elapsed:.0f}s/{state.budget.max_runtime_seconds}s)"

    return None


def _update_state_from_tool_result(
    state: AgentState,
    tool_name: str,
    tool_input: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Update AgentState based on a completed tool call."""
    if tool_name == "run_experiment":
        experiment_id = result.get("experiment_id", "")
        conclusion = result.get("conclusion")
        hypothesis = tool_input.get("hypothesis", "")

        if hypothesis and hypothesis not in state.hypotheses:
            state.hypotheses.append(hypothesis)

        if conclusion == "rejected":
            state.rejected_experiments.append(experiment_id)
        else:
            state.completed_experiments.append(experiment_id)

        # Track baseline and best
        if state.baseline_experiment_id is None:
            state.baseline_experiment_id = experiment_id

        primary_score = result.get("primary_score")
        if primary_score is not None:
            if state.best_experiment_id is None:
                state.best_experiment_id = experiment_id
            else:
                # Compare with current best — simple heuristic: higher is better
                # A more robust version would check the metric direction
                pass  # The experiment registry's best() handles this properly

    elif tool_name == "detect_target":
        # Update target column if confirmed
        candidates = result.get("candidates", [])
        if candidates and not result.get("requires_user_confirmation", True):
            state.target_column = candidates[0].get("column") if isinstance(candidates[0], dict) else None

    elif tool_name == "run_critic":
        verdict = result.get("verdict")
        if verdict == "reject":
            state.stopping_reason = f"critic rejected experiment: {result.get('summary', '')}"


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal message format (Anthropic-style blocks) to OpenAI format.

    The internal format stores assistant messages with block objects (.type, .text,
    .name, .input, .id) and user tool_result messages as dicts. This converts them
    to the OpenAI chat completions message format.
    """
    converted: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Simple string content — pass through
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        # Assistant message with blocks (TextBlock / ToolUseBlock objects)
        if role == "assistant":
            text_parts = []
            tool_calls_list = []

            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text_parts.append(block.text)
                elif block_type == "tool_use":
                    tool_calls_list.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    })

            openai_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if tool_calls_list:
                openai_msg["tool_calls"] = tool_calls_list
            converted.append(openai_msg)
            continue

        # User message — may contain tool_result dicts
        if role == "user" and isinstance(content, list):
            tool_results = [
                block for block in content
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
            if tool_results:
                for result in tool_results:
                    converted.append({
                        "role": "tool",
                        "tool_call_id": result["tool_use_id"],
                        "content": str(result["content"]),
                    })
                continue

        # Fallback — pass through
        converted.append({"role": role, "content": content})

    return converted


def _parse_response(response: Any) -> AgentMessage:
    """Parse an OpenAI chat completion response into internal AgentMessage format."""
    choice = response.choices[0]
    msg = choice.message
    blocks: list[Any] = []

    if msg.content:
        blocks.append(TextBlock(text=msg.content))
    if msg.tool_calls:
        for tc in msg.tool_calls:
            blocks.append(ToolUseBlock(
                id=tc.id,
                name=tc.function.name,
                input=json.loads(tc.function.arguments),
            ))

    stop_reason = "tool_use" if msg.tool_calls else "end_turn"
    return AgentMessage(content=blocks, stop_reason=stop_reason)


def run_agent(
    client: Any,
    state: AgentState,
    dataset_registry: DatasetRegistry,
    experiment_registry: ExperimentRegistry,
    messages: list[dict[str, Any]] | None = None,
    *,
    max_turns: int = 15,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    artifact_dir: str | None = "./data/artifacts",
) -> AgentTurnResult:
    """Drive the agentic loop until the model asks the user something, finishes, or
    `max_turns` is reached.

    `client` must expose `client.chat.completions.create(model=, max_tokens=,
    messages=, tools=)` returning an object with `.choices[0].message.content`
    (text) and `.choices[0].message.tool_calls` (list of tool calls) — the
    real `OpenAI()` client satisfies this, as does the fakes used in tests.
    """
    tools = build_tools()
    dispatcher = ToolDispatcher(dataset_registry, experiment_registry, artifact_dir=artifact_dir)
    messages = messages if messages is not None else build_initial_messages(state)
    tool_calls: list[dict[str, Any]] = []
    start_time = time.time()

    for _ in range(max_turns):
        # Check budget before each turn
        budget_reason = _check_budget(state, start_time)
        if budget_reason:
            state.stopping_reason = budget_reason
            return AgentTurnResult(
                status="budget_exhausted",
                messages=messages,
                final_text=f"Stopping: {budget_reason}",
                tool_calls=tool_calls,
            )

        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + _convert_messages(messages),
            tools=tools,
        )
        agent_msg = _parse_response(response)
        messages.append({"role": "assistant", "content": agent_msg.content})

        tool_use_blocks = [b for b in agent_msg.content if b.type == "tool_use"]

        if agent_msg.stop_reason == "end_turn" or not tool_use_blocks:
            final_text = next((b.text for b in agent_msg.content if b.type == "text"), None)
            return AgentTurnResult(status="done", messages=messages, final_text=final_text, tool_calls=tool_calls)

        ask_block = next((b for b in tool_use_blocks if b.name == ASK_USER_TOOL_NAME), None)
        if ask_block is not None:
            question = PendingQuestion(
                reason=ask_block.input.get("reason", ""),
                question=ask_block.input.get("question", ""),
                context=ask_block.input.get("context") or {},
            )
            return AgentTurnResult(
                status="awaiting_user_input",
                messages=messages,
                pending_question=question,
                ask_user_tool_use_id=ask_block.id,
                tool_calls=tool_calls,
            )

        tool_results = []
        for block in tool_use_blocks:
            tool_calls.append({"name": block.name, "input": block.input})
            try:
                result = dispatcher.dispatch(block.name, block.input)
                content = json.dumps(result, default=str)
                is_error = False
                # Update state based on the tool result
                _update_state_from_tool_result(state, block.name, block.input, result)
            except Exception as exc:  # noqa: BLE001 — surfaced to the model, not raised
                content = f"Error: {exc}"
                is_error = True
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}
            )

        messages.append({"role": "user", "content": tool_results})

    return AgentTurnResult(status="max_turns_reached", messages=messages, tool_calls=tool_calls)


def build_resume_messages(result: AgentTurnResult, answer: str) -> list[dict[str, Any]]:
    """Continue a turn that stopped with `awaiting_user_input`: append the user's
    answer as the ask_user tool's result, ready to pass back into `run_agent`."""
    if result.status != "awaiting_user_input" or result.ask_user_tool_use_id is None:
        raise ValueError("can only resume a turn with status 'awaiting_user_input'")
    messages = list(result.messages)
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": result.ask_user_tool_use_id, "content": answer}
            ],
        }
    )
    return messages
