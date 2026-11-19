# Copyright 2026 Gold Diamond Studios / PLE Titan Team
"""PLE-Titan Engine: High-performance multi-phase physical engineering agent loop."""

from __future__ import annotations

import asyncio
import json
import os
import posixpath
import re
from typing import Any

import openai
import taskbridge
from titan_compaction import ContextPruner
from titan_guard import DeliverablesGuard, RepeatToolGuard
from titan_prompts import (
    ENGINEERING_SYSTEM_DIRECTIVE,
    PLANNER_PROMPT,
    VERIFIER_PROMPT,
    WORKER_PROMPT,
)

MODEL = os.environ.get("MODEL_NAME", "claude-opus-5-5")
MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)
CACHE_MARK = "claude" in MODEL.lower()
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
COMPACT_AT_TOKENS = int(os.environ.get("COMPACT_AT_TOKENS", "90000"))

# Time shares
PLAN_SHARE = 0.10      # 10% for planning
CHECK_SHARE = 0.25     # 25% reserved for review & verify-fix rounds
WRAP_UP_SEC = 240      # 4 min final wrap-up margin
MIN_STEP_SEC = 120     # Minimum 2 min to initiate a step
MAX_FIX_ROUNDS = 2     # Up to 2 fix rounds
MAX_NUDGES = 3

RETRY_WAITS = (5, 15, 30, 60, 120)

CONTEXT_FULL = re.compile(
    r"context.?(length|window|size|limit)|maximum context|too many"
    r" tokens|prompt is too long|maximum prompt length"
    r"|(input|prompt|request|messages?)\b[^.]{0,40}(too long|exceed)"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)


def tool(name: str, desc: str, required: list[str], **props):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


CORE_TOOLS = [
    tool(
        "bash",
        "Run a shell command in the task container. Always use Python scripts for calculations.",
        ["command"],
        command={"type": "string"},
        timeout_sec={
            "type": "integer",
            "description": "Timeout in seconds (default 120).",
        },
    ),
    tool(
        "read_file",
        "Read a file from the task container.",
        ["path"],
        path={"type": "string"},
    ),
    tool(
        "write_file",
        "Create or overwrite a file in the task container. Parent directories are created automatically.",
        ["path", "content"],
        path={"type": "string"},
        content={"type": "string"},
    ),
    tool(
        "edit_file",
        "Replace exact text in a file.",
        ["path", "old_text", "new_text"],
        path={"type": "string"},
        old_text={"type": "string"},
        new_text={"type": "string"},
    ),
]

READ_ONLY_TOOLS = [CORE_TOOLS[0], CORE_TOOLS[1]]  # bash, read_file


def marked(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Marks messages with prompt caching control for Claude models."""
    if not CACHE_MARK or not messages:
        return messages
    res = [dict(m) for m in messages]
    if len(res) > 0:
        res[0] = dict(res[0])
        res[0]["cache_control"] = {"type": "ephemeral"}
    if len(res) > 1:
        res[-1] = dict(res[-1])
        res[-1]["cache_control"] = {"type": "ephemeral"}
    return res


def build_system_prompt(role_prompt: str, workdir: str) -> str:
    return (
        f"{ENGINEERING_SYSTEM_DIRECTIVE}\n\n"
        f"Container Working Directory: {workdir}\n\n"
        f"### YOUR CURRENT ROLE:\n{role_prompt}"
    )


class TitanEngine:
    def __init__(self, session: taskbridge.Session):
        self.session = session
        self.client = openai.AsyncOpenAI(
            base_url=os.environ.get("MODEL_BASE_URL", "http://localhost:4000/v1"),
            api_key=os.environ.get("MODEL_API_KEY", "dummy"),
            timeout=1200.0,
        )
        self.repeat_guard = RepeatToolGuard(thresholds=(3, 5, 8))
        self.pruner = ContextPruner(compact_at_tokens=COMPACT_AT_TOKENS)
        self.expected_deliverables = DeliverablesGuard.extract_expected_deliverables(
            session.instruction
        )

    def resolve_path(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return posixpath.join(self.session.workdir, path)

    async def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        # Loop hygiene guard check (DeepSeek style)
        repeat_warning = self.repeat_guard.record_call(name, args)

        output = ""
        try:
            if name == "bash":
                cmd = args.get("command", "")
                t_sec = int(args.get("timeout_sec", 120))
                left = self.session.time_left()
                if left is not None:
                    t_sec = min(t_sec, max(1, int(left - WRAP_UP_SEC)))
                res = await self.session.exec(cmd, timeout_sec=t_sec)
                timed_out = " (timed out)" if res.get("timed_out") else ""
                output = f"exit code {res.get('exit_code')}{timed_out}\n{res.get('output', '')}"
            elif name == "read_file":
                p = self.resolve_path(args.get("path", ""))
                raw = await self.session.read_file(p)
                output = raw.decode("utf-8", "replace")
            elif name == "write_file":
                p = self.resolve_path(args.get("path", ""))
                cnt = args.get("content", "")
                bytes_written = await self.session.write_file(p, cnt)
                output = f"wrote {bytes_written} bytes to {p}"
            elif name == "edit_file":
                p = self.resolve_path(args.get("path", ""))
                old = args.get("old_text", "")
                new = args.get("new_text", "")
                raw = (await self.session.read_file(p)).decode("utf-8", "replace")
                if old not in raw:
                    output = f"error: old_text not found in {p}"
                else:
                    updated = raw.replace(old, new, 1)
                    await self.session.write_file(p, updated)
                    output = f"successfully edited {p}"
            else:
                output = f"error: unknown tool '{name}'"
        except Exception as exc:
            output = f"error executing {name}: {type(exc).__name__}: {exc}"

        # Prune / truncate output if too large
        clipped = self.pruner.head_tail_truncate(output)

        if repeat_warning:
            clipped = f"{repeat_warning}\n\n[TOOL RESULT]:\n{clipped}"

        return clipped

    async def agent_loop(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        until_seconds: float,
        final: bool = False,
    ) -> str:
        nudges = 0
        head = ""
        waits = iter(RETRY_WAITS)

        while True:
            left = self.session.time_left()
            wrap_up = left is not None and left <= until_seconds

            # Dynamic context compaction
            est_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages)
            messages = self.pruner.prune_conversation(messages, est_tokens)

            timeout = min(WRAP_UP_SEC, left) if wrap_up else (None if left is None else left - until_seconds)
            timeout = max(5.0, timeout) if timeout is not None else 600.0

            try:
                reply = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=MODEL,
                        messages=marked(messages),
                        tools=tools,
                        tool_choice="none" if wrap_up else "auto",
                        reasoning_effort=EFFORT,
                        max_completion_tokens=MAX_OUTPUT_TOKENS,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                if wrap_up:
                    return head or "Wrap up deadline reached."
                continue
            except openai.APIError as exc:
                status = getattr(exc, "status_code", None)
                if status == 400 and CONTEXT_FULL.search(str(exc)):
                    # Force prune aggressively
                    messages = self.pruner.prune_conversation(messages, COMPACT_AT_TOKENS + 1000)
                    continue
                wait = next(waits, None)
                if wait is None or not (status is None or status in (408, 409, 429) or status >= 500):
                    raise
                await self.session.log(f"Model call failed ({exc}); retry in {wait}s")
                await asyncio.sleep(wait)
                continue

            waits = iter(RETRY_WAITS)

            # Record usage
            if reply.usage:
                details = getattr(reply.usage, "prompt_tokens_details", None)
                cached = getattr(details, "cached_tokens", 0) or 0
                await self.session.usage(
                    MODEL,
                    reply.usage.prompt_tokens,
                    reply.usage.completion_tokens,
                    cached,
                )

            choice = reply.choices[0]
            message = choice.message
            cut = choice.finish_reason == "length"
            text = head + (message.content or "")

            if final and text:
                self.session.final_message = text

            turn = message.model_dump(exclude_none=True)

            if wrap_up or not message.tool_calls:
                turn.pop("tool_calls", None)
                if message.content:
                    messages.append(turn)
                if wrap_up or (message.content and not cut) or nudges >= MAX_NUDGES:
                    return text
                nudges += 1
                head = text
                messages.append({
                    "role": "user",
                    "content": "Your reply was cut off; please continue and finalize." if cut else "Continue and finish with your deliverable summary."
                })
                continue

            nudges = 0
            head = ""
            messages.append(turn)

            for call in turn.get("tool_calls", []):
                fn = call.get("function", {})
                t_name = fn.get("name", "")
                try:
                    t_args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    t_args = {}

                await self.session.log(f"{t_name} {json.dumps(t_args)[:250]}")
                t_output = await self.execute_tool(t_name, t_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": t_output,
                })

            if left is not None:
                mins_left = max(0, int((self.session.time_left() - until_seconds) // 60))
                messages[-1]["content"] += f"\n\n[{mins_left} minutes remaining for this phase]"

    async def run(self) -> str:
        """Executes the full 3-phase Plan-Solve-Verify workflow."""
        total = self.session.time_left()

        def until(share: float) -> float:
            return 0.0 if total is None else total * share

        def has_time(margin_sec: float) -> bool:
            left = self.session.time_left()
            return left is None or left - margin_sec >= MIN_STEP_SEC

        task_instruction = self.session.instruction
        workdir = self.session.workdir

        # ========================================================
        # PHASE 1: PLANNER (10% Time)
        # ========================================================
        await self.session.log("=== PHASE 1: PLANNER START ===")
        planner_messages = [
            {"role": "system", "content": build_system_prompt(PLANNER_PROMPT, workdir)},
            {"role": "user", "content": f"Task Requirements:\n{task_instruction}\n\nExplore files and formulate your engineering plan."},
        ]
        plan = await self.agent_loop(
            planner_messages,
            READ_ONLY_TOOLS,
            until(1.0 - PLAN_SHARE),
        )
        await self.session.log(f"Plan produced: {plan[:600]}...")

        # Reset loop guard between phases
        self.repeat_guard.reset()

        # ========================================================
        # PHASE 2: WORKER (Implementation)
        # ========================================================
        await self.session.log("=== PHASE 2: WORKER START ===")
        worker_messages = [
            {"role": "system", "content": build_system_prompt(WORKER_PROMPT, workdir)},
            {
                "role": "user",
                "content": (
                    f"## Task Instructions\n{task_instruction}\n\n"
                    f"## Engineering Plan\n{plan}\n\n"
                    "Now execute the calculations using Python and generate all deliverables."
                ),
            },
        ]
        summary = await self.agent_loop(
            worker_messages,
            CORE_TOOLS,
            until(CHECK_SHARE),
            final=True,
        )
        await self.session.log(f"Worker finished summary: {summary[:600]}...")

        # ========================================================
        # PHASE 3: VERIFIER & QUALITY AUDIT (25% Time)
        # ========================================================
        await self.session.log("=== PHASE 3: VERIFIER AUDIT ===")

        for round_idx in range(MAX_FIX_ROUNDS):
            if not has_time(until(CHECK_SHARE / 2)):
                break

            # DeepSeek Deliverables Guard check
            deliv_status = await DeliverablesGuard.verify_deliverables(
                self.expected_deliverables,
                self.session.read_file,
                self.session.exec,
            )
            deliv_report = json.dumps(deliv_status, indent=2)

            audit_context = (
                f"## Task Instructions\n{task_instruction}\n\n"
                f"## Engineering Plan\n{plan}\n\n"
                f"## Worker Implementation Summary\n{summary}\n\n"
                f"## Automated Deliverables Check\n{deliv_report}\n\n"
                "Audit all deliverable files in the container. Execute solution scripts if applicable. Check units and math."
            )

            verifier_messages = [
                {"role": "system", "content": build_system_prompt(VERIFIER_PROMPT, workdir)},
                {"role": "user", "content": audit_context},
            ]

            verdict = await self.agent_loop(
                verifier_messages,
                READ_ONLY_TOOLS,
                until(CHECK_SHARE / 2),
            )
            await self.session.log(f"Verifier verdict (round {round_idx+1}): {verdict[:600]}...")

            if "STATUS: PASS" in verdict.upper() or not has_time(WRAP_UP_SEC):
                await self.session.log("Audit passed. Finalizing submission.")
                break

            # Fix round feedback to worker
            worker_messages.append({
                "role": "user",
                "content": (
                    f"[PEER REVIEW AUDIT DEFICIENCIES]:\n{verdict}\n\n"
                    "Fix every deficiency noted above, re-verify with Python, and output an updated final summary."
                ),
            })
            summary = await self.agent_loop(
                worker_messages,
                CORE_TOOLS,
                WRAP_UP_SEC,
                final=True,
            ) or summary

        return summary or self.session.final_message
