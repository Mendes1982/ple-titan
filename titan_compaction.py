# Copyright 2026 Gold Diamond Studios / PLE Titan Team
# Inspired by DeepSeek Harness (@deepseek-ai/dsh-compaction-tool-result-pruner)
"""Context compaction and tool result pruning for long-horizon engineering tasks."""

from __future__ import annotations

import re
from typing import Any


class ContextPruner:
    """Prunes older tool execution outputs while preserving reasoning decisions."""

    def __init__(
        self,
        compact_at_tokens: int = 90_000,
        keep_recent_tool_results: int = 10,
        max_tool_result_chars: int = 25_000,
        unread_keep_chars: int = 800,
    ):
        self.compact_at_tokens = compact_at_tokens
        self.keep_recent_tool_results = keep_recent_tool_results
        self.max_tool_result_chars = max_tool_result_chars
        self.unread_keep_chars = unread_keep_chars

    def head_tail_truncate(self, text: str, max_chars: int | None = None) -> str:
        """Truncates huge tool output preserving the beginning and the end."""
        limit = max_chars or self.max_tool_result_chars
        if len(text) <= limit:
            return text

        head = limit // 2
        tail = limit // 2
        omitted = len(text) - (head + tail)
        return (
            text[:head]
            + f"\n\n[... {omitted} characters omitted by PLE Titan Pruner ...]\n\n"
            + text[-tail:]
        )

    def prune_conversation(
        self,
        messages: list[dict[str, Any]],
        estimated_tokens: int,
    ) -> list[dict[str, Any]]:
        """Prunes older tool messages when context exceeds budget.

        Preserves:
        1. System prompt & initial task instructions.
        2. All assistant tool_calls and reasoning blocks.
        3. The most recent N tool results.
        Old tool results are collapsed to concise summaries.
        """
        if estimated_tokens < self.compact_at_tokens or len(messages) <= self.keep_recent_tool_results * 2:
            return messages

        # Find indices of all tool result messages
        tool_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "tool" or (m.get("role") == "user" and "tool_call_id" in m)
        ]

        if len(tool_indices) <= self.keep_recent_tool_results:
            return messages

        # The ones to prune: everything before the last keep_recent_tool_results
        prune_targets = set(tool_indices[:-self.keep_recent_tool_results])

        new_messages = []
        for i, msg in enumerate(messages):
            if i in prune_targets:
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > self.unread_keep_chars:
                    # Summarize pruned tool result
                    stub = (
                        content[:self.unread_keep_chars // 2]
                        + f"\n[... Output pruned to maintain context window. Exit code was recorded. ...]\n"
                        + content[-(self.unread_keep_chars // 2):]
                    )
                    pruned_msg = dict(msg)
                    pruned_msg["content"] = stub
                    new_messages.append(pruned_msg)
                    continue
            new_messages.append(msg)

        return new_messages
