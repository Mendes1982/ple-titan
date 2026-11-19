# Copyright 2026 Gold Diamond Studios / PLE Titan Team
# Inspired by DeepSeek Harness (@deepseek-ai/dsh-repeat-tool-reminder & deliverables guard)
"""Safety and execution guards for long-horizon engineering tasks."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Callable


class RepeatToolGuard:
    """Detects and breaks repetitive tool-call loops (DeepSeek style).

    When a model calls the exact same tool with identical arguments repeatedly
    (e.g., due to recurring syntax errors, missing packages, or infinite bash loops),
    this guard fires an advisory prompt to guide the model out of the loop.
    """

    def __init__(self, thresholds: tuple[int, ...] = (3, 5, 8)):
        self.thresholds = sorted(thresholds)
        self.call_history: list[tuple[str, str]] = []  # (tool_name, args_hash)
        self.consecutive_count = 0
        self.last_call: tuple[str, str] | None = None

    def record_call(self, tool_name: str, arguments: dict[str, Any] | str) -> str | None:
        """Records a tool call and returns an advisory warning if threshold is reached."""
        if isinstance(arguments, dict):
            args_str = json.dumps(arguments, sort_keys=True)
        else:
            args_str = str(arguments)

        args_hash = hashlib.sha256(args_str.encode("utf-8")).hexdigest()
        current_call = (tool_name, args_hash)

        if self.last_call == current_call:
            self.consecutive_count += 1
        else:
            self.last_call = current_call
            self.consecutive_count = 1

        self.call_history.append(current_call)

        # Check if consecutive repeats hit one of our thresholds
        if self.consecutive_count in self.thresholds:
            return (
                f"[SYSTEM ADVISORY - LOOP HYGIENE GUARD]: You have called the tool '{tool_name}' "
                f"with identical arguments {self.consecutive_count} times in a row without progress. "
                "Carefully inspect the previous output, review error messages, and change your strategy. "
                "If attempting calculation, write a dedicated Python script to debug step-by-step. "
                "Do NOT repeat the exact same command without modifying parameters."
            )
        return None

    def reset(self):
        """Resets the guard state."""
        self.call_history.clear()
        self.consecutive_count = 0
        self.last_call = None


class DeliverablesGuard:
    """Ensures required task deliverables exist, are non-empty, and verified.

    Inspired by DeepSeek Harness deliverables validation and PLE dev_set criteria.
    """

    FILE_PATH_RE = re.compile(
        r"(?:(?:output|deliverables|results|solution|files)/[a-zA-Z0-9_\-\.\/]+|/app/[a-zA-Z0-9_\-\.\/]+|[a-zA-Z0-9_\-]+\.(?:py|sh|csv|txt|md|json|yaml|yml|ifc|dxf|pdf))"
    )

    @classmethod
    def extract_expected_deliverables(cls, instruction: str) -> list[str]:
        """Extracts probable deliverable file paths from the task instruction."""
        matches = set(cls.FILE_PATH_RE.findall(instruction))
        # Filter out common false positives
        ignored = {"requirements.txt", "package.json", "solution.py.bak", "README.md"}
        return [m for m in sorted(matches) if m not in ignored]

    @classmethod
    async def verify_deliverables(
        cls,
        expected_paths: list[str],
        file_reader: Callable[[str], Any],
        exec_bash: Callable[[str], Any],
    ) -> dict[str, Any]:
        """Verifies if the declared/detected deliverables actually exist on disk."""
        results = {}
        for path in expected_paths:
            status = {"exists": False, "size_bytes": 0, "path": path}
            try:
                # Check via bash test -s
                res = await exec_bash(f"[ -s '{path}' ] && wc -c < '{path}' || echo '0'")
                out = res.get("output", "").strip() if isinstance(res, dict) else str(res).strip()
                size = int(out.split()[-1]) if out.isdigit() else 0
                if size > 0:
                    status["exists"] = True
                    status["size_bytes"] = size
            except Exception:
                pass
            results[path] = status
        return results
