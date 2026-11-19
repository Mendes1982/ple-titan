# Copyright 2026 Gold Diamond Studios / PLE Titan Team
"""Entrypoint that serves PLE-Titan over the competition WebSocket protocol."""

import taskbridge
from titan_engine import TitanEngine


async def solve(session: taskbridge.Session) -> str:
    """Executes the multi-phase physical-engineering loop for session."""
    engine = TitanEngine(session)
    return await engine.run()


if __name__ == "__main__":
    taskbridge.run(solve)
