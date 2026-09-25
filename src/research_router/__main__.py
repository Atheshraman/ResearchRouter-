"""Allow running the server with ``python -m research_router``."""

from __future__ import annotations

import asyncio

from research_router.server import run_server

if __name__ == "__main__":
    asyncio.run(run_server())
