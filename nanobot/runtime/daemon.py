from __future__ import annotations

from pathlib import Path

from nanobot.runtime.subconscious_daemon import SubconsciousDaemon


class RuntimeDaemon:
    """
    Unified runtime daemon entrypoint.

    RuntimeDaemon is the canonical import path for the background scheduler.
    It currently reuses the existing SubconsciousDaemon implementation.
    """

    def __init__(self, agent_loop, workspace: Path):
        self._impl = SubconsciousDaemon(agent_loop, workspace)

    def start_background_tasks(self) -> None:
        self._impl.start_background_tasks()

    def stop(self) -> None:
        self._impl.stop()

    def register_energy_recovery(self, cron_service) -> None:
        self._impl.register_energy_recovery(cron_service)

    async def handle_energy_recovery(self) -> None:
        await self._impl.handle_energy_recovery()

    def __getattr__(self, item):
        return getattr(self._impl, item)

