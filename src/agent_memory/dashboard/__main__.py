"""CLI entry point for the Memory Dashboard."""

import sys
from pathlib import Path

from agent_memory.dashboard.app import DashboardApp


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m agent_memory.dashboard <db_path>")
        print("Example: python -m agent_memory.dashboard .db/memory.db")
        sys.exit(1)

    db_path = sys.argv[1]
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    app = DashboardApp(db_path=db_path)
    app.run()


if __name__ == "__main__":
    main()
