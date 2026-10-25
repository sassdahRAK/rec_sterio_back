"""Allow ``python -m app.worker`` to run the worker loop."""
from __future__ import annotations

from app.worker.main import main

if __name__ == "__main__":  # pragma: no cover - process entrypoint
    main()
