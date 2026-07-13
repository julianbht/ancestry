"""Entry point for the family-photos web frontend.

    uv run ancestry-web            # serve at http://127.0.0.1:8000
"""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
