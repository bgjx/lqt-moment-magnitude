# lqtmoment: agent notes

- Python package in `src/lqtmoment/`, tests in `tests/`, lint config in `ruff.toml`.
- The Python environment lives only in the devcontainer (`lqt-moment-magnitude-package-1`, workspace `/lqt-moment-magnitude`). The host has no `uv`/`pytest`.
- When opencode runs on the host, execute tooling through the container:
  - Tests: `docker exec -w /lqt-moment-magnitude lqt-moment-magnitude-package-1 uv run pytest -q`
  - Lint: `docker exec -w /lqt-moment-magnitude lqt-moment-magnitude-package-1 uv run ruff check`
  - Missing env: `docker exec -w /lqt-moment-magnitude lqt-moment-magnitude-package-1 uv sync --extra testing`
- The project folder is bind-mounted, so host edits are visible in the container.
- When opencode runs inside the devcontainer, drop the `docker exec` prefix.
