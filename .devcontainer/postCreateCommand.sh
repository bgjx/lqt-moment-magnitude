#!/bin/bash
set -euo pipefail

# zsh/oh-my-zsh already ship in the base image; extra is named "testing" in pyproject.toml
uv sync --extra testing