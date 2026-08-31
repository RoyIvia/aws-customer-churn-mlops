#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="$REPO_ROOT/.venv"

echo "==> AWS Customer Churn MLOps project bootstrap"
echo "==> Repository root: $REPO_ROOT"

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

echo
echo "==> Checking prerequisites..."

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: Git is required but was not found."
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN was not found."
    echo "Install Python 3.12 or set PYTHON_BIN to its executable."
    exit 1
fi

PYTHON_VERSION="$(
    "$PYTHON_BIN" -c \
    'import sys; print(".".join(map(str, sys.version_info[:3])))'
)"

if [[ "$PYTHON_VERSION" != 3.12.* ]]; then
    echo "ERROR: Python 3.12 is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "    ✓ Git available"
echo "    ✓ Python $PYTHON_VERSION"

# ---------------------------------------------------------------------------
# Git repository
# ---------------------------------------------------------------------------

echo
echo "==> Validating Git repository..."

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: $REPO_ROOT is not a Git repository."
    exit 1
fi

echo "    ✓ Git repository detected"

# ---------------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------------

DIRECTORIES=(
    "data/raw"
    "data/processed"
    "data/external"
    "src"
    "tests"
    "scripts"
)

echo
echo "==> Creating project directories..."

for directory in "${DIRECTORIES[@]}"; do
    mkdir -p "$directory"
    echo "    ✓ $directory"
done

# ---------------------------------------------------------------------------
# Python virtual environment
# ---------------------------------------------------------------------------

echo
echo "==> Configuring Python virtual environment..."

if [[ -d "$VENV_DIR" ]]; then
    echo "    ✓ .venv already exists"
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "    ✓ .venv created"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

VENV_VERSION="$(
    "$VENV_PYTHON" -c \
    'import sys; print(".".join(map(str, sys.version_info[:3])))'
)"

if [[ "$VENV_VERSION" != 3.12.* ]]; then
    echo "ERROR: .venv is not using Python 3.12."
    echo "Found: $VENV_VERSION"
    exit 1
fi

echo "    ✓ Virtual environment uses Python $VENV_VERSION"

# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

echo
echo "==> Bootstrap complete."
echo
echo "Activate the environment with:"
echo "    source .venv/bin/activate"
echo
echo "Next step:"
echo "    Configure project dependencies in requirements.txt"
