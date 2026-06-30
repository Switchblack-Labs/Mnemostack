#!/usr/bin/env bash
set -e

# ==============================================================================
# run.sh — canonical launcher for Mnemostack
#
# Project profile (auto-determined from the repository):
#   Primary language      : Python (requires-python >=3.11)
#   Runtime/interpreter    : CPython 3.11+
#   Framework              : MCP / FastMCP (mcp.server.fastmcp)
#   Entry point            : mnemostack.mcp.server:run  (console script "mnemostack")
#   Dependency manager     : pip + pyproject.toml (PEP 621)
#   Build system           : hatchling (editable install; no compile step)
#   Serves network traffic : NO — MCP "stdio" transport (server.transport: stdio)
#   Default port           : N/A (no network listener)
#   Required env vars       : none (defaults in mnemostack/config/defaults.yaml;
#                            LLM/embeddings default to local Ollama, no API key)
# ==============================================================================

# Resolve symlinks to locate this script, then the repository root, so the
# script works no matter which directory it is invoked from.
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"

REPO_ROOT=""
VENV_DIR=""
PYTHON_BIN=""

# ------------------------------------------------------------------------------
locate_repo_root() {
    echo "==> Locating repository root..."
    if command -v git >/dev/null 2>&1 &&
        git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
        REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
    else
        # Fallback: walk upward until pyproject.toml is found.
        local dir="$SCRIPT_DIR"
        while [ "$dir" != "/" ] && [ ! -f "$dir/pyproject.toml" ]; do
            dir="$(dirname "$dir")"
        done
        REPO_ROOT="$dir"
    fi
    if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
        echo "ERROR: could not locate repository root (no pyproject.toml found)." >&2
        exit 1
    fi
    VENV_DIR="$REPO_ROOT/.venv"
    echo "    repository root: $REPO_ROOT"
}

# ------------------------------------------------------------------------------
check_system_tools() {
    echo "==> Verifying required system tools..."

    # A Python >=3.11 interpreter is the only hard requirement to launch.
    local candidate found=""
    for candidate in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' >/dev/null 2>&1; then
                found="$candidate"
                break
            fi
        fi
    done

    if [ -z "$found" ]; then
        echo "ERROR: required tool 'python3' (>=3.11) not found on PATH." >&2
        echo "       Install Python 3.11 or newer and retry." >&2
        exit 1
    fi
    PYTHON_BIN="$(command -v "$found")"
    echo "    python: $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"

    # Ollama powers local embeddings + consolidation. The server boots without
    # it (calls are lazy), so this is a warning, not a hard failure.
    if command -v ollama >/dev/null 2>&1; then
        echo "    ollama: $(command -v ollama) (local LLM/embeddings available)"
    else
        echo "    WARNING: 'ollama' not found — indexing/consolidation will fail" >&2
        echo "             until an embedding/LLM provider is configured." >&2
    fi
}

# ------------------------------------------------------------------------------
setup_python_env() {
    echo "==> Preparing isolated Python environment..."

    # Reuse an existing virtualenv; never delete or recreate it.
    if [ ! -d "$VENV_DIR" ]; then
        echo "    creating virtualenv at $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        echo "    reusing existing virtualenv at $VENV_DIR"
    fi

    local venv_python="$VENV_DIR/bin/python"
    if [ ! -x "$venv_python" ]; then
        echo "ERROR: virtualenv interpreter missing at $venv_python" >&2
        exit 1
    fi

    # Install/refresh the package quietly. Editable install is idempotent and
    # cheap to re-run; pip is a no-op when nothing changed.
    echo "    installing project dependencies (quiet)..."
    "$venv_python" -m pip install --quiet --upgrade pip >/dev/null
    "$venv_python" -m pip install --quiet -e "$REPO_ROOT" >/dev/null
    echo "    dependencies ready"
}

# ------------------------------------------------------------------------------
check_env_vars() {
    echo "==> Checking required environment variables..."

    # No .env.example/.sample/.template ship with this repo, and every runtime
    # knob has a default in mnemostack/config/defaults.yaml. The default LLM and
    # embedding providers are local Ollama, which needs no API key. Therefore no
    # environment variable is strictly required to launch.
    local missing=0
    local example
    for example in "$REPO_ROOT/.env.example" "$REPO_ROOT/.env.sample" "$REPO_ROOT/.env.template"; do
        if [ -f "$example" ]; then
            echo "    found $(basename "$example"); validating required keys..."
            # Lines of form KEY=... with no default value (empty RHS) are required.
            while IFS= read -r line; do
                case "$line" in
                    ''|\#*) continue ;;
                esac
                local key="${line%%=*}"
                local val="${line#*=}"
                if [ -z "$val" ] && [ -z "${!key:-}" ]; then
                    echo "    MISSING required environment variable: $key" >&2
                    missing=1
                fi
            done <"$example"
        fi
    done

    if [ "$missing" -ne 0 ]; then
        echo "ERROR: one or more required environment variables are unset." >&2
        exit 1
    fi
    echo "    no required environment variables missing"
}

# ------------------------------------------------------------------------------
configure_port() {
    echo "==> Configuring network port..."

    # Determined from mnemostack/config/defaults.yaml -> server.transport: stdio.
    # The MCP server communicates over stdio and opens no network listener, so
    # there is no port to detect, free, or export. Port logic is skipped.
    echo "    transport is stdio — application serves no network port; skipping port management"
}

# ------------------------------------------------------------------------------
build_project() {
    echo "==> Running build step..."

    # hatchling builds are not required to run from an editable install, and
    # there is no compiled artifact or asset bundling step. Nothing to do.
    echo "    no build step required (editable install, pure Python)"
}

# ------------------------------------------------------------------------------
launch() {
    echo "==> Launching mnemostack MCP server (stdio transport)..."

    local entry="$VENV_DIR/bin/mnemostack"
    cd "$REPO_ROOT"
    if [ -x "$entry" ]; then
        exec "$entry"
    fi
    # Fallback to the module entry point if the console script is unavailable.
    exec "$VENV_DIR/bin/python" -c 'from mnemostack.mcp.server import run; run()'
}

# ------------------------------------------------------------------------------
main() {
    locate_repo_root
    check_system_tools
    setup_python_env
    check_env_vars
    configure_port
    build_project
    launch
}

main "$@"
