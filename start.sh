#!/usr/bin/env bash
set -e

# ==============================================================================
# start.sh — one-command bootstrap for Mnemostack.
#
# Sets up a clean machine to run the Mnemostack MCP server:
#   1. verifies tooling
#   2. creates/reuses a Python virtualenv and installs the project (+ dev deps)
#   3. runs the test suite
#   4. checks (and offers to pull) the local Ollama models it uses
#   5. prints how to register the server with an MCP client
#
# Idempotent and safe to run repeatedly. Never uses sudo.
# ==============================================================================

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
REPO_ROOT="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON_BIN=""

# Embedding + consolidation models Mnemostack uses by default (see defaults.yaml).
EMBED_MODEL="nomic-embed-text"
# defaults.yaml ships ollama/llama3:8b for consolidation; pull it for parity.
CONSOLIDATION_MODEL="llama3:8b"

# ------------------------------------------------------------------------------
check_tools() {
    echo "==> [1/5] Verifying required tools..."
    local candidate found=""
    for candidate in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' >/dev/null 2>&1; then
            found="$candidate"
            break
        fi
    done
    if [ -z "$found" ]; then
        echo "ERROR: required tool 'python3' (>=3.11) not found on PATH." >&2
        exit 1
    fi
    PYTHON_BIN="$(command -v "$found")"
    echo "    python: $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"

    if command -v ollama >/dev/null 2>&1; then
        echo "    ollama: $(command -v ollama)"
    else
        echo "    WARNING: 'ollama' not found. Install it from https://ollama.com to"
        echo "             enable local embeddings + consolidation (no API key needed)."
    fi
}

# ------------------------------------------------------------------------------
setup_venv() {
    echo "==> [2/5] Setting up Python virtualenv..."
    if [ ! -d "$VENV_DIR" ]; then
        echo "    creating $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        echo "    reusing existing $VENV_DIR"
    fi
    echo "    installing project + dev dependencies (quiet)..."
    "$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip >/dev/null
    "$VENV_DIR/bin/python" -m pip install --quiet -e "$REPO_ROOT[dev]" >/dev/null
    echo "    dependencies ready"
}

# ------------------------------------------------------------------------------
run_tests() {
    echo "==> [3/5] Running test suite..."
    if "$VENV_DIR/bin/python" -m pytest -q "$REPO_ROOT/tests"; then
        echo "    tests passed"
    else
        echo "ERROR: test suite failed." >&2
        exit 1
    fi
}

# ------------------------------------------------------------------------------
check_models() {
    echo "==> [4/5] Checking local Ollama models..."
    if ! command -v ollama >/dev/null 2>&1; then
        echo "    skipping (ollama not installed)"
        return 0
    fi
    local installed
    installed="$(ollama list 2>/dev/null || true)"
    local model
    for model in "$EMBED_MODEL" "$CONSOLIDATION_MODEL"; do
        if printf '%s\n' "$installed" | grep -q "$model"; then
            echo "    present: $model"
        else
            echo "    missing: $model"
            if [ "${MNEMO_PULL_MODELS:-0}" = "1" ]; then
                echo "      pulling $model ..."
                ollama pull "$model"
            else
                echo "      run 'ollama pull $model' (or re-run with MNEMO_PULL_MODELS=1)"
            fi
        fi
    done
}

# ------------------------------------------------------------------------------
print_next_steps() {
    echo "==> [5/5] Setup complete."
    cat <<EOF

Mnemostack is installed. To launch the MCP server:

    ./run.sh

It speaks the MCP protocol over stdio (no network port). Register it with an
MCP client. Example for Claude Code:

    claude mcp add mnemostack -- "$REPO_ROOT/run.sh"

Or add to your client's MCP config (e.g. mcpServers in the client settings):

    {
      "mcpServers": {
        "mnemostack": {
          "command": "$REPO_ROOT/run.sh"
        }
      }
    }

Then call index_project(root_dir) once, and query_codebase / get_full_context
from your assistant.
EOF
}

# ------------------------------------------------------------------------------
main() {
    cd "$REPO_ROOT"
    check_tools
    setup_venv
    run_tests
    check_models
    print_next_steps
}

main "$@"
