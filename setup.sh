#!/bin/sh
# Create the module venv and install requirements. Sourced by run.sh (which
# uses $PYTHON afterwards); safe to run standalone too.
cd "$(dirname "$0")"

VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"
ENV_ERROR="This module requires Python >=3.10, pip, and virtualenv to be installed."

if ! python3 -m venv $VENV_NAME >/dev/null 2>&1; then
    echo "Failed to create virtualenv."
    if command -v apt-get >/dev/null; then
        echo "Detected Debian/Ubuntu, attempting to install python3-venv automatically."
        SUDO="sudo"
        if ! command -v $SUDO >/dev/null; then
            SUDO=""
        fi
        if ! apt info python3-venv >/dev/null 2>&1; then
            echo "Package info not found, trying apt update"
            $SUDO apt -qq update >/dev/null
        fi
        $SUDO apt install -qqy python3-venv >/dev/null 2>&1
        if ! python3 -m venv $VENV_NAME >/dev/null 2>&1; then
            echo $ENV_ERROR >&2
            exit 1
        fi
    else
        echo $ENV_ERROR >&2
        exit 1
    fi
fi

# Reinstall whenever requirements.txt changes (marker holds its checksum).
MARKER=".installed"
CHECKSUM=$(cksum requirements.txt)
if [ ! -f "$MARKER" ] || [ "$(cat $MARKER)" != "$CHECKSUM" ]; then
    echo "Installing/upgrading Python packages (torch download can take a while)..."
    if ! $PYTHON -m pip install -r requirements.txt -q; then
        echo "pip install failed" >&2
        exit 1
    fi
    echo "$CHECKSUM" > "$MARKER"
fi
