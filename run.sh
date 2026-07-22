#!/bin/sh
# Module entrypoint: bootstrap the venv (first run) and exec the module.
cd "$(dirname "$0")"

. ./setup.sh

exec "$PYTHON" src/main.py "$@"
