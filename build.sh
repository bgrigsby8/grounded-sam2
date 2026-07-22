#!/bin/sh
# Package the module as a source tarball. PyTorch cannot be reasonably
# bundled with PyInstaller; instead setup.sh installs dependencies into a
# venv on the target machine the first time the module starts.
cd "$(dirname "$0")"
set -e

mkdir -p dist
tar -czf dist/archive.tar.gz \
    --exclude '__pycache__' \
    meta.json \
    run.sh \
    setup.sh \
    requirements.txt \
    src
