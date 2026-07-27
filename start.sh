#!/bin/sh
# Double-click this (or run ./start.sh) on Mac or Linux.
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
    python3 start.py
else
    python start.py
fi
