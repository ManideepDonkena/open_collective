#!/bin/sh
# macOS: double-click this file in Finder to start the app.
# (Finder runs .command files in Terminal; plain .sh files it just opens in an editor.)
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
    python3 start.py
else
    python start.py
fi
