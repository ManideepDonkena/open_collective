#!/usr/bin/env python3
"""
start.py  --  the easy button.

You do NOT need to know anything technical. Just run this file:

    python3 start.py         (Mac / Linux)
    python start.py          (Windows)

...or double-click "start.sh" (Mac/Linux) or "start.bat" (Windows).

The first time, it quietly sets things up (this can take a few minutes). After
that it opens the app straight away. Nothing to configure.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"                 # a private workspace for this app's pieces


def venv_python() -> Path:
    """Where Python lives inside the private workspace (differs on Windows)."""
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _run(cmd):
    print(">", " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd])


def ensure_workspace() -> Path:
    py = venv_python()
    if py.exists():
        return py
    print("First-time setup: creating a private workspace...")
    try:
        _run([sys.executable, "-m", "venv", str(VENV)])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nCould not create the workspace.")
        print("On Linux, install the venv tool first, then run start again:")
        print("    sudo apt install -y python3-venv")
        raise SystemExit(1)
    return venv_python()


def has_everything(py: Path) -> bool:
    """True if the app's pieces are already installed in the workspace."""
    check = subprocess.run(
        [str(py), "-c",
         "import PySide6, numpy, scipy, matplotlib, PIL, yaml, h5py"],
        capture_output=True,
    )
    return check.returncode == 0


def install(py: Path):
    print("Installing the pieces the app needs (only happens once)...")
    print("This can take a few minutes. Please wait.\n")
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    # core (numpy/scipy/matplotlib) + GUI (PySide6) + pictures/animation (pillow)
    # + saving formats (pyyaml for configs, h5py for large trajectory files)
    _run([str(py), "-m", "pip", "install",
          "-r", str(HERE / "requirements.txt"),
          "PySide6", "pillow", "pyyaml", "h5py"])


def launch(py: Path) -> int:
    print("\nStarting the app... (close its window to come back here)\n")
    result = subprocess.run([str(py), str(HERE / "run_gui.py")],
                            capture_output=True, text=True)
    if result.returncode != 0:
        # show whatever the app said, then a friendly hint if we recognise it
        sys.stdout.write(result.stdout or "")
        sys.stderr.write(result.stderr or "")
        _hint(result.stderr or "")
    return result.returncode


def _hint(err: str):
    if "xcb" in err or "platform plugin" in err:
        print("\n" + "=" * 62)
        print("The window could not open. On Linux you usually just need one")
        print("system package. Copy this line, run it, then start again:\n")
        print("    sudo apt install -y libxcb-cursor0")
        print("\n(Fedora:  sudo dnf install -y xcb-util-cursor)")
        print("(Arch:    sudo pacman -S xcb-util-cursor)")
        print("=" * 62)


def main() -> int:
    py = ensure_workspace()
    if not has_everything(py):
        install(py)
    else:
        print("Everything is already set up.")
    # OC_LAUNCH=0 is used only by the automated test, to set up without opening a window.
    if os.environ.get("OC_LAUNCH", "1") != "1":
        print("Setup finished (not opening the window this time).")
        return 0
    return launch(py)


if __name__ == "__main__":
    raise SystemExit(main())
