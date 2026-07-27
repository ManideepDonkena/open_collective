"""Application entry point: build the QApplication and show the main window."""

from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .main_window import MainWindow


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
