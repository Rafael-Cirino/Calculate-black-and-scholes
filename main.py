"""Compatibility wrapper for the legacy project entrypoint.

This file keeps the original `main.py` location working while delegating all actual
runtime behavior to the new package-based CLI.
"""

from black_scholes.cli import app


if __name__ == "__main__":
    app()
