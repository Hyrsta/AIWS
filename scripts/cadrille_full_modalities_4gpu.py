#!/usr/bin/env python3
"""Backward-compatible wrapper for the old Cadrille batch-run entrypoint.

Canonical entrypoint: scripts/cadrille_batch.py
"""
from cadrille_batch import main


if __name__ == "__main__":
    main()
