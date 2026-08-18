"""
Offline demo server.

Starts the exact same Flask app but with NCBI replaced by the canned records
from `tools/selftest.py`. Useful for developing the frontend, giving a demo
without internet, or checking the UI on a machine behind a firewall.

    python tools/demo_server.py
    -> http://127.0.0.1:5001
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.selftest as selftest  # noqa: E402,F401  (applies the monkeypatch)

from app import app  # noqa: E402
from config import config  # noqa: E402

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    print("=" * 66)
    print("  OFFLINE DEMO MODE — PubMed is mocked, no network calls are made.")
    print(f"  http://127.0.0.1:{port}")
    print("=" * 66)
    app.run(host="127.0.0.1", port=port, debug=False)
