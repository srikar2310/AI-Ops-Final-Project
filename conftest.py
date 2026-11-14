"""Repo-root pytest configuration.

Lives at the repo root (next to dvc.yaml), so that:
  * the repo root is on sys.path for every test, and `import api`,
    `import features`, `from prepare import ...` work with plain `pytest`
    as well as `python -m pytest`;
  * the custom markers are registered even without a pytest.ini.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("PARAMS_FILE", os.path.join(ROOT, "params.yaml"))

MARKERS = {
    "e2e": "needs the docker compose stack running",
    "engine": "needs a local Stockfish binary (params.yaml eval.engine_path)",
    "slow": "takes more than a few seconds",
    "drift": "changes the backend's drift window; run on its own, after the other e2e tests",
    "data": "checks real pipeline outputs in data/, metrics/, reports/ (skipped if absent)",
}


def pytest_configure(config):
    for name, help_text in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {help_text}")
