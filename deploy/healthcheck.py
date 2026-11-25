"""Docker HEALTHCHECK: the API answers /health, the dashboard answers /_stcore/health."""
import os
import sys
import urllib.request

ui = os.getenv("SERVICE") == "ui"
port = os.getenv("PORT", "8501" if ui else "8000")
path = "/_stcore/health" if ui else "/health"
try:
    urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=4)
except Exception:
    sys.exit(1)
