#!/bin/sh
# Starts the API (default) or the dashboard (SERVICE=ui). Cloud hosts set $PORT; locally the defaults are used.
# XSRF protection is off for the dashboard: on a deployed server its cookie check makes file uploads fail with
# "403", and the dashboard has no user accounts for it to protect.
set -e
if [ "$SERVICE" = "ui" ]; then
  exec streamlit run app/ui.py --server.port "${PORT:-8501}" --server.address 0.0.0.0 \
       --server.headless true --server.maxUploadSize 200 --browser.gatherUsageStats false \
       --server.enableXsrfProtection false --server.enableCORS false
else
  exec uvicorn app.api:app --host 0.0.0.0 --port "${PORT:-8000}"
fi