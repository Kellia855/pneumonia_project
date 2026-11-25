#!/usr/bin/env bash
# Run the same Locust test against 1, 2 and 3 API containers and write a results table.
#
#   bash scripts/load_test.sh                    # defaults below
#   USERS=30 API_CPUS=2 bash scripts/load_test.sh
#   CONTAINERS="1 2" bash scripts/load_test.sh   # if the machine has little memory
#
# Build the image once before running this (docker compose build); the rounds reuse it, so no internet is needed.
#
# Each API container is limited to API_CPUS cores so that adding containers adds capacity, as it would
# with separate servers. Results: results/locust/<N>_containers_*.csv / .html and results/locust/summary.md
set -euo pipefail
cd "$(dirname "$0")/.."

USERS=${USERS:-20}
SPAWN_RATE=${SPAWN_RATE:-10}
DURATION=${DURATION:-2m}
CONTAINERS=${CONTAINERS:-"1 2 3"}
HOST=${HOST:-http://localhost:8000}
export API_CPUS=${API_CPUS:-1}
OUT=results/locust
mkdir -p "$OUT"

# The dashboard is not part of the test: stop it to leave its memory to the API containers.
docker compose stop ui

for n in $CONTAINERS; do
  echo "=== $n API container(s), $API_CPUS CPU each, $USERS users for $DURATION ==="
  docker compose up -d --no-build --scale api="$n" --wait mongo api nginx
  sleep 10                                   # let nginx pick up every replica and warm the models
  for i in $(seq 1 10); do curl -s -o /dev/null "$HOST/health"; done
  locust -f locustfile.py --host "$HOST" --headless -u "$USERS" -r "$SPAWN_RATE" -t "$DURATION" \
         --csv "$OUT/${n}_containers" --html "$OUT/${n}_containers.html" --only-summary
done

python scripts/summarize_locust.py "$OUT"

docker compose up -d --no-build --scale api=1 ui    # back to normal: 1 API container + dashboard