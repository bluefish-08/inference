#!/usr/bin/env bash
# Build the prod image from the current checkout and roll it out.
#
#   scripts/deploy-prod.sh              # full build (rebuilds the Web UI)
#   scripts/deploy-prod.sh --fast       # reuse the existing Web UI export
#   TAG=mytag scripts/deploy-prod.sh    # override the image tag
#
# Run this on a machine with npm. build_web() only *warns* when npm is missing,
# so building on the server silently produces a wheel with no Web UI.
set -euo pipefail

HOST=${HOST:-szhdeploy@10.189.109.220}
DEPLOY_DIR=${DEPLOY_DIR:-/opt/szh_Center/xinference}
TAG=${TAG:-$(date +%Y%m%d-%H%M)}
IMAGE="xinference-prod:${TAG}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

echo "==> building wheel ($([ $FAST = 1 ] && echo 'reusing Web UI' || echo 'with Web UI'))"
rm -rf dist
if [ $FAST = 1 ]; then
  [ -f xinference/ui/web/dist/index.html ] || {
    echo "no existing Web UI export; run without --fast" >&2; exit 1; }
  NO_WEB_UI=1 uv build --wheel
else
  command -v npm >/dev/null || { echo "npm not found: the wheel would ship without the Web UI" >&2; exit 1; }
  uv build --wheel
fi

WHEEL=$(ls dist/xinference-*.whl)
python3 - "$WHEEL" <<'EOF'
import sys, zipfile
n = zipfile.ZipFile(sys.argv[1]).namelist()
assert any(x.endswith("ui/web/dist/index.html") for x in n), "wheel has no Web UI"
print(f"    ok: {sys.argv[1]} ({sum(1 for x in n if '/ui/web/dist/' in x)} UI files)")
EOF

echo "==> uploading to $HOST"
ssh "$HOST" "mkdir -p $DEPLOY_DIR/dist && rm -f $DEPLOY_DIR/dist/xinference-*.whl"
scp -q "$WHEEL" scripts/Dockerfile.prod "$HOST:$DEPLOY_DIR/"
ssh "$HOST" "mv $DEPLOY_DIR/$(basename "$WHEEL") $DEPLOY_DIR/dist/"

echo "==> building $IMAGE"
ssh "$HOST" "cd $DEPLOY_DIR && docker build -q -f Dockerfile.prod -t $IMAGE . >/dev/null"

echo "==> switching and restarting"
ssh "$HOST" "cd $DEPLOY_DIR && \
  cp .env .env.bak.\$(date +%Y%m%d%H%M%S) && \
  sed -i 's|^XINFERENCE_IMAGE=.*|XINFERENCE_IMAGE=$IMAGE|' .env && \
  docker compose up -d --force-recreate xinference xinference-wheels >/dev/null 2>&1"

echo "==> waiting for health"
for _ in $(seq 30); do
  sleep 5
  status=$(ssh "$HOST" "docker inspect -f '{{.State.Health.Status}}' xinference-xinference-1" 2>/dev/null || echo starting)
  [ "$status" = healthy ] && break
done
[ "$status" = healthy ] || { echo "container is $status; check: ssh $HOST docker logs --tail 50 xinference-xinference-1" >&2; exit 1; }

ssh "$HOST" "docker exec xinference-xinference-1 python3 -c 'import xinference; print(\"deployed:\", xinference.__version__)'" 2>/dev/null | tail -1
echo "==> $IMAGE is live (rollback: set XINFERENCE_IMAGE in $DEPLOY_DIR/.env to a previous tag, then docker compose up -d --force-recreate xinference)"
