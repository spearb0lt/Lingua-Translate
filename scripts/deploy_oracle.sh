#!/usr/bin/env bash
#
# Provision the full edition on an Oracle Cloud Always Free ARM instance.
#
# Target: Ampere A1 (aarch64), Ubuntu 22.04 or 24.04. As of June 2026 the
# Always Free ARM allocation is 2 OCPU / 12 GB (it used to be 4 / 24) -- still
# comfortably enough for IndicTrans2 plus Whisper on CPU.
#
# Usage, on the instance:
#   curl -fsSL <raw-url>/scripts/deploy_oracle.sh | bash
# or, from a clone:
#   bash scripts/deploy_oracle.sh
#
set -euo pipefail

APP_USER="${APP_USER:-$(whoami)}"
APP_DIR="${APP_DIR:-$HOME/lingua}"
REPO="${REPO:-}"                 # set to your git remote to clone instead of copy
PORT="${PORT:-8000}"
WHISPER_MODEL="${WHISPER_MODEL:-base}"   # 'small' is better but slower on 2 OCPU

say() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }

say "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
  python3 python3-venv python3-dev python3-pip \
  build-essential git curl ca-certificates

say "Application directory: $APP_DIR"
if [[ -n "$REPO" && ! -d "$APP_DIR/.git" ]]; then
  git clone "$REPO" "$APP_DIR"
elif [[ ! -d "$APP_DIR" ]]; then
  echo "No repo set and $APP_DIR does not exist."
  echo "Copy the project there first, or set REPO=<git url>."
  exit 1
fi
cd "$APP_DIR"

say "Virtual environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip wheel setuptools

say "Server dependencies"
pip install --quiet -r requirements-server.txt

say "Local model dependencies (CPU build of torch -- this box has no GPU)"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet -r requirements-local.txt

say "Pre-downloading model weights"
# Doing this now means the first user request is fast instead of a 5-minute
# download, and it fails loudly here rather than silently at runtime.
python scripts/download_models.py --whisper "$WHISPER_MODEL" || {
  echo "Model download failed. The app will still start and fall back to the"
  echo "network providers, but local translation will be unavailable."
}

say "Opening port $PORT in the instance firewall"
# Oracle images ship with a default-deny iptables policy that surprises people.
# You must ALSO allow the port in the VCN security list in the OCI console.
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport "$PORT" -j ACCEPT || true
sudo netfilter-persistent save >/dev/null 2>&1 || true

say "Installing systemd service"
sudo tee /etc/systemd/system/lingua.service >/dev/null <<UNIT
[Unit]
Description=Lingua translation server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment="LINGUA_EDITION=full"
Environment="LINGUA_DEVICE=cpu"
Environment="LINGUA_WHISPER_MODEL=${WHISPER_MODEL}"
Environment="LINGUA_MODEL_CACHE=2"
Environment="LINGUA_PRELOAD=1"
ExecStart=${APP_DIR}/.venv/bin/uvicorn server.main:app --host 0.0.0.0 --port ${PORT} --workers 1
Restart=always
RestartSec=5
# 12 GB box: keep one worker and let the model cache bound memory instead.
MemoryMax=9G

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now lingua.service
sleep 3
sudo systemctl status lingua.service --no-pager -l | head -20 || true

say "Done"
cat <<INFO

  Local check :  curl http://localhost:${PORT}/api/health
  Public URL  :  http://<your-instance-public-ip>:${PORT}

  If the public URL times out, it is almost always the VCN security list, not
  this machine. In the OCI console: Networking > Virtual Cloud Networks >
  your VCN > Security Lists > add an ingress rule for TCP ${PORT}.

  Logs        :  sudo journalctl -u lingua -f
  Restart     :  sudo systemctl restart lingua

INFO
