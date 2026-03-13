#!/bin/bash
###############################################################################
# ONE-SHOT VM SETUP SCRIPT
# Run this once on instance-20260313-014807 (project: webdev2025-1)
#
# Usage:
#   gcloud compute ssh instance-20260313-014807 --project=webdev2025-1
#   # upload this script, then:
#   chmod +x setup_vm.sh && ./setup_vm.sh
###############################################################################
set -euo pipefail

echo "=== RepoAI Experiment VM Setup ==="
echo "Started: $(date)"

# ── 1. System packages ──
echo "[1/8] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip python3-venv python3-dev \
    git tmux curl wget unzip \
    openjdk-17-jdk maven \
    docker.io docker-compose

# Docker permissions
sudo usermod -aG docker $USER || true
sudo systemctl enable docker
sudo systemctl start docker

# ── 2. Disk space check ──
echo "[2/8] Checking disk space..."
AVAIL=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available disk space: ${AVAIL}GB"
if [ "$AVAIL" -lt 200 ]; then
    echo "WARNING: Less than 200GB free. SWE-bench needs 120GB+ for Docker images."
    echo "Consider resizing the disk:"
    echo "  gcloud compute disks resize $(hostname) --size=300GB --project=webdev2025-1"
    echo "  sudo growpart /dev/sda 1 && sudo resize2fs /dev/sda1"
fi

# ── 3. Clone experiment repo ──
echo "[3/8] Cloning experiment repo..."
cd ~
if [ -d "Repo-AI" ]; then
    echo "Repo-AI already exists, pulling latest..."
    cd Repo-AI && git pull origin main && cd ~
else
    git clone https://github.com/cnacha-mfu/Repo-AI.git
fi
cd ~/Repo-AI
git checkout -b experiment 2>/dev/null || git checkout experiment

# ── 4. Python environment ──
echo "[4/8] Setting up Python environment..."
python3 -m venv ~/repoai-env
source ~/repoai-env/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt -q 2>/dev/null || true

# Additional dependencies for experiments
pip install -q \
    langchain-openai \
    langchain-google-genai \
    google-generativeai \
    openai \
    scipy \
    seaborn \
    matplotlib \
    pandas \
    scikit-learn \
    tiktoken \
    faiss-cpu

# ── 5. Install SWE-bench ──
echo "[5/8] Installing SWE-bench..."
cd ~
if [ ! -d "SWE-bench" ]; then
    git clone https://github.com/princeton-nlp/SWE-bench.git
fi
cd SWE-bench
pip install -e . -q

# ── 6. Install RefactorBench ──
echo "[6/8] Installing RefactorBench..."
cd ~
if [ ! -d "RefactorBench" ]; then
    git clone https://github.com/microsoft/RefactorBench.git
fi

# ── 7. API keys setup ──
echo "[7/8] Setting up API keys..."
ENV_FILE=~/Repo-AI/.env

if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << 'ENVEOF'
# NVIDIA NIM API Keys (round-robin for rate limit distribution)
NVIDIA_API_KEY=nvapi-REPLACE_ME
NVIDIA_API_KEY_2=nvapi-REPLACE_ME
NVIDIA_API_KEY_3=nvapi-REPLACE_ME

# Google Gemini API Key ($50 student credit)
GOOGLE_API_KEY=REPLACE_ME

# GitHub Token (for pushing results)
GITHUB_TOKEN=REPLACE_ME
ENVEOF
    echo ""
    echo "========================================="
    echo "  IMPORTANT: Edit ~/Repo-AI/.env with your actual API keys!"
    echo "  nano ~/Repo-AI/.env"
    echo "========================================="
    echo ""
else
    echo ".env already exists, skipping."
fi

# ── 8. Copy experiment scripts ──
echo "[8/8] Setting up experiment scripts..."
cd ~/Repo-AI

# Create experiment directories
mkdir -p experiments/results experiments/adapters experiments/logs

echo ""
echo "=== VM Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit API keys:        nano ~/Repo-AI/.env"
echo "  2. Copy experiment scripts from your local machine to ~/Repo-AI/experiments/"
echo "  3. Start experiments:    tmux new -s experiment"
echo "                           cd ~/Repo-AI && ./experiments/run_all_experiments.sh"
echo "                           # Ctrl+B, D to detach. Close laptop."
echo ""
echo "Check specs:"
echo "  CPUs: $(nproc)"
echo "  RAM:  $(free -h | awk '/Mem:/ {print $2}')"
echo "  Disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo ""
