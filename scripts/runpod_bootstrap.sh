#!/usr/bin/env bash
# RunPod bootstrap for emotion_probes.
#
# Assumes the pod was launched from a PyTorch base image and the project
# directory has been uploaded to /workspace/emotion_probes_project (or
# whatever directory you cd into before running this).
#
# Idempotent: safe to re-run after a partial setup or a pod restart.
#
# Usage:
#   cd /workspace/emotion_probes_project
#   bash scripts/runpod_bootstrap.sh

set -euo pipefail

banner() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# ---- 1. Environment ----

banner "1. Environment"
if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not found. This script is for GPU pods only."
    exit 1
fi
nvidia-smi | head -15
echo ""
python3 --version
PYV=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
case "$PYV" in
    3.10|3.11|3.12) echo "Python $PYV: ok";;
    *) echo "WARNING: tested on Python 3.10-3.12, got $PYV";;
esac

# ---- 2. Disk space ----

banner "2. Disk space"
df -h "$(pwd)" .
echo ""
# Gemma 2 27B weights ~54 GB bf16, plus activations ~12 GB, plus venv ~3 GB.
# Need ~80 GB free on the project volume.
AVAIL_GB=$(df -BG --output=avail "$(pwd)" | tail -1 | tr -d 'G ')
if [ "$AVAIL_GB" -lt 80 ]; then
    echo "WARNING: only ${AVAIL_GB}G free; need ~80G for Gemma 2 27B + activations."
fi

# ---- 3. Virtual env ----

banner "3. Python virtual environment"
if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip

# ---- 4. Install project + deps ----

banner "4. Install emotion_probes"
pip install --quiet -e ".[dev]"
echo "Installed."

# ---- 5. Verify CUDA & PyTorch ----

banner "5. PyTorch + CUDA"
python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'device: {torch.cuda.get_device_name(0)}')
    print(f'vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'bf16: {torch.cuda.is_bf16_supported()}')
"

# ---- 6. HuggingFace auth ----

banner "6. HuggingFace auth"
if [ -z "${HF_TOKEN:-}" ]; then
    if [ -f "$HOME/.cache/huggingface/token" ]; then
        echo "HF_TOKEN env var unset but ~/.cache/huggingface/token exists; using cached login."
    else
        echo "ERROR: no HF_TOKEN set and no cached login."
        echo "  export HF_TOKEN=hf_yourtokenhere"
        echo "  # or:"
        echo "  huggingface-cli login"
        echo ""
        echo "See docs/GPU_SETUP.md for token creation instructions."
        exit 1
    fi
else
    echo "HF_TOKEN is set."
fi

python -c "
from huggingface_hub import whoami
try:
    info = whoami()
    print(f'Logged in as: {info[\"name\"]}')
except Exception as e:
    print(f'ERROR: HF auth failed: {e}')
    raise
"

# ---- 7. Verify Gemma 2 27B access ----

banner "7. Gemma 2 27B access"
python -c "
from huggingface_hub import model_info
try:
    info = model_info('google/gemma-2-27b')
    print(f'Gemma 2 27B reachable: {info.modelId}')
    if info.gated:
        print('  (gated repo; license acceptance is in place)')
except Exception as e:
    msg = str(e)
    print(f'ERROR accessing google/gemma-2-27b: {msg}')
    if 'gated' in msg.lower() or '403' in msg or '401' in msg:
        print()
        print('You probably have not accepted the Gemma license yet.')
        print('Open https://huggingface.co/google/gemma-2-27b in a browser,')
        print('click \"Acknowledge license\", and try again.')
    raise
"

# ---- 8. Smoke tests ----

banner "8. Smoke tests (under 1 minute)"
pytest tests/ -q --no-header

# ---- 9. Story corpus check ----

banner "9. Story corpus check"
if [ -f data/stories/stories.jsonl ]; then
    LINES=$(wc -l < data/stories/stories.jsonl)
    echo "stories.jsonl: $LINES lines (expected 14400)"
else
    echo "WARNING: data/stories/stories.jsonl missing — upload before running the pipeline."
fi
if [ -f data/stories/neutral_dialogues.jsonl ]; then
    LINES=$(wc -l < data/stories/neutral_dialogues.jsonl)
    echo "neutral_dialogues.jsonl: $LINES lines (expected 500)"
else
    echo "WARNING: data/stories/neutral_dialogues.jsonl missing — upload before running."
fi

# ---- Done ----

banner "Bootstrap complete"
cat <<'EOF'
Next:
    source .venv/bin/activate
    python scripts/run_extraction_pipeline.py

The driver is resumable — interrupt with Ctrl-C and re-run to pick up
where it left off.
EOF
