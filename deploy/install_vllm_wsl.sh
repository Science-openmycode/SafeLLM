#!/usr/bin/env bash
set -euo pipefail

VENV=/opt/aloepri-vllm
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
python3 -m venv "$VENV"
"$VENV/bin/pip" install --index-url "$PIP_INDEX_URL" --upgrade pip
"$VENV/bin/pip" install --index-url "$PIP_INDEX_URL" vllm==0.11.1
"$VENV/bin/pip" install --ignore-requires-python --no-deps --editable /mnt/e/AloePri
"$VENV/bin/python" -c 'import vllm; print(vllm.__version__)'
