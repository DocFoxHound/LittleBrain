#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
output=${1:-dist}
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
python3 -m venv "$work/venv"
python="$work/venv/bin/python"
"$python" -m pip install --only-binary=:all: numpy==2.5.4 pygame-ce==2.5.8 pyinstaller==6.22.3
"$python" -m PyInstaller --noconfirm --onefile --name littlebrain \
    --distpath "$output" --workpath "$work/build" --specpath "$work" \
    --paths "$PWD/prototype" --paths "$PWD/prototype/brain" prototype/viz/brainview.py
cp LICENSE "$output/LICENSE"
SLOP_SAVE_DIR="$work/saves" SDL_AUDIODRIVER=dummy OPENBLAS_NUM_THREADS=1 \
    "$output/littlebrain" --shots 2
test -s "$work/saves/shots/littlebrain_scale1_turn2.png"
