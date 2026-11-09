#!/usr/bin/env bash
set -euo pipefail
root=$(pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps --wheel-dir "$tmp/wheels" .
python -m venv "$tmp/env"
"$tmp/env/bin/pip" install --no-deps "$tmp"/wheels/*.whl
cd "$tmp"
"$tmp/env/bin/python" "$root/tools/verify_installed.py" "$root/docs/controls.json" "$tmp/env/bin/warplab"

"$tmp/env/bin/python" "$root/tools/verify_lengths_installed.py" "$root/docs/length-controls.json" "$tmp/env/bin/warplab"

"$tmp/env/bin/python" "$root/tools/verify_inverse_installed.py" "$root/docs/inverse-controls.json" "$tmp/env/bin/warplab"
