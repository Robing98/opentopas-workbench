#!/usr/bin/env bash
# cacheF from the viewer's page, for a case whose aligned volumes are gone. Every path comes as an argument:
#   run_cachef.sh PAGE VIEWERDATA STANDLOG OUT
# Lines starting with "## STEP" and "## VOL" are read by the case assistant as progress.
set -euo pipefail
unset CDPATH        # a CDPATH makes cd print the folder, which would end up in HERE
win() { case "$1" in [A-Za-z]:\\*|[A-Za-z]:/*) wslpath -a "$1" ;; *) printf '%s\n' "$1" ;; esac; }
PAGE=$(win "$1"); VIEW=$(win "$2"); LOG=$(win "$3"); OUT=$(win "$4")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${MRT_PY:-}"
if [ -z "$PY" ]; then
  for c in "$HOME/.venvs/mrt-overlay/bin/python" "$HOME/.venvs/pyradplan/bin/python"; do
    if [ -x "$c" ] && "$c" -c "import SimpleITK, pydicom, scipy, PIL" 2>/dev/null; then PY=$c; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "Lege die Python-Umgebung ~/.venvs/mrt-overlay an (einmalig, etwa 300 MB)."
  python3 -m venv "$HOME/.venvs/mrt-overlay"
  "$HOME/.venvs/mrt-overlay/bin/pip" install -q numpy scipy pydicom SimpleITK pillow
  PY="$HOME/.venvs/mrt-overlay/bin/python"
fi
mkdir -p "$OUT"
cd "$HERE/.."
export PYTHONUNBUFFERED=1
"$PY" -m mrt_overlay.cli cachef "$PAGE" "$VIEW" "$LOG" "$OUT"
