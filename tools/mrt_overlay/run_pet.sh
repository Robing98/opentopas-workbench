#!/usr/bin/env bash
# PET layer on the viewer grid. Every path comes as an argument:
#   run_pet.sh PETDIR CACHEF VIEWERDATA OUT [REF] [SIDE]
# REF: brain (default), striatum (needs $LABELS, the SynthSeg file of the viewer's reference T1) or
# contralateral with SIDE links or rechts (the side not to use). PET_STARTS: start rotations, see cli pet --starts.
set -euo pipefail
unset CDPATH        # a CDPATH makes cd print the folder, which would end up in HERE
win() { case "$1" in [A-Za-z]:\\*|[A-Za-z]:/*) wslpath -a "$1" ;; *) printf '%s\n' "$1" ;; esac; }
PET=$(win "$1"); CACHE=$(win "$2"); VIEW=$(win "$3"); OUT=$(win "$4"); REF=${5:-brain}; SIDE=${6:-}
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
ARGS=(--out "$OUT" --ref "$REF")
if [ "$REF" = striatum ]; then ARGS+=(--labels "$(win "${LABELS:?LABELS: SynthSeg-Datei fuer --ref striatum angeben}")"); fi
if [ -n "$SIDE" ]; then ARGS+=(--side "$SIDE"); fi
if [ -n "${PET_STARTS:-}" ]; then ARGS+=(--starts "$PET_STARTS"); fi
mkdir -p "$OUT"
cd "$HERE/.."
"$PY" -m mrt_overlay.cli pet "$PET" "$CACHE" "$VIEW" "${ARGS[@]}"
echo "Fertig: pet.json und pet_<Datum>.js im Datenordner des Viewers, Einzelheiten in $OUT."
