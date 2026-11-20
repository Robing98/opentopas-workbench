#!/usr/bin/env bash
# LETd layer on the follow-up MRIs of the viewer. Every path comes as an argument:
#   run.sh RUN CT RTSTRUCT CACHEF VIEWERDATA PRESCRIPTION FRACTIONS OUT KENNZAHLEN
# OUT receives the CT frame layers, the layers on the viewer grid and overlay.json; KENNZAHLEN numbers only.
set -euo pipefail
RUN=$1; CT=$2; RS=$3; CACHE=$4; VIEW=$5; RX=$6; FX=$7; OUT=$8; KZ=$9
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
"$PY" -m mrt_overlay.cli export "$RUN" "$OUT/ct_frame" --fractions "$FX"
SEG=()
if [ -n "${SYNTHSEG:-}" ] || command -v mri_synthseg >/dev/null; then SEG=(--synthseg "${SYNTHSEG:-mri_synthseg}"); fi
"$PY" -m mrt_overlay.cli viewer "$CACHE" "$CT" "$OUT/ct_frame" "$OUT/viewer_grid" --rtstruct "$RS" \
  --prescription "$RX" --viewer-data "$VIEW" --kennzahlen "$KZ" "${SEG[@]}"
echo "Fertig: $OUT/viewer_grid/overlay.json (Registrierung je Untersuchung, Hinschau-Liste)."
