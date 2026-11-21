#!/usr/bin/env bash
# LETd layer on the follow-up MRIs of the viewer. Every path comes as an argument:
#   run.sh RUN CT RTSTRUCT CACHEF VIEWERDATA PRESCRIPTION FRACTIONS OUT KENNZAHLEN
# OUT receives the CT frame layers, the layers on the viewer grid and overlay.json; KENNZAHLEN numbers only.
set -euo pipefail
# Windows paths (C:\...) from the start file are turned into WSL paths here: wsl.exe drops the quotes of a
# path with spaces when cmd converts it itself.
win() { case "$1" in [A-Za-z]:\\*|[A-Za-z]:/*) wslpath -a "$1" ;; *) printf '%s\n' "$1" ;; esac; }
RUN=$(win "$1"); CT=$(win "$2"); RS=$(win "$3"); CACHE=$(win "$4"); VIEW=$(win "$5"); RX=$6; FX=$7
OUT=$(win "$8"); KZ=$(win "$9")
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
# FreeSurfer 7.4.1 unpacked under ~/opt; mri_synthseg needs FREESURFER_HOME set up.
FS="$HOME/opt/freesurfer-7.4.1/freesurfer"
if ! command -v mri_synthseg >/dev/null && [ -d "$FS" ]; then
  export FREESURFER_HOME="$FS"; source "$FS/SetUpFreeSurfer.sh" >/dev/null 2>&1 || true
fi
if [ -n "${SYNTHSEG:-}" ] || command -v mri_synthseg >/dev/null; then SEG=(--synthseg "${SYNTHSEG:-mri_synthseg}"); fi
"$PY" -m mrt_overlay.cli viewer "$CACHE" "$CT" "$OUT/ct_frame" "$OUT/viewer_grid" --rtstruct "$RS" \
  --prescription "$RX" --viewer-data "$VIEW" --kennzahlen "$KZ" "${SEG[@]}"
echo "Fertig: $OUT/viewer_grid/overlay.json (Registrierung je Untersuchung, Hinschau-Liste)."
