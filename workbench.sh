#!/usr/bin/env bash
# OpenTOPAS Workbench launcher without installation. Usage: ./workbench.sh [WORKSPACE_FOLDER] [options]
# Default workspace: the folder workspace next to this script if it exists, else the current folder.
here="$(cd "$(dirname "$0")" && pwd)"
def="$PWD"; [ -d "$here/workspace" ] && def="$here/workspace"
ws="$(cd "${1:-$def}" && pwd)"
cd "$(dirname "$0")" && exec python3 -m opentopas_workbench --workspace "$ws" "${@:2}"
