#!/usr/bin/env bash
# OpenTOPAS Workbench launcher without installation. Usage: ./workbench.sh [WORKSPACE_FOLDER] [options]
ws="$(cd "${1:-$PWD}" && pwd)"
cd "$(dirname "$0")" && exec python3 -m opentopas_workbench --workspace "$ws" "${@:2}"
