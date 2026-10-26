# OpenTOPAS Workbench

A local web UI for [OpenTOPAS](https://github.com/OpenTOPAS/OpenTOPAS) and TOPAS 3.9: edit parameter files, run them in a queue, and plot scorer output. Pure Python standard library, no dependencies.

![Results tab: depth-dose curves with Bragg peak metrics](docs/results_dark_en.png)

> Community project, not affiliated with the TOPAS collaboration or TOPAS MC Inc.

## Features

- **Files**: browse, edit, rename, copy, and trash files in a workspace folder. Parameter files get a light syntax check (missing type prefix, unquoted strings, missing units, vector counts).
- **Runs**: queue parameter files for one or more TOPAS installations. Live log, stop button, output detection, optional output subfolder, skip if results exist.
- **Results**: plot TOPAS CSV scorer output. 1D curves overlay with normalisation, log axis, and Bragg peak metrics (R80, BPW80, D80-D20). 2D and 3D binnings show as a heat map. Images in the folder show as a gallery.
- **Generators**: plugins that write parameter sets from a form.
  - `pencil_beam`: any particle from protons to oxygen, an energy list, phantom material, depth dose, and an optional 2D map. Sets `EMRangeMax` for light ions above the default table range.
  - `ripple_filter`: depth-dose studies with 1D, crossed, and 2D-pin ripple filters (pins written as binary STL).
- English and German UI, light and dark theme.

| Files | Runs | Generators |
|---|---|---|
| ![](docs/files_light_en.png) | ![](docs/run_dark_en.png) | ![](docs/gen_dark_en.png) |

## Requirements

- Python 3.8 or later
- OpenTOPAS or TOPAS 3.9 on Linux, macOS, or Windows via WSL

## Install and start

```
pip install git+https://github.com/Robing98/opentopas-workbench
opentopas-workbench --workspace PATH_TO_YOUR_CASES
```

Without installing, from a clone of this repository:

```
./workbench.sh PATH_TO_YOUR_CASES
```

On Windows with TOPAS in WSL, run `workbench.bat PATH_TO_YOUR_CASES` from a Command Prompt, or double-click it to use the current folder.

The browser opens `http://localhost:8765`. Options: `--port`, `--no-browser`.

On first start the workbench looks for TOPAS binaries in `~/.local/bin/topas4`, `~/Applications/TOPAS/OpenTOPAS-install/bin/topas`, `~/topas/topas/bin/topas`, and `topas` on `PATH`. To add or change one, open **Settings**. Environment variables such as `TOPAS_G4_DATA_DIR` go there too.

Settings, job history, and logs live in `WORKSPACE/.workbench/`.

## Writing a generator plugin

Put a Python file into `WORKSPACE/.workbench/generators/` (or into `opentopas_workbench/generators/` to ship it). It needs four names:

```python
NAME = {"en": "My study", "de": "Meine Studie"}
DESCRIPTION = {"en": "What it generates."}
FIELDS = [
    {"key": "energies", "type": "text", "label": {"en": "Energies [MeV]"}, "default": "100, 150"},
    {"key": "histories", "type": "number", "label": {"en": "Histories"}, "default": 100000},
    {"key": "material", "type": "select", "label": {"en": "Material"}, "default": "G4_WATER",
     "options": [["G4_WATER", "Water"], ["G4_PLEXIGLASS", "PMMA"]]},
]

def generate(values):
    """Return a list of (relative path, content). Content is str for text or bytes for binary files."""
    return [(f"case_{e.strip()}.txt", f"d:So/Demo/BeamEnergy = {e.strip()} MeV\n")
            for e in values["energies"].split(",")]
```

Field types: `text`, `number`, `select`, `multi` (checkboxes, value is a list), and `bool`. Labels and option names are plain strings or `{"en": ..., "de": ...}`.

## Security

The server listens on `127.0.0.1` only, rejects requests with a foreign `Host` header, requires a per-session token for every API call, and only reads or writes inside the workspace. It runs the TOPAS binaries you configure, so do not expose it to a network.

## Known limits

- Plotting reads TOPAS CSV output only. Binary and ROOT output are not plotted yet.
- The syntax check is heuristic and does not follow `includeFile` chains.
- One job runs at a time. TOPAS itself uses all cores with `i:Ts/NumberOfThreads = 0`.

## Contributing

Issues and pull requests are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
