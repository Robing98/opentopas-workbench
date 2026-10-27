# Changelog

## 0.4.0

- Export: plot as PNG (1x to 3x) and SVG in a light print style, plotted data as CSV, plot to the clipboard; selected results and filtered images as ZIP; download button in the image viewer.
- Saved views: plot selection, reference, zoom, axes, and normalisation stored by name in the workspace.
- Notifications: browser notification and sound for failed runs and the end of the queue; Windows notification from WSL when no page is open.
- Parameter variations: copies of a parameter file with one parameter set to a list of values, with matching scorer output names, optionally queued right away.
- Keep awake: system sleep is blocked while runs are queued (Windows via WSL, macOS, Linux with systemd). Can be switched off in Settings.
- Ion comparison: depth axis absolute, depth minus R80, or depth over R80; metrics BPW80/R80, (D80-D20)/R80, and fragmentation tail (dose at R80 + max(10 mm, 4 x (D80-D20))); generator `pencil_beam` can take water ranges instead of energies so every particle gets the same range.
- Overview tab: all results in the workspace in one sortable, filterable table (particle, energy, folder, tags), with R80, BPW80, deviations to a reference file, Bragg-Kleeman range check, custom formula columns, CSV export, and an image gallery with full-screen view.
- Free port is chosen automatically if the default port is taken; under WSL the Windows browser opens.
- Runs: existing results are detected in the case folder and its subfolders, so cases computed outside the workbench show as done. Filter for cases without results.
- Runs: progress bar and remaining time from the TOPAS history count, error lines highlighted in the log, suspicious lines shown for failed runs.
- Results: drag to zoom, zoom to peak, hover read-out, difference or ratio to a reference curve, ΔR80 and filter BPW80 (quadrature) against the reference, selection kept across folders.
- Results: measured curves from PTW .mcc, IBA .asc, and column files (also with decimal commas) overlay as points.
- Editor: syntax highlighting, live check, autocompletion of parameter names from the workspace and common TOPAS parameters, includeFile links and missing-include check.
- Runs: live panel with phase, history count, and CPU load; queue remaining time from earlier runs (rate per history and MeV); select several files with Shift-click or by dragging.
- Speed: file and folder scans are cached; search folders can be limited in Settings (include top-level folders, exclude patterns), and the Runs and Overview tabs each search only the folder you pick.
- Runs: a warning appears when the page has lost the connection to the workbench.
- Tools: `compare_idd_matrad.py` compares TOPAS depth doses with a matRad machine file, `calibrated_cases.py` writes cases with energy and spread fitted to it, `matload.py` reads MATLAB v7.3 files.

## 0.3.0

- First public release as OpenTOPAS Workbench.
- Files, Runs, Results, Generators, and Settings tabs.
- Generators `pencil_beam` and `ripple_filter`.
- English and German UI, light and dark theme.
