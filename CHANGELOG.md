# Changelog

## Unreleased

- Beam profile monitors: the `pencil_beam` and `ripple_filter` generators can add thin scoring planes in a parallel world at a list of positions or before and after every component, each with a 2D fluence map (bin width and extent as options), an optional split into protons, electrons, and other particles, and an energy spectrum.
- Beam profile tab: x and y profiles (linear and logarithmic) and the 2D map per monitor plane, the envelope along the beam line for several cases, spectra, lateral dose profiles from 3D dose grids at a chosen depth, and sigma against depth. Fits: Gaussian core sigma, FWHM, centroid, rms, share outside 3 sigma, and a double Gaussian for the halo. Export as PNG, SVG, and CSV. New API: `/api/beam/scan`, `/api/beam/monitor`, `/api/beam/envelope`, `/api/beam/spectrum`, `/api/beam/dose`. Keys 1 to 7 switch tabs.
- Fix: 2D and 3D maps in the Results tab drew no heat map, since the color scale function was missing.
- MCP server: `opentopas-workbench mcp` is a stdio MCP server over the HTTP API of a running workbench, for Claude Code, Claude Desktop, and other MCP clients. Tools read cases, results, curves, the queue, and logs, and write files, generate cases, queue, remove, and stop runs; `--read-only` offers only the reading tools. Files in `.workbench` (settings, generator plugins, the token) cannot be written or read through it. It speaks protocol versions 2024-11-05 to 2025-11-25 with `initialize` and 2026-07-28 with `server/discover`. See [MCP server](README.md#mcp-server).
- Discovery: a running workbench writes `.workbench/server.json` (address, token, process id; readable only by the owner) and removes it on exit. The MCP server and the new commands find the workbench through it and start one in the background when none runs (`--no-start` turns that off).
- Commands: `opentopas-workbench enqueue FILES...` queues parameter files and `opentopas-workbench jobs` prints the queue, both without the browser.
- Runs tab: TOPAS processes started outside the queue that work in the workspace show as "external" (Linux and WSL), without a stop button, and count in the tab badge. `/api/jobs` has a new field `external`.
- Fix: `--port 0` printed and stored port 0 instead of the port the system chose.
- Fix: on macOS the workbench could take half a minute to start, waiting for a reverse DNS lookup of 127.0.0.1.
- A workbench now also ends cleanly, stopping the running job and removing `server.json`, when its terminal is closed (SIGHUP).
- Fix: keep awake ended two minutes into every run. The heartbeat for the Windows helper was not written while a job ran, so the helper released its power requests and a laptop could go to sleep during a long run. The helper now stays for the whole queue and ends when the queue is empty; `.workbench/keepawake.log` then gets a line "stopped by the workbench".
- Restart: the queue continues after a restart of the workbench. Waiting jobs keep waiting; before, they were marked "aborted" together with the running job. The job that ran when the workbench ended is queued again once, with the note "re-queued after server restart" in the queue, in `opentopas-workbench jobs`, and in the MCP tools (new job fields `restarts` and `note`). If the workbench ends again during that second attempt, the job is marked "aborted".
- Ctrl+C: while a job runs, the first Ctrl+C only warns, and a second one within 3 seconds ends the workbench. A job stopped this way runs again at the next start. A closing terminal and `kill` end the workbench at once, as before.
- Fixes: `rifi_kernel_fit.py` also runs with numpy older than 2.0. It called `np.trapezoid`, which exists only from numpy 2.0 on.
- Tools: `topas2matrad.py` can let the spot broadening of a ripple filter grow with depth. Until now it entered only the spot size in air, which is constant in depth. `--filter-distance CM` treats the filter as an angular spread at that distance before the isocentre and adds the variance `inc2 * ((1 + z / L)^2 - 1)` to `sigma`, `sigma1`, and `sigma2` of the template. `--filter-wet MIN MAX` with `--open-idd PATTERN` weights it with the depth dose of each filter thickness (triangular ridges): the spot widens towards the peak and narrows at its distal edge, where a kernel never gets narrower than zero. The depth doses of `--open-idd` must reach as deep as the dose with the filter plus the thickest part of the filter, otherwise the tool stops and names the file. With a range shifter in the beam, `--sigma-base COLUMN` names the spot size with the range shifter and without the filter, so that only the filter's own share grows from the filter's distance, and `--base-distance CM` lets the range shifter's share grow from its own distance. Without `--filter-distance` the machine file is unchanged.

## 0.5.0

- Speed: the Runs tab no longer resolves every case path with realpath and lists result sub folders once per folder instead of once per case. For 300 cases this drops from about 4,600 file system calls to about 360 after a change (seconds on WSL /mnt/c before), and to none while cached.
- Speed: parsed parameter files and result metrics are kept in `.workbench/index.json`, so after a restart the Runs and Overview tabs only check modification times. Large text files that are not parameter files are no longer read completely; 2D and 3D scorer files are indexed from their header alone.
- Speed: caches are filled in the background right after start. An expired cache answers at once and refreshes in the background; the page asks again when an answer was stale. A finished run refreshes the lists the same way instead of forcing a full walk.
- Speed: the queue sends every waiting job but only the latest history and renders running, waiting, and finished jobs as separate sections that are rewritten only when they change. The Runs filter and the Overview search wait for a pause in typing, the Overview renders rows on demand, the Results filter no longer reloads the plotted curves, plotted curves and thumbnails are reused instead of downloaded again.
- Navigation: tabs follow the workflow (Files, Generators, Runs, Overview, Results, Settings), the Runs tab shows the number of active jobs, and the run badge opens it.
- Navigation: the address keeps the tab, the open file, and the plotted curves (`#results?f=...`), so Back and Forward switch tabs, a reload keeps the plot, and plots can be bookmarked. Keyboard shortcuts: 1 to 6 or Alt+1 to Alt+6 switch tabs, / focuses the filter, ? lists the shortcuts.
- Runs: plot the results of the selected cases or of one case, open a case in the editor, run a finished or failed job again, remove a single waiting job, show only failures; history and long queues are paged. Filters and options are remembered.
- Generators: after creating files, one click selects the new cases in Runs. Overview: open the parameter file that wrote a result.
- Connection: a banner on every tab when the workbench cannot be reached, with a retry button; after the workbench was restarted the page says so and offers a reload. Errors that only went to the browser console now show as a short message.
- Safety: Move to trash, Stop current run, Remove waiting, Clear history, and removing a saved view ask for a second click (no browser dialogs). The editor warns before unsaved changes are discarded.
- Loading indicators and empty states in Runs, Overview, Results, and the queue; a compact header in narrow windows.
- Fixes: "Remove waiting" also removes waiting jobs beyond the 200 most recent jobs; the live run box no longer shows as an empty bar when nothing runs; the Last run column follows the selected program; English UI text uses US spelling.
- Keep awake on Windows: the helper now also holds power requests (PowerSetRequest, system and execution required), visible in `powercfg /requests`. New option in Settings to keep the display on as well, for laptops with Modern Standby, where Windows can enter standby once the display is off or the computer is locked.
- Tools: `rifi_kernel_fit.py` fits the ripple filter kernel (mean water equivalent thickness and its spread) from depth dose curves with and without the filter, as Gaussian, uniform, pyramid, and free-form kernel, and reports the best shape. The true kernel comes from the TOPAS parameter file (TsRidgeFilter, crossed filters as convolution, TsCAD STL by ray casting through the mesh) or from `--pin`, `--stl`, `--heightmap`; `--show-kernel` only analyses a filter geometry. Pairs are found automatically (every `*rifi0*` curve with all variants, also from the parent folder), PTW .mcc and two-column files are read. `rifi_shapes_figure.py` draws filter shape against thickness distribution and the convolution of a depth dose. `topas2matrad.py`: `--min-spacing` together with `--spacing-bpw` sets the lower limit of the peak distance.
- Ripple filter generator: 2D straight pyramids (`pyr3`, `pyr4`, `pyr6`, STL) and 1D ridges with the same thickness distribution as the 2D pins (`pin4e`, `pyr4e`), to check that only the distribution matters for the depth dose.
- API, all additions optional (existing endpoints unchanged): `/api/jobs?hist=N` returns all active jobs plus the last N finished ones, with `counts`, `finished`, and `rev`; `/api/dequeue` accepts `{"all": true}`; `/api/outputs` lists the plottable outputs of parameter files; `/api/params` and `/api/index` carry a `stale` flag; `/file?...&v=MTIME` may be cached by the browser.
- Python 3.9 or later: the package metadata needs setuptools 77, which does not support Python 3.8. CI also runs the ripple filter tool tests, with NumPy and SciPy.

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
