# Contributing

- Keep the server free of third-party dependencies. Python 3.8 standard library only.
- Run the tests before opening a pull request: `pip install pytest . && pytest -q`.
- New generators go into `opentopas_workbench/generators/`. Add a test that the generated files pass the syntax check.
- UI strings live in the `I18N` table in `ui.html`. Add every new key in English and German.
- Report TOPAS or Geant4 physics issues to the OpenTOPAS project, not here.
