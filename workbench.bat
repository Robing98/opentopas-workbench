@echo off
rem OpenTOPAS Workbench: starts the server in WSL and opens it in the browser.
rem Usage: workbench.bat [WORKSPACE_FOLDER]   (default: the current folder)
rem Keep this window open; closing it stops the workbench.
title OpenTOPAS Workbench
set "WS=%~1"
if "%WS%"=="" set "WS=%CD%"
for /f "usebackq delims=" %%P in (`wsl.exe -e wslpath -a "%~dp0."`) do set "R=%%P"
for /f "usebackq delims=" %%P in (`wsl.exe -e wslpath -a "%WS%"`) do set "W=%%P"
wsl.exe --cd "%R%" -e python3 -m opentopas_workbench --workspace "%W%"
pause
