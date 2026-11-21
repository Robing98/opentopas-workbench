@echo off
rem PET-Ebene im MRT-Viewer. Fragt jeden Pfad ab und ruft run_pet.sh in WSL.
rem Kopie nach 00_Start als "25 PET in den Viewer.bat". Kein Pfad ist hier fest eingetragen.
setlocal EnableExtensions
chcp 65001 >nul

set "REPO=C:\dev\opentopas-workbench"
set /p "REPO=Ordner des Repositorys [%REPO%]: "
set /p "PET=PET-Ordner (DICOM-Serie, Modality PT): "
set /p "CACHE=cacheF-Ordner des Viewers (mit ref_t1.nii.gz, mask.npy und log.json): "
set /p "VIEW=Datenordner des Viewers (dort landen pet.json und pet_Datum.js): "
set /p "OUT=Ausgabeordner fuer Einzelheiten: "
set "REF=brain"
set /p "REF=Referenzregion: brain, striatum oder contralateral [brain]: "
set "SIDE="
if /i "%REF%"=="contralateral" set /p "SIDE=Welche Seite NICHT als Referenz (links oder rechts): "
if /i "%REF%"=="striatum" set /p "LABELS=SynthSeg-Datei der Referenz-T1 (synthseg.nii.gz): "
if /i "%REF%"=="striatum" set "WSLENV=LABELS/p"

wsl -e bash -c "exec bash \"$(wslpath -a \"$0\")/tools/mrt_overlay/run_pet.sh\" \"$@\"" "%REPO%" "%PET%" "%CACHE%" "%VIEW%" "%OUT%" "%REF%" "%SIDE%"
if errorlevel 1 goto :fail
echo.
echo Fertig. Verlauf.html danach mit build_page.py neu erzeugen; der Schalter heisst PET-Ebene.
pause
exit /b 0

:fail
echo.
echo Abbruch. Die Meldung oben nennt den Schritt (bei der Seitenpruefung: links und rechts sind nicht sicher).
pause
exit /b 1
