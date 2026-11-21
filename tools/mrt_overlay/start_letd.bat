@echo off
rem LETd-Ebene auf den Verlaufs-MRTs. Fragt jeden Pfad ab und ruft run.sh in WSL.
rem Kopie nach 00_Start als "24 LETd auf Verlaufs-MRTs.bat". Kein Pfad ist hier fest eingetragen:
rem welche Ordner Eingabe und Ausgabe sind, entscheidest du beim Start.
setlocal EnableExtensions
chcp 65001 >nul

set "REPO=C:\dev\opentopas-workbench"
set /p "REPO=Ordner des Repositorys [%REPO%]: "
set /p "RUN=plan_mc-Laufordner (mit letd.npy und rbe_dose_1_1.dcm): "
set /p "CT=Planungs-CT (DICOM-Ordner): "
set /p "RS=RT Structure Set (Datei): "
set /p "CACHE=cacheF-Ordner des Viewers (mit ref_t1.nii.gz und log.json): "
set /p "VIEW=Datenordner des Viewers (dort landen letd_all.js, marks_all.js, letd.json): "
set /p "RX=Verschreibung in Gy(RBE), etwa 54: "
set /p "FX=Anzahl Fraktionen, etwa 30: "
set /p "OUT=Ausgabeordner: "
set /p "KZ=Datei fuer Kennzahlen (kennzahlen.json): "

rem wsl -e gibt die Argumente unveraendert weiter; run.sh setzt die Windows-Pfade selbst um.
wsl -e bash -c "exec bash \"$(wslpath -a \"$0\")/tools/mrt_overlay/run.sh\" \"$@\"" "%REPO%" "%RUN%" "%CT%" "%RS%" "%CACHE%" "%VIEW%" "%RX%" "%FX%" "%OUT%" "%KZ%"
if errorlevel 1 goto :fail
echo.
echo Fertig. Verlauf.html danach mit build_page.py neu erzeugen.
pause
exit /b 0

:fail
echo.
echo Abbruch. Die Meldung oben nennt den Schritt.
pause
exit /b 1
