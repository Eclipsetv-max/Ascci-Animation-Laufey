@echo off
rem ============================================================
rem  LOVER GIRL (ASCII) - lanzador
rem  creado por Eclipsetv (Fernando Abraham Peña Muñoz)
rem  doble clic para ejecutar. Tambien acepta opciones:
rem     Start.bat --cols 320
rem ============================================================
title LOVER GIRL - ASCII (Eclipsetv)
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 goto pypath
python lover_girl.py %*
goto end
:pypath
py lover_girl.py %*
:end
if errorlevel 1 pause
