@echo off
title Dispatcher Insights
echo ==========================================
echo  Dispatcher Insights - interactive dashboard
echo ==========================================

set PYEXE=python
if exist ".venv\Scripts\python.exe" set PYEXE=.venv\Scripts\python.exe

start cmd /k "%PYEXE% C:\Jagadish\ASBG_AI_ASSET3_WS\Dispatcher_Analysis_Agent\dispatcher_agent\app.py"
timeout /t 4 /nobreak > nul
start http://127.0.0.1:5710/
echo Dashboard launching at http://127.0.0.1:5710/
