@echo off
chcp 65001 > nul
cd /d %~dp0
echo Starting OmniVoice Studio...
.venv\Scripts\python.exe app.py
pause
