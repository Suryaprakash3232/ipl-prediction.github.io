@echo off
rem Change to the project directory (handles spaces)
cd /d "C:\Users\Hi\Desktop\Ipl prediction"
rem Activate the virtual environment and run Flask app
".venv\Scripts\python.exe" app.py
pause

