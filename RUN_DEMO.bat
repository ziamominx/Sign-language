@echo off
cd /d "%~dp0SIGN_AI"
py -3.13 app.py
if errorlevel 1 pause
