@echo off
title Vision Desktop Automation
cd /d "%~dp0"

echo Starting Vision Desktop Automation GUI...
uv run python -m vision_desktop_automation.launcher
