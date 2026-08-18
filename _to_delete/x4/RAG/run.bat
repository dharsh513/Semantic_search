@echo off
REM ===================================================================
REM  PubMed Semantic Search (RAG) - Windows launcher
REM  Creates a virtual environment on first run, installs dependencies,
REM  then starts the Flask server.
REM ===================================================================

cd /d "%~dp0"

if not exist ".venv\" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create the virtual environment.
        echo Make sure Python 3.9+ is installed and on your PATH.
        pause
        exit /b 1
    )
    echo [2/3] Installing dependencies ^(this takes a few minutes the first time^)...
    call .venv\Scripts\python.exe -m pip install --upgrade pip
    call .venv\Scripts\pip.exe install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment found.
    echo [2/3] Skipping install ^(delete the .venv folder to reinstall^).
)

echo [3/3] Starting server...
echo.
echo     Open  http://127.0.0.1:5000  in your browser.
echo     Press CTRL+C to stop.
echo.
call .venv\Scripts\python.exe app.py

pause
