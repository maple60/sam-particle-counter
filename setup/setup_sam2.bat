@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  SAM2 setup for Windows
REM ============================================================

REM ---- Configuration -----------------------------------------
REM Assume this .bat is placed in "scripts\" under the project root.
cd /d "%~dp0.." || (
    echo [Error] Failed to move to project root.
    exit /b 1
)

set "PROJECT_ROOT=%CD%"
set "VENV_DIR=.venv"

set "SAM2_DIR=sam2_repo"
set "SAM2_REPO=https://github.com/facebookresearch/sam2.git"
set "SAM2_REF=main"

REM Choose one or more models: tiny small base_plus large
REM If you want only one, e.g.:
REM set "SAM2_MODELS=base_plus"
set "SAM2_MODELS=tiny small base_plus large"

REM 1 = run uv sync before setup, 0 = skip
set "SYNC_PROJECT=1"

REM 1 = skip SAM2 CUDA extension build (safer on some Windows setups)
REM 0 = try normal install
set "SKIP_SAM2_CUDA=0"

echo.
echo ============================================================
echo Project root: %PROJECT_ROOT%
echo SAM2 repo:    %SAM2_REPO%
echo SAM2 ref:     %SAM2_REF%
echo Models:       %SAM2_MODELS%
echo ============================================================
echo.

REM ---- Prerequisite checks -----------------------------------
call :require_command uv "uv is not installed or not on PATH." || exit /b 1
call :require_command git "git is not installed or not on PATH." || exit /b 1
call :detect_downloader || exit /b 1

REM ---- Ensure .venv exists -----------------------------------
if not exist "%VENV_DIR%" (
    echo [Info] Creating virtual environment with uv...
    uv venv || (
        echo [Error] Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo [OK] Virtual environment already exists: %VENV_DIR%
)

REM ---- Sync project dependencies -----------------------------
if "%SYNC_PROJECT%"=="1" (
    echo [Info] Syncing project environment...
    uv sync || (
        echo [Error] uv sync failed.
        exit /b 1
    )
) else (
    echo [Skip] uv sync skipped.
)

REM ---- Clone or update SAM2 repo -----------------------------
if not exist "%SAM2_DIR%\.git" (
    echo [Info] Cloning SAM2 repository...
    git clone "%SAM2_REPO%" "%SAM2_DIR%" || (
        echo [Error] Failed to clone SAM2 repository.
        exit /b 1
    )
) else (
    echo [OK] SAM2 repository already exists.
)

echo [Info] Fetching latest refs from SAM2 repository...
git -C "%SAM2_DIR%" fetch --all --tags --prune || (
    echo [Error] Failed to fetch SAM2 repository updates.
    exit /b 1
)

echo [Info] Checking out SAM2 ref: %SAM2_REF%
git -C "%SAM2_DIR%" checkout "%SAM2_REF%" || (
    echo [Error] Failed to checkout SAM2 ref "%SAM2_REF%".
    exit /b 1
)

if /I "%SAM2_REF%"=="main" (
    echo [Info] Pulling latest changes for main...
    git -C "%SAM2_DIR%" pull --ff-only origin main || (
        echo [Error] Failed to pull latest main branch.
        exit /b 1
    )
)

REM ---- Ensure checkpoints directory exists -------------------
if not exist "%SAM2_DIR%\checkpoints" (
    mkdir "%SAM2_DIR%\checkpoints" || (
        echo [Error] Failed to create checkpoints directory.
        exit /b 1
    )
)

REM ---- Download requested checkpoints ------------------------
echo [Info] Ensuring SAM2 checkpoints...
for %%M in (%SAM2_MODELS%) do (
    call :download_model "%%~M" || exit /b 1
)

REM ---- Reinstall SAM2 editable -------------------------------
echo [Info] Reinstalling SAM2 in editable mode...
uv pip uninstall -y SAM-2 >nul 2>&1

pushd "%SAM2_DIR%" || (
    echo [Error] Failed to enter SAM2 directory.
    exit /b 1
)

if "%SKIP_SAM2_CUDA%"=="1" (
    echo [Info] Installing SAM2 with SAM2_BUILD_CUDA=0 ...
    set "SAM2_BUILD_CUDA=0"
    uv pip install -e . || (
        popd
        echo [Error] Failed to install SAM2.
        exit /b 1
    )
    set "SAM2_BUILD_CUDA="
) else (
    uv pip install -e . || (
        popd
        echo [Error] Failed to install SAM2.
        exit /b 1
    )
)

popd

echo [OK] SAM2 setup completed successfully.
exit /b 0


REM ============================================================
REM Functions
REM ============================================================

:require_command
where %~1 >nul 2>&1
if errorlevel 1 (
    echo [Error] %~2
    exit /b 1
)
exit /b 0

:detect_downloader
where curl >nul 2>&1
if not errorlevel 1 (
    set "DOWNLOADER=curl"
    echo [OK] Downloader: curl
    exit /b 0
)

where pwsh >nul 2>&1
if not errorlevel 1 (
    set "DOWNLOADER=pwsh"
    echo [OK] Downloader: pwsh
    exit /b 0
)

where powershell >nul 2>&1
if not errorlevel 1 (
    set "DOWNLOADER=powershell"
    echo [OK] Downloader: powershell
    exit /b 0
)

echo [Error] Neither curl, pwsh, nor powershell was found.
exit /b 1

:download_model
set "MODEL=%~1"
set "FILE="
set "URL_BASE=https://dl.fbaipublicfiles.com/segment_anything_2/092824"

if /I "%MODEL%"=="tiny"      set "FILE=sam2.1_hiera_tiny.pt"
if /I "%MODEL%"=="small"     set "FILE=sam2.1_hiera_small.pt"
if /I "%MODEL%"=="base_plus" set "FILE=sam2.1_hiera_base_plus.pt"
if /I "%MODEL%"=="large"     set "FILE=sam2.1_hiera_large.pt"

if "%FILE%"=="" (
    echo [Error] Unknown model name: %MODEL%
    exit /b 1
)

set "TARGET=%SAM2_DIR%\checkpoints\%FILE%"
set "URL=%URL_BASE%/%FILE%"

if exist "%TARGET%" (
    echo [Skip] %FILE% already exists.
    exit /b 0
)

echo [Download] %FILE%

if /I "%DOWNLOADER%"=="curl" (
    curl -L --fail -o "%TARGET%" "%URL%" || (
        echo [Error] Failed to download %FILE% with curl.
        if exist "%TARGET%" del /f /q "%TARGET%" >nul 2>&1
        exit /b 1
    )
    exit /b 0
)

if /I "%DOWNLOADER%"=="pwsh" (
    pwsh -NoLogo -NoProfile -Command ^
        "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%URL%' -OutFile '%TARGET%'" || (
        echo [Error] Failed to download %FILE% with pwsh.
        if exist "%TARGET%" del /f /q "%TARGET%" >nul 2>&1
        exit /b 1
    )
    exit /b 0
)

powershell -NoLogo -NoProfile -Command ^
    "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%URL%' -OutFile '%TARGET%'" || (
    echo [Error] Failed to download %FILE% with powershell.
    if exist "%TARGET%" del /f /q "%TARGET%" >nul 2>&1
    exit /b 1
)
exit /b 0