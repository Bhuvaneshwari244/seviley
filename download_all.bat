@echo off
echo ============================================================
echo  SEP-28K AUDIO DOWNLOAD - AUTOMATED
echo ============================================================
echo.

REM Check if wget exists
where wget >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] wget is not installed!
    echo.
    echo Please install wget first:
    echo   Option 1: winget install GnuWin32.Wget
    echo   Option 2: Download from https://eternallybored.org/misc/wget/
    echo.
    pause
    exit /b 1
)

REM Check if ffmpeg exists
where ffmpeg >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ffmpeg is not installed!
    echo.
    echo Please install ffmpeg first:
    echo   Option 1: winget install Gyan.FFmpeg
    echo   Option 2: Download from https://www.gyan.dev/ffmpeg/builds/
    echo.
    pause
    exit /b 1
)

echo [OK] wget and ffmpeg are installed
echo.

REM Create directories
if not exist "raw_audio" mkdir raw_audio
if not exist "processed_clips" mkdir processed_clips

echo ============================================================
echo  DOWNLOAD INFORMATION
echo ============================================================
echo  - SEP-28k: ~28,000 clips (~32 GB raw, ~2.6 GB processed)
echo  - FluencyBank: ~4,000 clips (~6 GB raw, ~400 MB processed)
echo  - Estimated time: 4-8 hours
echo  - Total disk space: ~40 GB
echo ============================================================
echo.

set /p CONFIRM="Do you want to continue? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Download cancelled.
    pause
    exit /b 0
)

echo.
echo ============================================================
echo  STEP 1/4: Downloading SEP-28k audio...
echo ============================================================
py download_audio.py --episodes SEP-28k_episodes.csv --wavs raw_audio
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to download SEP-28k audio
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  STEP 2/4: Extracting SEP-28k clips...
echo ============================================================
py extract_clips.py --labels SEP-28k_labels.csv --wavs raw_audio --clips processed_clips --progress
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to extract SEP-28k clips
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  STEP 3/4: Downloading FluencyBank audio...
echo ============================================================
py download_audio.py --episodes fluencybank_episodes.csv --wavs raw_audio
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to download FluencyBank audio
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  STEP 4/4: Extracting FluencyBank clips...
echo ============================================================
py extract_clips.py --labels fluencybank_labels.csv --wavs raw_audio --clips processed_clips --progress
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to extract FluencyBank clips
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  DOWNLOAD COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo   1. py clean_data.py              # Verify audio quality
echo   2. py split_data.py              # Create train/val/test splits
echo   3. py train_transfer_hubert.py   # Train model
echo.
pause
