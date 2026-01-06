@echo off
setlocal enabledelayedexpansion

REM === Jump to the folder where this .bat resides ===
pushd "%~dp0"

REM --- Use the venv's python to avoid PATH / execution-policy issues ---
set "PYEXE=%~dp0video_env\Scripts\python.exe"
if not exist "%PYEXE%" (
    echo [ERROR] Cannot find venv Python at "%~dp0video_env\Scripts\python.exe"
    echo Create it first:
    echo     py -3.11 -m venv .\video_env
    echo     .\video_env\Scripts\python -m pip install -r requirements.txt
    exit /b 1
)

echo [INFO] Using Python: %PYEXE%
echo.

REM === Loop over your Ciena video IDs (no underscore in filenames) ===
for %%i in (4 14 17 18 21 22 23 24 25 29 32 36) do (
    set "VID=Ciena%%i"
    set "VIDPATH=Ciena_Video\!VID!.mp4"

    echo ==========================================================
    echo [STEP 1] GPT-5 answer generation for !VID!  (^.!VIDPATH!^)
    echo ----------------------------------------------------------
    "%PYEXE" ".\qa_generate_gpt5.py" "!VIDPATH!"
    if errorlevel 1 (
        echo [WARN] GPT-5 generation failed for !VID!. Skipping evaluation.
        echo.
        goto :continue_%%i
    )

    echo.
    echo [STEP 2] Evaluate results for !VID!
    echo ----------------------------------------------------------
    "%PYEXE%" ".\eval_main.py" --video_name "!VID!"
    if errorlevel 1 (
        echo [WARN] Evaluation failed for !VID!.
    )

    :continue_%%i
    echo.
)

REM --- Restore original directory and exit ---
popd
echo Done.
