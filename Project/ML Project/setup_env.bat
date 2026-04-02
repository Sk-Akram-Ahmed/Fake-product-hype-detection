@echo off
REM ============================================================
REM setup_env.bat
REM Windows environment bootstrap script
REM Run once: setup_env.bat
REM ============================================================

echo.
echo ============================================================
echo   Multimodal Fake Product Hype Detection — Environment Setup
echo ============================================================
echo.

REM ── 1. Create virtual environment ───────────────────────────
echo [1/6] Creating Python virtual environment ...
python -m venv hype_env
if %errorlevel% neq 0 (
    echo ERROR: python not found. Install Python 3.9+ and retry.
    pause & exit /b 1
)

REM ── 2. Activate environment ──────────────────────────────────
echo [2/6] Activating environment ...
call hype_env\Scripts\activate.bat

REM ── 3. Upgrade pip ───────────────────────────────────────────
echo [3/6] Upgrading pip ...
python -m pip install --upgrade pip setuptools wheel

REM ── 4. Install dependencies ──────────────────────────────────
echo [4/6] Installing requirements (this may take several minutes) ...
pip install -r requirements.txt

REM ── 5. Download NLTK resources ───────────────────────────────
echo [5/6] Downloading NLTK corpora ...
python -c "import nltk; [nltk.download(r, quiet=True) for r in ['punkt','stopwords','wordnet','averaged_perceptron_tagger','omw-1.4']]"

REM ── 6. Download spaCy model ──────────────────────────────────
echo [6/6] Downloading spaCy en_core_web_sm ...
python -m spacy download en_core_web_sm

echo.
echo ============================================================
echo   Setup complete!
echo   Activate with:  hype_env\Scripts\activate
echo ============================================================
pause
