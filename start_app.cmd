@echo off
setlocal
cd /d "%~dp0"

set "APP_PYTHON=%CD%\.venv\Scripts\python.exe"
set "MODEL_FILE=%CD%\data\qwen2.5-3b-instruct-q4_k_m.gguf"

if not exist "%APP_PYTHON%" (
  echo [ERROR] Project virtual environment was not found:
  echo         %APP_PYTHON%
  pause
  exit /b 1
)

if not exist "%MODEL_FILE%" (
  echo [ERROR] Sentence model was not found:
  echo         %MODEL_FILE%
  pause
  exit /b 1
)

echo [1/3] Starting PostgreSQL...
docker compose up -d db
if errorlevel 1 (
  echo [ERROR] PostgreSQL could not be started. Please open Docker Desktop.
  pause
  exit /b 1
)

echo [2/3] Checking the project Python environment...
"%APP_PYTHON%" -c "import llama_cpp; import google.genai; print('[OK] llama_cpp and google-genai are available')"
if errorlevel 1 (
  echo [ERROR] Required AI packages are unavailable in .venv.
  pause
  exit /b 1
)

echo [3/3] Starting AI Hakka learning app...
echo Open http://localhost:8000 after startup completes.
echo Keep this window open. Press Ctrl+C to stop the app.
echo.
"%APP_PYTHON%" -m uvicorn main:app --reload

endlocal
