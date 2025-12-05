@echo off
REM Installation script for Ultra-Low Bitrate Semantic Speech Coding
REM Tested on Windows 10/11

echo ========================================
echo Ultra-Low Bitrate Semantic Speech Coding
echo Installation Script
echo ========================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

python --version
echo.

REM Check for FFmpeg
echo Checking for FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo FFmpeg not found. Please install FFmpeg manually:
    echo 1. Download from: https://www.ffmpeg.org/download.html
    echo 2. Add to PATH
    pause
    exit /b 1
)
echo FFmpeg found
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
echo Virtual environment created
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Check for NVIDIA GPU
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected. Installing CPU-only PyTorch...
    pip install torch torchvision torchaudio
) else (
    echo NVIDIA GPU detected. Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)
echo.

REM Install requirements
echo Installing Python dependencies...
pip install -r requirements.txt
echo.

REM Create directories
echo Creating project directories...
if not exist "audio_files" mkdir audio_files
if not exist "output" mkdir output
if not exist "logs" mkdir logs
if not exist "temp" mkdir temp
echo Directories created
echo.

REM Test installation
echo Testing installation...
python -c "import whisper; import sounddevice; import websockets; import jiwer; print('All core modules imported successfully')"
echo.

echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To activate the environment:
echo   venv\Scripts\activate.bat
echo.
echo To run the demo:
echo   python demo_end_to_end.py --mode interactive
echo.
echo To start the receiver server:
echo   python main_receiver.py --mode server
echo.
echo To start the sender client:
echo   python main_sender.py --mode interactive --connect
echo.
pause