@echo off
REM Download and install the Whisper executable and model for the project on Windows
REM HEMMINGWAY BRIDGE:  Need to make sure the right libraries download, so that's the
REM ffmpeg binaries, that need to make it onto the PATH, and then check if you actually need
REM to download the whisper library, or if the python cpu module does that

REM Define directories
set "WHISPER_DIR=%cd%\submodules\whisper.cpp"
set "MODEL_DIR=%cd%\models"
set "WHISPER_ZIP=%cd%\whisper-cublas-12.2.0-bin-x64.zip"
set "VAD_ZIP=%cd%\silero_vad_v1.zip"

REM Create necessary directories
if not exist "%WHISPER_DIR%" mkdir "%WHISPER_DIR%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

echo Downloading Whisper executable...
curl -L "https://github.com/ggerganov/whisper.cpp/releases/download/v1.5.4/whisper-cublas-12.2.0-bin-x64.zip" --output "%WHISPER_ZIP%"

if exist "%WHISPER_ZIP%" (
    echo Unzipping Whisper executable...
    PowerShell -Command "Expand-Archive -Path '%WHISPER_ZIP%' -DestinationPath '%WHISPER_DIR%'"
    echo Cleaning up Whisper executable zip file...
    del "%WHISPER_ZIP%"
) else (
    echo Failed to download Whisper executable.
    exit /b 1
)

echo Downloading Whisper model...
curl -L "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin" --output "%MODEL_DIR%\ggml-medium-32-2.en.bin"

if exist "%MODEL_DIR%\ggml-medium-32-2.en.bin" (
    echo Whisper setup complete!
) else (
    echo Failed to download Whisper model.
    exit /b 1
)

REM Set up Silero VAD
echo Downloading Silero VAD model...
curl -L "https://github.com/snakers4/silero-vad/releases/download/v0.4/silero_vad_v1.zip" --output "%VAD_ZIP%"

if exist "%VAD_ZIP%" (
    echo Unzipping Silero VAD model...
    PowerShell -Command "Expand-Archive -Path '%VAD_ZIP%' -DestinationPath '%MODEL_DIR%'"
    echo Cleaning up Silero VAD model zip file...
    del "%VAD_ZIP%"
) else (
    echo Failed to download Silero VAD model.
    exit /b 1
)

echo Silero VAD setup complete!

echo Done!
pause
