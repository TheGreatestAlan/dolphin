@echo off
REM build_image.bat - A script to build the Docker image for my-python-app

REM Define variables
SET IMAGE_NAME=translator
SET TAG=latest
SET DOCKERFILE_PATH=..
SET CONTEXT_PATH=..

REM Optional: Allow custom tag via command-line argument
IF NOT "%1"=="" (
    SET TAG=%1
)

REM Display build information
echo Building Docker image: %IMAGE_NAME%:%TAG%
echo Using Dockerfile at: %DOCKERFILE_PATH%\Dockerfile
echo Build context: %CONTEXT_PATH%

REM Run the Docker build command
docker build -t %IMAGE_NAME%:%TAG% -f %DOCKERFILE_PATH%\Dockerfile %CONTEXT_PATH%

REM Check if the build was successful
IF %ERRORLEVEL% EQU 0 (
    echo Docker image %IMAGE_NAME%:%TAG% built successfully.
) ELSE (
    echo Failed to build Docker image %IMAGE_NAME%:%TAG%.
    EXIT /B %ERRORLEVEL%
)

REM Optional: List the built image
docker images %IMAGE_NAME%
