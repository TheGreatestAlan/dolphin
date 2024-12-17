@echo off
REM Script to run a Docker container with FIREWORKS_API_KEY as an environment variable

REM Check if the FIREWORKS_API_KEY is provided
IF "%1"=="" (
    echo Error: FIREWORKS_API_KEY is required as the first parameter.
    echo Usage: run_translator.bat YOUR_FIREWORKS_API_KEY
    EXIT /B 1
)
SET FIREWORKS_API_KEY=%1

REM Define fixed values for the image and container
SET IMAGE_NAME=translator
SET TAG=latest
SET PORT=5000
SET CONTAINER_NAME=translator_container

REM Stop and remove the existing container if it exists
docker ps -a --filter "name=%CONTAINER_NAME%" --format "{{.ID}}" > tmp_container_id.txt
SET /p CONTAINER_ID=<tmp_container_id.txt
DEL tmp_container_id.txt
IF NOT "%CONTAINER_ID%"=="" (
    echo Stopping and removing existing container: %CONTAINER_NAME%
    docker stop %CONTAINER_ID%
    docker rm %CONTAINER_ID%
)

REM Run the Docker container with the FIREWORKS_API_KEY
echo Running Docker image: %IMAGE_NAME%:%TAG% with FIREWORKS_API_KEY set
docker run -p %PORT%:8080 --name %CONTAINER_NAME% -e FIREWORKS_API_KEY=%FIREWORKS_API_KEY% %IMAGE_NAME%:%TAG%

REM Check if the container started successfully
IF %ERRORLEVEL% EQU 0 (
    echo Docker container %CONTAINER_NAME% is running with FIREWORKS_API_KEY set.
) ELSE (
    echo Failed to start Docker container %CONTAINER_NAME%.
    EXIT /B %ERRORLEVEL%
)
