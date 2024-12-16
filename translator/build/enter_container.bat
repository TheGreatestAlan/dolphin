@echo off
REM Script to remove the entrypoint and start the container with a shell

REM Define default container name and image
SET CONTAINER_NAME=translator_container
SET IMAGE_NAME=translator
SET TAG=latest

REM Check if a container name is provided as an argument
IF NOT "%1"=="" (
    SET CONTAINER_NAME=%1
)
IF NOT "%2"=="" (
    SET IMAGE_NAME=%2
)

REM Stop and remove the existing container if it exists
docker ps -a --filter "name=%CONTAINER_NAME%" --format "{{.ID}}" > tmp_container_id.txt
SET /p CONTAINER_ID=<tmp_container_id.txt
DEL tmp_container_id.txt
IF NOT "%CONTAINER_ID%"=="" (
    echo Stopping and removing existing container: %CONTAINER_NAME%
    docker stop %CONTAINER_ID%
    docker rm %CONTAINER_ID%
)

REM Start a new container with the entrypoint overridden to /bin/bash
echo Starting container with entrypoint removed...
docker run -it --name %CONTAINER_NAME% --entrypoint /bin/bash %IMAGE_NAME%:%TAG%

REM If bash is unavailable, fallback to sh
IF %ERRORLEVEL% NEQ 0 (
    echo Bash not available, trying sh...
    docker run -it --name %CONTAINER_NAME% --entrypoint /bin/sh %IMAGE_NAME%:%TAG%
)
