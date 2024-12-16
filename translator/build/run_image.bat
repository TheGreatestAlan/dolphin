@echo off
REM Script to run the Docker image

REM Define default image name, tag, and container name
SET IMAGE_NAME=translator
SET TAG=latest
SET PORT=8080
SET CONTAINER_NAME=translator_container

REM Check if custom arguments are provided (optional)
IF NOT "%1"=="" (
    SET IMAGE_NAME=%1
)
IF NOT "%2"=="" (
    SET TAG=%2
)
IF NOT "%3"=="" (
    SET PORT=%3
)

REM Run the Docker container
echo Running Docker image: %IMAGE_NAME%:%TAG%
docker run -p %PORT%:8080 --name %CONTAINER_NAME% -d %IMAGE_NAME%:@echo off
REM Script to run the Docker image

REM Define default image name, tag, port, and container name
SET IMAGE_NAME=translator
SET TAG=latest
SET PORT=8080
SET CONTAINER_NAME=translator_container

REM Check if custom arguments are provided (optional)
IF NOT "%1"=="" (
    SET IMAGE_NAME=%1
)
IF NOT "%2"=="" (
    SET TAG=%2
)
IF NOT "%3"=="" (
    SET PORT=%3
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

REM Run the Docker container
echo Running Docker image: %IMAGE_NAME%:%TAG%
docker run -p %PORT%:8080 --name %CONTAINER_NAME% -d %IMAGE_NAME%:%TAG%

REM Check if the container is running
IF %ERRORLEVEL% EQU 0 (
    echo Docker container %CONTAINER_NAME% is running on port %PORT%.
) ELSE (
    echo Failed to start Docker container %CONTAINER_NAME%.
    EXIT /B %ERRORLEVEL%
)
%TAG%

REM Check if the container is running
IF %ERRORLEVEL% EQU 0 (
    echo Docker container %CONTAINER_NAME% is running on port %PORT%.
) ELSE (
    echo Failed to start Docker container %CONTAINER_NAME%.
    EXIT /B %ERRORLEVEL%
)
