@echo off

REM Set the image name and tag you want to push
SET IMAGE_NAME=translator:latest

REM Set the DockerHub repository (replace "yourusername" with your actual DockerHub username)
SET DOCKERHUB_REPO=happydance/%IMAGE_NAME%

REM Login to DockerHub (this assumes you've already logged in or will be prompted)
docker login

REM Tag the image for DockerHub
docker tag %IMAGE_NAME% %DOCKERHUB_REPO%

REM Push the image to DockerHub
docker push %DOCKERHUB_REPO%

REM Print a message to indicate the image has been pushed
echo Image %IMAGE_NAME% has been pushed to DockerHub repository %DOCKERHUB_REPO%.
