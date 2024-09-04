@echo off
cd ..
docker rm agent-server
docker run --env-file ./script/.env -p 8080:8080 --name agent-server --entrypoint /bin/bash -it agent-server:latest
