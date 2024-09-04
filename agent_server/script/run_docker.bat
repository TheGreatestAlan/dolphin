@echo off
cd ..
docker rm agent-server
docker run --env-file ./script/.env -p 5001:5001 --name agent-server agent-server:latest
