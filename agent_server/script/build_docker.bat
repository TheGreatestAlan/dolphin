@echo off
cd ../..
docker build -f AgentServerDockerfile -t agent-server:latest .

