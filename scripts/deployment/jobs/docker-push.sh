#!/bin/bash
# Builds and pushes locally-built images.
# cuda-base, ssh-cuda, and job-cuda are built on the cluster via kaniko — not here.

set -e
cd "$(dirname "$0")/../../.."
docker build -f deployment/Dockerfile.ssh -t julianbht/ancestry-pipeline:ssh .
docker push julianbht/ancestry-pipeline:ssh
