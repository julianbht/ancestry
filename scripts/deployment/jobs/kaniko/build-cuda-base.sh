#!/bin/bash
# Rebuilds ancestry-pipeline:cuda-base on the cluster via kaniko.
# Only needed when changing the base CUDA image or apt packages.
set -e
cd "$(dirname "$0")/../../../.."
bash scripts/deployment/jobs/rerun-job.sh kaniko-build-cuda-base deployment/k8s/kaniko-build-cuda-base-job.yml
