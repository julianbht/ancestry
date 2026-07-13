#!/bin/bash
set -e
cd "$(dirname "$0")/../../../.."
bash scripts/deployment/jobs/rerun-job.sh dsw-ancestry-pipeline-job-hpopt-face-recognition-deepface deployment/k8s/jobs/hpopt-face-recognition-deepface "$@"
