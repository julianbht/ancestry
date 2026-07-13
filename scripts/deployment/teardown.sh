#!/bin/bash
# Usage: ./scripts/deployment/teardown.sh
set -e

kubectl delete \
  deployment/dsw-ancestry-ssh-cuda \
  job/kaniko-build-cuda-base \
  job/kaniko-build-cuda-dev \
  job/kaniko-build-job-cuda \
  job/dsw-ancestry-pipeline-job-frame-crop \
  job/dsw-ancestry-pipeline-job-face-crop \
  job/dsw-ancestry-pipeline-job-face-recognition \
  job/dsw-ancestry-pipeline-job-hpopt-canny \
  job/dsw-ancestry-pipeline-job-hpopt-sam \
  job/dsw-ancestry-pipeline-job-hpopt-face-recognition-deepface \
  job/dsw-ancestry-pipeline-job-hpopt-face-recognition-insightface \
  --ignore-not-found
