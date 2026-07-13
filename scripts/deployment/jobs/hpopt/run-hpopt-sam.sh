#!/bin/bash
set -e
cd "$(dirname "$0")/../../../.."
bash scripts/deployment/jobs/rerun-job.sh dsw-ancestry-pipeline-job-hpopt-sam deployment/k8s/jobs/hpopt-sam "$@"
