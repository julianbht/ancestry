#!/bin/bash
set -e
git clone https://x-access-token:${GITHUB_TOKEN}@github.com/julianbht/ancestry.git /tmp/ancestry
cp -a /tmp/ancestry/. /app
rm -rf /tmp/ancestry
cd /app

# Git commit identity for interactive work in this dev box. Sourced from env
# (supplied via the k8s secret) and written to /root/.gitconfig, which SSH login
# shells read — unlike env vars, which sshd drops. Required: boot fails loudly if
# unset rather than committing under a wrong identity.
git config --global user.name "${GIT_USER_NAME:?GIT_USER_NAME must be set (see deployment/k8s/secrets.example.yml)}"
git config --global user.email "${GIT_USER_EMAIL:?GIT_USER_EMAIL must be set (see deployment/k8s/secrets.example.yml)}"
# The dev box is a full interactive environment, so install every feature group
# (web/annotation/experiments). Pipeline *jobs* stay core-only — see their patch.yml.
uv sync --frozen --no-dev --group web --group annotation --group experiments
# persist Claude auth across pod restarts
mkdir -p /app/data/.claude
ln -sf /app/data/.claude /root/.claude

exec /usr/sbin/sshd -D
