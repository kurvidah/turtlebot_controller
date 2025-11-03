#!/usr/bin/env bash
set -euo pipefail

# upload.sh -- send everything in the current directory except this script
# to pi@192.168.17.1:~/turtlebot3_controller using password "piword".

REMOTE_USER="pi"
REMOTE_HOST="192.168.17.1"
REMOTE_DIR="~/turtlebot3_controller"
PASSWORD="piword"

# require rsync and sshpass
if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required. Install it and retry." >&2
    exit 1
fi
if ! command -v sshpass >/dev/null 2>&1; then
    echo "sshpass is required. Install it and retry." >&2
    exit 1
fi

EXCLUDE="$(basename "$0")"

# Sync current directory contents to remote, excluding this script.
# Using ssh options to avoid host-key prompt on first connect.
sshpass -p "$PASSWORD" rsync -av -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude="$EXCLUDE" ./ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"