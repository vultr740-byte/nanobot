#!/usr/bin/env sh
set -e

PORT_VALUE="${PORT:-18790}"

if [ "$#" -gt 0 ]; then
  exec nanobot "$@"
fi

exec nanobot gateway --port "$PORT_VALUE"
