#!/bin/bash
# Launch TrackingEye in its own virtualenv.
cd "$(dirname "$0")" || exit 1
if [ "$1" = "--calibrate" ]; then
  shift
  exec .venv/bin/python calibrate.py "$@"
fi
exec .venv/bin/python main.py "$@"
