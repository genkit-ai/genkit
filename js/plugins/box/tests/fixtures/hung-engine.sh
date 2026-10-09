#!/bin/sh
# Stands in for a container engine whose container never gets healthy: `run`
# stays in the foreground (briefly, so it doesn't hold up the test process)
# without serving anything; cleanup commands (`stop`, `rm`) succeed at once.
if [ "$1" = "run" ]; then
  # Lets tests count container starts.
  [ -n "$HUNG_ENGINE_LOG" ] && echo run >> "$HUNG_ENGINE_LOG"
  exec sleep 3
fi
exit 0
