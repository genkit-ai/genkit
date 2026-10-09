#!/bin/sh
# Stands in for a container engine whose container never gets healthy: `run`
# stays in the foreground (briefly, so it doesn't hold up the test process)
# without serving anything; cleanup commands (`stop`, `rm`) succeed at once.
[ "$1" = "run" ] && exec sleep 3
exit 0
