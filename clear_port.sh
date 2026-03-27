#!/bin/bash

# Default port
PORT=${1:-9092}

echo "Checking processes on port $PORT..."

PIDS=$(lsof -ti :$PORT)

if [ -z "$PIDS" ]; then
    echo "No process is using port $PORT."
    exit 0
fi

echo "Processes using port $PORT: $PIDS"

echo "Sending TERM signal..."
kill -TERM $PIDS 2>/dev/null

sleep 2

# Check again
REMAINING=$(lsof -ti :$PORT)

if [ -n "$REMAINING" ]; then
    echo "Force killing remaining processes: $REMAINING"
    kill -9 $REMAINING 2>/dev/null
fi

# Final check
FINAL=$(lsof -ti :$PORT)

if [ -z "$FINAL" ]; then
    echo "Port $PORT is now free."
else
    echo "Warning: Port $PORT is still in use:"
    lsof -i :$PORT
fi