#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration ---
SRC="mpi.cpp"
EXEC="m"
MACHINEFILE="machinefile"
NP=4

# --- Compilation ---
echo "Compiling $SRC..."
mpicxx "$SRC" -o "$EXEC" -lmetis
echo "Compilation finished."

# --- Execution ---
echo "Running with $NP processes on machines listed in $MACHINEFILE..."
mpiexec -n "$NP" -f "$MACHINEFILE" ./"$EXEC"
echo "Execution complete."
