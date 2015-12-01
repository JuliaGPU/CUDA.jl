#!/usr/bin/env bash
set -e

DEBUG=1 \
julia --color=yes test/runtests.jl

BUILDS=( $(shopt -s nullglob; echo /tmp/JuliaCUDA_*/) )
if [[ -z "$BUILDS" ]]; then
    echo "ERROR: no build files found!"
    exit 1
fi
set -u
if [[ ${#BUILDS[@]} < 2 ]]; then
    echo "ERROR: only a single build found, cannot compare output yet."
    exit 1
fi

SORTED=( $(
    for el in "${BUILDS[@]}"
    do
        echo "$el"
    done | sort -t _ -k 2,2 -n) )

$(set -x; diff -Nur ${SORTED[0]} ${SORTED[-1]})
