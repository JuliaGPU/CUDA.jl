#!/bin/bash

cat /proc/$1/task/*/stat | awk '{print $14}' | sort -n
