#!/bin/bash


mkdir -p tmp/results

name="results"

julia-1.6 --startup-file=no rpc-benchmarks/run.jl run des tmp/results/des_${name}.csv
julia-1.6 --startup-file=no rpc-benchmarks/run.jl run des2 tmp/results/des_2_${name}.csv

julia-1.6 --startup-file=no rpc-benchmarks/run.jl run despoller tmp/results/poller_${name}.csv

julia-1.6 --startup-file=no rpc-benchmarks/run.jl run malloc ttmp/results/malloc_$name.csv
julia-1.6 --startup-file=no rpc-benchmarks/run.jl run print tmp/results/print_$name.csv

julia-1.6 --startup-file=no rpc-benchmarks/run.jl run malloc_count tmp/results/malloc_count_$name.csv
