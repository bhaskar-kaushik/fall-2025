#!/usr/bin/env bash
set -euo pipefail
echo "== Running pipeline (Q1→Q4) =="
julia --project=. PS1_Kaushik.jl --profile
echo "== Running unit tests =="
julia --project=. PS1_Kaushik_UnitTest.jl
