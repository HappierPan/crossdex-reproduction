#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")"
if [ -n "${CROSSDEX_PYTHON:-}" ]; then
  runner="$CROSSDEX_PYTHON"
elif [ -x /home/siyu/miniforge3/envs/crossdex_gym/bin/python ]; then
  runner=/home/siyu/miniforge3/envs/crossdex_gym/bin/python
else
  runner=python3
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$runner" play.py "$@"
