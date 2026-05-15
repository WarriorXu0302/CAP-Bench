#!/usr/bin/env bash
# Batch-generate evaluation scripts by iterating over task JSON files.
#
# Usage:
#   ANSWERS_DIR=/path/to/answers/<agent_name> \
#   JSON_BASE_DIR=/path/to/tasks_json \
#   bash run_eval_batch.sh
#
# Optional:
#   MAX_JOBS    Maximum concurrency (default: 7)
#   PYTHON      Python interpreter (default: python)

set -euo pipefail

ANSWERS_DIR="${ANSWERS_DIR:-${1:-}}"
JSON_BASE_DIR="${JSON_BASE_DIR:-${2:-}}"
MAX_JOBS="${MAX_JOBS:-7}"
PYTHON="${PYTHON:-python}"

if [[ -z "${ANSWERS_DIR}" || -z "${JSON_BASE_DIR}" ]]; then
    echo "Usage: ANSWERS_DIR=<answers_dir> JSON_BASE_DIR=<tasks_json_dir> bash run_eval_batch.sh" >&2
    echo "   or: bash run_eval_batch.sh <answers_dir> <tasks_json_dir>" >&2
    exit 2
fi

if [[ ! -d "${ANSWERS_DIR}" ]]; then
    echo "Error: ANSWERS_DIR not found: ${ANSWERS_DIR}" >&2
    exit 1
fi
if [[ ! -d "${JSON_BASE_DIR}" ]]; then
    echo "Error: JSON_BASE_DIR not found: ${JSON_BASE_DIR}" >&2
    exit 1
fi

echo "Generating evaluation scripts (max parallel jobs: ${MAX_JOBS})"

current_jobs=0
for dir in "${ANSWERS_DIR}"/task-*/; do
    task_id="$(basename "${dir}")"
    json_path="${JSON_BASE_DIR}/${task_id}.json"

    if [[ ! -f "${json_path}" ]]; then
        echo "Skip: ${task_id} (no JSON at ${json_path})"
        continue
    fi

    echo "Launching: ${task_id}"
    "${PYTHON}" generate_eval.py --json_path "${json_path}" &

    current_jobs=$((current_jobs + 1))
    if (( current_jobs >= MAX_JOBS )); then
        wait -n
        current_jobs=$((current_jobs - 1))
    fi
done

wait
echo "Done."
