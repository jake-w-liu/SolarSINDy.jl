#!/bin/bash
set -euo pipefail

[ "$#" -eq 1 ] || { echo "one fresh evidence path required" >&2; exit 1; }
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
parent="$(cd -- "$(dirname -- "$1")" && pwd -P)"
runner_temp="$(cd -- "$RUNNER_TEMP" && pwd -P)"
name="$(basename -- "$1")"
[ "$parent" = "$runner_temp" ] || { echo "evidence must be directly under RUNNER_TEMP" >&2; exit 1; }
case "$name" in julia-source-syntax-*) ;; *) echo "owned syntax evidence name required" >&2; exit 1 ;; esac
output="$parent/$name"
[ ! -e "$output" ] && [ ! -L "$output" ] || { echo "fresh evidence directory required" >&2; exit 1; }
available_kib="$(awk '$1 == "MemAvailable:" {print $2; found++} END {if (found != 1) exit 1}' /proc/meminfo)"
disk_bytes="$(df --output=avail -B1 -- "$runner_temp" | tail -n 1 | tr -d '[:space:]')"
[[ "$available_kib" =~ ^[0-9]+$ ]] && [[ "$disk_bytes" =~ ^[0-9]+$ ]] || { echo "native resource observation failed" >&2; exit 1; }
available_bytes=$((available_kib * 1024))
mkdir -- "$output"
printf 'available_physical_bytes\t%s\nfree_disk_bytes\t%s\nrequired_physical_bytes\t8589934592\nrequired_disk_bytes\t17179869184\n' \
    "$available_bytes" "$disk_bytes" > "$output/ADMISSION.tsv"
[ "$available_bytes" -ge 8589934592 ] && [ "$disk_bytes" -ge 17179869184 ] || {
    echo "8 GiB available physical memory and 16 GiB free disk required before Julia" >&2
    exit 1
}
echo "Native 8 GiB memory / 16 GiB disk admission passed before Julia setup"
