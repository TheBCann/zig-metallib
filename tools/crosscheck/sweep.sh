#!/usr/bin/env bash
# Run the differential test bench over a range of seeds and sort each outcome.
#
#   tools/crosscheck/sweep.sh FROM TO
#   CROSSCHECK_ARGS="-Dcrosscheck-floats=true" tools/crosscheck/sweep.sh FROM TO
#
# CROSSCHECK_ARGS is appended to every `zig build crosscheck` (float mode,
# `-Dcrosscheck-known=true`, ...) and mirrored into the reproducer's generation.
# Every seed leaves about 50 MB of build products in .zig-cache (its GPU object,
# library and Debug driver), so the cache is cleared every CROSSCHECK_CACHE_EVERY
# seeds (default 25; 0 never); the seed after a clear rebuilds the tools, about a
# minute. Do not run other `zig build`s in this tree during a sweep.
#
# For every seed, `zig build crosscheck -Dseed=N` generates a kernel, runs it on
# the Metal device and on the CPU, and compares. Each seed lands in one bucket:
#
#   match     GPU and CPU agree on every element
#   MISMATCH  they disagree: a codegen bug (the most valuable finding)
#   REFUSED   this project's assembler refused a construct (a feature gap)
#   METAL     Metal's compiler rejected or failed on our library
#   SKIP      no Metal device (e.g. a CI runner without a GPU)
#   ERROR     anything else, e.g. the generator emitted invalid Zig
#
# Anything but a match is saved under zig-out/crosscheck/<seed>/ (the generated
# body.zig and the full log) so it can be reproduced and reduced. Exits 1 if
# any seed was a MISMATCH, REFUSED, METAL or ERROR, else 0.
set -u

from=${1:?usage: sweep.sh FROM TO}
to=${2:?usage: sweep.sh FROM TO}
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
out_dir=zig-out/crosscheck

# Plain counters, not an associative array: macOS's /bin/bash is 3.2.
n_match=0 n_MISMATCH=0 n_REFUSED=0 n_METAL=0 n_SKIP=0 n_ERROR=0
cache_every=${CROSSCHECK_CACHE_EVERY:-25}
n_run=0
for seed in $(seq "$from" "$to"); do
  n_run=$((n_run + 1))
  if [ "$cache_every" -gt 0 ] && [ $((n_run % cache_every)) -eq 1 ] && [ "$n_run" -gt 1 ]; then rm -rf .zig-cache; fi
  # shellcheck disable=SC2086
  log=$(zig build crosscheck -Dseed="$seed" ${CROSSCHECK_ARGS:-} 2>&1)
  if grep -q '^ok  crosscheck seed' <<<"$log"; then
    bucket=match
  elif grep -q '^FAIL crosscheck seed .* mismatched' <<<"$log"; then
    bucket=MISMATCH
  elif grep -q '^SKIP ' <<<"$log"; then
    bucket=SKIP
  elif grep -qE '^FAIL (newLibraryWithData|newFunctionWithName|newComputePipelineState|command buffer)' <<<"$log"; then
    bucket=METAL
  elif grep -qE '^air-splice: (line [0-9]+|.*Unsupported)' <<<"$log"; then
    bucket=REFUSED
  else
    bucket=ERROR
  fi
  eval "n_$bucket=\$((n_$bucket + 1))"
  if [ "$bucket" != match ] && [ "$bucket" != SKIP ]; then
    mkdir -p "$out_dir/$seed"
    printf '%s\n' "$log" > "$out_dir/$seed/log.txt"
    # Regenerate the exact body for the seed as a standalone reproducer.
    # The same spellings `zig build` treats as true: the bare flag or `=true`.
    gen_args=""
    # shellcheck disable=SC2086
    for tok in ${CROSSCHECK_ARGS:-}; do
      case "$tok" in
        -Dcrosscheck-floats|-Dcrosscheck-floats=true) gen_args="$gen_args --floats";;
        -Dcrosscheck-known|-Dcrosscheck-known=true) gen_args="$gen_args --include-known-findings";;
      esac
    done
    # shellcheck disable=SC2086
    zig run tools/crosscheck/gen.zig -- --seed="$seed" $gen_args "$out_dir/$seed/body.zig" >/dev/null 2>&1
    reason=$(grep -m1 -E '^(FAIL|air-splice:|error:)' <<<"$log" | cut -c1-140)
    printf 'seed %-6s %-9s %s\n' "$seed" "$bucket" "$reason"
  fi
done

total=$((to - from + 1))
printf '\nseeds %s..%s (%s): match %s, MISMATCH %s, REFUSED %s, METAL %s, SKIP %s, ERROR %s\n' \
  "$from" "$to" "$total" "$n_match" "$n_MISMATCH" "$n_REFUSED" "$n_METAL" "$n_SKIP" "$n_ERROR"
[ $((n_MISMATCH + n_REFUSED + n_METAL + n_ERROR)) -eq 0 ]
