#!/bin/zsh
# Runtime validation of the SPIR-V lowering (docs/spirv.md): every probe kernel
# of tools/spirv/probes.zig and every compute entry point of the corpus is
# lowered on its own into a library (spirv-probe) and Metal builds a compute
# pipeline from it (metallib-check --kernels). Prints one row per kernel:
#   <name> <TAB> ok | refused: <why> | Metal: <error>
# and then compares the rows with tools/spirv/corpus/EXPECTED.txt (a refusal
# recorded there is a pass; any other difference fails, exit 1).
# Usage (from the repository root): tools/spirv/run_probes.sh <outdir> [--fast-math=true]
# The libraries land in <outdir>; delete it afterwards. The GPU execution
# check of the probes is zig-out/bin/spirv-probe-run (built by the same step).
set -u
out=${1:?usage: run_probes.sh <outdir> [--fast-math=true]}
fm=${2:---fast-math=false}
mkdir -p "$out" || exit 1
zig build spirv-splice || exit 1
bin=zig-out/bin
{
  perl -e 'alarm 600; exec @ARGV' $bin/spirv-probe "$fm" probes "$out"
  for f in tools/spirv/corpus/*.spvasm; do
    perl -e 'alarm 600; exec @ARGV' $bin/spirv-probe "$fm" corpus "$f" "$out"
  done
} 2>"$out/stderr.txt" | while IFS=$'\t' read -r name lib why; do
  if [[ "$lib" == REFUSED ]]; then
    print -r -- "$name	refused: $why"
    continue
  fi
  kernel=${name#*:}
  res=$(perl -e 'alarm 120; exec @ARGV' $bin/metallib-check --kernels="$kernel" "$lib" 2>&1 | grep -E '^(ok  kernel|FAIL)' | head -1)
  if [[ "$res" == "ok  kernel"* ]]; then
    print -r -- "$name	ok"
  else
    print -r -- "$name	Metal: ${res:-no result (timeout or crash)}" | tr '\n' ' '
    print
  fi
done > "$out/table.tsv"
cat "$out/table.tsv"
if ! diff -u tools/spirv/corpus/EXPECTED.txt "$out/table.tsv" > "$out/expected.diff"; then
  print -u2 "run_probes.sh: outcomes differ from tools/spirv/corpus/EXPECTED.txt (see $out/expected.diff)"
  exit 1
fi
print -u2 "run_probes.sh: $(grep -c '	ok$' "$out/table.tsv") ok, $(grep -c '	refused:' "$out/table.tsv") refused, all as in tools/spirv/corpus/EXPECTED.txt"
