export const meta = {
  name: 'crosscheck-review',
  description: 'Review the crosscheck generator, the block-reordering pass and the bench harness; adversarially verify every finding',
  phases: [
    { title: 'Review', detail: 'three independent reviewers, one per area' },
    { title: 'Verify', detail: 'three skeptics per finding, majority rules' },
  ],
}

// Re-run from Claude Code with /crosscheck-review, optionally passing
// { repo: '<path>' } as args. Reviewers experiment under .zig-cache/review/,
// which git ignores.
const REPO = (args && args.repo) || '/Users/bcann/programming/zig-programming/examples/window'
const SCRATCH = `${REPO}/.zig-cache/review`

const COMMON = `
You are reviewing the current working tree of the git repository ${REPO}. Use the Zig on
PATH (check 'zig version'; build.zig.zon pins the version the project expects).
RULES: never modify, format or build-install anything inside the repository except under
${SCRATCH}/; running 'zig build test', 'zig build crosscheck -Dseed=N' and
'zig build crosscheck -Dcrosscheck-body=<file>' inside it is allowed (they only write to
.zig-cache and zig-out). For experiments, copy files into your own directory under
${SCRATCH}/<your-label>/ and work there (zig test on copies etc). Set
_ZO_DOCTOR=0 in the environment before running zig.

Context. The project compiles Zig shaders to Metal libraries (docs/pipeline.md). The
"crosscheck" bench (docs/crosscheck.md) is a differential tester: tools/crosscheck/gen.zig
writes a random i32 program body(a, b) for a seed; build.zig compiles that ONE file twice,
into a GPU kernel (src/engine/crosscheck_harness.zig -> tools/air_splice.zig text rewrite
-> tools/air/assembler.zig bitcode -> .metallib, built ReleaseFast for nvptx64) and into
the host driver (tools/crosscheck/driver.zig, a Debug build) which runs both over 4096
input pairs and compares. tools/crosscheck/sweep.sh classifies each seed into
match/MISMATCH/REFUSED/METAL/SKIP/ERROR from the driver's and splice's output lines.
The contract that matters most: a MISMATCH must only ever mean a real codegen bug. That
requires every generated program to be defined (no undefined behaviour) and terminating
for ALL inputs, on BOTH builds. The Debug host build traps UB on the inputs it sees, so
UB that Debug catches lands in ERROR (bad but visible); UB that Debug does NOT catch, or
behaviour that legitimately differs between Debug and ReleaseFast, or between aarch64 and
nvptx64, could create a false MISMATCH or hide a real one (worst case). Also bad: the
generator emitting invalid Zig (compile error -> ERROR), or a legal program the splice or
assembler refuses for a reason that is our bug.

Report only concrete defects with a failure scenario (inputs/state -> wrong outcome). Do
not report style, naming or documentation issues. If you find nothing real, return an empty
list; do not pad. For each finding give file, 1-based line, severity (high = can cause a
false verdict or silent miscompile; medium = causes ERROR/REFUSED on some seeds or breaks a
documented guarantee; low = minor), the problem, the evidence (what you ran or traced), and
a concrete fix.`

const DIMENSIONS = [
  {
    key: 'generator',
    prompt: `${COMMON}

YOUR AREA: tools/crosscheck/gen.zig (read all of it) together with
src/engine/crosscheck_harness.zig. Questions to settle, with experiments where possible
(generate many seeds with 'zig build-exe -O Debug tools/crosscheck/gen.zig -femit-bin=...'
in your scratch dir and inspect/compile the outputs; a Debug test that calls body() over
edge inputs is a good UB detector but NOT a complete one):
1. Can any emitted construct be undefined behaviour that a Debug build would NOT trap?
   Think: shifts, truncation/@intCast, division helpers and their guards (minInt/-1 for
   @rem and @mod too), saturating ops, @abs, 64-bit helpers, @intCast(k1) of loop
   indices, switch on masked values, anything whose Debug and ReleaseFast semantics
   legitimately differ.
2. Can a program fail to terminate? Loops are claimed bounded (for over a range <= 8).
   Check runtime-bounded ranges like 0..(u(x) & 7) and nested loops.
3. Can the generator emit invalid Zig for some seed: unused parameters/locals, a 'var'
   that is never mutated (Zig rejects it), comptime-known break/continue conditions
   making later statements unreachable, an unused loop capture, a switch that is not
   exhaustive, shadowing, an expression of the wrong type, use of a local before
   assignment?
4. Does the printed program depend only on the seed (determinism), so a reproducer is
   exact?
5. Anything in the harness kernel or the shared helpers that could differ between the
   GPU (ReleaseFast, nvptx64) and CPU (Debug, aarch64) evaluation of the SAME Zig.`,
  },
  {
    key: 'reorder',
    prompt: `${COMMON}

YOUR AREA: the block-reordering pass 'reorderBlocks' and 'blockLabel' in
tools/air_splice.zig and how emitHelper/emitShader use them; also the assembler's value
and block handling in tools/air/assembler.zig (FunctionState.lookup, define, createBlocks,
the PendingPhi/resolvePhis mechanism, lowerBody) to the extent the pass interacts with it.
Background: the assembler resolves instruction operands in text order and reported
"use of undefined value" when LLVM laid a rotated loop's latch before the header whose phi
it used. The pass emits basic blocks in a reverse postorder from the entry so every
definition precedes its non-phi uses. Settle, with traced code paths and experiments
(you can feed hand-written IR to convertWith through a copy of the unit tests, or run
'zig build test' in the repo):
1. Is the correctness argument airtight for the code AS WRITTEN: entry detection (lines
   before the first label; what if the body's first line IS a label; what if the entry is
   a branch target), successor detection (every 'label %x' on non-label lines; could a
   non-terminator line contain 'label %'? comments? multi-line switch case lists?
   quoted labels? labels that are numeric after renaming, e.g. '__7:'), the DFS and
   its reversal, duplicate targets, blocks unreachable from the entry (appended last:
   can one of them define a value used by a reachable block? could an unreachable block
   be referenced by a phi?), and the 'n < 3' early return.
2. Interactions: implicitEntryLabel (it scans the body for phi references to the entry's
   number and prints a label FIRST), sret rewriting, renameNumbered (runs before or after
   this pass?), sanitizeBodyLine, isLifetimeIntrinsic skipping, the uniformity analysis
   in tools/air/divergence.zig (does it assume anything about block order?), and the
   assembler's createBlocks (first label = entry? predecessor counts).
3. Does reordering change program meaning in any case (it must not), and can it make a
   previously assembling input fail?
4. Since the pass changed the bytes of default.metallib, is there any way it changed
   what the shaders compute? (The repo's docs/crosscheck.md claims the printed IR is the
   same multiset of unchanged blocks; verify the claim independently if you can.)`,
  },
  {
    key: 'bench',
    prompt: `${COMMON}

YOUR AREA: the bench's verdict path: tools/crosscheck/driver.zig,
src/engine/crosscheck_harness.zig, tools/crosscheck/sweep.sh, and the crosscheck section
of build.zig (search for 'crosscheck'). Settle, with experiments where cheap:
1. Can the driver report 'match' when the GPU and CPU actually disagree, or 'mismatch'
   when they agree? Consider: how inputs are generated and copied, the output buffer
   poison value, the count/grid (n = 4096, threadgroup 64, Params.count), whether every
   element is compared, the i32 compare, and the two Params definitions (driver's extern
   struct vs the harness's struct: must agree in layout).
2. Are the exit codes and printed lines exactly what sweep.sh's regexes expect, for every
   bucket? Could a REFUSED or METAL outcome be misclassified as ERROR or vice versa, or a
   failure be classified as match? Could an 'ok' line from a different step confuse it?
3. -Dcrosscheck-body: is the hand-written body compiled for both sides from the same
   file; is the reproduce hint right; does the -Dseed input stream still vary
   independently of the body?
4. The crosscheck-apple step: does it test what docs/crosscheck.md says (library built by
   xcrun metal from our printed IR, same driver)? Any way it could pass trivially?
5. Determinism and reproducibility of a reported seed on the same machine.`,
  },
]

const FINDINGS = {
  type: 'object',
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['file', 'line', 'severity', 'problem', 'evidence', 'fix'],
        properties: {
          file: { type: 'string' },
          line: { type: 'integer' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          problem: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const VERDICT = {
  type: 'object',
  required: ['refuted', 'reason'],
  properties: {
    refuted: { type: 'boolean' },
    reason: { type: 'string' },
    corrected_fix: { type: 'string' },
  },
}

const LENSES = [
  'REPRODUCE: write the smallest concrete case that would show the problem and actually run it (in your scratch copy, never modifying the repo). If you cannot make it happen, refuted=true.',
  'TRACE: read the actual code paths the finding depends on, line by line, and check every claim in the problem statement against them. If any load-bearing claim is false or the code already handles the case, refuted=true.',
  'IMPACT: assume the claim is literally true and determine whether it can produce a false MISMATCH, hide a real mismatch, cause REFUSED/METAL/ERROR on real seeds, or break a documented guarantee. If it cannot, refuted=true (a true but inconsequential observation is refuted for our purposes), and say so.',
]

function verifyPrompt(f, dim, lens) {
  return `${COMMON}

A reviewer of the '${dim}' area reports this finding. Your job is to try to REFUTE it.
Default to refuted=true if you remain uncertain after real effort. Your lens: ${lens}

FINDING
file: ${f.file}
line: ${f.line}
severity claimed: ${f.severity}
problem: ${f.problem}
evidence offered: ${f.evidence}
proposed fix: ${f.fix}

Return refuted (boolean), reason (what you did and saw, with file:line references), and,
if the finding stands but the proposed fix is wrong or incomplete, corrected_fix.`
}

const results = await pipeline(
  DIMENSIONS,
  d => agent(d.prompt, { label: `review:${d.key}`, phase: 'Review', schema: FINDINGS }),
  (review, d) => {
    if (!review || !review.findings.length) return []
    log(`${d.key}: ${review.findings.length} finding(s) to verify`)
    return parallel(review.findings.map(f => () =>
      parallel(LENSES.map((lens, i) => () =>
        agent(verifyPrompt(f, d.key, lens), { label: `verify:${d.key}:${f.file.split('/').pop()}:${f.line}:${i}`, phase: 'Verify', schema: VERDICT })))
        .then(votes => {
          const v = votes.filter(Boolean)
          const standing = v.filter(x => !x.refuted).length
          return { ...f, dim: d.key, votes: v, confirmed: standing >= 2 }
        })))
  },
)

const all = results.flat(2).filter(Boolean)
const confirmed = all.filter(f => f.confirmed)
const refuted = all.filter(f => !f.confirmed)
log(`${confirmed.length} confirmed, ${refuted.length} refuted`)
return { confirmed, refuted }