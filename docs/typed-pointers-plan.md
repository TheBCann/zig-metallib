# Typed-pointer emission for zig-metallib: final implementation plan

**Status: shelved (2026-09-26), not implemented.** This is a design only. It is kept for the case where zig-metallib needs to load on macOS 13-15 or on GitHub's virtual GPU. The research inputs it cites (reports, prototype diff, check logs) were local working files and are not in this repository; the measured facts they contributed are restated below with how each was measured.

**Scope: Apple-silicon Macs only.** Every host, GPU, test, CI leg and evidence claim in this plan is Apple silicon, meaning arm64 Macs with Apple GPUs. Intel Macs are out of scope. The plan therefore has no x86_64 hosts, no Intel or AMD GPUs, and no Intel-specific code paths, tests, claims or CI legs. The four profiles (macOS 13, 14, 15 and 26) all run on Apple silicon, so they stay. macOS guests running in Virtualization.framework on an Apple-silicon host (tart, UTM) are in scope as an optional source of evidence (§6.4); their GPU shows up as "Apple Paravirtual device". See §0, constraint 6.

**Code anchors.** Line numbers refer to HEAD `52595f0` and to the std of the pinned Zig, `0.17.0-dev.2307+392b17125`.

**Inputs** (local, not in the repository):
- the six research reports (`typed/*/report.md`), respecting every "Refuted" item in their skeptic sections;
- the critic's working prototype, `critic/proto/typed_prototype_apple.diff`. It passes 42/42 on the M3 at AIR 2.8, 2.7, 2.6 and 2.5 (`critic/proto/libs/typed_macos{26,15,14,13}.check.txt`);
- the three plans and the three judgements;
- the 15 gaps from the plan review (`plan/critic_gaps.json`), and the findings of the later verification rounds. The last two tables in §9 say where each one is resolved.

**Facts measured while writing this plan** (read-only; no project change):

| # | Fact | How it was measured |
|---|---|---|
| F-A | std `Builder.zig` is byte-identical in 0.17.0-dev.2257 and 2307. Judge 2 found the same for `ir.zig`, `bitcode_writer.zig` and `BitcodeReader.zig`. | sha256 of both local installs: `48946b6075becf7d…` |
| F-B | CI run 36266828343, macos-15 leg (macOS 15.7.9, "Apple Paravirtual device"): Apple's **typed** control kernel for macOS 15.0 (AIR 2.7) builds a pipeline (`ok  kernel control`). The same kernel built with `-Xclang -opaque-pointers` fails (`CompilerError Code=2`). Our opaque macos15 library also fails, at `scaleKernel`. So the macos-15 runner's control passes for typed AIR 2.7, which means a typed macos15 row on that runner can be interpreted. | `gh run view 36266828343 --log` |
| F-C | The shader's sampler word is `[2 x i64] [i64 34901797601020489, i64 0]`, where word 0 is `0x7bff0000080a49`. Its tagged AIR ≤2.6 form is **`-9188470239253755319`** (`0x807bff0000080a49`), the value apple-encoding §10 reports from Apple's macOS 14 compile. Two of the plans quoted `-9188470239253755246`. That is the skeptic's probe word `0x807bff0000080a92`, not this shader's word. | `zig-out/bin/shader.air.ll:13` |
| F-D | The real shader uses only these intrinsics: sample, read, get_read_sampler and write; the `.u.i32` and `xchg.i32` atomics; `air.atomic.fence`; and `llvm.assume`. It contains 0 constant GEPs. | grep of `zig-out/bin/shader.air.ll` |
| F-E | The real shader contains none of the constructs in §3 rows 11-15. It also has no `i1` memory access and no `{}` type. All 15 of its `define`s are entry points, so it has no helper functions. It has 0 `phi`, `select` or `icmp` over pointers, 0 `ptrtoint`/`inttoptr`, 0 `insertvalue`/`extractvalue` of pointers, 0 pointer-typed `null`/`undef`/`poison`, and 0 `load`/`store` of `i1`. (The 10 `{}` matches are `!{}` metadata nodes.) Its `air.atomic` declarations are exactly the nine forms declared at `gpu.zig:195-203`, plus `air.atomic.fence(i32, i32, i32)`, which takes no pointer. | grep of `zig-out/bin/shader.air.ll`. Its `default.metallib` hashes to `1969ef4d…cb74`, the HEAD value. |
| F-F | Three behaviours that stop at the first failure shape the probe and xcheck design. (1) `air_splice.zig` `main` assembles every manifest entry and returns on the first error (`:156-166`). (2) `metallib-check` exits 1 on the first failing compute pipeline (`:193-194`) or render pipeline (`:274`), and `--kernel=` builds one pipeline without dispatching it (`:156-179`). (3) `zig build` turns metallib-check's exit 77 into a failure, so CI reads `SKIP` from the output instead (`ci.yml:82-85`). | read of the files at HEAD |
| F-G | The only global used as a sampler operand anywhere in the existing assembler tests is `@sampler.state`, typed exactly `[2 x i64]` (`assembler.zig:2841-2848`). The `{ [2 x i64], i8 }` global in the literals test (`:3450`) is only loaded, never passed as a sampler. So A5's sampler part (Phase 2) changes no existing test. | read of `tools/air/assembler.zig` at HEAD |

---

## 0. The hard constraints, and how the plan meets each

**1. The default output stays byte-identical (`1969ef4d…cb74`) and all 90 tests pass.**
- The pointer mode is an explicit parameter, and opaque is the default for every profile. In opaque mode `ModuleState.typed == null`.
- `assemble()`, `build()`, `sanitizeBodyLine(gpa, line)`, `samplerGlobals(gpa, ir)`, `textureIntrinsic(name)` and `textureDeclareText(gpa, ti)` keep their signatures as wrappers, so no existing test body changes.
- §2 lists the edits to shared paths. Each commit that touches them is gated by three checks:
  - the default library's sha256;
  - `cmp` of `shader.air.ll` against the saved copy;
  - `zig build golden` (§1.4, D16).
- `verified: bool` and the CLI refusal of `--target=macos15` do not change. This keeps the assertions at `target.zig:186` and `air_splice.zig:2020` passing.

**2. The build never requires Apple tools.**
- Typed emission, the splice and the verifier are pure Zig/std.
- `air-xcheck --golden` and `--expect-pointers` are pure Zig too. They never spawn a process and they exit 0 or 1.
- `xcrun` is used only behind `air-xcheck --apple` and `--reassemble`, the optional `zig build xcheck` modes. Without the toolchain these print `SKIP no Metal toolchain`.
  - A direct call then exits 77.
  - `zig build xcheck` exits 0, because build.zig passes `--skip-status=0`. `zig build` treats any non-zero exit as a failure (F-F).
  - The SKIP line is read from the output, the same way CI's `state()` already reads it.
- `xcrun` is also used in CI.

**3. Re-pins are caught loudly.** Every Builder or BitcodeReader internal the design relies on is pinned three ways (§4):
- a compile-time field or declaration access;
- a named guard test;
- a check on every typed build (`typed.verify`, V1-V7).

So if an internal changes, the result is a compile error, a failing test or `error.Unsupported`, and never silently bad bitcode.

**4. Project conventions.**
- Refusals are `error.Unsupported`, printed with the offending line.
- Tests live in the file they test.
- A construct counts as done only after `zig build check` has executed it on the M3. Typed mode applies this **per construct class**:
  - A class is accepted when the 42 checks execute at least one member of it, and every other member produces the same kinds of typed records in the same positions. These are §3 rows 1-10 and the four generalisations listed after them ("Accepted by class", a-d).
  - A class the 42 checks do not execute at all is refused until its own Phase 5 probe executes it (D15, §3 rows 11-21).
  - The generalisations rest on an argument about record kinds, not on an execution of each member. §3 states the argument for each one, and Q24 tracks the risk.
- Docs quote only logged measurements.

**5. The unknowns** (GitHub's paravirtual GPU, native macOS 13-15) are answered by named CI rungs, an optional local-VM rung (§6.4), a field-report procedure and a promotion rule (§6, §8). No evidence level changes without a measurement.

**6. Apple silicon only.**
- Every evidence level names an Apple GPU (§5):
  - `native` means a real Apple-silicon GPU running that macOS major;
  - `native_virtual` means the Apple Paravirtual device of a Virtualization.framework guest on an Apple-silicon host. That can be a GitHub arm64 runner or a local VM (§6.4).
  - A report from any other GPU does not count as evidence.
- CI uses arm64 runner images only: `macos-15`, `macos-latest`, and optionally `macos-14`. It never uses `macos-13` or any `-intel` label.
  - The Runner step asserts that `uname -m` is `arm64`, so a relabelled image fails loudly.
  - No job runs on an x86_64 host. The determinism comparison runs on `macos-latest` (§6.1 step 8).
- The field-test `metallib-check` is built for `aarch64-macos.13.0` only, with no universal or x86_64 slice.
- The plan adds no Intel-specific code path, test, doc claim or fallback. The only x86_64 item in the earlier draft was the determinism job on `ubuntu-latest`, which has moved.

---

## 1. Decision and rationale

### 1.1 What the judges decided

**Totals:** productionize 42 + 43 + 42 = **127**; staged-risk-first 37 + 38 + 39 = 114; own-the-type 31 + 29 + 31 = 91.

All three judges picked productionize, so the architecture is not in dispute. Their graft recommendations diverged in four places. Each is resolved here:

| Divergence | Resolution |
|---|---|
| Test helper. Judge 3 valued productionize's dual-mode helper. Judge 1 wanted separate typed helpers. Judge 2 wanted the dual-mode helper to accept known refusals. | Keep a dual-mode `testRoundTripAs` / `testFailsAs`, changing only the helpers and never a test body. A `typed.Diag` carries the refusal id. The helper accepts a construct refusal only when an expected-refusal table lists that module with that exact id, and requires success for every other module. See D18. |
| CI sha pin. productionize D15 had none; staged used a hard pin. | All three judges proposed the version-keyed golden check, which is adopted (D16). |
| Pointer load/store and native atomics. productionize shipped them in Phase 2 on unit tests alone. | Refused until a probe kernel executes them. Judges 1 and 2 grafted this and Judge 3 did not object (D15). This revision extends it to every construct class the 42 checks do not execute, including §3 rows 11-15. |
| The "in-memory typed verifier" (Judge 1, taken from own-the-type). | Implemented as construction-time checks in the assembler's emission helpers (K1-K3, §4.3). A post-hoc walk over `Builder.Function` would need its private `extraData` accessors (the ones own-the-type's edit E38 had to make `pub`), or a re-implemented LLVM reader. |

### 1.2 Architecture

The critic's option 4, hardened, on the stock `std.zig.llvm.Builder`. Data flow:

1. **Text rewrite** (`air_splice.convertWith`): unchanged, plus the AIR-version forms, which are applied here and **only here**. The printed `shader.air.ll` is exactly what the assembler reads.
2. **Assembler** (`assembler.assembleWith(profile, pointers)`) builds a module with Builder.
   - `opaque`: `toBitcode` produces the output bytes. This is today's path.
   - `typed`: pointers are *handles*, meaning Builder pointer types in synthetic address spaces `0x100 + i`. Each handle stands for one `(pointee, real address space)` pair. `toBitcode` then produces opaque-framed words, and `typed.finish` runs these steps in order:
     - `normalizeCeCasts`, only once contingency C1 is wired (§1.5; off by default);
     - `splice`: in TYPE_BLOCK only, `OPAQUE_POINTER [as]` becomes `POINTER [pointee, real as]`;
     - `verify`: checks V1-V7 on the result, with the splice's input as V1's reference (§4.2);
     - the bytes are returned.
3. **Packer** (`metallib.pack`): unchanged. A new `metallib.unpack` serves `xcheck`, the checker and tests.

### 1.3 Typed rules (enforced in `assembler.zig`; helpers in `tools/air/typed.zig`)

- **T1. Canonical pointers.** `parseType("ptr [addrspace(N)]")` returns `canon(N)` = `handle(i8, N)`.
  - This covers parameters, returns, aggregate fields, phi/select values, `null`/`undef`/`poison`, `inttoptr` results and call results. §3 rows 11-14 use this rule, but they stay refused until their probes execute (D15).
  - Every pointer `Value` passed to `define` is `canon(realSpace)`; check K1 enforces this.
  - The one exception is the AIR 2.8 `air.get_read_sampler` result, typed `handle(%struct._sampler_t, 2)`. It may only feed the read's sampler parameter, which the existing type-equality checks enforce.
- **T2. Views only at use sites.** `view(v, T)` is `wip.cast(.bitcast, v, handle(T, realSpace(v)))` followed by check K2. Builder folds a view that changes nothing. Views are applied at:
  - load and load atomic (the loaded type);
  - store (the value type);
  - GEP (the base goes to the source element type, and the result goes straight back to i8);
  - call arguments (the declared parameter's pointee);
  - atomicrmw and cmpxchg (Phase 5).
- **T3. Globals and functions.**
  - A global enters a function only through `enterGlobal(g)`, an INST_CAST from the reader's `G addrspace(N)*` to `i8 addrspace(N)*`.
  - Constant bitcasts are emitted only by `Handles.noopRef`, for metadata. Their destination is the global's own typed pointer (`fnty*` or `valty addrspace(N)*`).
  - Pointer-changing constant bitcasts and constant GEPs over globals are **never emitted**. Critic G1 and the builder-feasibility refutation show that Apple's reader rejects Builder's encoding of them.
- **T4. Allocas** are placed in real address space 0 (`wip.alloca(..., .default)`) and then `view`ed to `canon(0)`. An alloca whose mapped address space is not 0 is refused (R7).
- **T5. Apple-shaped declarations.**
  - Texture parameters are `handle(%struct._texture_2d_t, 1)`. Sampler parameters and returns are `handle(%struct._sampler_t, 2)`.
  - Pointer parameters of the executed `air.atomic` forms are `handle(i32, AS)`. The executed forms are `intrinsics.typed_atomics`, the nine declared in gpu.zig. `air.atomic` declarations without pointer parameters, such as `air.atomic.fence(i32, i32, i32)`, are left unchanged (F-E).
  - Entry parameters stay canonical i8. That is the measured configuration (critic G3).
  - A `{}` or other zero-sized pointee is never emitted: `view` refuses it (R17, K2) and V2 rejects it. prior-art found that a `{}*` texture crashes MTLCompilerService. An `i1` pointee is refused (R18) until a probe executes one; metal-ir-pipeline reports that `i1*` crashes.
- **T6. Handle interning.** A handle is interned after its pointee, in first-use order, so the output is deterministic and pointees come before pointers.
- **T7. Reader-typed values.** Globals, functions and allocas have reader types that differ from their Builder type. They may appear only as the operand of `enterGlobal`, as the operand of an alloca's view, or as a callee. Check K3 enforces this.

### 1.4 Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Option 4: stock Builder, synthetic-address-space handles, a TYPE_BLOCK splice and a verifier. **Fallback 4b, an unmeasured design:** vendor Builder and patch 3 writer sites, with a spike before switching (§1.5). **Further fallback:** option 1 (a real typed Type kind, as in own-the-type), only if the handle trick itself becomes the obstacle. | Measured 42/42 at four AIR versions in about 320 lines. Nothing needs re-applying on a re-pin; the dependencies can be listed, and each is guarded (§4). F-A shows the last re-pin left Builder untouched. |
| D2 | Canonical i8 for every SSA pointer; views only at use sites; no pointee inference. | There are no undecidable cases (inference). All existing type-equality checks stay valid. Every pointee policy passed on the M3 (oracle, inference). |
| D3 | Apple-shaped intrinsic declarations, i8 everywhere else. | i8 texture, sampler and atomic signatures fail `air-opt -verify` (refutations in verify-apple-encoding and verify-prior-art). Apple shapes pass both air-opt and the runtime (critic G3). |
| D4 | Instruction bitcasts for every use of a global. Constant bitcasts only when they change nothing, and only for metadata. | Critic G1: a Builder CE_CAST writes the destination type into its operand-type field, and Apple's reader checks that field. |
| D5 | Allocas in real address space 0, then a view. | Removes the prototype's dependency on "ALLOCA records carry no address space". Re-measured in the Phase 2 gate by the three sret renders. The prototype already relied on the reader deriving `valty addrspace(3)*` for threadgroup globals whose typed pointer is missing from the type table, and it still passed. |
| D6 | Constant-GEP operands become instruction GEPs. Phi incoming globals and constant GEPs are materialised once at the start of the entry block. Pointer-valued initializers are refused permanently under option 4, whether scalar or aggregate. | One code path and no CE_GEP. The entry block dominates every predecessor, and inserting at `cursor = {entry, 0}` needs no terminator bookkeeping, which is simpler than productionize's predecessor placement. |
| D7 | The mode is an explicit parameter: `assembleWith(..., pointers)`, `buildWith(..., mode)`. No global `pub var`. | Tests can run both modes in one process, and the opaque path is structurally untouched. |
| D8 | Every typed build runs the splice, lockstep verification and lint (V1-V7), plus construction checks (K1-K3). | Guard tests cover only the shapes they build. `verify` checks the real output of every build (constraint 3). |
| D9 | The AIR-version forms (the ≤2.7 read_texture and the ≤2.6 sampler word) live in profile fields and apply in both pointer modes. The **text rewrite is the only place that transforms them**; the assembler only validates, keyed on the same profile fields. **Phasing:** the `sampler_word` field and the assembler's sampler handling (D10, A5's sampler part) land in Phase 2 with `.pair` on every profile. The `read_texture` field, the text rewrite's transforms, A5's read-form part and the 14/13 `sampler_word` values land in Phase 3. | They are facts about AIR, not about pointers (apple-encoding §10). A typed 2.7 library with the 2.8 read form crashes the compiler service. The printed IR then equals what is assembled. This removes the double application the judges found between productionize §5.3 and §5.6. Landing the sampler field in Phase 2 means every assembler branch that reads it exists from the first commit that reads it. |
| D10 | Sampler registration follows `profile.sampler_word`, in both pointer modes and in both the assembler and the text rewrite. The field and both assembler branches land in Phase 2, with `.pair` on every profile; Phase 3 sets `.tagged` for 14/13 only if its rule holds. **`.pair`**: the globals whose value type is **exactly** `[2 x i64]`, in definition order. A5's sampler part (also Phase 2) refuses a global of any other type at a sampler operand, so every directly used sampler is on that list, and so is a sampler that reaches the intrinsic through select, phi or a helper argument. **`.tagged`**: `defineGlobal` registers nothing; `lowerCall` registers the globals at direct sampler operands, de-duplicated, in first-use order; A5 requires them to be `i64`; the text rewrite refuses any `[2 x i64]` constant referenced in any other way (A4). | Fixes the `{ [2 x i64], i8 }` substring misdetection at `assembler.zig:487` (reproduced by verify-inference). It does so without dropping samplers that reach the intrinsic through select, phi or a helper argument (`*const Sampler` is an ordinary pointer, gpu.zig:367-384). Dropping one from `!air.sampler_states` crashes pipeline creation (gpu.zig:370), and no gate would notice. The text rewrite's `samplerGlobals` already matches the exact type (`air_splice.zig:928`). Under `.tagged` the rewritten word is `i64`, so the type cannot identify a sampler, and refusing the ambiguous case keeps the failure loud. The default shader has one sampler, used directly, so its bytes do not change (sha gate); F-G shows A5's sampler part changes no existing test. A `[2 x i64]` data table is still listed under `.pair`, as it is today (Q21). |
| D11 | The default pointer mode stays **opaque for every profile**. `-Dpointers=typed` opts in. Typed at `unverified` is refused without `--allow-unverified`. At `upgrader_only` or better it builds and prints a caveat. | Keeps the `target.zig:186` and `air_splice.zig:2020` assertions intact. Flipping the macos13-15 default is a user decision with a stated test consequence (§8, Q9). |
| D12 | Evidence is recorded per (profile, pointers). `verified: bool` keeps meaning "opaque, verified". The new `typed_verification` takes `unverified`, `upgrader_only`, `native_virtual` or `native`. | Honest semantics: macos13-15 typed is proven only through macOS 26's upgrader until CI, a local VM or field runs say more. |
| D13 | `shader.air.ll` stays the opaque assembler input for every mode, with the AIR-version forms applied. Typed builds prepend a comment saying so. Typed cross-checks run `air-xcheck` over our own blobs. | A typed text printer would duplicate the lowering. The `-Xclang -opaque-pointers` re-assembly rung cannot represent typed output (critic N1). |
| D14 | Constructs outside the sample shader are proven through separate probe libraries, one per construct (D20). A strict grep checks that each probe kernel really contains its construct. | Adding kernels to `my_shader.zig` would change `default.metallib`. |
| D15 | Refuse until executed, **per construct class**. In typed mode, every construct class the 42 checks do not execute is refused: pointer load/store; native atomics; the texture queries, gather and grad (`typed_proven = false`); helper pointer parameters and returns; pointer phi/select/icmp and pointer `null`/`undef`/`poison`; `ptrtoint`/`inttoptr`; aggregates holding pointers; `air.atomic` forms other than the nine declared in gpu.zig; and `i1` memory access. A function used as a value is refused too. The generalisations of executed classes that typed mode accepts are listed in §3 ("Accepted by class", a-d) and tracked by Q24. | Convention 4, applied per construct class: rows 11-15 are no longer "allowed by construction". F-E shows the real shader contains none of them, so typed 42/42 is unaffected. Each is lifted by its own probe library (Phase 5). |
| D16 | A golden default-library check keyed on the pinned Zig version: `air-xcheck --golden`, recorded in `tools/air/golden.zig`. Strict under the pinned Zig; a note (CI `::warning::`) under any other Zig. It runs in CI and in every gate, but not inside `zig build test`, so tests do not start compiling the shader. `--golden` never spawns a process and exits 0 or 1. | All three judges proposed it. It does not fail on a legitimate re-pin, which was productionize's D15 objection. |
| D17 | Keep relying on the CE_CAST quirk for the metadata no-op casts, guarded by G4 (test) and V4 (every build). Contingency **C1** (normalise CE_CAST operand types) is **built and tested in Phase 1** as the `normalize_ce_casts` option of `finish`, default false. Wiring it means flipping that default. Report the quirk upstream. | The quirk is the most likely thing to break on a re-pin, and the user re-pins under time pressure. With C1 built and its exact sequence (normalize, splice, verify) already tested, the fix is one line plus a re-measurement. |
| D18 | Typed tests assert on bitcode records (`typed.records`, `typeTable`, `countCode`, `stats`), never on `Builder.print`. The dual-mode helpers run every existing round-trip and failure module in typed mode too, against an **expected-refusal table** keyed by a hash of the entry name and module text: a listed module must fail with exactly its listed refusal id; every other round-trip module must succeed; a failure module must fail with the same kind of refusal as in opaque mode. | Critic N2: `Builder.print` shows `ptr addrspace(257)`. Dual mode catches typed crashes and asserts across the whole existing corpus without editing test bodies. A typed regression that starts refusing a module it used to accept fails the test. A listed module that succeeds also fails, so each Phase 5 lift must delete its entries. |
| D19 | Determinism checks. A unit test builds twice in one process and compares the bytes. The Phase 2 gate rebuilds with fresh cache directories, because a cached air-splice run would otherwise be replayed. A CI job compares typed-library hashes across the two arm64 runners. | llvm-downgrade had a nondeterminism bug (prior-art). |
| D20 | Probe constructs are built, checked and lifted **one library per construct** (`air-splice --entry`, `metallib-check --only`), each with its own opaque macos26 control, each in its own process. A construct whose opaque control fails **on the M3** is out of scope for the lift and stays refused. **Scope decisions come only from the M3** (`probe_matrix.sh gpu`). CI runs the presence and refusal checks strictly and the GPU rows as data only, without opaque controls, using Apple's typed control as the runner check (§6.1 step 11). A lift may cover only the upper part of the AIR range: `probe_expect.min_air` records it, and every check is per profile. | air-splice stops at the first refused entry, and metallib-check stops at the first failing pipeline (F-F). With one shared library, the probe gate would be circular (nothing builds until every lift has succeeded), and one failing construct would hide every later one, including its opaque control. GitHub's paravirtual device rejects all opaque output, Apple's own included (F-B), so an opaque control run there would mark every construct out of scope. |

### 1.5 Fallback, and when to switch to it

**Fallback 4b (vendored Builder): an unmeasured design.** Nobody has built this variant. builder-feasibility built a different one: it patched the `.pointer` branch at Builder.zig:14620 plus the address-space fields of the global and function records around 15013-15081, which was needed because its prototype put globals in handle address spaces. builder-feasibility's skeptic did not rebuild that one either. This plan's three sites differ because globals stay in real address spaces (T3, V4), and because 4b's purpose here is pointer-changing constant casts (R6).
- Copy the pinned `Builder.zig`, `ir.zig` and `bitcode_writer.zig` into `tools/air/vendor_llvm/`. The directory must not be named `llvm`, because that name fails with "llvm intrinsics cannot be defined!" (builder-feasibility). own-the-type measured that the module name `air_llvm` is safe.
- Patch three writer sites:
  - the TYPE_BLOCK pointer record, written from the handle table;
  - the CE_CAST operand type, written as the operand's typed pointer;
  - the CE_GEP base type.
- Delete the splice. The assembler's handle scheme stays. V1 (splice identity) no longer applies; V2-V7 still run.
- The opaque default stays on std's Builder, so there are two compilations and the default sha cannot move.
- Adopt own-the-type's tooling:
  - `revendor --check`, with an upstream sha256 record and an edit table whose anchors must each match exactly once;
  - an in-process opaque-parity test against `std.zig.llvm.Builder`.

**Trigger: switch to 4b when any of these holds.** Each is read from the Re-pin log (§4.5) or from CI.
1. A re-pin breaks a guard or V-check, and the fix cannot be made inside `typed.zig` or the assembler's typed paths within about one day (the log records the hours). Two consecutive re-pins that each need a new workaround also count.
2. A consumer needs a record the stock writer cannot produce. For example: the Apple re-write rung (§6) shows that framing is the cause and the fix needs SYMTAB or a sync-scope block; or users need pointer-valued initializers (R6), which require pointer-changing constant casts.
3. std removes or reshapes Builder or BitcodeReader in a way that no fallback in §4 absorbs.

**Before switching: the 4b spike (0.5 d or less, on a branch).** Vendor Builder, patch the three sites, then:
- build the four typed default libraries and run V2-V7;
- run `zig build xcheck -- --expect-pointers=typed --apple` on each;
- run `zig build check` on the M3 and require 42/42 at each profile.

Record the result in the Re-pin log. Adopt 4b only if the spike passes. If it fails, the next step is option 1, not 4b.

**Contingency C1.** When G4 or V4 fires on a re-pin (that is, upstream starts writing the operand type in CE_CAST), flip the default of `finish`'s `normalize_ce_casts` to true. `finish` then runs `normalizeCeCasts`, then `splice`, then `verify`. V1's reference is the normalized stream (the splice's input), so V1 still checks that the splice changed only TYPE_BLOCK. `normalizeCeCasts` patches the length words of the CONSTANTS block and the MODULE block. Then flip G4's expectation and re-run `xcheck` and the M3 checks.

### 1.6 Deviations from the prototype (typed_prototype_apple.diff)

| Prototype | Final | Why |
|---|---|---|
| `pub var typed_pointers` global | explicit mode parameter | D7 |
| `abbrev_width = 4` hard-coded | read from the reader's block state; alignment and length word asserted | G10 |
| unchecked `handles[as - 0x100]` | bounds-checked, `error.Unsupported` | V2 |
| alloca in a handle address space | real address space 0, then a view | D5 |
| `phiPointerType` skipped in typed mode | a typed variant returns `canon(realSpace(global))` | D1: loop-carried phis over relocated tables need it |
| ≤2.7 read transform done in `lowerCall` | done in the text rewrite; the assembler validates | D9 |
| `.f32` atomics mapped to `float*` | refused (R10) | D15: no evidence |
| helper pointers, SSA pointer phi/select, `ptrtoint`, pointer aggregates and `.s.i32` atomics accepted | refused (R13-R16, R10) until each one's probe executes | D15: not executed by the 42 checks |
| splice only | splice, then verify (V1-V7), plus K1-K3 at construction | D8 |
| `resolving_phis` flag suppresses casts | R2 refusal in Phase 2, entry-block materialisation in Phase 5 | D6 |

---

## 2. Phased implementation

### Common gate for every commit (M3, macOS 26.3)

1. `zig build test` passes: the 90 existing tests with bodies unmodified, plus the new tests. Record the count in the commit message.
2. `zig build metallib && shasum -a 256 zig-out/bin/default.metallib` prints `1969ef4d02c303c9079fab89d2c2b0b6bf7fc5912471016fe10df7d2eea1cb74`, and `zig build golden` passes (from Phase 1 on).
3. `cmp zig-out/bin/shader.air.ll` against the copy saved in Phase 0 reports no difference.
4. `zig build check -- zig-out/bin/default.metallib` reports `42 ok`.
5. GPU hygiene:
   - run checks one at a time, never in parallel;
   - before a gate run, move `~/Library/Logs/DiagnosticReports/MTLCompilerService*.ips` into an archive folder instead of deleting them. Crash reports are rate-limited (verify-oracle, verify-prior-art).

### Phase 0: baseline (0.25 d, no code)

- Record the 90/90 test count, the sha, 42/42, and a copy of `shader.air.ll`. Also record `uname -m` (`arm64`), the macOS build and the GPU name.
- Negative control: `zig build metallib -Dmetal-target=macos15 -Dallow-unverified-target=true --prefix /tmp/o15`, then a check. Expect 7 ok, then `FAIL newComputePipelineState scaleKernel … "Failed to upgrade function bitcode"`. Keep the log.

### Phase 1: safety net, no behaviour change (2.5 d)

**New `tools/air/typed.zig`.** It imports only std, so `air_xcheck` and `metallib_check` can import it too.
- A `comptime` pin block (§4.4).
- A `code` namespace of LLVM bitcode constants, each citing LLVMBitCodes.h:
  - blocks: MODULE 8, CONSTANTS 11, FUNCTION 12, METADATA 15, TYPE 17, SYMTAB 25;
  - TYPE records: NUMENTRY 1, OPAQUE 6, INTEGER 7, POINTER 8, ARRAY 11, VECTOR 12, STRUCT_ANON 18, STRUCT_NAME 19, STRUCT_NAMED 20, FUNCTION 21, OPAQUE_POINTER 25;
  - constants: SETTYPE 1, AGGREGATE 7, CE_CAST 11, CE_GEP 12 and 20;
  - METADATA VALUE 2; MODULE GLOBALVAR 7, FUNCTION 8, VSTOFFSET 13;
  - FUNCTION records: INST_CAST 3, INST_PHI 16, INST_ALLOCA 19, INST_LOAD 20, INST_GEP 43.
- `Handle` and `Handles`:
  - `init`, `deinit`;
  - `get(pointee, space)`, `canon(space)`, `isHandle`, `realSpace`, `pointeeOf`;
  - `namedOpaque(.texture_2d | .sampler)`, cached per module;
  - `noopRef(b, k, value_ty, space)`, which also counts itself for V7.
- `Refusal = enum { R1, …, R18 }` and `Diag { kind: enum { none, construct, internal }, id: ?Refusal }`. Every typed refusal message starts with its id: `typed pointers: R<n>: …`.
- `splice(gpa, bc, handles)`, adapted from the prototype's `spliceTypes`:
  - takes the abbreviation width from `reader.stack.items[top].abbrev_id_width`;
  - asserts that the length word at `seek - 4` equals `blk.len` and that the bit offset is 0;
  - counts type ids against each handle's `ty` index;
  - checks the handle bounds;
  - refuses a nested block, an unknown TYPE code or a pre-existing POINTER record;
  - checks NUMENTRY against the id count;
  - patches the MODULE length word.
- `verify(gpa, reference_bc, typed_bc, handles)`: V1-V7 (§4.2). `reference_bc` is the splice's input. Each failure prints `air-splice: typed pointers: V<n> failed: <detail>` and returns `error.Unsupported`.
- `finish(gpa, words, handles, opts)`. `opts.normalize_ce_casts` defaults to false (contingency C1). `finish` runs `normalizeCeCasts` when the option is set, then `splice`, then `verify` against the splice's input.
- `normalizeCeCasts(gpa, bc)`: contingency C1.
  - It re-encodes the module-level CONSTANTS block unabbreviated, with CE_CAST `opty := current SETTYPE`.
  - It then patches the length word of that CONSTANTS block and the MODULE length word. V6 guarantees there are no absolute offsets to fix. V4 guarantees that function-level constants blocks hold no CE_CAST.
- Record helpers: `records`, `typeTable` (ids, codes, names), `countCode(bc, block, code)`, `stats` (counts of OPAQUE_POINTER, POINTER, INST_CAST by opcode, CE_CAST, CE_GEP and METADATA VALUE).
- **Tests** (same file):
  - guards G1-G12 and G14, one test each, named "guard Gn: …" (§4.1);
  - "splice: only TYPE_BLOCK pointer records change; every other byte is identical; the module length word is updated; the output is deterministic";
  - "splice: refuses a nested block, an unknown handle, an existing POINTER record and an unknown TYPE code";
  - "verify rejects: a byte changed outside TYPE_BLOCK, a leftover OPAQUE_POINTER, a forward pointee, a synthetic address space";
  - "verify rejects: a forward element reference from a STRUCT_ANON, STRUCT_NAMED, ARRAY, VECTOR or FUNCTION record to anything but a named struct; a POINTER whose pointee is `{}` or `i1`";
  - "verify rejects: a pointer-changing CE_CAST, a CE_CAST whose operand type is the opaque ptr (a 'fixed' Builder), a CE_GEP, an aggregate holding a global, a GLOBALVAR initializer that names a global value id or a CE_CAST";
  - "verify rejects: a METADATA VALUE that names a global directly or is typed differently from its constant; VSTOFFSET, FNENTRY or SYMTAB present";
  - "contingency C1: `finish(.{ .normalize_ce_casts = true })` on a simulated fixed-Builder module runs normalize, splice and verify in that order and passes; the CONSTANTS and MODULE length words match the re-encoded sizes; V1 is checked against the normalized stream; with the option off, the same module fails V4".
- Every guard failure message ends with `std.zig.llvm.Builder changed (Zig re-pin?): see tools/air/typed.zig register entry Gn; fallback: …`.

**`tools/air/metallib.zig`.**
- New `Unpacked { name, stage, bitcode }` and `unpack(gpa, bytes) ![]Unpacked`. It walks the function list and strips the darwin wrapper and padding.
- Test: "unpack returns what pack wrote, for every profile".

**New `tools/air_xcheck.zig`** (its own executable, compiled and unit-tested by `zig build test`).

Usage: `air-xcheck [--expect-pointers=opaque|typed] [--golden [--nondefault]] [--apple] [--reassemble=<out>] [--skip-status=<n>] <lib>…`

- `--expect-pointers` and `--golden` are pure Zig. They never spawn a process, and on their own they exit 0 (all ok) or 1 (any FAIL).
  - `--expect-pointers`: for each blob, `typed` expects 0 code-25 records and at least one code-8 record in TYPE_BLOCK; `opaque` expects the reverse.
  - `--golden` compares the library's sha256 with `tools/air/golden.zig` (`zig_version`, `default_metallib_sha256`). It does so only when `builtin.zig_version_string` equals the recorded version and the build used default options. Otherwise it prints a note and passes.
- `--apple` opts in to Apple's tools. When `xcrun --find air-opt` succeeds, it runs `xcrun air-opt -verify` on each extracted blob with the output discarded, and `xcrun metal-objdump -d` on the library.
  - Each result prints as `ok  air-opt <lib>:<fn>` or `FAIL air-opt <lib>:<fn>: <first stderr line>`.
  - Confirm the exact flags (`-disable-output` or `-o /dev/null`) at implementation.
  - Without the toolchain it prints `SKIP no Metal toolchain`.
- Exit codes: 0 if every check that ran is ok; 1 on any FAIL. If every check that ran passed but an Apple part was skipped, it exits with the skip status: `--skip-status`, default 77, the convention metallib-check uses.
- Look up the pinned spawn API (`std.process`) and `std.crypto.hash.sha2.Sha256` with `zfn`.
- **Tests:**
  - argument parsing and output formatting;
  - the exit-status table (FAIL gives 1; SKIP gives `--skip-status`);
  - the action plan computed from `--golden` or `--expect-pointers` alone contains no spawn;
  - the golden decision table (pinned vs other Zig, default vs non-default options).

  None of them spawn processes.

**New `tools/air/golden.zig`**: the two constants, plus a comment giving the re-record procedure.

**`build.zig`.**
- New `xcheck` step: pass-through arguments, plus `--skip-status=0`. `zig build` fails on any non-zero exit (F-F), so without this flag a machine that lacks the Metal toolchain would fail `zig build xcheck` instead of skipping. The `SKIP` line stays in the output for CI and for people to read.
- New `golden` step: runs `air-xcheck --golden` on the configured `default.metallib`. It passes `--nondefault` when any of `-Dmetal-target`, `-Dpointers` or `-Dshader-optimize` differs from its default. It never spawns a process, so it cannot skip.
- The test step also runs `air_xcheck`'s tests.

**`assembler.zig`**: only `_ = @import("typed.zig");` in the `test {}` block at `:2192`.

**Docs**: `docs/pipeline.md` gets a new "Re-pinning Zig" section (§4.5) with an empty Re-pin log. It states only "guards pass on 0.17.0-dev.2307".

**Gate:**
- `zig build test`: splice artifact at about **115** (90 + 25); `air_xcheck` artifact at about 4.
- The common gate, with `zig build golden` ok.
- `zig build xcheck -- --expect-pointers=opaque --apple zig-out/bin/default.metallib`: air-opt **15/15 ok**, metal-objdump ok. verify-prior-art measured all 15 AIR 2.8 blobs as clean.
- Discrimination control: `zig build xcheck -- --apple /tmp/o15/bin/default.metallib` gives **exactly one** FAIL, copyTextureKernel `invalid AIR function air.get_read_sampler` (measured by verify-prior-art).
- Skip path: `DEVELOPER_DIR=/nonexistent zig build xcheck -- --apple zig-out/bin/default.metallib` prints `SKIP no Metal toolchain` and exits 0. `DEVELOPER_DIR=/nonexistent zig build golden` passes and spawns nothing.

### Phase 2: typed lowering and sampler registration, proven on macos26 (3.25 d)

**`tools/air/target.zig`.**
- New `pub const Pointers = enum { @"opaque", typed };` and `pub const Verification = enum { unverified, upgrader_only, native_virtual, native };`.
- New `pub const SamplerWord = enum { pair, tagged };` and `Profile.sampler_word: SamplerWord = .pair`. **Every profile is `.pair` in this phase**; Phase 3 sets `.tagged` for macos14/13 only if its rule holds (D9, D10).
- New `Profile.typed_verification: Verification = .unverified`.
- New functions:
  - `verification(p, pointers)`: opaque returns `.native` when `verified`, else `.unverified`;
  - `pointersReason(p, pointers) ?[]const u8`: opaque returns `unverifiedReason(p)`; typed returns non-null exactly when the typed verification is `.unverified`;
  - `caveat(p, pointers)`: non-null for `upgrader_only` and `native_virtual`.
- `unverifiedReason(p)` keeps its signature. Its text is rewritten only after Phase 3.
- **Tests:**
  - "typed verification table holds only measured entries": pins the per-phase values, so every flip has to edit this test in the same commit;
  - "verification, pointersReason and caveat agree";
  - "sampler_word is `.pair` on every profile" (Phase 3 replaces it with the AIR-minor rule).

**`tools/air/intrinsics.zig`.**
- Each texture row gains `typed_proven: bool`: true for sample, read, get_read_sampler and write; false for gather, sample_grad, get_width and get_height.
- New:
  - `typedPointee(param_text) ?enum { texture_2d, sampler }` (`ptr addrspace(1)` gives texture, `ptr addrspace(2)` gives sampler);
  - `atomicElement(name) ?Type` (a `.i32` suffix gives i32; anything else gives null). It is consulted only for `air.atomic.*` declarations that take pointer parameters;
  - `typed_atomics`: the nine `air.atomic` names declared at gpu.zig:195-203, which the 42 checks execute (F-E);
  - `texture_2d_struct = "struct._texture_2d_t"` and `sampler_struct = "struct._sampler_t"`.
- **Test:** "typed pointees: textures and samplers by position; atomics by `.i32` suffix, accepted only for the nine executed forms; `air.atomic.fence` (no pointer parameter) is not an element-typed atomic; others refused; typed_proven set".

**`tools/air/assembler.zig`.**

*Module level:*
- `:127` `assemble`: now returns `assembleWith(…, .@"opaque")`.
  - New `pub fn assembleWith(gpa, ir, comptime opts, profile, pointers) Error![]u8`. It builds `typed.Handles` only in typed mode, calls `buildWith`, then `toBitcode`, then `typed.finish` (typed) or a dupe (opaque, unchanged).
- `:137` `build`: an opaque wrapper around `buildWith(b, gpa, ir, opts, profile, .@"opaque")`. `buildWith` takes `mode: union(enum) { @"opaque", typed: struct { handles: *typed.Handles, diag: ?*typed.Diag } }`.
- `:196` metadata references become `Constant`s:
  - opaque: `func.toConst(b)` and `g.toConst()`, identical to what `lowerNode` built before;
  - typed: `noopRef(fn, fn_ty, .default)` and `noopRef(g, g.typeOf(b), g.addr_space)`.
- `:255` `ModuleState` gains `profile`, `typed: ?*typed.Handles = null`, `diag: ?*typed.Diag = null`, `init_refs_global: bool`, and `failTyped(c, id, msg)` / `failInternal(c, msg)`. Both print through `fail()` with the prefix `typed pointers: R<n>:` (or `typed pointers: internal:`), set `diag.kind` and `diag.id`, and return `error.Unsupported`.

*Global definitions and declarations:*
- `:434` `defineGlobal`:
  - replace the `[2 x i64]` substring registration at `:487` with D10's rule, switching on `m.profile.sampler_word`. Under `.pair`, register the global when the parsed value type is exactly `[2 x i64]` (`ty == arrayType(2, .i64)`). Under `.tagged`, register nothing here, because `lowerCall` registers by use. Both branches land in this phase;
  - typed: **R6** when the initializer refers to a global or function. While an initializer is parsed, the `@` and `getelementptr` arms of `parseConst` set `m.init_refs_global`. This covers the scalar case `@p = constant ptr @q`, which `checkElement` never sees. (Today `setInitializer` at `:474-477` accepts it unchecked.)
- `:491` `declare`:
  - typed `air.atomic.*` **with pointer parameters**: when the name is in `typed_atomics`, pointer parameters become `handle(atomicElement(name), realSpace)`. Otherwise it records `Define.typed_refusal = .R10`. That covers `.s.i32` until its probe runs, `.f32`, and every other form;
  - typed `air.atomic.*` with no pointer parameter or return (`air.atomic.fence(i32, i32, i32)`) is declared unchanged, with no refusal;
  - any other typed `air.*` or `llvm.*` declaration with pointer parameters or a pointer return records `.R8`. The exceptions are texture intrinsics and `llvm.mem*`, which R3 refuses at the call;
  - refusals are raised **at the call**, so an unused declaration never fails.
- `:640` `textureFnType`, typed: pointer parameters and returns come from `typedPointee` as `handle(namedOpaque(…), 1 or 2)`. This replaces the prototype's string compare.
- `:655` `memIntrinsicFor`, typed: R3.
- `:756` `parseType` ptr branch: `if (m.typed) |t| return t.canon(space)`. A vector of pointers is R11.
- `:788` `parseConst`:
  - the `@` arm returns the raw global constant and sets `m.init_refs_global` while an initializer is being parsed;
  - `getelementptr` (`:832`) is R1 in typed mode, or R6 inside an initializer;
  - aggregate or vector literals with a global or function address among their elements are R6. This uses typed wording in `checkElement` (`:904`) but the same `error.Unsupported`.

*Function level:*
- `:1087` `FunctionState` gains:
  - `view(v, pointee, c)`: the cast plus K2. Before building the cast it refuses a zero-sized pointee (`{}`, `[0 x T]`) with R17 and an `i1` or `<N x i1>` pointee with R18;
  - `enterGlobal(k, c)`: `view(k, .i8)`, allowed only for a global variable. A function gives R12;
  - `operandInfo(c, ty) → { value, global: ?Global.Index }`, which `operandOfType` wraps.
- `:1118` `define`, typed: K1.
- Lowering a reached non-entry `define`, typed: R13 when a parameter or the return type is, or contains, a pointer (§3 row 11).
- `:1283`/`:1314` phi and `:1816` `phiPointerType`, typed: the `@` and `getelementptr` arms return `canon(realSpace(base global))` without building constants. In Phase 2 every pointer phi is still refused: R2 when an incoming value is a global or constant expression, R14 otherwise.
- Pointer `select`, pointer `icmp`, and pointer-typed `null`/`undef`/`poison` operands, typed: R14 (row 12).
- `ptrtoint` and `inttoptr`, typed: R15 (row 13).
- `insertvalue`/`extractvalue` on an aggregate type that contains a pointer, typed: R16 (row 14).
- `:1853` `resolvePhis`, typed: an incoming value that is not local and not null/undef/poison gives R2.
- `:1423` GEP: `view(base, src)`, the GEP, then `define(view(g, .i8))`.
- `:1442` load and `:1461` store:
  - `view(ptr, T)`;
  - a `T` that is or contains a pointer gives R5;
  - `atomic` gives R4.
- `:1479` atomicrmw and `:1497` cmpxchg: R4.
- `:1516` alloca: T4 and R7.
- `:1626` `lowerCall`:
  - raise `Define.typed_refusal` (R8, R10);
  - a texture row with `typed_proven == false` gives R9;
  - texture branch (`:1737`) and general path (`:1760`): an argument that is a pointer in the same real space as its parameter is `view`ed to the parameter's pointee. The existing `p != at` checks then run unchanged;
  - **both modes, A5's sampler part:** a global at `ti.sampler_param` (from `operandInfo`) whose value type is not exactly `[2 x i64]` under `.pair`, or not `i64` under `.tagged`, is refused. F-G shows no existing test is affected;
  - **both modes, `.tagged` only:** that global is then appended to `m.samplers`, de-duplicated, in first-use order (D10). Under `.pair`, `defineGlobal` has already registered it, since A5 has just checked that its type is `[2 x i64]`.
- `:1789` `operandOfType`, typed: a global constant goes through `enterGlobal`.

*Header doc comment:* add T1-T7, the G1 reason and the refusal list.

*Test helpers (D18):*
- `typed_expected: []const struct { key: u64, id: typed.Refusal, note: []const u8 }`, where `key = std.hash.Wyhash.hash(0, entry ++ "\x00" ++ module)`.
  - It is filled once in Phase 2 by running the whole corpus in typed mode. On an unexpected result the helper prints `typed: unexpected R<n> for module key 0x…` (or `…expected R<n>, got success`), so entries are copied from the output, never guessed.
  - The table is expected to include D5.1 and D5.7 (R1), the two D1 phi-over-globals tests (R2), D5.10 (R3), D5.11 (R4), D5.12c's `get_width` (R9), D5.8's sret helper and the D5.13 `ticket` helper (R13), the D1 SSA-pointer phi (R14), and the D1 insertvalue module (R16). The committed table is whatever the run shows, reviewed against this list.
- `:2204` `testRoundTripAs`:
  - the opaque part is unchanged and returns the opaque `Builder.print` text as before;
  - **plus** a typed run of the same module with the default profile and a `Diag`:
    - on success: the module must not be in `typed_expected`, `typed.finish` must pass, and there must be 0 code-25 records;
    - on `error.Unsupported`: the module must be in `typed_expected`, with `diag.kind == .construct` and `diag.id` equal to the listed id;
    - anything else fails the test, including a listed module that now succeeds ("remove the entry").
- `:2223` `testFailsAs`: additionally expects `error.Unsupported` in typed mode. `diag.kind` must be `.none`, meaning the same pre-existing refusal fired as in opaque mode. The exception is a module listed in `typed_expected`, whose `diag.id` must then match the listed id.

*New tests*, which assert on records through `typed.stats` and `typeTable`:
- "typed: kernel buffers are i8 pointers; loads, stores and GEPs go through INST_CAST opcode 11; GEP results return to i8";
- "typed: the sret alloca is allocated in address space 0 and cast to i8*";
- "typed: globals enter through instruction bitcasts; metadata uses no-op constant bitcasts; the CE_CAST count is 1 + samplers; no function-level CE_CAST";
- "typed: Apple-shaped texture, sampler and atomic declarations (STRUCT_NAME spellings, parameter type ids)";
- "typed: a module calling `air.atomic.fence` (no pointer parameters) assembles, splices and verifies, and the declaration is unchanged";
- "typed: each refusal R1-R18 names itself and sets Diag.construct and Diag.id";
- "typed: an unexecuted air.atomic form (`.s.i32`, `.f32`) is refused with R10 at the call, not at the declaration";
- "typed: `@p = constant ptr @q` is refused with R6 in defineGlobal";
- "typed: two builds are byte-identical";
- "samplers under `.pair` are exactly the `[2 x i64]` globals: a `{ [2 x i64], i8 }` global is not listed; a sampler that reaches the intrinsic through `select` or a helper argument is listed". This test runs in opaque mode; the typed run of that module is refused (R14 or R13) through `typed_expected`;
- "A5 sampler part under `.pair`: a global at a sampler operand whose value type is not exactly `[2 x i64]` (for example `{ [2 x i64], i8 }`) is refused, in both pointer modes";
- "sampler_word `.tagged` (a synthetic profile with `air_minor = 6` and `sampler_word = .tagged`): an `i64` sampler global at a direct sampler operand is listed by use, de-duplicated, in first-use order; `defineGlobal` registers nothing; a `[2 x i64]` global at a sampler operand is refused (A5)". No shipped profile uses `.tagged` until Phase 3, so this test is what exercises that branch here.

**`tools/air/metadata.zig`.**
- `:539` `lowerModule(b, fm, func: Builder.Constant, samplers: []const Builder.Constant, profile)`.
- `:563` `lowerNode(…, func: Builder.Constant)`, with `.func => b.metadataConstant(func)`.
- The text printer (`:490`) is unchanged (D13).

**`tools/air_splice.zig`.**
- `:64` `Cli.pointers: target.Pointers = .@"opaque"`.
- `:82` `parseCli`: accept `--pointers=opaque|typed`, with a new `CliError.UnknownPointers`. Refuse when `pointersReason(profile, pointers) != null` and `--allow-unverified` is absent (`UnverifiedTarget`). For opaque this is today's logic.
- `:116` `main`:
  - print `pointersReason` as a warning when building unverified, and `caveat` as `air-splice: note: …`;
  - call `assembler.assembleWith(…, cli.pointers)`;
  - in typed mode, prepend `; air-splice: the .metallib holds typed-pointer bitcode; this text is the assembler's opaque-pointer input (cross-check with zig build xcheck)` to the text output.
- Update the usage string and file header.
- **Test:** "--pointers parsing and the refusal matrix". The existing CLI test is untouched.

**`build.zig`.**
- `const pointers = b.option(air_target.Pointers, "pointers", "Pointer representation of the bitcode: opaque (default) or typed");`
- `run_splice.addArg(b.fmt("--pointers={t}", .{p}))` **only when the option is set**, so the default argv and cache keys do not change.

**`tools/metallib_check.zig`.** The info line gains `(profile macos15, typed pointers: <caveat or verified>)`, computed with `container.unpack` and `typed.countCode`. The CI greps (`^info device`, `^SKIP`, `^FAIL`) are untouched.

**Gate:**
- `zig build test` ≈ **134**. Every existing module also runs typed. It either succeeds or fails with exactly the id the expected-refusal table lists for it.
- The common gate.
- `zig build metallib -Dpointers=typed -Dallow-unverified-target=true --prefix /tmp/t26`, then `zig build check -- /tmp/t26/bin/default.metallib`: **42 ok**, including `render vertexShader + fragmentShader (constexpr sampler): 0 mismatches` and `dispatch copyTextureKernel: 0 mismatches`. This also re-measures D5 through the three sret renders.
- `zig build xcheck -- --expect-pointers=typed --apple /tmp/t26/bin/default.metallib`: air-opt **15/15** clean, metal-objdump ok.
- Determinism: rebuild into `--prefix /tmp/t26b` with fresh `--cache-dir /tmp/zc2 --global-cache-dir /tmp/zgc2`, so air-splice really runs again instead of being replayed from the cache. Then `cmp` both `default.metallib` and `shader.air.ll` against `/tmp/t26`. If the text were equal and the bytes different, that would isolate the assembler from nvptx codegen.
- Over `/tmp/t26/bin/shader.air.ll`, `grep -cE 'atomicrmw|cmpxchg|load atomic|store atomic|getelementptr[a-z ]*\(|@llvm\.mem|load (<[0-9]+ x )?i1>?,|store (<[0-9]+ x )?i1>? |phi ptr|select i1 [^,]*, ptr|icmp [a-z]+ ptr|ptrtoint|inttoptr|(insert|extract)value [^,]*ptr'` is **0**. F-E measured all of these as 0 on HEAD's text. A successful typed build already implies the result; the grep cross-checks the refusal detectors against the text.

**Exit:** in the same commit, `macos26.typed_verification = .native`. The commit message records the date, macOS build, GPU and log path.

### Phase 3: AIR-version forms and typed macos15/14/13 (1.5 d)

**`tools/air/target.zig`.**
- New `ReadTexture = enum { sampler_offset, lod_access }`.
- `Profile.read_texture` is `.lod_access` for macos15/14/13 (apple-encoding §10, oracle, prior-art verify).
- `Profile.sampler_word` (added in Phase 2, `.pair` everywhere) becomes `.tagged` for macos14/13, **subject to the rule below**.
- **Test:** "AIR-version forms follow the AIR minor": `lod_access` iff `air_minor < 8`, and `tagged` iff `air_minor < 7`, unless the rule below keeps `pair`. It replaces Phase 2's "sampler_word is `.pair` on every profile".

**`tools/air/intrinsics.zig`.**
- Rows gain `air_minor_min: u16 = 0` and `air_minor_max: u16 = 99`.
- A new row `air.read_texture_2d.v4f32` for AIR ≤ 7:
  - `.params = { tex_ptr, "<2 x i32>", "i32", "i32" }`;
  - `.from_28 = { .keep = {0,2,4,5}, .read_sampler_operand = 1, .zero_operand = 3 }`;
  - `wraps_status`, `typed_proven = true`.
- `air.get_read_sampler` gets `air_minor_min = 8`.
- New `textureIntrinsicFor(name, air_minor)`. `textureIntrinsic(name)` stays the 2.8 view, so the existing test passes. `textureDeclareText(gpa, ti)` is unchanged; the 2.7 row is just another row.
- **Test:** "texture forms per AIR version: 2.8 read has 6 operands and get_read_sampler; ≤2.7 read has 4, keeps (0,2,4,5), no get_read_sampler".

**`tools/air_splice.zig`.** The text rewrite is the only place these forms are transformed (D9).
- `Error` gains `Unsupported`.
- `ConvertOptions` gains `samplers_out: ?*std.ArrayList([]const u8) = null`, so existing calls are unchanged.
- Pre-scan `samplerUses(gpa, src)`, used only under `.tagged`: the names found at a texture call's `sampler_param` **and** defined as `constant [2 x i64]`. The existing `samplerGlobals` becomes the candidate filter, so its test still passes. The names are returned through `samplers_out` in first-use order. Under `.tagged`, `main` prints these through `printModule`; under `.pair` it keeps printing `samplerGlobals`, which is today's behaviour (D10).
- Declarations (`:386-395`) use `textureIntrinsicFor(name, profile.air_minor)`. At ≤2.7 the `air.get_read_sampler` declaration is dropped.
- Global lines (`:396-413`): the `addrspace(2)` insertion is unchanged. For a `samplerUses` name under `sampler_word == .tagged`, the line is printed as `… addrspace(2) constant i64 <signed(W0 | 1<<63)>, align 8`. Refusals, all under `.tagged`:
  - W1 ≠ 0 (a LOD bias): A3;
  - W0 already has bit 63 set: A3;
  - a `constant [2 x i64]` global referenced anywhere other than as a direct sampler operand of a texture call: A4. This covers three cases. A sampler global that is also referenced elsewhere, and a sampler that reaches the intrinsic only through select, phi or a helper, cannot be retagged. A data table of the same type cannot be told apart from a sampler.
- Body lines: `sanitizeBodyLine(gpa, line)` stays as a 2.8 wrapper over the new `sanitizeBodyLineIn(gpa, line, *BodyRewrite) Error!?[]u8`, where `null` means "drop the line". At ≤2.7:
  - `%x = … @air.get_read_sampler()` lines are dropped and `%x` is recorded;
  - `rewriteTextureCall` keeps operands (0,2,4,5) only if operand 1 is a recorded `%x` and operand 3 is `zeroinitializer` or all zero; otherwise A1;
  - after each function, any other use of a recorded `%x`, matched at a token boundary, is A2.
- **Tests:**
  - "printed IR follows the AIR version: macos15 4-operand read and no get_read_sampler; macos14 `constant i64 -9188470239253755319` for word `0x7bff0000080a49` (Apple's macOS 14 value, apple-encoding §10); macos26 unchanged";
  - "AIR ≤2.7 refusals: non-zero offset, foreign sampler operand, stray get_read_sampler use";
  - "AIR ≤2.6 refusals (`.tagged`): LOD-bias word, bit 63 already set, a `[2 x i64]` constant used other than as a direct sampler operand (including one passed through `select`)".

**`tools/air/assembler.zig`** (validation only).
- `declare` uses `textureIntrinsicFor`.
- A5's read-form part, in `lowerCall` (both modes): at air_minor ≤ 7, a 6-operand read or any `air.get_read_sampler` call is refused.
- The sampler-word branches (D10's registration and A5's sampler part) exist since Phase 2 and are not changed here. Phase 3 only sets the field for 14/13, so if the sampler-word rule falls back to `.pair` at 14/13, nothing in the assembler changes.
- **Tests:**
  - "AIR-version validation refuses the 2.8 read form and `air.get_read_sampler` below AIR 2.8";
  - "typed: the converted macos15 module assembles, splices and verifies (4-operand read)";
  - "the converted macos14 module under `sampler_word = .tagged` (a `constant i64` sampler) assembles in both pointer modes and lists the sampler by use". It uses a copy of the macos14 profile with `.tagged` set explicitly, so it holds whichever way the rule goes.

**Sampler-word rule (U9).**
1. Build typed macos14 and macos13 with `.tagged`. Run the full check and `xcheck --apple`.
2. **Adopt `.tagged`** iff both give 42/42 with `constexpr sampler: 0 mismatches over 64 pixels` **and** air-opt reports 15/15 clean.
3. Otherwise:
   - set `.pair` for 14/13. D10's registration, A4 and A5 follow the field automatically;
   - re-measure `.pair` with the production binary (the critic's prototype measured 42/42 with `.pair`);
   - classify the known air-opt message `metadata AISamplerState is corrupted` on fragmentShader as a `KNOWN` line in xcheck for `pair` at air_minor ≤ 6;
   - document the disagreement.

**Gate:**
- `zig build test` ≈ **142**.
- The common gate. macos26 bytes and text are unchanged by construction; typed macos26 still gives 42 ok.
- For each t in {macos15, macos14, macos13}:
  - `zig build metallib -Dmetal-target=$t -Dpointers=typed -Dallow-unverified-target=true --prefix /tmp/t-$t`, then check: **42 ok**;
  - `xcheck --expect-pointers=typed --apple`: air-opt **15/15** (for 14/13, only under `.tagged`).
- Controls from the same binary:
  - opaque macos15 still fails at `scaleKernel` with `Failed to upgrade function bitcode`, so pointers remain the cause even with the 2.7 forms;
  - `xcheck --apple` on opaque macos15 no longer reports `invalid AIR function air.get_read_sampler`. This verifies the AIR-version fix on its own.

**Exit:** `typed_verification = .upgrader_only` for macos15/14/13. The `unverifiedReason` text is updated to point at `-Dpointers=typed`.

### Phase 4: CI rungs and docs for the measured state (1.75 d, plus CI turnaround)

- The `.github/workflows/ci.yml` changes in §6.1, arm64 legs only (§0, constraint 6).
- The docs in §7, limited to what Phases 1-3 measured.
- New `zig build install-check`, which installs `metallib-check` for the field-test bundle.
- The field-test bundle also carries Apple-compiled typed control libraries for 13/14/15 (§6.1 step 9). That way a field or local-VM run has Apple's typed control in the same run.
- `air-xcheck --reassemble=<out.metallib>` (Apple's writer applied to our IR):
  - `xcrun air-opt -S` on each blob, then `xcrun air-as`;
  - strip the darwin wrapper, because air-as output is already wrapped and a double wrap crashed MTLCompilerService (verify-builder-feasibility);
  - `metallib.pack`, with the profile from `readStamp` and the names and stages from `unpack`;
  - without the toolchain it prints `SKIP` and ends with the skip status, like `--apple`.
  - Confirm the exact flags at implementation.
- Optional, not a gate: the local VM rung (§6.4). It adds 0.5-1 d when done.

**Gate:**
- One green run on both arm64 legs, with the new summary tables filled in.
- The determinism job, on `macos-latest`, is green.
- Docs cite only logged runs.
- Evidence flips follow in separate commits once the promotion rule is met (§6.3).

### Phase 5: probe libraries and refusal lifts (4.75-5.75 d)

**New `src/engine/probe_shader.zig`**, beside `air.zig` and `gpu.zig`, with its own `functions` manifest. It is never embedded in the app. It holds one kernel per construct. Everything else in a kernel must already be accepted in typed mode (§3 rows 1-10, the class generalisations a-d, or an earlier lift), so that a construct's library is refused only for its own construct. When Zig emits a construct only together with another refused one, the kernel lists that one in `needs` (below).
- `probeConstGep`: constant-index reads of comptime tables, i.e. constant-GEP operands (R1);
- `probePhiGlobals`: a pointer selected by parity between two comptime tables, i.e. a phi over globals (R2);
- `probeMemcpy`: a large struct copied from a comptime table (`llvm.memcpy`, R3);
- `probeNativeAtomics`: `@atomicRmw` and `@cmpxchgStrong` on a device `u32` (R4);
- `probePtrStore`: a pointer stored to threadgroup memory and loaded back (R5);
- `probeGetWidth`, `probeGetHeight`, `probeGather` and `probeSampleGrad`: one kernel per texture row (R9). Each row is its own `typed_proven` entry and is lifted on its own evidence, so each gets its own library. A failing row cannot hide another, and a row that passes only at AIR 2.8 does not hold the others back;
- `probeSignedAtomic`: `air.atomic.global.*.s.i32` and `air.atomic.local.*.s.i32` calls (R10);
- `probeHelperPtr`: a `noinline` helper that takes and returns a pointer (R13);
- `probeSsaPhiSelect`: a phi and a select over two SSA buffer pointers, plus a null test of an optional pointer (R14);
- `probePtrInt`: an `@intFromPtr`/`@ptrFromInt` round trip on a buffer pointer, with results read back (R15, new);
- `probePtrAggregate`: a `{ pointer, length }` aggregate built and taken apart inside one function (R16, new).
- `pub const probe_expect` gives each kernel, keyed by kernel name, four fields:
  - `id`: the refusal the kernel lifts, as a string such as `"R4"` (the shader module cannot import `typed.zig`). Several kernels can share an id (the four R9 kernels), so no rule is keyed on it. Dependencies and states are keyed on kernel names;
  - `needs`: the **names of other probe kernels** whose constructs Zig emits only together with this one. For example, if a pointer aggregate appears only across a helper call, `probePtrAggregate` needs `probeHelperPtr` (R13). It stays empty unless the presence grep shows the construct cannot be emitted alone;
  - `status`: `.refused`, `.lifted` or `.out_of_scope`. `.out_of_scope` is set only in a commit that cites the M3 `probe_matrix.sh gpu` log in which the kernel's opaque control failed;
  - `min_air`: for a `.lifted` kernel, the lowest AIR minor from which the lift holds. It is 5 when the lift holds at all four typed profiles, and 8 when it holds only at macos26 (the Q6 split). A profile is **lifted for the kernel** when `status == .lifted` and the profile's `air_minor >= min_air`; every other profile still refuses. `min_air` is null unless the kernel is lifted. It is set from the M3 `gpu` log, never guessed:
    - a lift branch starts at 5;
    - if macos26 passes on the M3 but a lower profile fails, the lift commit raises `min_air` to the lowest minor from which every profile passed, and cites the log;
    - if macos26 fails, there is no lift.
- **Expected refusal at a profile.** At a profile that is not lifted for a kernel, the kernel's typed build must fail with an id from its **allowed set** there. The allowed set is the kernel's own `id`, plus the `id` of every `needs` kernel that is not lifted at that profile.
  - The set is computed from `probe_expect`, not stored. A lift therefore never edits another kernel's entry, and a need lifted only from AIR 2.8 simply keeps its id in the set below 2.8.
  - An id outside the set means the kernel holds another refused construct.
- R18 (`i1` memory) gets a probe kernel only if Zig ever emits `i1` memory access (Q8). Until then R18 stays.

**`tools/air_splice.zig`.**
- `--entry=<name>[,<name>…]` assembles and packs only the named manifest entries, in manifest order. An unknown name gives `CliError.UnknownEntry`. The text output still holds the whole module. With no flag, every entry is packed, so the default argv, cache keys and bytes do not change.
- `--list-entries` prints the manifest's entry names, plus each kernel's `probe_expect` fields when the shader module declares them, then exits 0.
- **Test:** "--entry filters the packed functions, keeps manifest order and refuses unknown names; --list-entries".

**`build.zig`.**
- A second `air-splice` executable with `shader` = the probe module.
  - `zig build probe -Dprobe=<kernel> [-Dmetal-target] [-Dpointers]` passes `--entry=<kernel>` and installs `zig-out/probe/<kernel>/probe.metallib` and `probe.air.ll`.
  - Without `-Dprobe`, it builds one library holding every probe kernel. That library is used for the presence grep and for the end state where everything is lifted.
  - `zig build probe-list` runs `--list-entries`.
  - The probe module's nvptx compile is shared, so a build for one kernel re-runs only air-splice.
- A second `metallib-check` with `shader` = the probe module: `zig build check-probe -- [--only=<kernel>] <lib>`. Optionally, `zig build install-check-probe` installs it for the field or VM bundle.
- Probe libraries never touch `default.metallib` or the default build graph.

**`tools/metallib_check.zig`.**
- New `--only=<name>[,<name>…]`, in both checkers. It restricts function lookup, compute and render pipelines, and self-tests to the named manifest entries. That lets a library holding one kernel be checked, and stops one failing construct from hiding another. An unknown name exits 2. With no flag, behaviour and output are unchanged, so the default checker stays at 42 lines.
- Probe self-tests, gated at comptime by `findFunction` and at run time by `--only`:
  - `testProbeConstGep`, `testProbePhiGlobals`, `testProbeMemcpy`, `testProbeNativeAtomics`, `testProbePtrStore`, `testProbeGetWidth`, `testProbeGetHeight`;
  - `testProbeGather`, `testProbeSampleGrad`, `testProbeSignedAtomic`, `testProbeHelperPtr`, `testProbeSsaPhiSelect`, `testProbePtrInt`, `testProbePtrAggregate`.

  Each dispatches with known inputs and compares the results it reads back. By contrast, `--kernel=` only builds a pipeline (F-F).
- The test step gains a run of metallib_check's argument-parsing test. It is pure: no device is created in tests.

**New `tools/probe_matrix.sh`.** A bash script used by the M3 gate and by CI; the build itself never needs it. It reads each kernel and its `probe_expect` fields from `zig build probe-list`, handles every kernel in separate processes, one at a time, and prints one table row per kernel: presence, refusal state, opaque control, and typed result per profile. GPU results are read from `^ok`/`^FAIL`/`^SKIP` lines, the same way CI's `state()` reads them. It has three modes, and only `static` is ever strict in CI:

1. **`static`: presence and refusal state.** Pure Zig, no GPU; strict on the M3 and in CI.
   - Presence: the per-kernel grep listed in the gate below, over the all-kernels opaque `probe.air.ll`.
   - Every kernel is built typed at **each of the four profiles**, and each result is checked against that profile's state for the kernel:
     - At a profile lifted for the kernel (`.lifted` and `air_minor >= min_air`), the build must succeed, splice and verify included.
     - At any other profile (`.refused`, `.out_of_scope`, or below `min_air`), the build must fail with an id from the kernel's allowed set there.
       - An id outside the set means the kernel holds another refused construct. Fix the probe, or name the kernel that lifts that construct in `needs`.
       - A build that succeeds means the detector missed the construct or `min_air` is stale. Either is a bug.
   - Consistency of `probe_expect`:
     - every `needs` name is a probe kernel, and the `needs` graph has no cycle;
     - `min_air` is set exactly when `status == .lifted`, and is one of the profiles' AIR minors (5-8);
     - every `needs` kernel of a `.lifted` kernel is `.lifted`, with a `min_air` no higher than the dependent's. So a dependent is never lifted at a profile where one of its needs still refuses.
   - Exits 1 on any presence miss, refusal mismatch or inconsistency, and 0 otherwise.
2. **`gpu`: the M3 run.** It is the only source of scope decisions.
   - **Opaque control.** Build the kernel's opaque macos26 library and run `check-probe -- --only=<kernel>`. A FAIL here on the M3 marks the construct **out of scope** for this change set:
     - it stays refused in typed mode, and a commit that cites this log sets its `status = .out_of_scope`;
     - the result table and the docs say "opaque control failed";
     - the failure is filed as an opaque-mode finding, because either the opaque assembler or the probe kernel is wrong. Native `atomicrmw`, for example, has never been executed in opaque mode either.
   - **Typed runs** for every `.lifted` kernel, at each profile lifted for it (`air_minor >= min_air`).
     - On a lift branch, the kernel being lifted is already `.lifted` with `min_air = 5`, so all four profiles run.
     - For each profile, build the typed library, then run `check-probe -- --only=<kernel>` and `xcheck --expect-pointers=typed --apple`.
     - A profile below `min_air` is not run. Its row reads "refused below AIR 2.<min_air>", and the static mode checks that refusal.
   - Exits 1 on a typed FAIL at a profile lifted for a kernel (on a lift branch, the signal to raise `min_air` or drop the lift), on an opaque-control FAIL for a `.lifted` kernel (a regression), or on any SKIP (no device, so the gate did not run). An opaque-control FAIL for a kernel that is not lifted is a scope finding: it is printed, and the exit status stays 0.
3. **`gpu --ci --control-state=<state>`: CI's informational run.** It always exits 0.
   - **No opaque controls.** GitHub's Apple Paravirtual device rejects all opaque output, Apple's own control included (F-B, §6.2 baseline). An opaque row there says nothing about a construct, and would mark every construct out of scope.
   - **Runner check:** Apple's typed control from the same leg's control step (`CONTROL_STATE`, §6.1 step 1). If it is not `ok`, every row reads "runner cannot answer" (§6.2, last row), and no probe library is run.
   - Otherwise, for each `.lifted` kernel, it builds typed libraries for the profiles that are lifted for it and whose major is at most the runner's. It checks each with `check-probe -- --only=<kernel>` and classifies the result with `state()`. A kernel that is not lifted gets the row "refused (static check)", and so does each profile below a lifted kernel's `min_air`.
   - It never edits `probe_expect` and never marks a construct out of scope. Its rows are data, read with §6.2. A probe row becomes strict only under §6.3.

**`tools/air/assembler.zig` lowerings.** A lift happens only after its own probe has passed its opaque control and its typed runs on the M3 (`probe_matrix.sh gpu`). **Lifts are independent unless a kernel declares `needs`.** A kernel with `needs` is lifted only after every kernel it names, and never from a lower AIR minor than they are, because its typed library cannot build where a need still refuses. If a needed kernel is out of scope, the dependent stays refused and the docs say "blocked by <kernel> (R<n>)".

A lift commit does three things:
1. It sets its kernel's `status = .lifted` and `min_air`.
2. It deletes its `typed_expected` entries (D18).
3. When `min_air > 5`, it keeps the refusal for the profiles below `min_air`. For most refusals, typed.zig records a minimum AIR minor per lifted id (`lifted_from`). Texture rows do the same through their per-AIR `typed_proven` rows.

No other kernel's entry changes, because allowed refusal sets are computed. `probe_matrix.sh static` checks the state against every profile's build, and `zig build test` fails on a stale `typed_expected` entry. The lowerings:
- `lowerConstGep(text)`: `enterGlobal(base)`, then `view(src)`, then an instruction GEP with the constant indices, then a view back to i8. Nested constant GEPs recurse. `operandOfType` routes a leading `getelementptr` here. R1 stays for initializers (R6).
- Phi incoming globals and constant GEPs: `resolvePhis` sets `wip.cursor = .{ .block = entry, .instruction = 0 }` and materialises each distinct incoming text once. It keeps one `Value` per text, so duplicate `switch` edges share it (G13). R2 is lifted.
- `llvm.mem*`: in typed mode, always routed through `memIntrinsicFor`, with the name from `intrinsics.typedMemIntrinsicName` (`llvm.memcpy.p<N>i8.p<M>i8.i64`, `llvm.memset.p<N>i8.i64`) and canonical parameters. R3 is lifted.
- Pointer load/store: the address is viewed to `handle(canon(N), AS)`. R5 is lifted.
- Native atomics: the pointer is viewed to the value (or compare) type. R4 is lifted.
- Rows 11-14 (R13-R16): no new lowering is needed, because they use canonical i8 values (T1). The lift removes the refusal once `probeHelperPtr`, `probeSsaPhiSelect`, `probePtrInt` or `probePtrAggregate` passes, respectively.
- Row 15: the `air.atomic` names that `probeSignedAtomic` executed join `intrinsics.typed_atomics`. R10 is lifted for those names.
- `intrinsics.zig`: `typed_proven` flips for each row whose probe passed, only in the AIR-range rows at or above its kernel's `min_air`. A row that passes at 2.8 but not at 2.7 is split by AIR range, and its kernel's `min_air` becomes 8.

**Tests:**
- typed.zig: "guard G13: WipFunction inserts at cursor.instruction".
- assembler.zig:
  - "typed: constant GEP operands become instruction GEPs";
  - "typed: phi incoming globals are materialised once in the entry block; duplicate switch edges share one value";
  - "typed: memcpy is renamed p<N>i8 in STRTAB";
  - "typed: pointer load/store view the address as i8**";
  - "typed: native atomics view the pointer as the value type";
  - "typed: helper pointer parameters, pointer select/icmp, phis over SSA pointers, ptrtoint/inttoptr and pointer aggregates need no extra casts (after the R13-R16 lifts)".
- intrinsics.zig: "typed memcpy mangling p<N>i8".
- air_splice.zig: the `--entry` / `--list-entries` test above. metallib_check.zig: the `--only` parsing test.

**Gate:**
- `zig build test` ≈ **153**. Existing modules whose refusals were lifted now take the typed success path, and their `typed_expected` entries were deleted in the lift commits.
- The common gate.
- `tools/probe_matrix.sh static` exits 0 (strict, no GPU). It checks:
  - **presence**: in the all-kernels `probe.air.ll`, each kernel's own body, or a helper it calls, contains its construct:
    - `getelementptr inbounds (` (probeConstGep);
    - a `phi ptr` over two globals (probePhiGlobals);
    - `@llvm.memcpy` (probeMemcpy);
    - `atomicrmw` and `cmpxchg` (probeNativeAtomics);
    - `store ptr` and `load ptr` (probePtrStore);
    - `@air.get_width_texture_2d` (probeGetWidth) and `@air.get_height_texture_2d` (probeGetHeight);
    - `@air.gather_texture_2d` (probeGather) and `@air.sample_texture_2d_grad` (probeSampleGrad);
    - a `.s.i32` `air.atomic` call (probeSignedAtomic);
    - a non-entry `define` with a `ptr` parameter or return (probeHelperPtr);
    - `phi ptr`, `select i1 … ptr` and `icmp … ptr … null` (probeSsaPhiSelect);
    - `ptrtoint` and `inttoptr` (probePtrInt);
    - `insertvalue`/`extractvalue` over `{ ptr …` (probePtrAggregate).

    If Zig stops emitting one, adjust the probe;
  - **refusal state**, checked at each of the four typed profiles:
    - a profile lifted for a kernel builds, splice and verify included;
    - every other profile refuses with an id from the kernel's allowed set;
    - `probe_expect` is consistent.
- GPU (M3, `tools/probe_matrix.sh gpu` exits 0): every construct's opaque macos26 control is recorded. For every lifted construct, typed runs are ok at every profile from its `min_air` up (all four when `min_air = 5`), and `xcheck --expect-pointers=typed --apple` is clean. A construct whose opaque control fails is out of scope and stays refused.
- All four typed default libraries are re-checked at 42/42, because the lowering code changed.
- Any construct whose probe fails stays refused, and the docs say so. Each construct has its own library and process, so a failure blocks only its own lift and the lifts of kernels that list it in `needs`.

### Phase 6: contingent

Triggered by the CI or VM results (§6.2), a re-pin (§4.5) or a user decision:
- the fidelity layer (§8, Q1). Its step (2) adds named structs with bodies. Every body element type must be interned **before** `opaqueType(name)`, and the body set afterwards. Builder writes types in index order (Builder.zig:14563), and LLVM 15 rejects forward references to anything but named structs. Today the assembler avoids the problem only because it resolves named types structurally. V2's forward-reference rule (Phase 1) already checks STRUCT_NAMED element ids and catches a violation at build time;
- framing experiments, including the macOS 13 variants (§6.2);
- wiring C1;
- 4b, after its spike (§1.5);
- flipping the macos13-15 default to typed.

### Effort summary

| Phase | Work | Days |
|---|---|---|
| 0 | baseline | 0.25 |
| 1 | typed.zig safety net, C1 (built, off), unpack, air-xcheck (pure checks, `--apple`, skip status), golden | 2.5 |
| 2 | typed lowering, refusals R1-R18, expected-refusal table; `sampler_word` (`.pair` everywhere), D10 in both branches, A5's sampler part; typed macos26 42/42 | 3.25 |
| 3 | AIR-version forms keyed on profile fields (read form, text-rewrite retag, A5's read-form part, 14/13 `sampler_word`); typed macos15/14/13 42/42 | 1.5 |
| 4 | CI rungs (arm64 only), docs, reassemble, field bundle with Apple typed controls | 1.75 + CI turnaround |
| 5 | per-construct probe libraries (`--entry`, `--only`), `probe_matrix.sh` with `static`, `gpu` and `gpu --ci` modes, `probe_expect` state (`min_air`) and dependencies, 14 probe kernels (one per texture row), lifts | 4.75-5.75 |
| **Total** | | **14.0-15.0 d**, plus CI iteration. The optional local VM rung (§6.4) adds 0.5-1 d. |

**Size:** about 1,860 production lines, plus about 360 for the probe kernels and the matrix script. With tests, CI and docs, about 3,650-4,050.

| File | Lines (about) |
|---|---|
| typed.zig | 600 code + 500 tests |
| assembler.zig | 340 |
| air_xcheck.zig | 280 |
| air_splice.zig | 210 |
| probe_shader.zig | 200 |
| metallib_check.zig | 200 |
| probe_matrix.sh | 140 |
| intrinsics.zig | 100 |
| target.zig | 60 |
| metallib.zig | 60 |

---

## 3. Construct table (typed mode)

Every typed refusal is `error.Unsupported`, printed as `air-splice: line N: typed pointers: R<n>: <text>` followed by the offending line, with `Diag.kind = .construct` and `Diag.id = R<n>`.

**Implemented in Phase 2/3 and executed by the 42 checks.** Each row is a construct class. The 42 checks execute at least one member of each; the generalisations accepted beyond the executed members are listed in the next table.

| # | Construct | Lowering | Evidence gate |
|---|---|---|---|
| 1 | Entry pointer parameters: buffers in AS 1/2/3, textures, samplers | `canon(AS)` (T1) | typed 42/42 at 26 (P2) and 15/14/13 (P3) |
| 2 | Byte and stride GEPs over SSA pointers | `view(base, src)`, GEP, back to i8 | same |
| 3 | Plain load/store of non-pointer, non-`i1` values | `view(ptr, T)` | same |
| 4 | sret alloca | real AS 0 alloca, then view to i8 (D5) | the three render readbacks |
| 5 | Threadgroup and constant globals as operands | `enterGlobal` INST_CAST | reduce, reverse, twoStage, tgBuf, atomicOps |
| 6 | Constexpr sampler global | `enterGlobal`, then `view(%struct._sampler_t)`; `!air.sampler_states` via `noopRef` | constexpr render, 0 mismatches |
| 7 | Entry function in the stage metadata | `noopRef` CE_CAST to `fnty*` | every pipeline |
| 8 | sample; read (2.8 and ≤2.7 forms); get_read_sampler (2.8); write | Apple-shaped declarations, arguments viewed | renders plus copyTextureKernel |
| 9 | The nine `air.atomic` forms gpu.zig declares (`typed_atomics`): `air.atomic.global.{add,sub,max,min,or,and,xor}.u.i32`, `air.atomic.global.xchg.i32`, `air.atomic.local.add.u.i32`; plus `air.atomic.fence`, which has no pointer parameters and is declared unchanged | `i32 addrspace(N)*` parameters | atomicOps, count |
| 10 | ≤2.6 tagged sampler word (if adopted) | text rewrite; `noopRef` to `i64 addrspace(2)*` | constexpr render at typed 14/13 |

**Accepted by class, without an execution of their own.** Typed mode accepts these because they produce the same kinds of typed records, in the same positions, as members the 42 checks execute. This is an argument, not a measurement (§0, constraint 4); Q24 tracks the risk.

| # | Generalisation | Executed members it relies on | Why the typed records are the same kind |
|---|---|---|---|
| a | GEPs of any shape over SSA pointers: any index count or index width, and any source type except zero-sized ones (`{}`, `[0 x T]`) and `i1`/`<N x i1>` | the byte and stride GEPs of row 2 | each is `view(base, src)`, one INST_GEP and a view back to i8; only the source type id differs. `view` refuses zero-sized source types (R17) and `i1` ones (R18), as in (b). |
| b | Loads and stores of any non-pointer, non-`i1`, non-zero-sized type: scalars, vectors, and aggregates without pointers | the loads and stores of row 3 | one view to `handle(T, AS)` and the same INST_LOAD/INST_STORE records; only the pointee type id differs. Pointer-holding types are R5, `i1` is R18, zero-sized types are R17. |
| c | Allocas in address space 0 of any type, not only sret | the three sret allocas of row 4 | the same real-AS-0 INST_ALLOCA followed by a view (D5) |
| d | Helper (non-entry) functions whose parameters and return hold no pointer | every executed `air.*` call (the call shape) and the 15 entry bodies (the body lowering); the real shader has no helper itself (F-E) | a FUNCTION record whose type holds no pointer, and an INST_CALL with an explicit function type against the callee's reader type `fnty*`, the shape every `air.*` call already has; the body is lowered as an entry body is. A helper with pointers is R13. |

**Refused in Phase 2 because the 42 checks execute no member of their class; lifted in Phase 5 once their own probe executes.** They need no new lowering (canonical i8 values, T1), so the lift only removes the refusal. The real shader contains none of them (F-E). No air-opt or unit-test result counts as evidence for them.

| # | Construct | Refusal text until then | Probe |
|---|---|---|---|
| 11 | Helper (non-entry) function whose parameters or return are, or contain, a pointer | R13: `a helper function taking or returning a pointer has not been run with typed pointers on a GPU yet; build with -Dpointers=opaque for macos26` | `testProbeHelperPtr` |
| 12 | Pointer phi/select/icmp over SSA pointers; pointer-typed `null`/`undef`/`poison` | R14: `a pointer phi, select, comparison or null/undef/poison pointer has not been run with typed pointers on a GPU yet; build with -Dpointers=opaque for macos26` | `testProbeSsaPhiSelect` |
| 13 | `ptrtoint` / `inttoptr` | R15: `ptrtoint/inttoptr have not been run with typed pointers on a GPU yet; build with -Dpointers=opaque for macos26` | `testProbePtrInt` (new) |
| 14 | `insertvalue` / `extractvalue` of aggregates that hold pointers | R16: `an aggregate value holding a pointer (insertvalue/extractvalue) has not been run with typed pointers on a GPU yet; build with -Dpointers=opaque for macos26` | `testProbePtrAggregate` (new) |
| 15 | `air.atomic.*.s.i32` (same shape as `.u.i32`) | R10 (row 25) | `testProbeSignedAtomic` |

**Refused in Phase 2, lowered in Phase 5 once the probe passes**

| # | Construct | Refusal text until then | Phase 5 lowering |
|---|---|---|---|
| 16 | Constant GEP as an instruction operand (tests D5.1, D5.7) | R1: `a getelementptr constant expression is not lowered yet (Builder writes its base type as an opaque pointer, which Apple's typed reader rejects: 'Type mismatch in constant table'); index at run time, or build with -Dpointers=opaque for macos26` | `lowerConstGep` |
| 17 | Phi incoming global or constant GEP (the D1 tests) | R2: `a phi whose incoming value is a global or constant expression needs its bitcast materialised outside the phi, which is not implemented yet; build with -Dpointers=opaque for macos26` | entry-block materialisation (G13) |
| 18 | `llvm.memcpy` / `memmove` / `memset` (D5.10) | R3: `llvm.memcpy/memmove/memset need LLVM 15 mangling (llvm.memcpy.p<N>i8.p<M>i8.i64), not implemented yet; build with -Dpointers=opaque for macos26` | `typedMemIntrinsicName` |
| 19 | Native `atomicrmw`, `cmpxchg`, `load atomic`, `store atomic` (D5.11) | R4: `native atomicrmw/cmpxchg/load atomic/store atomic have no GPU-proven typed lowering yet; use gpu.atomic* (air.atomic.*) or -Dpointers=opaque` | view to the value type |
| 20 | Load/store whose type is or contains a pointer | R5: `loading or storing a pointer value has no GPU-proven typed lowering yet` | address viewed as `i8 addrspace(N)* addrspace(M)*` |
| 21 | `get_width`, `get_height`, `gather`, `sample_grad` | R9: `<name> has not been run with typed pointers on a GPU yet (typed_proven = false in intrinsics.zig); build with -Dpointers=opaque for macos26` | flip `typed_proven` per AIR row (the kernel's `min_air`) |

**Refused permanently under option 4.** They would need a pointer-changing constant bitcast.

| # | Construct | Refusal text |
|---|---|---|
| 22 | Pointer-valued global initializers, whether scalar (`@p = constant ptr @q`) or aggregate; constant aggregates or vectors holding global or function addresses; constant GEPs in initializers | R6: `a global initializer or constant aggregate that holds the address of a global needs a pointer-changing constant bitcast, which Builder encodes in a way Apple's typed reader rejects ('Type mismatch in constant table'); build this shader with -Dpointers=opaque (macos26)`. Enforced in `defineGlobal` for scalars and in `checkElement` for literals. V4 rejects any GLOBALVAR initializer that names a global value id or a CE_CAST. Lifting it is a 4b trigger (§1.5). |

**Refused for lack of reader support or evidence**

| # | Construct | Refusal text |
|---|---|---|
| 23 | Alloca in an address space other than 0 | R7: `an alloca outside address space 0: the bitcode ALLOCA record carries no address space and the reader places it in address space 0` |
| 24 | Unknown `air.*` or `llvm.*` declaration with pointer parameters or a pointer return (raised at the call) | R8: `<name> takes or returns a pointer and has no typed signature (known: air texture intrinsics, the executed air.atomic forms); build with -Dpointers=opaque` |
| 25 | An `air.atomic.*` declaration **with pointer parameters** outside `typed_atomics`: `.s.i32` until its probe runs; `.f32` and other elements permanently, for lack of evidence. Raised at the call. `air.atomic.fence` and other `air.atomic.*` declarations without pointers are not affected. | R10: `<name>: only the air.atomic forms executed on a GPU have a typed signature (i32 addrspace(N)*); build with -Dpointers=opaque` |
| 26 | Vectors of pointers | R11: `vectors of pointers are not supported` |
| 27 | A function used as a value (not a callee) | R12: `a function used as a value has no typed lowering` |
| 28 | A zero-sized access, view or parameter pointee (`{}`, `[0 x T]`) | R17: `a zero-sized memory access has no typed pointer form ({}* pointees crash MTLCompilerService)`. V2 is the backstop. |
| 29 | An `i1` or `<N x i1>` load, store or view | R18: `an i1 memory access has not been run with typed pointers on a GPU (i1* is reported to crash MTLCompilerService); store the value as i8`. It stays refused until a probe exists (Q8). V2 is the backstop. |

**AIR-version and sampler rules (both pointer modes; the text rewrite prints `air-splice: AIR 2.<n> (<profile>): …` plus the line)**

| # | Rule | Where |
|---|---|---|
| A1 | ≤2.7 read with a non-zero offset, or with a sampler operand that is not a dropped `air.get_read_sampler` result: `AIR 2.<n> air.read_texture_2d has no sampler or offset operand (Apple's form is (texture, coord, lod, access))` | text rewrite (Phase 3) |
| A2 | ≤2.7 `air.get_read_sampler` result used other than as that operand: `AIR 2.<n> has no air.get_read_sampler; its result may only feed air.read_texture_2d` | text rewrite (Phase 3) |
| A3 | Under `sampler_word == .tagged` (AIR ≤2.6 profiles): a sampler with word 1 ≠ 0, or with word 0 bit 63 already set: `AIR 2.<n> writes a constexpr sampler as one i64; <reason> has no measured single-word encoding` | text rewrite (Phase 3) |
| A4 | Under `.tagged`: a `constant [2 x i64]` global referenced anywhere other than as a direct sampler operand of a texture call (a sampler also used elsewhere, a sampler passed through select, phi or a helper, or a data table of that type): `AIR 2.<n> retags constexpr samplers by their direct use; @<name> cannot be classified` | text rewrite (Phase 3) |
| A5 | Assembler validation, keyed on profile fields, in two parts. **Sampler part (Phase 2, with D10):** a global at a sampler operand that is not exactly `[2 x i64]` under `.pair`, or not `i64` under `.tagged`. **Read-form part (Phase 3):** the 2.8 read form or `get_read_sampler` at air_minor ≤ 7. | `lowerCall` |

**Unchanged existing refusals:** addrspacecast, fence, `llvm.nvvm.*`, `llvm.ctpop.i4`, mutable address-space-0 globals, and aggregates holding relocated pointers (D1).

**Internal checks.** `typed pointers: internal: <rule>` with `Diag.kind = .internal`. These are bugs, and tests fail on them (K1-K3, §4.3).

---

## 4. Builder and std internals, guard register (constraint 3)

### 4.1 Guard tests (`tools/air/typed.zig`, "guard Gn: …")

| ID | Internal relied on | Pinned by | Fallback |
|---|---|---|---|
| G1 | `ptrType` keys on address space only (Builder.zig:11974-11979) and accepts values ≥ 0x100; `Type.pointerAddrSpace` round-trips them | test; splice bounds check | 4b |
| G2 | `WipFunction.cast(.bitcast)` keeps a cast between two different pointer types as an instruction and asserts nothing about address spaces; a same-type cast folds (Builder.zig:7099) | test (INST_CAST code 3, opcode 11 counted; a same-type cast adds none); K2 at every view; Debug asserts fire in normal builds | 4b |
| G3 | `castConst(.bitcast, g, handle)` stays a CE_CAST with opcode 11 and does not fold. `convConstTag` would pick addrspacecast for pointer to pointer (Builder.zig:12826), which is why the tag is explicit. | test; V4 | 4b |
| G4 | **Load-bearing.** A CE_CAST's operand-type field holds the destination type (Builder.zig:12864, 15251-15256) | test; V4 (opty == SETTYPE == operand's typed pointer) | **C1** (built and tested, §1.5), then 4b |
| G5 | A METADATA VALUE record writes `constant.typeOf` (Builder.zig:15743-15748) | test; V5 | a C1-style metadata rewrite, or 4b |
| G6 | TYPE_BLOCK lists type items in index order, so type id = `@intFromEnum(Type)`; one defining record per item; STRUCT_NAME takes no id; NUMENTRY = the count (Builder.zig:14560-14622) | test; the splice counts ids, asserts that the handle's `ty` equals the record id, and checks NUMENTRY; V2's forward-reference rule | a Type-to-position map, or 4b |
| G7 | Pointers are written only as OPAQUE_POINTER code 25 `[as]` (ir.zig:2194-2200); POINTER (8) is never written | test; V1 (0 code 25 after the splice; the rewritten count equals the code-25 count; a pre-existing code 8 is refused) | if Builder starts writing POINTER, re-evaluate: the splice may become unnecessary |
| G8 | `i8` (id 14) comes before the pre-interned `ptr` (21) and `ptr addrspace(4)` (22) (Builder.zig:991-992, 9930-9934) | test; V2 (the record at `@intFromEnum(Type.i8)` is INTEGER [8] and precedes every pointer) | map real-AS pointers to an earlier pointee |
| G9 | No absolute offsets: no VSTOFFSET, FNENTRY or SYMTAB | test; V6 | an offset fix-up in the splice, or 4b |
| G10 | BitcodeReader: `stack.items[top].abbrev_id_width` (BitcodeReader.zig:12, 442); block starts are word-aligned; the length word sits at `seek - 4` | comptime field access; test; runtime assertions in the splice | a small local bit reader |
| G11 | `DataLayout.getPointerSpec` falls back to address space 0's spec for unknown spaces (Builder.zig:828-830), so handles are 64-bit | test | add the synthetic specs to the in-memory datalayout |
| G12 | `opaqueType(name)` keeps the exact name for the first type with that name (Builder.zig:10170) | test (STRUCT_NAME spells `struct._texture_2d_t`); V2 checks the names of named-struct handle pointees | cache per module (already done) |
| G13 | `WipFunction` inserts at `cursor.instruction` within the block (Builder.zig:8104). Phase 5 uses `cursor = {entry, 0}`. | comptime `WipFunction.Cursor` fields; test (Phase 5) | keep R2 |
| G14 | Documentary: a constant GEP writes `typeOf(base)` as its base type (Builder.zig:15283). This is why D6 lowers constant GEPs. | test (informational failure: "R1 may be liftable") | none needed |

**Removed dependency:** "ALLOCA carries no address space" is no longer needed, because of D5.

### 4.2 Runtime checks on every typed build (`typed.verify`)

- **V1 Lockstep identity.**
  - The reference stream is the splice's input: the `toBitcode` words, or the `normalizeCeCasts` output once C1 is wired.
  - The bytes before TYPE_BLOCK match the reference, except the MODULE length word, which must equal the old value plus the size delta.
  - The bytes after TYPE_BLOCK are identical.
  - Inside TYPE_BLOCK, each record matches the reference one, except that code 25 becomes 8.
  - 0 code-25 records remain.
- **V2 Type table.**
  - Forward references: every POINTER pointee id, and every element id of a STRUCT_ANON, STRUCT_NAMED, ARRAY, VECTOR or FUNCTION record, is lower than the record's own id, unless it names a named struct. LLVM 15's reader accepts forward references only to named structs. This also covers Phase 6's named structs with bodies.
  - No POINTER pointee is an empty struct (`{}`) or `i1`. This is the backstop for R17/R18: K2 refuses these first, so a V2 hit means a K2 check was missed.
  - No address space ≥ 0x100 remains anywhere.
  - NUMENTRY equals the number of defining records.
  - `i8` comes before every pointer.
  - Named-struct pointees of handles are spelled exactly.
  - No handle has a real-AS pointer as its pointee.
- **V3 Leak check.** No real-AS pointer type id appears in SETTYPE, the INST_CAST destination type, the INST_LOAD type, the INST_GEP source type, the INST_PHI type or the INST_ALLOCA type. (A real-AS pointer type id is one that came from a code-25 record with space < 0x100.)
- **V4 Constants** (grafted from staged-risk-first L3/L4).
  - Derive each global value id's reader type in record order: `valty addrspace(N)*` from GLOBALVAR, `fnty*` from FUNCTION.
  - Every CE_CAST has opcode 11, a global-value operand, and `opty == current SETTYPE ==` that operand's typed pointer. The comparison is structural, because duplicate POINTER records are legal (critic U4).
  - No CE_GEP (codes 12/20) anywhere.
  - No CE_CAST in any function-level constants block.
  - No AGGREGATE element that is a global value id.
  - No GLOBALVAR initializer (its `initid` field) names a global value id or a module constant whose record is CE_CAST or CE_GEP.
  - GLOBALVAR and FUNCTION records have their explicit-type bit set and an address space < 0x100.
- **V5 Metadata.** No METADATA VALUE refers directly to a global value id. A VALUE naming a module constant carries exactly that constant's type.
- **V6 Offsets.** No VSTOFFSET (MODULE code 13), no VST FNENTRY and no SYMTAB block.
- **V7 Counts.** The CE_CAST count equals `Handles.noop_refs` (1 for the entry point, plus one per sampler).

### 4.3 Construction-time checks (assembler, typed mode, `Diag.internal`)

- **K1.** Every pointer passed to `define` is `canon(realSpace)`, or is the 2.8 `get_read_sampler` handle.
- **K2.** After each `view`, the operand's handle pointee equals the access type, source type or parameter pointee. Every pointer operand of a load, store, GEP, atomic or call argument is a handle. Before the view is built, a zero-sized pointee is refused with R17 and an `i1` pointee with R18. These are construct refusals, not internal failures.
- **K3.** A reader-typed value (a global, a function or an alloca result) only ever appears as the operand of `enterGlobal`, as the operand of the alloca's view, or as a callee.

These checks catch a case no bitstream check can see: a raw global or alloca used where the reader would type it differently. They also catch a missed view in the Phase 5 lowerings at the offending IR line. Without them, oracle found, a missing cast surfaces only as "Failed to materializeAll".

### 4.4 Compile-time pins

A `comptime` block at the top of `typed.zig` references:
- `@FieldType(std.zig.llvm.BitcodeReader, "stack")`;
- `std.zig.llvm.BitcodeReader.Item`;
- `std.zig.llvm.Builder.WipFunction.Cursor`;
- `Builder.Constant.Tag.bitcast`;
- `@hasDecl(Builder, "castConst")` and `@hasDecl(Builder, "opaqueType")`.

A rename then becomes a compile error that names `typed.zig`.

### 4.5 Re-pin procedure and Re-pin log (docs/pipeline.md "Re-pinning Zig")

1. Update `build.zig.zon` `minimum_zig_version` and the README pin. Always use an exact nightly.
2. Record `shasum -a 256` of std's `Builder.zig`, `ir.zig`, `bitcode_writer.zig` and `BitcodeReader.zig` in the log. If the files are unchanged, the guards pass by construction (F-A).
3. `zig build test`:
   - a compile error in `typed.zig` means a pinned field moved;
   - a failing "guard Gn" names the behaviour that changed and its fallback.
4. Build all four typed libraries. `verify` refuses with a named V-check if the writer changed.
5. On the M3, run `zig build check` on the default library and the four typed libraries, plus `zig build xcheck -- --apple` on each. Once the probe libraries exist, run `tools/probe_matrix.sh static` and then `tools/probe_matrix.sh gpu`.
6. If nvptx codegen changed the default bytes, re-run the check, then update `tools/air/golden.zig` deliberately.
7. Append a log row with these columns: date; from → to; guard/V failures; files and lines touched; hours; default hash changed?; M3 result.
8. If the §1.5 trigger fires, run the 4b spike. Switch to 4b if it passes, or to option 1 if it fails.

---

## 5. Profile semantics (`tools/air/target.zig`)

| profile | AIR | read_texture | sampler_word | `verified` (opaque) | default pointers | typed after P2 | typed after P3 | typed on a virtual-GPU pass | typed on field report |
|---|---|---|---|---|---|---|---|---|---|
| macos26 | 2.8 | sampler_offset | pair | **true** (M3) | opaque | **native** (M3 42/42) | native | paravirtual result recorded in docs; the CI row becomes strict | – |
| macos15 | 2.7 | lod_access | pair | false | opaque | unverified | **upgrader_only** | native_virtual (macos-15 runner, control ok; or a local VM, §6.4) | native |
| macos14 | 2.6 | lod_access | tagged* | false | opaque | unverified | **upgrader_only** | native_virtual (macos-14 arm64 leg if offered, or a local VM) | native |
| macos13 | 2.5 | lod_access | tagged* | false | opaque | unverified | **upgrader_only** | native_virtual from a local VM only (no Apple-silicon runner) | native |

\* `tagged` only if the Phase 3 rule holds; otherwise `pair`, with the air-opt disagreement documented. In Phase 2 every profile is `pair`; the `read_texture` column exists from Phase 3.

**Meaning of each level.**

| Level | Meaning | Build behaviour |
|---|---|---|
| `unverified` | no 42/42 run | refused without `--allow-unverified` |
| `upgrader_only` | 42/42 on macOS 26.3 (Apple M3) through its AIR upgrader | builds, with the caveat "never run on macOS N itself" |
| `native_virtual` | 42/42 on the target's own macOS on a Virtualization.framework virtual GPU ("Apple Paravirtual device") on an Apple-silicon host, meaning a GitHub arm64 runner or a local VM guest (§6.4), with Apple's typed control passing in the same run | builds, with the caveat "not yet on a real macOS N GPU" |
| `native` | 42/42 on a real Apple-silicon GPU running that macOS major, from a recorded report (date, OS build, Mac model and chip, GPU, who ran it) | builds silently |

**Opaque mode.** `verified` is unchanged. Opaque macos13-15 stays unverified. The measured failures are the M3 upgrader and the macOS 15 virtual GPU (F-B). On that virtual GPU Apple's own opaque control also fails, so that failure says nothing about real hardware.

**When each value flips.** Only in the commit that records the measurement: the log path for M3 runs, the run ID for CI, the VM log for local-VM runs (§6.4), and the report for field runs. The test "typed verification table holds only measured entries" is edited in the same commit.

**AIR-version fields** apply in both pointer modes. macos26's values match today's behaviour, so default bytes and text are unchanged. Opaque macos15/14/13 bytes change on purpose; those profiles are unverified.

---

## 6. CI changes (`.github/workflows/ci.yml`)

Every leg and job runs on an arm64 (Apple-silicon) image. No leg uses `macos-13` or an `-intel` label (§0, constraint 6).

### 6.1 Steps

1. **Keep:**
   - unit tests;
   - the default build;
   - the control step, including its opaque-text Apple re-assembly rung (the default is still opaque). The step now also exports `CONTROL_STATE` (Apple's typed control on this runner: `ok`, `fail`, `nodevice`, `notrun` or `unavailable`) and `OPAQUE_STATE` to `$GITHUB_ENV`. Steps 5 and 11 read `CONTROL_STATE` as the runner check;
   - the default runtime check, with its informational paravirtual downgrade.

   The Runner step also runs `test "$(uname -m)" = arm64`, so a runner image that is not Apple silicon fails the job.
2. **New, strict: "Golden default library".** `zig build golden`. Under a Zig other than the pinned one it passes with a `::warning::`. It never spawns a process.
3. **New, strict: "Build typed-pointer libraries (all profiles)".** For t in macos26/15/14/13: `zig build metallib -Dmetal-target=$t -Dpointers=typed --prefix "$RUNNER_TEMP/typed-$t" --cache-dir "$RUNNER_TEMP/zc-typed"`.
   - The fresh cache directory guarantees that each leg runs the assembler in this run, which the determinism job relies on.
   - `-Dallow-unverified-target=true` is passed only for a profile that is still `unverified`. After Phase 3 there are none, so CI also exercises the refusal logic.
   - `verify` runs inside each build, so a Builder drift fails CI without a GPU.
4. **New, strict: "Bitcode checks".**
   - `zig build xcheck -- --expect-pointers=typed --apple "$RUNNER_TEMP"/typed-*/bin/default.metallib`;
   - `zig build xcheck -- --expect-pointers=opaque --apple zig-out/bin/default.metallib`.
   - The Zig part always runs. air-opt and objdump run where the toolchain exists, reusing the `downloadComponent MetalToolchain` fallback.
   - `zig build xcheck` exits 0 on a skip (build.zig passes `--skip-status=0`), so the step reads the output. A `FAIL` line fails the step; `SKIP` or `KNOWN` lines produce a warning.
5. **New, informational (`continue-on-error: true`): "Experiment, typed libraries on this runner's GPU".** For each typed library whose target major is ≤ the runner's major, run `zig build check -- <lib>` and classify it with the existing `state()`.
   - The job summary gets a `### Typed libraries on <os>` table: library, AIR, result, the first 200 characters of the first FAIL, and the reading from §6.2.
   - macos-latest answers **U5** (typed macos26 on the paravirtual GPU) and records 15/14/13 through that GPU's upgrader as data.
   - macos-15 answers **U6 on a virtual GPU** (typed macos15, macOS 15's native reader) and records 14/13 through macOS 15's upgrader as data.
6. **New, informational: "Typed IR re-written by Apple's writer".** Where the toolchain exists: `zig build xcheck -- --reassemble="$RUNNER_TEMP/apple-typed-$t.metallib" "$RUNNER_TEMP/typed-$t/bin/default.metallib"`, then check the result. t is macos26 on macos-latest and macos15 on macos-15. This rung tells framing apart from IR policy.
7. **Changed:** the macos15 experiment passes an explicit `-Dpointers=opaque`, so it keeps asking its original question.
8. **New job `determinism`**, strict, on `macos-latest` (arm64, so no x86_64 host is involved):
   - each build leg writes the `sha256` of the typed libraries and of the default library to `hashes-${{ matrix.os }}.txt` and uploads it;
   - the job downloads both files and runs `diff`.
9. **New artifacts (field-test bundle, U6 on real GPUs and in local VMs):**
   - the typed macos13/14/15 libraries;
   - Apple-compiled typed control libraries `control-macos{13,14,15}.metallib` (the workflow's control kernel through `xcrun metal -mmacosx-version-min=N.0` on the macos-latest leg), so a field or VM run has Apple's typed control in the same run;
   - `metallib-check` built with `zig build install-check -Dtarget=aarch64-macos.13.0` (arm64 only).

   On the macos-15 leg, that binary runs `--kernel=control` on the control library to prove it launches there (informational). Its launch on macOS 13 is recorded by the first macOS 13 report, from a VM or the field: the `info library targets` and `info device` lines show it started.
10. **Optional matrix leg `macos-14`**, only if actions/runner-images still offers it as an arm64 image (check at implementation; never substitute an Intel image). The existing steps generalise through `MACOS_MAJOR`, and its typed rows answer U6 and U9 on macOS 14's native reader (virtual GPU). If the image is gone, drop the leg, say so in the docs, and use a local macOS 14 VM (§6.4) instead.
11. **Phase 5: two probe steps in the build legs.** Only the static step can fail the job. No CI result changes a construct's scope or evidence; scope decisions come only from the M3 `probe_matrix.sh gpu` run (§2 Phase 5, D20).
    - **Strict: "Probe static checks".** `tools/probe_matrix.sh static` on the macos-latest leg. It is pure Zig, so the result does not depend on the runner. It fails the job on a presence miss, a build or refusal mismatch at any profile, or an inconsistent `probe_expect`.
    - **Informational (`continue-on-error: true`): "Probe kernels on this runner's GPU".** On both legs, `tools/probe_matrix.sh gpu --ci --control-state="$CONTROL_STATE"`. This mode:
      - runs no opaque controls, because this device rejects all opaque output (§6.2 baseline);
      - uses Apple's typed control as the runner check, and prints "runner cannot answer" rows when it is not `ok`;
      - runs only lifted constructs, at the profiles lifted for them (`min_air`) whose major is at most the runner's;
      - writes a `### Probe kernels on <os>` summary table;
      - always exits 0, so an unanswered U5 or a paravirtual failure cannot fail the job.

      A probe row becomes strict only under §6.3.

macOS 13 has no Apple-silicon runner, so its virtual-GPU evidence can come only from a local VM (§6.4).

### 6.2 How to read a typed GPU row

The same table applies to CI rows, local-VM rows and field reports.

| Apple typed control | ours typed | Apple re-write of ours | Meaning | Action |
|---|---|---|---|---|
| ok | ok | – | this consumer accepts our typed output | Record the run ID or log. macos15 row on macos-15 or in a VM: `typed_verification → native_virtual` (after promotion). macos26 row on macos-latest: record it in the docs; CI gains its first strict GPU gate. |
| ok | FAIL at pipeline creation | ok | **framing**: producer string, no SYMTAB or sync-scope block, unabbreviated TYPE_BLOCK, the pre-interned type records Builder always writes | Phase 6: try the framing variants one at a time on a branch. The producer string can be set through `toBitcode`'s `Producer`; SYMTAB and sync-scope need 4b (trigger 2). |
| ok (Apple's typed control for 13.0, same guest or machine) | macos13 FAIL at pipeline creation, while typed 14/15 pass on their own OS | ok, or not yet run | **macOS 13 framing.** target.zig:32-36 records that Apple's macOS 13 output writes `undef` where newer targets write `poison`, drops `noundef` and uses an older function-attribute set. The `frame-pointer` flag is already omitted by the profile (`frame_pointer_flag = false`). builder-feasibility lists these as open questions. | On the same rung, one variant at a time, each on a branch: (1) confirm the library carries no `frame-pointer` flag (grep the text); (2) an undef-for-poison variant, in which the macos13 text rewrite replaces `poison` operands with `undef`; (3) a variant closer to Apple's macOS 13 attribute set; (4) the Apple re-write of ours (`--reassemble` on the host, checked in the guest or on the machine). The first variant that passes names the cause, and it becomes a macos13 profile field. If (4) fails, read the row as IR policy. |
| ok | FAIL at pipeline creation | FAIL | **IR policy** | Phase 6 fidelity layer, in order, each re-tested on the same rung: (1) texture/sampler entry parameters as the named structs; (2) buffer entry pointees from the manifest (named structs built with `@offsetOf`, inference rules; body elements interned first, §2 Phase 6); (3) GEP retyping. |
| ok | FAIL dispatch/render mismatch | – | a real bug (the M3 passes) | investigate |
| FAIL / nodevice | – | – | the runner, guest or machine cannot answer | retry later |

**Baseline already measured (F-B).** In run 36266828343 Apple's typed control passed on both runners and Apple's opaque control failed on both. So both typed rungs can be interpreted as soon as they exist. No opaque row on these runners can be interpreted, including a per-construct opaque probe control, which is why the CI probe mode runs none (§6.1 step 11).

### 6.3 Promotion rule

- A typed GPU row becomes strict (its failure fails the step) after **3 consecutive passing runs** on that runner image, spanning at least 2 days.
- The promotion commit records the run IDs in docs/pipeline.md.
- An evidence flip follows the same rule. For a local VM (§6.4), the rule is two passing runs from separate guest boots, with Apple's typed control passing in each and both logs recorded.
- The docs quote only rows that ran.

### 6.4 Local VM rung (optional, outside CI)

- **What.** macOS 13, 14, 15 and 26 guests in Virtualization.framework (tart or UTM) on an Apple-silicon host. Such a guest's GPU is expected to report "Apple Paravirtual device", the same GPU GitHub's arm64 runners expose. The first run records the `info device` line to confirm it.
- **Why.** It is the only virtual-GPU source for macOS 13, which has no Apple-silicon runner. It is a native-reader source for macOS 14 if the macos-14 leg is unavailable. It also gives faster iteration on U5 and the §6.2 variants than CI turnaround.
- **How.**
  - Copy the field bundle (§6.1 step 9) into the guest: the typed libraries, Apple's typed controls, and `metallib-check` built for `aarch64-macos.13.0`.
  - Run these one at a time: `metallib-check --kernel=control control-macosNN.metallib`, then `metallib-check typed-macosNN/default.metallib`.
  - Record the guest OS build, the host Mac model, chip and macOS build, the VM tool and its version, and the full output.
  - Probe libraries can be checked the same way with `install-check-probe`, read like the CI probe mode: only lifted constructs at the profiles lifted for them, and opaque rows count only if Apple's opaque control passes in the same guest. Guest results never set a construct's scope.
- **Reading.** Use the §6.2 table. A result counts only if Apple's typed control passes in the same guest session. A guest without a Metal device prints `SKIP` and answers nothing.
- **Evidence.** A pass gives `native_virtual` for that major under §6.3's VM rule, and never `native`. **Assumption, not measured:** no report has measured how much of the paravirtual GPU's behaviour comes from the host. This plan assumes it may depend on the host Mac's GPU and macOS build, so the host build is part of the record. A VM result supports the GitHub rows for U5 but does not replace them.
- **macOS 13 launch.** The first macOS 13 run, whether in a VM or in the field, also records that the bundle's `metallib-check` launched there (its `info` lines).
- **Guest images.** Whether macOS 13-15 restore images or prebuilt guest images are available is checked at implementation. The rung is optional, and no gate depends on it.

---

## 7. Docs changes (written from the logs of each gate only)

**README.md**
- Quick start: the new test count.
- "What is verified": add rows for typed libraries at 26/15/14/13 (M3, 42/42; 15/14/13 through the macOS 26 upgrader), `zig build xcheck` (air-opt N/15), `zig build golden`, and CI or VM rows with run IDs or logs once promoted.
- "Scope and honest limits":
  - state the Apple-silicon-only scope: arm64 Macs with Apple GPUs; Intel Macs are neither supported nor tested;
  - replace "macOS 26 and newer only" with the per-(profile, pointers) evidence table and the typed refusal list, including what is "not executed" and the class generalisations accepted without their own execution (§3 a-d);
  - add paravirtual and native status only as measured.
- Layout: `typed.zig`, `air_xcheck.zig`, `golden.zig`, the probe and `probe_matrix.sh`.

**docs/pipeline.md**
- §4: a new subsection "Typed pointers (`tools/air/typed.zig`)":
  - T1-T7, D4, and why constant casts are forbidden (G1);
  - the splice and V1-V7, K1-K3;
  - the construct table, including the per-class rule, the rows refused until executed, and the expected-refusal table used by the tests.
- Build graph: `-Dpointers`, `xcheck` (`--apple`, `--skip-status`), `golden`, `install-check`, `probe` (`-Dprobe`), `probe-list`, `check-probe` (`--only`).
- "Deployment targets": rewritten around the evidence levels (§5).
- "Continuous integration": the new rungs, the arm64-only legs and the `uname -m` assertion, the §6.2 reading table (including the macos13 row), the promotion rule, and why the CI probe step runs no opaque controls and never sets the exit status.
- "When something fails", new rows:
  - `typed pointers: V<n> failed` (a Builder change: run `zig build test`, where guard Gn names it);
  - a failing `guard Gn` test;
  - `typed pointers: internal:` (an assembler bug);
  - each R and A refusal, with its workaround;
  - an xcheck `FAIL` or `SKIP`;
  - `probe_matrix.sh` outcomes by mode: in `static`, a refusal mismatch (an impure probe, a missing `needs` entry, or a stale `min_air`); in `gpu` on the M3, "opaque control failed" (construct out of scope); in `gpu --ci`, "runner cannot answer" (Apple's typed control not ok);
  - crash-report rate limiting and archiving old `.ips` files.
- A new "Re-pinning Zig" section with the Re-pin log (§4.5), including the 4b spike.
- A new "Field test" recipe: `zig build metallib -Dmetal-target=macosNN -Dpointers=typed` (or the CI bundle), then on an Apple-silicon Mac running that OS, `metallib-check --kernel=control control-macosNN.metallib` followed by `metallib-check <lib>`. Post the output with the date, OS build, Mac model and chip, and GPU.
- A new "Local VM" recipe (§6.4).

**docs/air-format.md**
- §1.7: add rows for the read_texture and sampler-word differences per target, with provenance.
- §2: the project emits typed pointers on request. Document:
  - the record-level changes: OPAQUE_POINTER → POINTER, INST_CAST, the CE_CAST rule, typed METADATA VALUE;
  - that typing is all-or-nothing (one OPAQUE_POINTER is fatal);
  - pointee ordering and the forward-reference rule;
  - that `{}` texture pointees crash;
  - that air-opt is necessary but not sufficient.
- §2.4: the per-AIR-version texture rows and the operand map (0,2,4,5).
- §2.5: correct "63 | never set by Apple" to "never set in the two-word form; at AIR ≤2.6 Apple writes one `i64` with bit 63 set", with this shader's value (F-C).
- §3.5: the typed spelling of the sampler-state reference, and D10's registration rule per sampler word.
- §4: `zig build xcheck` and its limits, the `--reassemble` rung, and that the `-Xclang -opaque-pointers` re-assembly applies to opaque builds only. Never use `xcrun metallib` to build test libraries, because it re-optimises (verify-inference).

**Code comments**
- the `target.zig` file comment, replacing the paragraph that begins "One difference is NOT captured…";
- the headers of `assembler.zig`, `intrinsics.zig` and `air_splice.zig`, including the usage string;
- the `build.zig` comments at lines 22-25.

---

## 8. Risks and open questions

| # | Risk or question | How it is answered | Action on each outcome |
|---|---|---|---|
| Q1 (U5) | Does GitHub's paravirtual GPU (macOS 26.6.x) accept Builder-written typed AIR 2.8 (i8 values, Apple-shaped intrinsics)? | CI §6.1 step 5 (typed macos26 row) and step 6 (Apple re-write). A macOS 26 guest in a local VM (§6.4) allows faster iteration; it is supporting data, and the CI row answers U5. | §6.2 table |
| Q2 (U6) | Do native macOS 13-15 readers accept typed libraries? | The macos-15 runner (F-B shows its control passes for typed AIR 2.7); the optional macos-14 arm64 leg; local VM guests for 13/14/15 (§6.4); field reports through the CI bundle, which carries Apple's typed controls | `native_virtual` on a runner or VM pass; `native` only from a report on real Apple-silicon hardware. macOS 13 has no Apple-silicon runner: a local VM gives `native_virtual`, and a field report gives `native`. |
| Q3 (U9) | Is the tagged ≤2.6 sampler word needed, or accepted? | The Phase 3 rule on the M3, then macos-14, VM and field reports | `tagged` or `pair`, as in Phase 3; D10 and A4/A5 follow the field |
| Q4 (U7) | Does the runtime remangle opaque-style `llvm.mem*` names? | Moot: the plan emits LLVM 15 mangling. `testProbeMemcpy` covers it. | fail: R3 stays |
| Q5 (U8) | Are pointee checks applied to native atomics at runtime? | `testProbeNativeAtomics` at the four typed profiles, in its own library after its opaque control on the M3 | fail: R4 stays and points to `gpu.atomic*`. Opaque control fails: out of scope (Q23) |
| Q6 (N6) | Do gather and sample_grad survive the ≤2.7 upgrader? | `testProbeGather` and `testProbeSampleGrad`, each in its own library, at typed macos15/14/13 | fail: raise that kernel's `min_air` to 8, flip only its 2.8 `typed_proven` row, and keep R9 at ≤2.7. The static mode then checks that macos26 builds and macos15/14/13 refuse. Their ≤2.7 forms come from MSL probes compiled with `xcrun metal -mmacosx-version-min=15.0`, a research step outside the build. |
| Q7 | Upstream fixes the CE_CAST operand-type field | G4 fails in `zig build test`, and V4 fails at build time | wire C1 (flip `normalize_ce_casts`; the sequence is already tested), flip G4, re-measure; file the bug upstream now |
| Q8 | Are `i1` pointee views safe? metal-ir-pipeline says `i1*` crashes; this is untested here. | R18 refuses them in typed mode (K2), and V2 rejects an `i1` pointee. The Phase 2 gate greps `shader.air.ll` for `i1` loads and stores (expect 0; F-E measured 0). A probe kernel is added if Zig ever emits them. | probe passes: lift R18. Probe fails: view i1 memory as i8 plus trunc/zext |
| Q9 | Flip the macos13-15 default to typed (the critic's recommendation)? | User decision after Q2 | If typed `upgrader_only` keeps building without the flag, the flip breaks the `air_splice.zig:2020` assertion. The alternative is to require `--allow-unverified` below `native_virtual`. The user picks one. |
| Q10 | Make typed macos26 the default after a Q1 pass? | User decision | re-baselines `golden.zig` and the default sha |
| Q11 | Keep `zig build golden` after this change set? | User preference | it adds friction when shader sources change |
| Q12 | Builder internals drift on a re-pin | §4 register, the Re-pin log, the §1.5 trigger and the 4b spike | fix in `typed.zig`, or 4b after a passing spike |
| Q13 | D5 (real-address-space alloca) deviates from the measured prototype | Phase 2 gate: the three sret renders at typed macos26, then 15/14/13 | fail: revert to the prototype's handle-address-space alloca and add a guard "ALLOCA carries no address space" |
| Q14 | A shared-path refactor drifts the default output | Per commit: sha, `cmp` of the text, golden, 90 unmodified tests, 42/42 | revert the offending hunk |
| Q15 | The tagged `i64` sampler through the metadata no-op cast (`i64 addrspace(2)*`) was never run; the prototype used `[2 x i64]` | Phase 3 gate (constexpr render at typed 14/13) | fail: `pair` |
| Q16 | air-opt is insufficient as a gate (it misses the upgrader failure and the `{}*` texture crash) | The GPU check stays the gate; xcheck is necessary, not sufficient | – |
| Q17 | A probe stops exercising its construct after an optimizer change | The strict per-kernel presence grep in `probe_matrix.sh static`, in Phase 5 and CI | adjust the probe |
| Q18 | Typed output differs between machines | The CI determinism job (both arm64 runners, fresh caches) | investigate before trusting any cross-runner result |
| Q19 | Crash attribution (reports are rate-limited; parallel jobs muddy them) | Serial runs; archive old `.ips` files before each gate; `probe_matrix.sh` runs one process at a time | – |
| Q20 | Over-refusal makes typed mode impractical for ordinary shaders before Phase 5 | Refusals point to docs and to `-Dpointers=opaque`. Rows 11-15 are now refused too (D15). Phase 5 lifts one construct at a time, in order of occurrence (constant GEPs, memcpy, phis over globals and helper pointers first). | – |
| Q21 | `.pair` still lists a `[2 x i64]` data table as a sampler (today's behaviour, kept by D10 so that indirectly used samplers are not dropped) | Not measured; documented as a limit | a probe kernel holding such a table next to a constexpr sampler would measure it; not planned in this change set |
| Q22 | Local VM guests behave differently from GitHub runners or real hardware | The host build is recorded (an assumption, §6.4); VM evidence stops at `native_virtual`; the CI rows still answer U5 | – |
| Q23 | A probe construct's opaque control fails (for example, native `atomicrmw`, never executed in opaque mode) | `probe_matrix.sh gpu` on the M3, per construct. CI never runs the opaque probe controls, because the paravirtual device rejects all opaque output (F-B), so CI cannot put a construct out of scope. | the construct is out of scope: a commit citing the M3 log sets `status = .out_of_scope`, it stays refused, kernels that need it stay "blocked", the docs say so, and an opaque-mode finding is filed |
| Q24 | A class generalisation (§3 "Accepted by class", a-d) fails on a GPU although the executed members of its class pass | No gate measures each member. Phase 5 probe kernels exercise some members incidentally (other GEP shapes, load types and allocas); a field, VM or CI failure is read with §6.2. | refuse the failing member with a new R-id until a probe executes it, and narrow the §3 row |

---

## 9. Critic open issues: resolution map

| Critic item | Resolution |
|---|---|
| C1 constant bitcasts of globals | D4, R6 (scalar and aggregate initializers), V4 (no pointer-changing CE_CAST, no CE_GEP, no global-valued GLOBALVAR initializer) |
| C2 fork vs stock Builder | D1 option 4; 4b fallback (an unmeasured design, with a spike before switching) under the §1.5 trigger |
| C3 pointee policy vs the air-opt gate | D2 + D3; the Phase 2/3 gates require air-opt 15/15 |
| C4 air-opt as a gate | necessary, not sufficient; the GPU gate stays (Q16) |
| C5 pointer load/store accepted today | R5 until `testProbePtrStore` passes in its own library, then the Phase 5 lowering |
| U5 paravirtual GPU | Q1: CI §6.1 steps 5-6; local VM (§6.4) as supporting data |
| U6 native 13-15 | Q2: macos-15 rung (F-B), macos-14 arm64 leg, local VM guests for 13/14/15 (§6.4), field bundle with Apple's typed controls |
| U7 mem* remangling | Q4: LLVM 15 mangling plus probe |
| U8 native atomics | Q5: R4 plus its per-construct probe, after its opaque control on the M3 |
| U9 ≤2.6 sampler word | Phase 3 rule; Q3; D10/A4/A5 keyed on `sampler_word` (the field, D10's branches and A5's sampler part from Phase 2) |
| N1 debug text and the CI re-assembly rung | D13: the text stays the opaque input, with a typed comment; `air-xcheck` (plus `--reassemble`) for typed builds; the opaque rung is unchanged for the default |
| N2 `Builder.print` misleading | D18: record-level assertions; dual-mode helper with `Diag` ids and the expected-refusal table |
| N3 constructs to lower or refuse | §3 table (rows 1-29, plus the class generalisations a-d) |
| N4 Builder internals | §4 register: G1-G14, V1-V7 (V2 forward references and `{}`/`i1` pointees; V4 initializers), pins; ALLOCA dependency removed by D5 |
| N5 plumbing, `verified` semantics, docs | D11/D12, §5, §7 |
| N6 other texture intrinsics at ≤2.7 | R9 plus `testProbeGetWidth`, `testProbeGetHeight`, `testProbeGather` and `testProbeSampleGrad`, each in its own library, lifted per AIR range through `min_air` |
| Open question: guard the CE_CAST quirk, or splice METADATA VALUE? | D17: guard (G4/V4) plus C1, built and tested as a `finish` option; no metadata splice now |
| Open question: native atomics without evidence | R4 until its probe passes (own library, own opaque control on the M3) |
| Open question: implement or refuse constant GEPs, mem*, phis over globals, pointer initializers | rows 11-21 lifted in Phase 5 one construct at a time (in `needs` order where declared); row 22 refused permanently (a 4b trigger), including scalar initializers |
| Open question: typed text printer or air-opt/objdump? | `air-xcheck` over our blobs; no printer (D13) |
| Recommendation: typed default for 13-15 | not adopted in this change set (D11); Q9 names the test trade-off |
| Recommendation: mark 13-15 verified only for the upgrader path | `upgrader_only` (§5) |
| "Still to design": versioned intrinsic table | intrinsics AIR-range rows (Phase 3) |
| "Still to design": ≤2.6 sampler word | `sampler_word` field and assembler branches in Phase 2; text-rewrite retag and the 14/13 values in Phase 3 |
| "Still to design": guard tests | §4 |
| "Still to design": typed cross-check | xcheck: pure checks never spawn; Apple's tools sit behind `--apple`, with a skip status |
| "Still to design": typed test variants | D18, with the expected-refusal table |
| "Still to design": CI rungs | §6, including the strict probe static step, the informational probe GPU step, arm64-only legs and the optional local VM rung |
| "Still to design": docs | §7 |

**Plan-review gaps resolved** (`plan/critic_gaps.json`)

| Gap | Resolution |
|---|---|
| 1 (major) Phase 5 probe gate is circular; one failing construct hides the rest | D20: one probe library per construct (`air-splice --entry`, `zig build probe -Dprobe`), `metallib-check --only` with dispatch, a per-construct opaque control on the M3, out of scope when that control fails (Q23). `tools/probe_matrix.sh` has three modes: `static` (presence, refusal state and `probe_expect` consistency; strict on the M3 and in CI), `gpu` (the M3 run and the only source of scope decisions), and `gpu --ci` (no opaque controls, Apple's typed control as the runner check, lifted constructs only, always exit 0). CI runs `static` as a strict step and `gpu --ci` as an informational one (§6.1 step 11, §6.2 baseline). |
| 2 R10 would refuse `air.atomic.fence` | R10 and the `atomicElement` retyping apply only to `air.atomic.*` declarations with pointer parameters (Phase 2 `declare`, §3 row 25, T5); new test that a typed module calling `air.atomic.fence` assembles |
| 3 D10 by-use registration drops indirectly used samplers | D10 rewritten: under `.pair`, the exact value type `[2 x i64]`, with A5's sampler part guaranteeing that every direct sampler operand has that type; under `.tagged`, direct use plus a wider A4. Both assembler branches and A5's sampler part land in Phase 2. New tests for a sampler passed through `select`/a helper and for the `.tagged` branch on a synthetic profile; Q21 |
| 4 A5 keyed on `air_minor` breaks the `.pair` fallback | A5 keyed on `profile.sampler_word`. `SamplerWord` (`.pair` on every profile) and A5's sampler part land in Phase 2 with D10; Phase 3 adds only A5's read-form part and the 14/13 values (§2 Phases 2-3, §3) |
| 5 Exit 77 does not survive `zig build` | `--golden`/`--expect-pointers` never spawn and exit 0/1; Apple's tools only behind `--apple`/`--reassemble`; build.zig passes `--skip-status=0`; a skip-path gate item (§0.2, Phase 1) |
| 6 The cached second build proves nothing | Phase 2 gate rebuilds with fresh `--cache-dir`/`--global-cache-dir` and compares both outputs; CI typed builds use a fresh cache dir (D19) |
| 7 Rows 11-15 accepted without GPU execution | Refused (R13-R16, R10) until their probes execute; new probes `probePtrInt` and `probePtrAggregate`; air-opt dropped from their evidence (D15, §3) |
| 8 No `i1` grep; no check behind T5's `{}` rule | `i1` grep in the Phase 2 gate; R17 (`{}`/zero-sized) and R18 (`i1`) in K2; V2 backstop |
| 9 Scalar pointer initializer not caught | R6 check in `defineGlobal`; V4 rejects a GLOBALVAR initializer that names a global value id or CE_CAST |
| 10 No local-VM source for macOS 13 | §6.4 local VM rung (13/14/15/26 guests, optional `native_virtual` source); bundle carries Apple typed controls; the macOS 13 launch is recorded |
| 11 No macOS 13 row in §6.2 | macos13 row: frame-pointer confirmation, undef-for-poison variant, attribute-set variant, Apple re-write, each on the same rung |
| 12 C1 wiring underspecified | `finish` runs normalize, then splice, then verify, with V1 against the splice's input; C1 patches the CONSTANTS and MODULE length words; the sequence is tested in Phase 1 |
| 13 Phase 6 named structs forward-reference their bodies | Phase 6 note (intern body elements first); V2's forward-reference rule covers STRUCT_NAMED elements from Phase 1 on |
| 14 4b is unmeasured | Labelled an unmeasured design, with the difference from builder-feasibility's variant explained; a 0.5 d spike added to the §1.5 procedure and to re-pin step 8 |
| 15 Dual-mode helper accepts any construct refusal | `Diag.id` plus the expected-refusal table keyed by module hash; success required everywhere else; lifts must delete entries (D18) |
| Scope (user) | Apple-silicon-only scope line at the top and §0 constraint 6; arm64-only CI with a `uname -m` assertion; the determinism job moved off `ubuntu-latest` (the only x86_64 item) to `macos-latest`; evidence levels name Apple GPUs only |

**Verification findings resolved in this revision**

| Finding | Resolution |
|---|---|
| Gap 1, CI part: the CI probe job would mark every construct out of scope (opaque controls on the paravirtual device), and a GPU failure could set the exit status | `probe_matrix.sh` modes (`static`, `gpu`, `gpu --ci`); scope decisions only from the M3 `gpu` run, recorded as `status = .out_of_scope` in a commit citing its log; CI splits into a strict static step and an informational GPU step that skips opaque controls, uses `CONTROL_STATE` (Apple's typed control) as the runner check and always exits 0 (§2 Phase 5, §6.1 steps 1 and 11, §6.2 baseline, D20, Q23) |
| Phase order: Phase 2's D10 code used `sampler_word` and A5 before Phase 3 defined them | `SamplerWord` with `.pair` on every profile, both D10 branches and A5's sampler part all land in Phase 2, with a synthetic-profile test for `.tagged`; Phase 3's assembler work is validation only (A5's read-form part) plus setting 14/13; D10's union with direct operands removed, since A5 makes it redundant from Phase 2 on (D9, D10, §2 Phases 2-3, §3 A5, F-G) |
| Phase 5 said lifts land in any order, yet `needs` creates dependencies and the refusal check wanted one exact id | `probe_expect` gains `status` (and `min_air`, from the round-2 row below). Lifts are independent unless `needs` is declared. Dependents are lifted after their needs, and stay "blocked" if a need is out of scope. The expected refusal is computed per profile, so a lift edits no other entry. `probe_matrix.sh static` checks all of it (§2 Phase 5). |
| §0 constraint 4 overclaimed "no exceptions" while §3 accepts generalisations | Per-class rule in §0 constraint 4 and D15; new §3 "Accepted by class" table (a-d) giving each generalisation and its argument; Q24 tracks the risk; README states the generalisations (§7) |
| §6.4 asserted that the paravirtual GPU is backed by the host's GPU and macOS | Restated as an unmeasured assumption, kept only as the reason to record the host build (§6.4, Q22) |
| Round 2: `probe_expect.status` was one value per kernel. The Q6 split (a row lifted at AIR 2.8 only) could therefore be neither `.lifted` nor `.refused`, and the strict static step would stay red. R9 was the id of two kernels, so `needs` by id was ambiguous, and gather and sample_grad shared one kernel. | `probe_expect.min_air` records the AIR range a lift covers, and `static`, `gpu` and `gpu --ci` check each profile against it. `needs` names kernels. The stored `refuses` is replaced by an allowed set computed per profile. There are four R9 kernels, one per texture row (§2 Phase 5, D20, Q6). |
| Round 2: §3 class (a) accepted GEPs over any source type, which contradicts R17/R18 in `view` | Class (a) now excludes zero-sized (`{}`, `[0 x T]`, R17) and `i1`/`<N x i1>` (R18) source types, as class (b) already did (§3) |
