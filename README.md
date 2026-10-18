# PFP Darwinism

**Pass, Fail, Possible — survival of the fittest reasoning.**
*"Nebius — survival of the fittest."*

A reasoning flow drops into an arena and falls. Wherever its continuations
genuinely disagree, it **branches** into parallel sub-agents; those branch again
at their own forks. Branches that contradict established truth **crash into the
verified main line** and die. Branches whose ideas align **merge** — cooperation,
not just competition. Whatever survives repeated, rotated falsification exits
into the **winner's box** with its proof trail.

Built for the Nebius x NVIDIA Global AI Hackathon on NVIDIA Nemotron models via
Nebius Token Factory. Full design: [`DESIGN.md`](DESIGN.md).

![arena](https://img.shields.io/badge/arena-Tron%20light--cycle-00f0ff) ![deps](https://img.shields.io/badge/runtime%20deps-stdlib%20only-00ff9d)

---

## Quick start

```bash
# No key needed: the mock backend speaks the exact OpenAI-compatible interface
python run_arena.py --mock --html-out arena_out/arena.html --stream-out arena_out/run.jsonl

# Open arena_out/arena.html — scrub, play, hover branches.
#   arena.html#seq=120  opens at event 120
#   arena.html#play     autoplays from ignition (demo-video capture)
```

Live on Nebius Token Factory:

```bash
export NEBIUS_API_KEY=...                       # Token Factory key
export NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1   # default
export NEBIUS_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B        # default (fan-out workhorse)
python run_arena.py --hard-cap 2.00 --state-dir .pfp_state --html-out arena_out/arena.html
```

`--state-dir` persists the version registry and the case bank, so each run starts
seeded with past wins on similar problems, on whichever config version is active
or pinned. Both files are atomically written and integrity-enveloped; a changed or
truncated recall bank is rejected before a model client is created. Legacy bare-list
case banks load once and upgrade on the next save. A recalled win's head start is
its past certainty scaled by how similar its problem is to this one, and a repeat win
(even after a reload truncated the problem or collapsed the fork's whitespace) bumps the
stored case's `wins` instead of piling up duplicates. Real scale is `--starters 9`
(odd: no arbitration ties).

Registry control data is validated before admission: rollback thresholds and scores must
be finite, sandbox coherence is a real boolean, prompts remain strings, and proposition /
sandbox payloads must be persistable JSON. A rejected rewrite leaves registry state intact,
so malformed self-improvement output cannot disable auto-revert or create an unsavable version.

| Flag | Default | Meaning |
|---|---|---|
| `--starters` | 3 | starter agents (3 demo, 9 real) |
| `--hard-cap` | 2.00 | absolute USD ceiling on true API spend |
| `--depth-dial` | 3 | bounds branch count per split |
| `--k-probes` | 3 | blind continuations per divergence check |
| `--gauntlet-rounds` | 3 | rotated falsification rounds to earn Pass |
| `--selection-rounds` | 3 | align → clash → merge rounds (merged children can merge again) |
| `--state-dir` | — | persist registry + case bank across runs (one run at a time: token-owned `.pfp.lock`, auto-reclaiming dead local PIDs) |
| `--pin-version` | — | pin a registry version in `--state-dir` before running |
| `--unpin` | off | clear any manual version pin in `--state-dir` before running |
| `--restore-version` | — | restore a registry version in `--state-dir` before running |
| `--list-versions` | off | print active, pinned, and historical versions in `--state-dir` as JSON and exit |
| `--no-improve` | off | skip the Piece 8 post-run loop |
| `--improve-cap` | — | USD ceiling for the Piece 8 loop (default: whatever `--hard-cap` leaves after the arena; the summary records `self_improvement_budget_usd`) |
| `--max-retries` | 3 | live retries on 429/5xx/network errors |
| `--max-concurrent` | 32 | max in-flight model calls across all branches |
| `--summary-out` | — | write the run summary JSON (spend, winners, self-improvement; also works with `--replay`) |
| `--report-out` | — | write the proof trail: `.md` Markdown or `.json` structured (also works with `--replay`) |
| `--replay` | — | re-render a saved `--stream-out` JSONL to ASCII / `--html-out` / `--summary-out`; runs post-replay improvement if `--state-dir` is given without `--no-improve` |
| `--verify` | — | check a saved `--stream-out` JSONL's hash chain and print its anchor (event count + head digest) as JSON; exit `1` on any break or a legacy unchained journal |
| `--expect-head` | — | with `--verify` / `--replay`: required journal head (64 hex, or a summary / `.json` proof report carrying `journal.head_digest`); a truncated, extended or replaced journal is refused |

Exit code is `0` for a completed run, `2` if the run halted (`hard_cap`,
`emergency_stop`, `backend_error`, `internal_error`, or `interrupted` by Ctrl-C) or
Ctrl-C cut the post-run self-improvement short, and `1` for an
unreadable replay, an unknown version ID, a corrupt `--state-dir`, or a `--state-dir`
another live run is using (all checked before any client exists, so nothing is spent
and the bad file is left in place for inspection). A halted run still writes its
stream, HTML and summary, so the money already spent is never lost to a crash.

```bash
# Re-render yesterday's live run for the demo video without spending a cent
python run_arena.py --replay arena_out/run.jsonl --html-out arena_out/replay.html

# Inspect or pin versions in a persisted state directory
python run_arena.py --state-dir .pfp_state --list-versions
python run_arena.py --state-dir .pfp_state --pin-version v000_initial --list-versions
```

## The proof trail

`--report-out proof.md` writes what the winner's box means: for every winner, the
claim it carried out, each falsification it survived (angle + judge), per-parent
merge provenance, its full ancestry back to the starters, and every crash into
the main line along that lineage. It is a pure reduction of the event stream
(`pfp/proof_report.py`), so a live run and a `--replay` of its JSONL produce the
same report; the live one adds the spend line. The problem statement rides on the
starter spawn events (inside the hash chain), so a replay recovers its title without
`--problem` and a forged title breaks the journal. Model text is escaped, so a reply
can't inject Markdown links, headings or HTML. Use a `.json` path for the
structured form.

```bash
python run_arena.py --replay arena_out/run.jsonl --quiet --report-out arena_out/proof.md
```

### Anchoring the journal

The hash chain proves each event is consistent with everything before it, but a copy
cut short at the tail is still a valid chain: drop a trailing `revocation` and a revoked
winner is back in the box. So every run records the journal's **anchor** — event count
plus the head digest — in the summary (`journal`), the proof report (`journal`, and a
`Journal:` line in Markdown) and the final stderr line. Keep or publish it apart from the
JSONL, then check any copy against it:

```bash
python run_arena.py --verify arena_out/run.jsonl --expect-head arena_out/summary.json
python run_arena.py --replay arena_out/run.jsonl --expect-head 3f9c…e1 --html-out arena_out/replay.html
```

A legacy v1 journal (no digests) still replays for old recordings, but with a warning and
`verified_chain: false`: anyone can strip a chained journal's digests, edit it and relabel
it v1. `--verify` refuses one unless `--expect-head` is given — it is re-hashed exactly as
the live run hashed it, so only an unedited copy matches the anchor.

## How it maps to the design

| Piece | Module | What it does |
|---|---|---|
| 1. Issue detector | `pfp/issue_detector.py` | K blind high-temperature probes; a split *is* the issue. Plaus (divergence + logprob entropy + critic leaps) and NS (blind variation on a budget-derived cadence). Novelty exhaustion across the whole lineage. |
| 2. Branch spawner | `pfp/branch_spawner.py` | Directives, recursive mini-harnesses running in parallel, single-round budget: root = 2 × difficulty (capped), ×0.5 depth decay, certainty-weighted sibling split with a floor, 10–15% finish reserve, graceful degradation (NS slows → stops → no spawns → conclude). |
| 3. PFP tracker | `pfp/tracker.py` | Thread-safe ledger; one-way transitions, no resurrection, transition history, lineage/DAG queries. |
| 4. Alignment detector | `pfp/alignment_detector.py` | Cheap embedding pre-filter (recall) → Super claim-by-claim judge: overlap / contradictory / unrelated. |
| 5. Merge synthesizer | `pfp/merge_synthesizer.py` | Super unifies aligned branches with per-parent provenance; parents are absorbed and their budgets transfer to the child, which starts Possible. |
| 6. Fail judge | `pfp/fail_judge.py` | Falsification on a schedule (post-merge, Pass gauntlet, contradiction, background, pre-exit), rotated angles, one appeal per kill with a different judge, revokable Pass that rewrites the main line and re-verifies dependents. |
| 7. Arena feed | `pfp/arena.py`, `pfp/arena_html.py` | Append-only event stream (JSONL in/out). ASCII renderer and a self-contained HTML/Canvas Tron player are pure functions of the stream. |
| 8. Self-improvement | `pfp/self_improvement.py`, `pfp/dials.py` | Mine the log → recall past wins → self-assessment → evidence-grounded anti-model → proposition → sandbox → hashed version registry with auto-revert, pin, run-boundary switching, emergency stop. |

Models: Nano (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) for fan-out; Super
(`nvidia/nemotron-3-super-120b-a12b`) for alignment, merge and falsification.

## Safety rails and honest accounting

- **True spend, not affordable spend.** Every model call goes through
  `MeteredClient`; the ledger records each call's real cost. Cost a branch can't
  cover is drawn from the finish reserve (verification work only) or reported as
  `overdraft_usd` — never dropped. Every spend event (including `problem_sizing_probe`
  and `merge_synthesis`) is recorded in the arena stream so stream-mined spend,
  `total_spent_usd`, and `api_spend_usd` match exactly.
- **The cap must be able to bind.** `--hard-cap nan` or `inf` is rejected by the CLI,
  `MainHarness`, `BudgetLedger`, `MeteredClient` and the sandbox: a NaN cap compares
  False against every spend and would silently admit unlimited calls.
- **Budget inputs fail closed.** Ledger configuration, provider-reported spend,
  reserve transfers, and branch-budget arithmetic reject booleans, negatives,
  `NaN`, and infinities before mutating state; invalid accounting data cannot
  poison totals or silently disable affordability checks.
- **Hard cap is enforced**, not advisory: every call atomically reserves a
  conservative upper-bound cost before it enters the parallel request pool.
  A call that would exceed `--hard-cap` (including spend already in flight) is
  refused; the run halts cleanly with `halted_reason: "hard_cap"` and a `halt`
  event in the arena feed. Unknown model IDs use the highest known catalog rate
  instead of an optimistic cheap-model fallback. Live reservations cover every
  potentially billable retry, not just the final attempt, and positional
  OpenAI-style arguments receive the same prompt / output / `n` preflight as
  keyword calls. The output bound honors both `max_tokens` and `max_completion_tokens`
  (the larger, if both are sent), and a call with no limit (`max_tokens=None`) reserves
  a full context window (131,072 tokens) rather than the 256-token default.
- **Meter configuration fails closed.** Hard caps, timeouts and retry backoff must be
  finite numbers (never booleans); retry counts and concurrency limits must be real
  integers in range. Inner-client cost multipliers and billable-attempt counts are
  validated before the meter starts, so malformed adapters cannot weaken reservations.
- **Token counts fail closed.** Negative or fractional request limits reserve the full
  context window instead of being clamped or truncated into a cheap call, and malformed
  sample counts are rejected before dispatch. Negative or fractional provider usage is
  treated as missing and charged at the conservative bound, so malformed accounting cannot
  turn a billed response into a free one.
- **The whole request is under custody.** Tool definitions, response schemas, metadata, and
  future OpenAI-compatible request fields participate in the input-token reservation and in
  missing-usage fallback billing. A tiny chat message cannot hide a large billable schema
  outside the hard-cap preflight.
- **The request contract fails closed.** `stream=True` (or any truthy `stream`) is refused
  by the live client, the mock, and the meter before a reservation or request: an SSE reply
  is billed but unreadable as JSON, so each retry would pay again for nothing. The mock now
  validates `n` exactly like the live client, so code that passes on the mock cannot send a
  sample count the live backend refuses.
- **One cap for the whole command.** The post-run self-improvement loop
  (assessment, anti-model, proposition, sandbox run) is metered against what the
  arena left under `--hard-cap`; the summary reports `total_api_spend_usd` =
  arena + improvement. A sandbox cut short by the cap is never adopted.
- **No free calls.** If the provider omits or garbles `usage`, the call is charged
  a conservative upper bound (`usage.estimated`,
  `api_estimated_usage_calls`) instead of $0. Calls billed above their
  reservation are counted in `reservation_overruns`; `hard_cap_overrun_usd`
  exposes any unavoidable provider-side overrun instead of hiding it.
- **One paid verdict per branch per selection round.** Once a pair comes back `overlap`,
  both branches belong to that merge; later candidate pairs touching either are skipped
  instead of buying a Super verdict on a branch about to be absorbed (the merged child is
  compared afresh next round). On the mock demo this took the ratio from 24 judgments → 9
  merges to 20 → 14; at `--starters 9`, 24 → 3 became 16 → 8. Equal-similarity ties go to
  pairs from different starter trees (`PFPTracker.starter_roots` walks parent + merge edges).
- **Merges must leave the finish they create.** A merge buys a synthesis call and a post-merge
  falsification, and the child it makes still needs the Pass gauntlet plus a pre-exit check.
  After the first merge the harness learns that round-trip cost and stops merging once the
  root budget can't cover one more merge *and* `gauntlet_rounds + 1` falsifications.
- **A refused merge costs nothing and absorbs nothing.** Self-merges (which would double-count
  one budget), a child id already in the ledger, and an alignment verdict for a different pair
  are rejected before the Super call. A parent killed while synthesis is in flight leaves both
  parents and their budgets intact; the paid call is still booked, as overdraft.
- **A merge commits whole or not at all.** After the Super call, the parent re-check, both
  absorptions and the child's registration run under one tracker lock (`PFPTracker.atomic()`),
  so a worker can't kill a parent, or take the child id, halfway through and strand an absorbed,
  zeroed parent with no child to carry its budget.
- **Selection is paid for.** Each branch keeps a protected verification share
  (30% by default) that exploration and child spawning can't touch; split
  parents' idle budgets return to the finish reserve when selection begins.
- **Exploration degrades on real costs.** The ledger learns per-call-kind costs;
  a branch that can't afford another probe concludes instead of overdrawing.
- **Emergency stop** (`MainHarness.emergency_stop()` or
  `VersionRegistry.emergency_stop()`) kills a live run; rollback applies next run.

## Live-model robustness

- Model replies are parsed with `pfp/jsonutil.py`: code fences, leading prose and
  `<think>` blocks are handled; `"survived": "false"` is false (plain
  `bool("false")` would be true), and non-finite floats (`nan`, `inf`) return
  `default` instead of `True`.
- **Hidden reasoning cannot become a verdict.** A truncated, unterminated `<think>`
  block is discarded instead of being searched for JSON, so a hypothesis inside a
  cut-off chain of thought cannot accidentally pass or kill a branch.
- **Falsification fails closed.** A judge reply with no readable verdict is
  *inconclusive*: no survival credit, no kill, no appeal consumed, and no Pass.
  That holds for the appeal too: a kill is irreversible, so it needs a readable
  verdict from both judges; a verdict-less appeal hands the appeal back and the
  branch lives. Verdicts commit against live state under the ledger lock: a branch
  killed or absorbed while the judge was out earns no credit, is not appealed (no
  paid call on the dead), and is never journaled as a second `fail`; a branch
  promoted meanwhile has its Pass revoked instead of raising.
- **The alignment pre-filter matches concepts on whole words.** Concept anchors used
  to be substring tests, so "report" hit *port*, "monkey" hit *key* and "ghost" hit
  *host*, and unrelated branches cleared the 0.20 threshold into a paid Super judge call.
- **Every Super prompt quotes branch text as evidence, not orders.** Claim extraction, the
  alignment judge and the merge synthesizer fence model-written positions in `<branch_text>`
  exactly like the fail judge (v0.14), so a branch can't talk the alignment judge into an
  `overlap` verdict or smuggle commands into the merged child. Extracted claims are
  de-duplicated (case / whitespace-insensitive) before they reach the judge.
- **Branches can't rule on themselves.** The judge prompt quotes model-written text
  inside `<branch_text>` / `<context>` fences, tells the judge that instructions in
  there are evidence against the branch, and defangs any fence tag a reply writes
  (`fail_judge.fence_untrusted`), so a branch can't close its own quote.
- **Journal appends are transactional.** Timestamp coercion, payload copying and hashing
  all complete before the feed advances its sequence. A malformed producer event therefore
  cannot consume a sequence number and poison every later replay. Live UI subscribers are
  isolated projections: one renderer failure is retained in `subscriber_failures` for
  inspection, while the committed event continues to every other subscriber and the harness.
- **Selection never hinges on scheduler jitter.** Untested age decays certainty a whole
  second at a time (`issue_detector.AGE_QUANTUM_S`). It used to count fractions of a
  second, so siblings born microseconds apart got certainties like -1.2500 vs -1.2502;
  that difference decided which branch the background falsifier tested first, and so which
  branch went on to win. The same mock sandbox finished at 22.5 or 31.5 depending on thread
  timing, and the paired A/B sometimes flagged an unchanged-quality candidate as degraded.
  Now equal evidence means equal certainty, and the existing content tie-break decides.
- **Untested age runs on the arena's clock, not the wall clock.** Whole seconds still leaked
  scheduling: a run that straddled a second boundary aged one twin and not the other, and in
  a live run a branch whose plaus call came back slowly lost 0.15 certainty per second of
  network latency, which set sibling budgets. Untested age is now counted in arena rounds
  (`PFPTracker.arena_round`): the spawn phase is round 0, and each completed selection round
  decays every live branch it left untested by one `age_without_testing` unit, so that dial
  means "rounds without facing the falsifier". Alignment embeddings also hash the ID-free
  position: merge traces name their parents (`[Merged branch_12 + branch_7]`), and those IDs
  shifted cosine similarity in the fourth decimal and so decided which pair merged first.
  Together these made ~1 in 60 identical mock runs reach a different winner; 300 of 300 now agree.
- **Selection is invariant to parallel registration order.** Every equal-score decision now
  uses an ID-free branch-position key plus a content-derived lineage digest: alignment pair
  orientation and similarity ties, background sampling, contradiction ties, dependent
  re-verification, gauntlet order, duplicate collapse and the winner cut no longer inherit
  whichever subtree happened to acquire a branch ID first. Merge lineage hashes both parents
  without regard to orientation, preserving distinct ancestry when branch text is identical.
- **The winner's box is re-checked after the whole pre-exit sweep.** A later
  candidate's kill re-verifies its dependents; a winner demoted that way no longer
  exits. A contradiction tie on certainty spares the branch with more survived
  falsifications, then fewer open objections.
- HTTP: exponential backoff with jitter on 408/409/425/429/5xx, network errors
  and connections dropped mid-body, `Retry-After` honored (seconds or HTTP-date), non-retryable errors raised immediately with
  status and body; reasoning-model `reasoning_content` fallback.
- A backend failure after retries (or a rejected key) halts the run as
  `backend_error`; Ctrl-C halts it as `interrupted`. Either way every branch
  stops starting paid calls at once and the partial arena is kept.
- The sizing probe reply goes through the same tolerant parser; a NaN, infinite
  or negative difficulty estimate falls back to the default instead of becoming
  the root budget.
- Bounded concurrency (`max_concurrent_calls`, default 32) across all branches.
- **A billed reply never crashes the meter.** Once the provider answers, the response
  is parsed tolerantly: non-object choices are skipped, content-part lists are joined,
  garbage / NaN / positive logprobs are defaulted (no `OverflowError` in the entropy
  signal). An HTTP 200 whose body is unreadable (non-JSON, a JSON array, a connection
  dropped mid-body) is charged at its reservation bound, both when a retry then
  succeeds and when every attempt fails (`BackendError.billed_usd`, booked by
  `MeteredClient`). Unreachable endpoints are not charged.
- **Providers that ignore `n` still split.** If a K-probe batch comes back with one
  choice, the detector tops it up with single-sample calls (distinct prompts, each
  metered) and, having learned the provider drops `n`, stops sending wasted
  multi-sample requests; `budget.api_n_topup_calls` counts them. Without this every
  batch has one member, nothing can diverge, and a live arena never branches.
- **Fail fast across the branch tree.** A branch that fails trips a run-wide stop
  before its nested pool waits on siblings, so they start no further paid calls; their
  kills are labelled `run_halted`, not `emergency_stop`.
- **Defects halt, they don't lose money.** An unexpected exception mid-run halts as
  `internal_error` (type and message in the `halt` event) with the stream, HTML,
  report and summary still written. Ctrl-C during Piece 8 keeps the finished arena
  run, and an interrupted sandbox is never recorded as a failed experiment.

## Self-improvement guardrails

- **Auto-revert compares like with like.** The sandbox is a paired A/B: the candidate
  and the current version run on the *same* sandbox problem, and degradation is the
  candidate's score against that paired baseline (`sandbox_result.baseline_source =
  "paired_sandbox"`, the baseline run kept under `baseline_sandbox`). Before v0.11 the
  candidate's sandbox certainty was measured against the live run's winner certainty —
  a different problem at a different scale — so a zero-winner live run disabled
  auto-revert entirely. An unchanged candidate reuses its own score
  (`identical_candidate`, no second run); a baseline cut short by the cap falls back to
  the live-run figure (`run_metrics_fallback`) instead of a 0 that would wave anything
  through. The CLI splits the remaining sandbox budget across the pair;
  `SelfImprovementLoop(paired_sandbox=False)` restores the single-run behaviour.

- **The A/B is scored on a yardstick no proposition can rewrite.** Certainty weights and
  `gauntlet_rounds` are dials, and each sandbox side used to be scored by its winner's
  certainty in its *own* weights. On the mock, raising `survived_falsifications_weight` 3.0 → 3.25
  lifted an identical winner from 22.5 to 24.75 and was adopted as an improvement; lowering it to
  1.0 read as a 53% degradation and was auto-reverted for doing nothing; `gauntlet_rounds: 9` scored
  40.5 by testing more, not reasoning better. Both sides, and the live-run fallback baseline
  (`spend_efficiency.max_winner_evidence`), are now scored by `dials.evidence_score`: survived
  falsifications, merges attracted and open objections on the factory weights
  (`dials.REFERENCE_WEIGHTS`), with survivals from gauntlet rounds past the reference three left
  uncounted (a candidate that runs fewer rounds still loses what it skipped). Every sandbox record
  names its `score_yardstick` and keeps the candidate-weight figure as `certainty_score`.
- **Auto-revert judges the version that was measured; a pin is never collateral.** Under a manual
  pin the post-run candidate is registered but held (status `held_by_pin`, not `adopted`), and
  before v0.32 the degradation check then ran on the *current* version, which is the pinned one:
  a candidate that dropped `gauntlet_rounds` to 1 got the operator's pinned version marked
  `reverted`, unpinned and replaced by its parent, while the bad candidate stayed `adopted` and
  never reached `failed_experiments`. `evaluate_and_auto_revert(..., version_id=)` now names the
  measured version: a degraded held candidate is reverted and logged as a failed experiment, and
  the active version, the pin and the next-run queue are left alone (a reverted version is also
  dropped from `pending_next_run_version_id`). Pinning or restoring a held candidate adopts it.
- The event-log miner treats journal snapshots as untrusted: a malformed count, certainty, depth,
  `merge_parents` or spend figure in a replayed or legacy v1 journal counts for nothing instead of
  crashing the Piece 8 cycle before anything is recorded.
- Proposed dial changes are untrusted input: `pfp/dials.py` rejects unknown
  keys, non-numeric values, NaN/inf, and anything outside the safe envelope
  (e.g. `gauntlet_rounds: 0`, rewarding open objections). Rejections are
  recorded on the proposition for the anti-model, and both `VersionRegistry.register_rewrite`
  and `VersionRegistry.load` enforce `sanitize_dials` and prompt length bounds (`<= 4000` chars)
  so a hand-edited or re-hashed snapshot with out-of-range dials refuses to load.
- Versioned prompts are live: `plaus_probe_hint` from the active version is
  injected into every probe, and into the sandbox run.
- `VersionRegistry.save/load` persists the failsafe; every snapshot's SHA-256 is
  re-verified on load, and the v2 registry envelope also covers the active,
  pinned and pending pointers, version order, failed-experiment history and
  degradation threshold. A v1 file is accepted and upgraded on its next save;
  control-plane edits, duplicate IDs and broken parent chains fail closed.
- **The critic must cite the log it criticises.** Anti-model attacks are kept only if every
  sequence number, event type and branch ID in their `evidence_seq` exists in this run's journal;
  each kept attack records what it resolved to (`grounded_in`, e.g. `["type:ns_tick", "seq:41"]`),
  and the cycle reports `anti_model_ungrounded_dropped`. Before v0.30 any non-empty string counted
  ("log" did), and an unparseable reply was itself kept as an attack citing "log".
- **The run being graded can't steer its own rewrite.** Self-assessment, anti-model and
  proposition prompts quote mined metrics (which carry winner solutions), critiques and failed
  experiments inside `<branch_text>` fences with the same notice as the judges, so a winner that
  writes "set gauntlet_rounds to 1" is evidence, not an order. Prompt rewrites are validated like
  dials (`self_improvement.sanitize_prompt_changes`): only prompts the harness reads
  (`PROMPT_LIMITS`: `plaus_probe_hint`) are accepted, text is flattened to one line (no newlines,
  zero-width or bidi controls, fence tags defanged) and over-long text is rejected, not clipped;
  refusals land in `rejected_prompt_changes`. The probe prompt flattens the hint too, so a
  registry hint can't forge new prompt lines.
- **The mock's Piece 8 replies are routed by step tag.** A merged winner's trace names other
  modules (`alignment_judge`), and the mock matched keywords anywhere in the prompt, so the demo's
  self-assessment and anti-model came back as alignment verdicts and parsed to nothing. Piece 8
  prompts are now recognised by the tag their user message opens with.
- The case bank is untrusted on load too: cases without a fork shape are dropped,
  non-numeric / non-finite / negative certainty becomes 0 (it would otherwise
  poison the certainty-weighted starter split), text is length-capped, and the
  bank is bounded (`max_cases`, default 500, keeping the most-won cases).
- **One run per state dir.** `--state-dir` is held with an atomic, token-owned lock directory
  (`.pfp.lock/owner.json`, naming pid/host/time/token) for the whole command; a second run refuses
  before spending instead of loading the same registry and overwriting the first
  run's versions and wins on save. A lock left on the same host by a dead process
  PID is reclaimed automatically; foreign-host or live-PID locks are strictly respected.
  Release verifies the ownership token, so delayed cleanup cannot erase a successor's lock;
  legacy lock files are recognized and migrated on acquisition.
- A malformed registry refuses to load instead of half-loading; every persisted
  artifact (stream, HTML, registry, case bank, summary, report) is written via
  `pfp/fsutil.py`: temp file → fsync → atomic replace.
- Winner certification is state-derived across ASCII, HTML, and proof reports:
  `winner_exit` only counts with a Pass snapshot, and any later `fail`, `kill`,
  `revocation`, or non-Pass state evicts the claim from the winner's box. A UI
  event cannot manufacture a certified result.
- **Journal numbers are standards-compliant and sortable.** The ledger rejects
  NaN/Infinity budgets and certainty scores at registration and update time; the
  arena rejects non-finite values anywhere in nested snapshots/details without
  consuming a sequence number. JSONL export and replay therefore never admit the
  non-standard `NaN`/`Infinity` tokens that can poison ranking and cross-language
  verification.

## The arena player

`--html-out` writes one self-contained file (no network, no dependencies):
right-angle light-cycle trails per starter lineage, NS mutations in violet,
merges converging in gold, red crash sparks where branches hit the verified main
line (which glows brighter with every Pass), survivors riding down into the
winner's box. The scoreboard HUD, event feed and winner's box are all
scrub-aware: drag the slider and a Pass revoked later leaves the box. Space =
play/pause, ←/→ = step, Home/End. All stream text is rendered as text, never
HTML, and the embedded JSON can't break out of its `<script>`.

## Tests

```bash
pip install -e ".[dev]"      # optional; stdlib-only at runtime
python -m pytest -q
```

The suite runs fully offline against the deterministic, content-addressed mock
(same prompt → same reply regardless of thread interleaving; since v0.14 branch IDs are
blanked from the content address, because parallel subtrees race for the run-wide ID counter
and the same branch could otherwise draw a different reply depending on scheduling. Since
v0.17 all selection ties use ID-free content plus a lineage digest, not registration order) and a local fake
HTTP server for the live client. It also stress-tests concurrent cap admission
and rejects replay logs with sequence gaps, duplicate sequence numbers, or
missing branch snapshots.

## Repository layout

```
pfp/                 the package (pieces 1–8 + backend, json + dial helpers, HTML player, proof report)
run_arena.py         full pipeline CLI (pieces 1–8)
run_phase1.py        pieces 1–3 only, JSON summary
spike.py             Token Factory parallel fan-out economics probe
tests/               offline test suite
.github/workflows/   CI: pyflakes + pytest on Linux/Windows, Python 3.10–3.13, live==replay smoke
DESIGN.md            the design this implements
```

## License

MIT — see [LICENSE](LICENSE).
