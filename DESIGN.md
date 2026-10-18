# PFP Darwinism — Design Document

**Pass, Fail, Possible — survival of the fittest reasoning.**

## Origin

This idea is ~3 years old (2023). It was scrapped for one reason: computing
power. Local hardware forced sequential processing; a trial of parallel
execution wasn't worth it (single-tenant local models serialize anyway — more
complexity, zero throughput gain).

Nebius changes the constraint. Hundreds of parallel model inferences are now
affordable, which makes this the right time to build it.

## The Concept

A reasoning flow starts at the top of an arena and falls downward. When it
hits an **issue** (uncertainty, a decision point), it **branches** — spawning
sub-agents per alternative. Sub-agents branch further at their own issues:
recursive fan-out, potentially hundreds deep. Branches that prove false are
pruned. Branches whose ideas **align merge together** — cooperation, not just
competition; the winner isn't the last one standing, it's what the survivors
built together. The flow continues until the solution is fully verified, then
exits into a winner's box.

Like a flowing river: splits around rocks, rejoins downstream, until the
solution is full proof.

**Not** token-probability sampling. **Not** round-robin turns. All branches
live at once, all the time.

## Vocabulary

- **Pass** — a branch that survived falsification. Verified.
- **Fail** — a branch proven wrong. Dies. No resurrection.
- **Possible** — a branch still alive, unresolved.
- **Plaus / plausibles** — known unknowns. Branches exploring visible forks.
- **NS / natural selection** — unknown unknowns. Blind variation: periodic
  fan-out with no trigger, the mutation rate.
- Tagline: **"Nebius — survival of the fittest."**

## The Arena (visualization + product UI)

Tron-like light-cycle aesthetic. Agents drop from the top, branch in different
directions, keep moving and splitting. A branch **fails by hitting the main
line** (the verified trunk) — contradiction with established truth kills it,
Tron rules. Survivors keep falling until they exit the bottom of the box into
a **winner's box** presenting the solution.

Colors are cosmetic only (agent distinction; can all be white).

**Scoreboard HUD:** tracks every agent — time alive + state. Battle-royale
spectator feel.

Demo scale: **3 starter agents** (sweet spot). Real scale: **9** (odd number —
no ties in arbitration). Fan-out math: 3 starters × ~12 branches × ~12 … ≈
3M total agents spawned (not concurrent — death and merge prune constantly).

The arena is both the demo video and the live product UI. The demo and the
deliverable are the same thing.

## The Eight Pieces

### 1. Issue detector — when to split

Branch first, read the issue off the divergence. At each step, fire K diverse
continuations blind (high temperature, cheap). If they cluster — no issue,
merge and flow straight. If they genuinely split — that split *is* the issue,
discovered not predicted.

Two split triggers:
- **Plaus:** measurable uncertainty — divergence across probe continuations
  (primary, behavioral), token entropy via logprobs (cheap, noisy), critic
  pass listing unjustified leaps.
- **NS:** blind fan-out on cadence (every 1–2s per agent, budget-derived).
  The mutation rate. Catches unknown unknowns — "we don't know what we
  don't know" is the argument *for* Darwinism: variation is blind, selection
  is the intelligence.

Branch count: bounded by novelty exhaustion (stop when new continuations are
remixes) and the depth dial.

Certainty ranking (drives compute allocation, scoreboard order, exit order):
survived falsifications (big plus), merges attracted (plus), open critic
objections (minus), age-without-testing (decays per arena round a live branch goes untested). Certainty is *revealed by
selection*, not computed upfront. Score weights are tunable dials.
When evidence ties, selection uses ID-free branch content and a content-derived lineage
digest. Parallel registration timing therefore cannot choose pair orientation, a
contradiction loser, falsification order, or which candidates make the winner cut.

### 2. Branch spawner — directives, mini-harnesses, budget

**Directive** (compact packet per branch): parent's reasoning so far
(compressed), the fork it's exploring, its state (Possible), its budget.
A branch never needs to ask what it's doing.

**Mini-harness:** every branch *is* a harness — carries the same machinery
recursively (own divergence detection, own sub-branching, own NS cadence).
Main harness runs the top level; mini-harnesses run subtrees; all events
bubble up. The fractal that scales 3 → hundreds with no central bottleneck.

**Budget** (single round — no escalation tranches):
- Root budget = **2 × difficulty-estimate** (one cheap probe call sizes the
  problem, then double as buffer). One round. Hard cap as safety rail
  (user-set, default ~$2).
- **Depth decay:** each level gets a fraction of its parent (×0.5). Deep
  branches are cheap by construction — this is why no max-depth kill is
  needed; the budget already bounds depth economically.
- **Certainty-weighted split** among siblings, with a floor (enough to prove
  itself or die, never zero). Selection governs resources, not just survival.
- **NS cadence derived from budget:** `cadence = expected_agent_seconds /
  budget_calls`. Tight budget → slower cadence.
- **Kill conditions:** budget exhausted, marked Fail, merged. (No max depth —
  vetoed; decay covers it.)
- **Finish reserve:** 10–15% of root held back; exploration can't touch it,
  only near-conclusion branches (high certainty, no open issues) can draw.
- **Graceful degradation:** as budget depletes, NS slows → stops → no new
  spawns → everything runs to conclusion. Exploration degrades into
  exploitation. Nothing dies mid-thought.

### 3. PFP tracker — the state ledger

Every branch carries: id, parent, state (Possible/Pass/Fail), certainty
score, time alive, budget remaining, fork under exploration. The branch's own
mini-harness updates it; the main harness aggregates.

Transitions are one-way: Possible → Pass (survived falsification),
Possible → Fail (contradiction proven / hit the verified line), anything →
absorbed on merge. No resurrection — dead stays dead; that's what keeps the
arena honest.

The arena feed and scoreboard read straight off these records.

### 4. Alignment detector — spotting mergeable branches

Two stages:
1. **Embedding pre-filter** (cheap): cosine similarity on branch positions.
   Recall only — similar-sounding ≠ compatible.
2. **Judge call** (Super): extract each branch's core claims, compare
   claim-by-claim. Overlap → alignable. Contradiction → leave them to fight.
   Disjoint → unrelated, walk away.

Safety net: a bad merge isn't fatal. Merged branches start Possible and must
survive falsification like everything else. The detector needs recall;
selection handles precision.

### 5. Merge synthesizer — combining aligned branches

A Super call unifies the branches: keeps what's compatible, drops what's
redundant, preserves provenance (which parent contributed what). The child
starts Possible, never Passed.

Economics of cooperation: parents are absorbed (terminal), and their
remaining budget **transfers to the child**. A merged branch is born richer
than either parent — "the better way to go" is literally true in the system.

### 6. Fail judge — falsification as a heartbeat

Not a gate — a schedule. Falsification runs on events:
- **After every merge** (new claims → verify before going anywhere)
- **Pass-readiness gauntlet** (certainty crosses threshold → 3 rounds, survive all)
- **Contradiction trigger** (alignment detector finds clash → falsify both, loser dies)
- **Background sampling** (every T seconds, probe the highest-certainty untested branch)
- **Pre-exit** (nothing enters the winner's box without a final pass)

**Pass is revokable.** A branch verified at t=20s can be demoted at t=40s;
the main line rewrites and dependents re-verify or die. Every Pass carries
provenance: what it survived, when, against which falsifier angle (logic,
counterexample, assumption audit — rotated so branches can't overfit).

**One appeal per kill:** re-test with a different judge; survive and the
verdict's overturned. Bounded, no infinite regress.

Plausible is the starting line, not the finish. If several branches all
survive, the higher-certainty one exits first. The system promises whatever
exits *survived* — not that only one can.

### 7. Arena feed — the single event stream

Every component emits events: spawn, branch, merge, fail, pass, NS tick,
budget change. The feed is append-only — history never rewrites (a revoked
Pass is itself an event). Every event carries a timestamp and the branch's
full state snapshot.

The Tron visualization, the scoreboard, and the winner's box are dumb
renderers on this stream. Record once, render anywhere: live UI, demo video,
post-run replay — same data.

### 8. Self-improvement loop — the system learns

1. **Log:** mine the arena stream for what worked — winning branch shapes,
   effective falsifier angles, merge success rates, spend-vs-certainty.
2. **Recall:** new runs start seeded with past wins (case-based). Fewer blind
   branches needed over time — this is how it gets faster.
3. **Self-assessment:** post-run summary — what worked, what to improve.
4. **Anti-model:** a purely negative critic that attacks every bit of the
   run. Grounded in the event log (evidence), not vibes — unbounded
   negativity without evidence is noise.
5. **Proposition:** synthesis of self-assessment + anti-model critique into
   concrete changes (dials, prompts, thresholds).
6. **Self-rewrite + sandbox:** changes applied to its own code, then run
   against a test problem in isolation. Adopted only if coherent.

**Failsafe (drift protection):**
- Version registry: every rewrite snapshots code/config/prompts, hashed,
  with proposition + sandbox result attached.
- Auto-revert on degradation past threshold (no human needed).
  Degradation is measured on a fixed yardstick (factory certainty weights, reference
  gauntlet length), never on the dials under test: a rewrite that only changes how
  certainty is counted must not look like better or worse reasoning.
- Manual pin: any version, anytime; all future runs use it.
  A candidate proposed under a pin is held, and only it can be reverted for its own
  sandbox score: the pinned version is never judged on evidence it didn't produce.
- Run-boundary switching: versions change between runs, never mid-arena.
  Emergency stop kills the run; rollback applies next.
- Failed versions feed back into the anti-model as failed experiments.

## Hackathon context

- **Contest:** Nebius x NVIDIA Global AI Hackathon ("Best AI Apps and Agents"
  track). Deadline: Oct 30, 2026, 12:00 PM CDT. Fully online. No payment
  required to enter or win (per official rules).
- **Model requirement:** at least one NVIDIA open-source model, on Nebius
  Token Factory or Nebius AI Cloud.
- **Model picks (live catalog):**
  - `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` ($0.06/$0.24 per 1M) — branch
    fan-out workhorse. Start here.
  - `nvidia/nemotron-3-super-120b-a12b` ($0.30/$0.90) — merge judge,
    alignment judge, falsifier.
  - Ultra ($1.00/$3.00) skipped — too pricey for fan-out.
- **Free path (verified, no card):** Token Factory $1 trial + Builder Program
  $25 Token Factory credit (email verification, 90-day expiry). ~$26 ≈
  140,000+ Nano calls. The $5 AI Cloud GPU trial needs a linked card —
  ruled out; PFP is pure inference orchestration and doesn't need it.
- **IP boundary:** the KUH framework stays proprietary. PFP uses the
  concept with renamed labels (Pass/Fail/Possible, plaus, NS). Clean-room:
  no KUH references in code, comments, or docs.

## Demo video plan (≤ 3 min)

1. The problem — one hard question, 10 seconds.
2. Ignition — 3 agents take off, legend in corner.
3. The run — ~90s of branching, crashing, merging. The money.
4. The exit — survivor enters winner's box, solution + proof trail.
5. The line — "Nebius — survival of the fittest."

## Build order

1. Spike: parallel Nano fan-out economics (`spike.py`).
2. Core harness: pieces 1–3 (split, spawn, track).
3. Selection: pieces 4–6 (align, merge, falsify).
4. Arena: piece 7 (event stream → Tron visualization + scoreboard).
5. Meta: piece 8 (learning loop + failsafe).
6. Video, repo polish, Devpost submission.
