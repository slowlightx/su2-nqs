# SU(2) NQS — Project Tracker

*Last updated: 2026-10-29. This is the lean, dashboard-facing tracker — open questions, decisions, and status. The full day-by-day engineering/GPU-job log lives in [SU2NQS_all.md](SU2NQS_all.md) and [RESULTS.md](RESULTS.md); this file deliberately does not duplicate that detail.*

## 0. Regenerating the dashboard & PDF

Edit this file directly, then ask Claude to rebuild the dashboard and PDF from it. Suggested prompt:

> Regenerate the SU(2) NQS dashboard and PDF from `projects/clebsch_tree_nqs/SU2NQS_tracker.md`, following the format in §8. Update the existing dashboard artifact in place (don't create a new one).

- Dashboard artifact: https://claude.ai/artifact/KJRmEHKCJrmLFVcCEaTU6X
- Full engineering log (not duplicated here): `projects/clebsch_tree_nqs/SU2NQS_all.md`
- Format reference for how each section below maps onto the dashboard/PDF: see §8 at the end of this file.
- Resources: [su2nqs.pptx (Slides, SharePoint)](https://cornellprod-my.sharepoint.com/:p:/r/personal/yx789_cornell_edu/Documents/su2nqs.pptx?d=w2fb242caa9004001b66af2d0d5412184&csf=1&web=1&e=4GdQI0) — live file, not a copy.

## 1. Project overview

This project builds neural quantum states (NQS) with exact, non-Abelian SU(2) spin symmetry built directly into the architecture, applied to frustrated quantum magnets — primarily the **triangular-lattice J1-J2 Heisenberg antiferromagnet** (the square-lattice J1-J2 case is a secondary reproduction check, not a main target).

Mission statement (`AGENTS.md`): *"Build neural quantum states directly in non-Abelian SU(2) multiplicity spaces. Group theory fixes magnetic-index structure; learned models represent only the remaining fusion-path amplitudes and correlations."*

**Current direction (as of 2026-10-29, following discussion with Eunah):** the goal is a controlled variational ansatz — one that is not only competitive in energy with other NQS methods, but is provably in the correct SU(2)-symmetric sector, so that extrapolating toward larger system sizes can be trusted to preserve the physical features (exact total spin, no cross-sector contamination, correct momentum/little-group quantum numbers) the model is known to require. Beating GCNN's ground-state energy everywhere is no longer the sole success criterion; strict SU(2) control plus competitive quality is what lets ED-validated low-energy structure be extended past exact diagonalization. Strict SU(2) guarantees total-spin identity, but it does *not* by itself guarantee correct magnetic order, phase identification, or QSL identity — finite-size AFM, VBS, and QSL ground states can all be singlets, so this is a controlled, reproducible route to physical inference, not an architecture that answers the question in advance.

**Route for next steps:**
1. Validate small-system (ED-scale) low-energy branches — energies plus their exact S / momentum / little-group labels, counts, and transition weights.
2. Reach competitive large-system energies while keeping those same exact quantum numbers, cross-checked against ordinary NQS / DMRG / iPEPS.
3. Use the resulting spectra to determine whether J2 = 0 → 0.125 stays organized as a single 120° rotor family, or develops a distinct, size-stable organization compatible with a QSL candidate.

Working title: *Symmetry-controlled neural spectroscopy of frustrated magnets beyond exact diagonalization.*

## 2. Open questions

Three architectural routes are being pursued in parallel. For each, the question is the same: **is it novel, and is it feasible?**

### Item 1 — Ordinary Spatial-ViT for J1-J2 (physics baseline)
- Raised: 2026-10-20 · Owner: Yi · Priority: high
- Status (as of Oct 20 11:48 EDT review): ViT6 (99,904→174,400 params) completed 4 accepted updates but is **blocked by a broken cost/time-estimator** in the job gate that double-charges one-time compile cost, so no new audited 600-step endpoint yet. ViT12 is actively training for both couplings (J0, J.125), with ~5.2h / 7.1h wall time estimated remaining.
- Novelty: none claimed for itself — this is the established-architecture baseline used to anchor numerical comparisons, not the paper's novelty claim.
- Feasibility: high. Proven at 6×6 and 12×12, actively training. The sole blocker is a diagnosed, fixable estimator bug, not a fundamental limit.

### Item 2 — Free ClebschTree for the triangular lattice J1-J2
- Raised: 2026-10-20 · Owner: Yi · Priority: high
- Status: seven newly verified CT6 endpoints (J0/J.125 × widths 16/18 × binary/joint variants), cross-checked to 2.4e-18 discrepancy. The project's own stated conclusion: *"There is no established energy-and-variance win over ordinary ViT, nor a symmetry-only or QSL-specific advantage"* — best case is only ~0.83–2.03 combined standard errors below ViT. CT12 has **no optimized endpoint yet**; compute-cost gated (forecasts of 5.46–6.21h exceed the 5h training-phase limit).
- Novelty: not a numerical win over the baseline — the claim rests on methodology: rigorously exact-SU(2) states with verified reproducibility at this scale. Oct 17 framing: *"Promote the strict finite-width triangular ClebschTree to a bounded parallel METHODS mainline, not yet a superior-state claim."*
- Feasibility: 6×6 is solidly feasible and running; 12×12 is blocked on compute cost (full-Hamiltonian action ≈89s/configuration on an A100) — *"no optimized free-ClebschTree12 state yet: real native sampling/full-H/full-gradient costs must be established first."*

### Item 3 — ClebschTree + attention (combined route)
- Raised: 2026-10-20 · Owner: unassigned · Priority: high
- Status: repeated tests (local J0 6×6, J.125 controller, Oct 19 8-attempt run) all report zero or statistically unresolved benefit: *"CT+attention6: ...8 attempts, 0 accepted, no parameter change — No trained benefit."* CT+attention12 is blocked by a rank-deficient initialization bug; a fix was queued (`attention_rankfix_PENDING_20261019.json`) and reported "locally verified" on Oct 19 but not yet re-run at scale.
- Novelty: explicitly disclaimed in the project's own docs: *"Static attention must earn a same-manifold optimization/parameter/transfer benefit; it is not itself a strict-SU2 or long-range-entanglement breakthrough."* Hoped-for value is parameter-efficiency / size-transfer (6→12), not a larger variational manifold.
- Feasibility: weakest of the three. Lowest priority, blocked at 12×12 by the init bug, and at 6×6 has produced zero accepted updates across several independent attempts to date.

*(Item 1–3 status above is from the Oct 20 review and is deliberately not re-synced to the day-by-day log in SU2NQS_all.md — see §7.)*

## 3. Decisions made

*(D1–D6 are earlier execution-level decisions recorded in the full engineering log, `SU2NQS_all.md` §3 — not duplicated here. D7–D8 below are the research-direction decisions from today's discussion with Eunah, condensed.)*

### D7 — Strict sectors + competitive accuracy as the controlling criterion (2026-10-29, after discussion with Eunah)
Beating GCNN energies everywhere is no longer the sole success criterion. Success now requires exact SU(2) total-spin control together with competitive variational quality, validated progressively through ED, quality, and size sequences. Updates D1's original success criterion.

### D8 — Validate CT12 on Kim before further B200 production (2026-10-29)
CT12 energies remain far from competitive even after 32 B200 steps. Before more B200 investment, use Kim resources to isolate the actual source of slow/no energy lowering (same parent/width first, then a single bottleneck expansion if needed). No additional long B200 production for now.

## 4. Closed / retired

_Not populated here. The full engineering log's closed/retired bookkeeping lives in `SU2NQS_all.md` §4._

## 5. Next meeting

No recurring review meeting is set up for this project yet. Once one exists, agenda items drawn from the three open routes above — whichever one needs a go/no-go call next — would go here.

## 6. Workflow / documentation conventions

_Not populated yet. Describe here, in your own words, how you want this lean tracker to relate to the full engineering log and your own check-ins._

## 7. Open flags / caveats

- This file is intentionally lean: it carries the open questions, decisions, and status that matter for a quick read or a dashboard, not the day-by-day GPU-job log. That detail lives in `SU2NQS_all.md` and `RESULTS.md`.
- D7–D8 above are condensed from `SU2NQS_all.md` §3 (same dates, same substance, shorter text) — refer to that file for the full rationale and execution boundaries.
- The "raised" dates on items 1–3 are the date the novelty/feasibility question was first posed in this form (2026-10-20); the underlying experimental work has been running since mid-September.
- The Decisions, Closed, and Workflow sections are for things actually decided or settled — not things inferred from code or job logs.

## 8. Format reference (for regenerating the dashboard)

This section documents the convention the sections above follow, so the dashboard/PDF can be rebuilt mechanically rather than re-interpreted from scratch each time.

**§2 Open questions** — each `### Item N — Title` becomes one open-question card. Expected bullet fields directly under the heading, in this order: `Raised` (date), `Owner`, `Priority` (high/normal/low), then a paragraph each for `Status:`, `Novelty:`, `Feasibility:` (omit a paragraph if genuinely not applicable). Item N is used as the card's `item` chip.

**§3 Decisions made** — each decision is its own `### D<n> — Title (date)` heading followed by a short paragraph, with a `Rationale:` line (or an unlabeled second paragraph) underneath. Empty (just the placeholder italic line) → dashboard/PDF show an empty-state card instead of a list.

**§4 Closed / retired** — a flat bullet list, each bullet `**short text.** optional note`. Empty → empty-state card.

**§5 Next meeting** — free-text paragraph(s), shown verbatim if non-empty; otherwise the empty-state card.

**§6 Workflow** — a numbered list, each item `**Step name** — description`, optionally followed by a closing paragraph (shown as a trailing note). Empty → empty-state card.

**§7 Open flags / caveats** — a flat bullet list; each becomes one flagged card in the dashboard/PDF's "Flags & caveats" area.

General rule: a section that's just the italic placeholder text ("_Not populated yet..._") should regenerate as that section's empty state — don't invent content for it by reading README.md, AGENTS.md, or RESULTS.md. Only §1 (overview) and §2 (open questions) are meant to be refreshed from the project's own documents; §3–§7 are yours to fill in by hand (or condense from the full engineering log, as §3 does here).
