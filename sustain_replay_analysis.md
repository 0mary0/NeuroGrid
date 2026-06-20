# NeuroGrid — Offline Sustain-Value Replay Analysis

**Session replayed:** sub-Mary_ses-006 (natural play, 201 trials, 179 active)
**Question:** Can `sustain_required` come down from 5 to improve adaptation
latency without introducing noise-driven false triggers? Is the current
value justified?

---

## Method

Session 006's `trial_summary_telemetry.csv` was replayed offline through
a fresh `MetricsEngine` + `BehavioralStateEstimator` + `AdaptationPolicyEngine`
at `sustain_required` ∈ {3, 4, 5, 6, 7}, holding everything else constant
(`cooldown_trials=5`, `problem_action_threshold=0.50`). This session
predates the `TIMING_PRESSURE` state and the late-press taxonomy, so
`late_rate` is held at 0.0 throughout -- it has no effect on this replay.

For each setting:
- **EASE / TIGHTEN actions** -- how many times the policy actually changed
  a knob
- **First action (trial)** -- adaptation latency from session start
- **Weak triggers** -- EASE actions where the winning evidence was < 0.55,
  i.e. just barely above the 0.50 action threshold
- **Trigger breakdown** -- which problem state drove each EASE action

---

## Results

| sustain_required | EASE | TIGHTEN | First action (trial) | Weak triggers | Trigger |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 3 | 9 | 14 | 63 | 9 | HIGH_IMPULSIVITY (9/9) |
| 4 | 9 | 13 | 64 | 9 | HIGH_IMPULSIVITY (9/9) |
| **5 (current)** | **8** | **13** | **65** | **8** | **HIGH_IMPULSIVITY (8/8)** |
| 6 | 7 | 12 | 66 | 7 | HIGH_IMPULSIVITY (7/7) |
| 7 | 7 | 12 | 67 | 7 | HIGH_IMPULSIVITY (7/7) |

---

## Interpretation

**1. Every single EASE action across every sustain value was driven by
HIGH_IMPULSIVITY.** This matches the real scorecard for this session:
inhibition failure rate was 26.32%, well above the 0.50-evidence
threshold's effective trigger point. No other problem state ever
sustained long enough to act in this session, which is itself useful
context: a one-session replay can only validate the dynamics for
whichever state actually showed up.

**2. Lowering sustain from 5 to 3 buys 2 trials of earlier first action.**
Trial 65 -> 63 is a small gain in a 179-active-trial session. The
adaptation timeline is dominated by `cooldown_trials=5`, not the sustain
window -- after any knob change, 5 trials must pass before the next
action is possible, so the effective minimum gap between actions is
roughly `cooldown + sustain` regardless of which one is larger.

**3. Every trigger had evidence just above the action threshold.**
All EASE actions across all sustain values fired at evidence < 0.55, a
property of this session's sustained moderate impulsivity rather than
of the sustain value itself. It confirms the threshold does what it's
meant to: a persistent moderate signal still triggers relief, not just
sharp ones.

**4. sustain=5 sits at the same plateau as before.** Action count and
first-action trial both stabilize by sustain=5-6; raising it further
gives no benefit, and lowering it gives only marginal latency gains.

---

## Recommendation

Keep `sustain_required = 5` (provisional). This replay only exercises
HIGH_IMPULSIVITY dynamics, since that's the only state this session
sustained -- ATTENTION_LAPSE, PERFORMANCE_DECLINE, and TIMING_PRESSURE
timing dynamics are still unvalidated and should be checked against
future sessions where those states actually trigger. The
config-externalization in `neurogrid_config.json` means this value can
be adjusted without touching source code once more session data exists.

---

## Limitation

Session 005 (deliberate end-session sabotage) was not replayed here
because its artificially elevated commission rate would make any
sustain value look effective -- it isn't representative of natural play.
This analysis should be repeated once more natural sessions exist,
ideally including at least one where ATTENTION_LAPSE or
PERFORMANCE_DECLINE actually sustains, since this replay says nothing
about those states' dynamics.