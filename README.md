# NeuroGrid Go/No-Go
**Cognitive game focused on reaction time with auto-adjusting difficulty**
Maryam Sadat Ashena · Bachelor's of Computer Science · June 2026

---

## Overview

This system was designed for clinicians and their patients for a fully transparent and explainable Go-NoGo game based on the SART research. It presents a detailed telemetry for each session while estimating the player's state of alertness and adapting the difficulty accordingly.

---

## Repository Structure

```
main.py                       # Main Game + Metrics Engine
behavioral_state_estimator.py # Estimates the player's live state
adaptation_policy_engine.py   # Adapts difficulty based on telemetry and the player's state
generate_plots.py             # Plots telemetry features
neurogrid_config.json         # A configuration file to set the base values
sustain_replay_analysis.md    # Replay analysis behind the sustain_required setting
test_behavioral_state_estimator.py # Unit tests for the estimator
test_adaptation_policy_engine.py   # Unit tests for the policy engine
```

---

## Running the Game

Needs Python 3.10+, pygame, and matplotlib (`pip install pygame matplotlib`).
Make sure all files are in the same folder, run main.py, enter name and session ID and the game starts.
Use "P" for pause and "Q" for quit.

To run the test suite:
```bash
python -m unittest test_behavioral_state_estimator.py test_adaptation_policy_engine.py
```

---

## Session Output Files

| File | Contents |
|---|---|
| `experimental_scorecard_summary.json` | The full per-session summary — completeness check, pauses, baseline, performance, vigilance profile, co-occurring states, alerts |
| `trial_summary_telemetry.csv` | One row per trial — outcome, RT, press counts |
| `raw_event_telemetry.csv` | Every event with a millisecond-precision timestamp |
| `estimator_trace.csv` | The estimator's output per trial — problem state, evidence, headroom, confidence |
| `policy_trace.csv` | Every knob change the policy made, with the reason and trigger |

How to generate plots from a session:
```bash
python generate_plots.py <file_prefix>
```
`<file_prefix>` is the shared prefix on all the files from one session, e.g.
`sub-Mary_ses-001_20260620_073815_`.

---

## Paradigm Design

NeuroGrid is a Go/No-Go task based on the Sustained Attention to Response
Task (SART). 75% of trials are Go (press space), 25% are No-Go (withhold).
The stimulus always appears in the same spot on screen, so spatial
attention isn't a factor — only sustained attention and inhibition are
being measured.

Go and No-Go stimuli use the Okabe-Ito colorblind-safe palette: a blue
circle for Go, and an orange-to-blue rounded square for No-Go that gets
visually closer to the Go stimulus as `similarity_level` increases.

A session has three phases. Buffer trials (10) are thrown away to let
the player settle in. Calibration trials (up to 25) establish the
player's baseline reaction time. After that the session is Active, and
the adaptation system starts adjusting difficulty.

One thing calibration alone doesn't account for: most people get faster
in the first few minutes just from warming up, not because they're more
attentive. If that's not corrected for, normal speeding-up gets read as
"the baseline is wrong" and later RTs look like a problem when they're
actually just a return to normal. To fix this, the system watches
block-by-block RT during the active phase and re-anchors the baseline
once it plateaus — the moment warm-up has clearly stopped.

---

## Adaptation Architecture

The adaptation system is three separate pieces, each with one job:

```
MetricsEngine  ->  features dict  ->  BehavioralStateEstimator
                                            |
                                       estimate dict
                                            |
                                   AdaptationPolicyEngine  ->  knob changes
```

MetricsEngine watches raw trial data and turns it into a small set of
numbers (omission rate, RT drift, inhibition failure rate, late-press
rate, etc). The Estimator turns those numbers into a state label and a
headroom vector. The Policy looks at that output and decides whether to
change a difficulty knob, and if so, which one.

There are two kinds of output from the Estimator, handled differently
on purpose. Problem states (ATTENTION_LAPSE, PERFORMANCE_DECLINE,
HIGH_IMPULSIVITY, TIMING_PRESSURE) are picked by argmax — only one is
reported as the active state per trial, even if more than one is
elevated, because acting on several at once would make it hard to tell
which knob change actually helped. If more than one state really is
elevated, that's still visible in the trace files and the scorecard's
`Co_Occurring_State_Elevations` section, just not acted on
simultaneously.

Headroom (speed, discrimination, inhibition) is the opposite — all
three are independent and can be true at once, since being fast, being
accurate, and being good at withholding aren't mutually exclusive.
Headroom is what lets the system tighten difficulty when the player is
doing well, not just ease it when they're struggling.

A few design choices worth calling out:

**Hysteresis** — switching the reported problem state from one active
state to another requires the new one to clearly beat the old one (by
more than 0.10 evidence), not just edge ahead by a hair. Without this,
two close states can flicker back and forth every trial on noise alone.

**Confidence gating** — a piece of evidence doesn't get used until
there's enough data behind it, but gating too bluntly can backfire: if
a severe lapse means fewer hits to compute statistics from, gating the
whole state on RT-confidence would hide the lapse exactly when it's
worst. So each component of evidence is gated on its own data
requirement, not the state as a whole.

**Rollback-first** — when a problem state triggers an ease, the policy
first checks whether it recently tightened the relevant knob itself. If
so, it reverts that change instead of stacking a new ease on top —
undoing your own action is safer than adding more changes when you
don't know which one is actually at fault.

**Dynamic precedence** — if more than one problem state is sustained at
once, the policy responds to whichever has the highest evidence right
now, not a fixed priority order. A fixed order would let a weak but
high-priority state mask a genuinely worse one.

### Behavioral State Estimator

Four problem states:

- **ATTENTION_LAPSE** — missed Go trials and inconsistent reaction times.
  Based on omission rate and RT coefficient of variation.
- **PERFORMANCE_DECLINE** — reaction times trending slower over the
  session (vigilance decrement). Based on RT drift (SDI) and omission
  rate.
- **HIGH_IMPULSIVITY** — pressing on No-Go trials, especially fast ones,
  plus anticipatory Go presses. Based on inhibition failure rate,
  anticipation rate, and commission speed.
- **TIMING_PRESSURE** — genuine motor responses landing just after the
  stimulus window closes. Based on late-press rate. This is distinct
  from a lapse — the player did respond, the window was just too short.

If none of the four clears the `flow_floor` threshold, the reported
state is `OPTIMAL_FLOW`.

### Adaptation Policy Engine

Each problem state has a default knob it eases (see the knob/signal
table below), but rollback is checked first. If the targeted knob is
already at its limit, the policy escalates to a broader ease instead of
holding indefinitely with no recourse.

A state has to clear its evidence threshold for `sustain_required`
consecutive trials before anything happens, and after any knob change
there's a `cooldown_trials` window before the next change is allowed.
Both exist to stop the system reacting to single-trial noise.

### Adaptive Knobs

| Knob | Range | Step | Targets |
|---|---|---|---|
| `exposure_ms` | 300–800 | 50 | TIMING_PRESSURE (ease), speed headroom (tighten), ATTENTION_LAPSE (global ease) |
| `isi_ms` | 600–1500 | 100 | PERFORMANCE_DECLINE (ease), speed headroom (tighten), ATTENTION_LAPSE (global ease), TIMING_PRESSURE (fallback if exposure_ms is maxed) |
| `go_probability` | 0.60–0.85 | 0.05 | HIGH_IMPULSIVITY (ease), inhibition headroom (tighten) |
| `similarity_level` | 1–5 | 1 | discrimination headroom (tighten), ATTENTION_LAPSE (global ease, reduces similarity) |

---

## Configuration

All the provisional numbers above — and a few more, like task timing
and confidence minimums — live in `neurogrid_config.json` instead of
being hardcoded. The game loads this file at startup. If a key is
missing, it falls back to a built-in default, so the file only needs to
contain whatever a clinician actually wants to change. If the file is
missing or malformed, the game still runs on defaults and prints a
warning instead of crashing.

---

## Clinician Controls (In-Session)

| Key | Effect |
|---|---|
| `SPACE` | Respond to a Go stimulus |
| `P` | Pause (only works between trials, not mid-stimulus) |
| `R` | Resume from pause — starts a fresh 10-trial buffer before estimation and adaptation resume |
| `Q` | Save all data and quit (only works while paused) |

---

## Scorecard

`experimental_scorecard_summary.json` has 13 top-level sections:

- **Session_Completeness_Check** — whether the session reached the
  active phase and collected enough trials to be analyzed
- **Session_Pause_Log** — number of pauses and how many buffer trials
  they cost
- **Protocol_Metadata_Header** — participant/session ID, timestamp, RNG
  seed and hash (for reproducing the exact stimulus sequence)
- **Protocol_Configuration_Logs** — every starting threshold and timing
  value used this session
- **Data_Quality_Verification** — calibration status and baseline
  confidence rating
- **Baseline_Reanchoring** — whether the plateau-based re-anchoring
  fired, and at which trial
- **Baseline_Phenotype_Anchor** — the player's effective RT baseline
  (mean, SD, CV)
- **Active_Phase_Performance_Profile** — accuracy counts, press
  taxonomy (redundant/late/noise), RT distribution
- **Vigilance_Decrement_Chronological_Profile** — early/mid/late
  tertile comparison (needs at least `min_trials_for_vigilance_profile`
  active hits)
- **Advanced_Attentional_Biomarkers** — trial-to-trial RT variability
  (MSSD) and post-error slowing
- **Co_Occurring_State_Elevations** — how often more than one problem
  state was elevated at the same time
- **Terminal_Active_Window_Snapshot** — the final rolling RT mean and
  drift index at the end of the session
- **Experimental_Alerts_Summary** — count of distinct lapse episodes
  and inhibition-failure bursts. Note: this uses its own older
  threshold logic, separate from the Estimator's, so its numbers won't
  line up exactly with the per-trial problem-state trace — needs
  reconciling into one system later.

---

## Provisional Thresholds — What Needs Clinical Validation

| Parameter | Current value | What validation is needed |
|---|---|---|
| `flow_floor` | 0.30 | What evidence level is genuinely too low to call a problem — needs comparison against normative SART data |
| `problem_action_threshold` | 0.50 | Sensitivity vs. false-trigger tradeoff, needs multi-participant testing |
| `sustain_required` | 5 | Justified by an offline replay of one real session (see [sustain_replay_analysis.md](sustain_replay_analysis.md)), not yet validated across participants |
| Feature weights | see `neurogrid_config.json` → `estimator.problem_weights` | Needs empirical derivation from labeled, multi-participant data, ideally EEG-anchored |
| Feature thresholds | see `neurogrid_config.json` → `estimator.thresholds` | Directionally based on SART literature, but the exact cutoff numbers are engineering guesses |
| `late_press_window_ms` | 200 | Needs comparison against motor response time literature for this population |
| `min_trials_for_vigilance_profile` | 15 | A judgment call for a defensible 3-way split, not derived from a power analysis |

---

## Explicitly NOT Doing (Deferred to Future Work)

1. **PCG distractor field** — a procedurally generated background was
   considered to increase realism, but it would mix attentional capture
   with the sustained-attention signal the task is meant to measure.
   Left for a separate paradigm variant.
2. **Cross-session persistence** — tracking a participant's history and
   knob trajectory across sessions needs a real database and, more
   importantly, ethical approval for storing longitudinal data.
   Deferred to the clinic phase.
3. **Audio cue for ATTENTION_LAPSE** — this was actually built and then
   removed. An audio alert when a lapse sustains risks changing the
   very attentional state it's trying to measure, so it got pulled
   rather than kept.
4. **Gaze dispersion / fixation duration** — eye-tracking would be a
   strong complement to this behavioral pipeline, but it needs hardware
   access this project doesn't have.
5. **QEEG / clinic connection** — the Estimator's interface is built to
   be swappable specifically so a neurophysiological signal could plug
   in later, but the actual hardware integration is clinic-phase work.
6. **ML-based estimator** — this was an active decision, not a time
   constraint. A model trained on one participant's unlabeled data
   would be less defensible than explicit, citeable thresholds, not
   more. The swappable interface means a learned estimator can replace
   the rule-based one later, once there's real labeled multi-participant
   data to train and validate it on.
7. **Full app/menu UI** — a patient-facing interface with session
   management is real future work, but it doesn't change the technical
   contribution here and isn't worth the time right now. The current
   interface assumes a researcher is running the session.

---

## Dependencies

```
python >= 3.10
pygame >= 2.0
matplotlib >= 3.5
```
No other libraries. No ML frameworks.

---

## Contact
mary.ashena@gmail.com

---

*All thresholds provisional. See [neurogrid_config.json](neurogrid_config.json) and
[sustain_replay_analysis.md](sustain_replay_analysis.md) for parameter justifications.*
