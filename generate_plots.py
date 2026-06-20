"""
NeuroGrid Session Plot Generator
==========================================================================
Reads a completed session's exported telemetry and produces diagnostic
plots:

  1. RT-over-time with policy-action markers (colour-coded by reason)
     and LATE_RESPONSE rug marks along the bottom axis
  2. Evidence-vector-over-time trace: all four problem states
     (ATTENTION_LAPSE, PERFORMANCE_DECLINE, HIGH_IMPULSIVITY,
     TIMING_PRESSURE) with shaded bands where >= 2 states are
     co-elevated above the flow_floor simultaneously
  3. Vigilance tertile chart (mean RT + inhibition-failure rate)
  4. Adaptive knob trajectory (exposure/ISI/go_probability/
     similarity_level as step plots)

Usage:
    python generate_plots.py <file_prefix>

`file_prefix` must match NeuroGrid's own export naming convention.
    the script looks for:
    <prefix>experimental_scorecard_summary.json
    <prefix>trial_summary_telemetry.csv
    <prefix>policy_trace.csv
    <prefix>estimator_trace.csv   (optional -- plots 2 skipped if absent)
"""
import sys
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mc
import numpy as np

# ======================================================================
# COLOUR PALETTE  (Okabe-Ito throughout for colorblind safety)
# ======================================================================
# Policy-action reason colours
REASON_COLORS = {
    "headroom_tighten": "#0072B2",   # blue   -- task got harder
    "default_ease":     "#E69F00",   # orange -- normal targeted ease
    "rollback":         "#9B59B6",   # purple -- undid a recent tighten
    "escalation":       "#D55E00",   # red    -- last-resort broad ease
}
DIRECTION_MARKER = {"tighten": "^", "ease": "v"}

# Problem-state evidence colours
STATE_COLORS = {
    "Ev_Attention_Lapse":     ("#009E73", "Attention Lapse"),
    "Ev_Performance_Decline": ("#D55E00", "Performance Decline"),
    "Ev_High_Impulsivity":    ("#CC79A7", "High Impulsivity"),
    "Ev_Timing_Pressure":     ("#56B4E9", "Timing Pressure"),   # sky blue -- new
}

CO_ELEVATION_COLOR = "#F0E442"   # yellow -- co-occurring elevated states band


# ======================================================================
# DATA LOADING
# ======================================================================
def load_scorecard(prefix):
    with open(f"{prefix}experimental_scorecard_summary.json") as f:
        return json.load(f)


def _load_csv(path):
    if not Path(path).exists():
        return None
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("Trial_ID")]


def load_trial_summary(prefix): return _load_csv(f"{prefix}trial_summary_telemetry.csv")
def load_policy_trace(prefix):  return _load_csv(f"{prefix}policy_trace.csv")
def load_estimator_trace(prefix): return _load_csv(f"{prefix}estimator_trace.csv")


# ======================================================================
# PLOT 1: RT TIMELINE WITH POLICY ACTIONS + LATE-PRESS RUG MARKS
# ======================================================================
def plot_rt_with_actions(trial_rows, policy_rows, scorecard, out_path):
    hit_trials  = [(int(r["Trial_ID"]), float(r["Primary_RT_MS"]))
                   for r in trial_rows if r["Accuracy_Outcome"] == "HIT"]
    comm_trials = [(int(r["Trial_ID"]), float(r["Primary_RT_MS"]))
                   for r in trial_rows if r["Accuracy_Outcome"] == "COMMISSION"]
    # LATE_RESPONSE: real motor responses that arrived after the window
    # closed. RT is stored as -1 (no valid window RT), so we show them
    # as rug marks along the bottom rather than on the RT axis.
    late_trial_ids = [int(r["Trial_ID"])
                      for r in trial_rows if r["Accuracy_Outcome"] == "LATE_RESPONSE"]

    fig, ax = plt.subplots(figsize=(13, 5))

    if hit_trials:
        xs, ys = zip(*hit_trials)
        ax.plot(xs, ys, color="#0072B2", linewidth=1, alpha=0.6, zorder=2)
        ax.scatter(xs, ys, color="#0072B2", s=14, zorder=3, label="Go HIT RT")
    if comm_trials:
        xs, ys = zip(*comm_trials)
        ax.scatter(xs, ys, color="#D55E00", marker="x", s=45, zorder=4,
                   label="No-Go COMMISSION RT")

    baseline    = scorecard["Baseline_Phenotype_Anchor"]["Effective_Mean_Processing_Speed_ms"]
    baseline_sd = scorecard["Baseline_Phenotype_Anchor"]["Effective_RT_Sigma_SD_ms"]
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Re-anchored baseline ({baseline:.0f} ms)")
    ax.axhspan(baseline - baseline_sd, baseline + baseline_sd,
               color="gray", alpha=0.08)

    rb = scorecard.get("Baseline_Reanchoring", {})
    if rb.get("Rebaselined") and rb.get("Rebaseline_Trial_ID"):
        rt_id = rb["Rebaseline_Trial_ID"]
        ax.axvline(rt_id, color="black", linestyle=":", linewidth=1.2, alpha=0.6)

    all_rts = [y for _, y in hit_trials] + [y for _, y in comm_trials]
    data_top  = max(all_rts) if all_rts else baseline + baseline_sd
    marker_y  = data_top * 1.12
    ax.set_ylim(bottom=0, top=marker_y * 1.08)

    if rb.get("Rebaselined") and rb.get("Rebaseline_Trial_ID"):
        ax.text(rb["Rebaseline_Trial_ID"], marker_y * 1.05, " re-baseline",
                fontsize=8, rotation=90, va="top")

    # Policy action markers
    seen_reasons = set()
    if policy_rows:
        for row in policy_rows:
            action = row.get("Action")
            if not action or action in ("HOLD", "COOLDOWN"):
                continue
            tid       = int(row["Trial_ID"])
            direction = action.lower()
            reason    = row.get("Reason") or "default_ease"
            color     = REASON_COLORS.get(reason, "#999999")
            marker    = DIRECTION_MARKER.get(direction, "o")
            ax.axvline(tid, color=color, linestyle="-", linewidth=0.8, alpha=0.3)
            ax.scatter([tid], [marker_y], color=color, marker=marker, s=55,
                       zorder=5, edgecolor="black", linewidth=0.5)
            seen_reasons.add((reason, direction))

    # Late-press rug marks: a short vertical tick at y=0 for each late
    # trial, so timing-pressure events are visible alongside the RT trace
    # without requiring a separate axis.
    if late_trial_ids:
        rug_y = data_top * 0.02   # just above zero
        ax.scatter(late_trial_ids, [rug_y] * len(late_trial_ids),
                   color="#56B4E9", marker="|", s=60, linewidths=1.5,
                   zorder=5, label="Late motor response (LATE_RESPONSE)")

    ax.set_xlabel("Trial")
    ax.set_ylabel("Reaction Time (ms)")
    ax.set_title("Reaction Time Over Session  —  with Adaptive Policy Actions")

    handles, labels = ax.get_legend_handles_labels()
    for reason, direction in sorted(seen_reasons):
        handles.append(mpatches.Patch(color=REASON_COLORS.get(reason, "#999999")))
        labels.append(f"{direction} ({reason})")
    ax.legend(handles, labels, loc="upper left", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ======================================================================
# PLOT 2: EVIDENCE-VECTOR TRACE  (requires estimator_trace.csv)
# ======================================================================
def plot_evidence_trace(estimator_rows, out_path):
    if not estimator_rows:
        return False

    xs = [int(r["Trial_ID"]) for r in estimator_rows]

    fig, ax = plt.subplots(figsize=(13, 4.5))

    # --- Four problem-state evidence lines ---
    for col, (color, label) in STATE_COLORS.items():
        if col not in estimator_rows[0]:
            continue   # graceful skip if column absent (old session file)
        ys = [float(r[col]) for r in estimator_rows]
        ax.plot(xs, ys, label=label, color=color, linewidth=1.4)

    # --- Co-elevation shading ---
    # When >= 2 states are simultaneously above the flow_floor (0.30),
    # shade the trial band in yellow to make mixed-state periods obvious.
    # Uses the Elevated_States column which lists all above-floor states
    # as pipe-separated names.
    if "Elevated_States" in estimator_rows[0]:
        co_xs = [int(r["Trial_ID"]) for r in estimator_rows
                 if len([s for s in r["Elevated_States"].split("|") if s]) >= 2]
        if co_xs:
            # Convert to a list of (start, end) segments for contiguous runs
            segments = []
            seg_start = co_xs[0]
            prev = co_xs[0]
            for x in co_xs[1:]:
                if x - prev > 2:          # gap -- close old segment
                    segments.append((seg_start, prev))
                    seg_start = x
                prev = x
            segments.append((seg_start, prev))

            for seg_start, seg_end in segments:
                ax.axvspan(seg_start - 0.5, seg_end + 0.5,
                           color=CO_ELEVATION_COLOR, alpha=0.25, zorder=0)

    # --- Reference lines ---
    ax.axhline(0.50, color="gray", linestyle="--", linewidth=0.9,
               label="Action threshold (0.50)")
    ax.axhline(0.30, color="gray", linestyle=":", linewidth=0.7,
               label="Flow floor (0.30)")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Problem-state evidence")
    ax.set_title("Behavioral State Estimator — Problem-State Evidence Over Session")

    # Build legend; insert co-elevation patch after the state lines if needed
    handles, labels = ax.get_legend_handles_labels()
    if "Elevated_States" in estimator_rows[0] and co_xs:
        co_handle = mpatches.Patch(color=CO_ELEVATION_COLOR, alpha=0.4,
                                   label="Co-elevated states (≥ 2 above floor)")
        handles.insert(len(STATE_COLORS), co_handle)
        labels.insert(len(STATE_COLORS), "Co-elevated states (≥ 2 above floor)")

    ax.legend(handles, labels, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ======================================================================
# PLOT 3: VIGILANCE TERTILE CHART
# ======================================================================
def plot_vigilance_tertiles(scorecard, out_path):
    profile  = scorecard.get("Vigilance_Decrement_Chronological_Profile")
    if not isinstance(profile, dict) or "Early_Tertile" not in profile:
        print("  (vigilance tertile plot SKIPPED -- insufficient trials for tertile profile)")
        return

    tertiles  = ["Early_Tertile", "Mid_Tertile", "Late_Tertile"]
    mean_rts  = [profile[t]["Mean_Hit_RT_ms"] for t in tertiles]
    nogo_tot  = [profile[t]["Total_Presented"] - profile[t]["Valid_Go_Hits_Count"]
                 for t in tertiles]
    fail_rates = [
        (profile[t]["Inhibition_Failure_Count"] / nogo_tot[i] * 100.0)
        if nogo_tot[i] > 0 else 0.0
        for i, t in enumerate(tertiles)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(3); labels = ["Early", "Mid", "Late"]

    ax1.bar(x, mean_rts, color="#0072B2")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Mean Hit RT (ms)")
    slope = profile.get("Fatigue_Slope_Decline_ms", float("nan"))
    ax1.set_title(f"RT by Tertile  (slope: {slope:+.1f} ms)")
    for i, v in enumerate(mean_rts):
        ax1.text(i, v + max(mean_rts) * 0.01, f"{v:.0f}", ha="center", fontsize=9)

    ax2.bar(x, fail_rates, color="#D55E00")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Inhibition Failure Rate (%)")
    ax2.set_title("No-Go Commission Rate by Tertile")
    for i, v in enumerate(fail_rates):
        ax2.text(i, v + max(fail_rates + [1]) * 0.02, f"{v:.0f}%", ha="center", fontsize=9)

    fig.suptitle("Vigilance Decrement Profile")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ======================================================================
# PLOT 4: KNOB TRAJECTORY
# ======================================================================
def plot_knob_trajectory(policy_rows, out_path):
    xs         = [int(r["Trial_ID"])      for r in policy_rows]
    exposure   = [float(r["Exposure_ms"]) for r in policy_rows]
    isi        = [float(r["ISI_ms"])      for r in policy_rows]
    go_prob    = [float(r["Go_Probability"]) for r in policy_rows]
    similarity = [int(r["Similarity_Level"]) for r in policy_rows]

    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True)
    panels = [
        (axes[0], exposure,   "Exposure (ms)",      "#0072B2"),
        (axes[1], isi,        "ISI (ms)",            "#009E73"),
        (axes[2], go_prob,    "Go Probability",      "#E69F00"),
        (axes[3], similarity, "Similarity Level",    "#CC79A7"),
    ]
    for ax, ys, label, color in panels:
        ax.step(xs, ys, where="post", color=color, linewidth=1.5)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Trial")
    fig.suptitle("Adaptive Difficulty Knob Trajectory Over Session")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_plots.py <file_prefix>")
        sys.exit(1)
    prefix = sys.argv[1]

    scorecard      = load_scorecard(prefix)
    trial_rows     = load_trial_summary(prefix)
    policy_rows    = load_policy_trace(prefix)
    estimator_rows = load_estimator_trace(prefix)

    base     = Path(prefix)
    out_dir  = base.parent
    stem     = base.name

    rt_path       = out_dir / f"{stem}plot_rt_timeline.png"
    evidence_path = out_dir / f"{stem}plot_evidence_trace.png"
    tertile_path  = out_dir / f"{stem}plot_vigilance_tertiles.png"
    knob_path     = out_dir / f"{stem}plot_knob_trajectory.png"

    plot_rt_with_actions(trial_rows, policy_rows, scorecard, rt_path)
    print(f"  {rt_path}")

    has_evidence = plot_evidence_trace(estimator_rows, evidence_path)
    if has_evidence:
        print(f"  {evidence_path}")
    else:
        print("  (evidence trace SKIPPED -- estimator_trace.csv not found)")

    plot_vigilance_tertiles(scorecard, tertile_path)
    print(f"  {tertile_path}")

    plot_knob_trajectory(policy_rows, knob_path)
    print(f"  {knob_path}")


if __name__ == "__main__":
    main()