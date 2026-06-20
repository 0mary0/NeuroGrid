"""
BehavioralStateEstimator
==========================================================================
Takes a dict of behavioral features and produces two things:

1. A problem-state label (ATTENTION_LAPSE, PERFORMANCE_DECLINE,
   HIGH_IMPULSIVITY, TIMING_PRESSURE, or OPTIMAL_FLOW) picked by argmax.
   These are treated as roughly separate explanations for what's going
   wrong right now, not as a perfect partition of behavior.

2. A headroom vector (speed, discrimination, inhibition). These three
   are independent and can all be true at once, so there's no argmax
   here. Each points at a different adaptation knob: speed_headroom
   feeds the timing knobs, discrimination_headroom feeds stimulus
   similarity, inhibition_headroom feeds the Go/No-Go ratio.

Background for the feature choices (see README for citations): omission
rate links to attention lapse (Robertson et al. 1997, SART literature),
inhibition failure links to impulsivity, fast pre-error RTs link to
impulsivity (McVay & Kane 2009), RT slowing (positive SDI) links to
performance decline, and RT-CV links to attention instability.

Every threshold in PROVISIONAL_THRESHOLDS is a placeholder. There's no
normative SART dataset to anchor these numbers to yet, so they're best
guesses pending clinical consultation, all collected in one place so
revising them later is a one-spot edit.

Confidence gating: problem-state evidence is a weighted sum, so each
component is gated on its own data requirement. A severe lapse with
few hit RTs still shows up through its omission term even with no data
for the CV term -- gating the whole state on every component at once
would hide lapses right when they're worst.

Headroom works differently. Its formulas multiply terms like
(1 - badness), and with no data "badness" defaults to 0, so missing
data would look like maximum headroom unless guarded against. Headroom
is gated all-or-nothing instead: if any required confidence flag is
false, that term is forced to 0 rather than trusting the math.

Confidence also separates GO-trial volume (go_trials_seen, includes
omissions) from GO-trial RT values (go_hits_seen, hits only), because
gating omission_rate on hit count alone would make its own confidence
flag get worse exactly as the lapse it's measuring gets worse.
"""

from copy import deepcopy


# ====================================================================== #
#  PROVISIONAL CONFIG  (to be revised after clinical consultation)
# ====================================================================== #
PROVISIONAL_THRESHOLDS = {
    # feature                zero_evidence_at full_evidence_at
    "omission_rate":        {"zero_at": 0.00, "one_at": 0.40},   # proportion of GO trials missed
    "rt_cv_multiplier":     {"zero_at": 1.00, "one_at": 2.00},   # window_cv / baseline_cv
    "sdi_decline":          {"zero_at": 0.00, "one_at": 2.00},   # positive SDI (slowing)
    "sdi_speed":            {"zero_at": 0.00, "one_at": -2.00},  # negative SDI (speeding up)
    "inhibition_failure":   {"zero_at": 0.00, "one_at": 0.50},   # commissions / NoGo trials
    "anticipation_rate":    {"zero_at": 0.00, "one_at": 0.30},   # sub-floor GO hits / GO hits
    "commission_speed":     {"zero_at": 1.00, "one_at": 0.50},   # commission_rt / baseline_mean_rt
    "late_rate":            {"zero_at": 0.00, "one_at": 0.20},   # late presses / GO trials in window
}

PROBLEM_WEIGHTS = {
    "ATTENTION_LAPSE":     {"omission": 0.60, "cv": 0.40},
    "PERFORMANCE_DECLINE": {"sdi_pos": 0.60, "omission": 0.40},
    "HIGH_IMPULSIVITY":    {"inhibition": 0.50, "anticipation": 0.30, "commission_speed": 0.20},
    "TIMING_PRESSURE":     {"late": 1.00},
}

FLOW_FLOOR = 0.30
MIN_NOGO_FOR_CONFIDENCE = 3
MIN_GO_TRIALS_FOR_CONFIDENCE = 3   # volume gate for omission_rate (includes omissions)
MIN_HITS_FOR_CONFIDENCE = 3        # value gate for anything needing real RT numbers
HYSTERESIS_MARGIN = 0.10    # PROVISIONAL: a challenger label must beat the
                            # currently-reported label by more than this to
                            # take over


# ====================================================================== #
#  HELPERS
# ====================================================================== #
def _lin(x, zero_at, one_at):
    if one_at == zero_at:
        return 1.0 if x == one_at else 0.0
    t = (x - zero_at) / (one_at - zero_at)
    return max(0.0, min(1.0, t))


# ====================================================================== #
#  ESTIMATOR
# ====================================================================== #
class BehavioralStateEstimator:

    PROBLEM_STATES = ("ATTENTION_LAPSE", "PERFORMANCE_DECLINE", "HIGH_IMPULSIVITY", "TIMING_PRESSURE")

    def __init__(self, thresholds=None, weights=None, flow_floor=FLOW_FLOOR,
                 hysteresis_margin=HYSTERESIS_MARGIN,
                 min_nogo=MIN_NOGO_FOR_CONFIDENCE,
                 min_go_trials=MIN_GO_TRIALS_FOR_CONFIDENCE,
                 min_hits=MIN_HITS_FOR_CONFIDENCE):
        self.thresholds = deepcopy(thresholds or PROVISIONAL_THRESHOLDS)
        self.weights = deepcopy(weights or PROBLEM_WEIGHTS)
        self.flow_floor = flow_floor
        self.hysteresis_margin = hysteresis_margin
        self.min_nogo = min_nogo
        self.min_go_trials = min_go_trials
        self.min_hits = min_hits
        self._last_label = "OPTIMAL_FLOW"

    def reset(self):
        """Clear hysteresis state. Call between independent sessions/tests --
        otherwise the label-switching memory would leak across them."""
        self._last_label = "OPTIMAL_FLOW"

    @staticmethod
    def required_features():
        """Documents the features dict contract (units matter)."""
        return {
            "omission_rate":          "proportion 0..1 (omissions / GO trials in window)",
            "window_rt_cv":           "float, CV of recent hit RTs (SD/mean)",
            "baseline_rt_cv":         "float, baseline CV (may be None pre-calibration)",
            "sdi":                    "float, signed standardized drift index",
            "inhibition_failure_rate":"proportion 0..1 (commissions / NoGo trials in window)",
            "anticipation_rate":      "proportion 0..1 (sub-floor GO hits / GO hits in window)",
            "recent_commission_rt_ms":"float or None (mean recent commission RT)",
            "baseline_mean_rt_ms":    "float or None (baseline GO mean RT)",
            "nogo_seen":              "int, NoGo trials in the inhibition window (confidence)",
            "go_trials_seen":         "int, GO trials INCLUDING omissions (confidence for omission_rate)",
            "go_hits_seen":           "int, GO HITS only -- actual RT samples (confidence for RT-value features)",
        }

    # -------------------------------------------------------------- #
    #  FEATURE -> 0..1 EVIDENCE SCORES
    # -------------------------------------------------------------- #
    def _feature_scores(self, f, confidence):
        t = self.thresholds

        om_s = _lin(f["omission_rate"], **t["omission_rate"]) if confidence["omission_confident"] else 0.0
        inhf_s = _lin(f["inhibition_failure_rate"], **t["inhibition_failure"]) if confidence["nogo_confident"] else 0.0
        ant_s = _lin(f["anticipation_rate"], **t["anticipation_rate"]) if confidence["rt_confident"] else 0.0

        # RT-CV scored relative to the player's own baseline CV.
        base_cv = f.get("baseline_rt_cv")
        if confidence["rt_confident"] and base_cv and base_cv > 0 and f.get("window_rt_cv") is not None:
            cv_mult = f["window_rt_cv"] / base_cv
            cv_s = _lin(cv_mult, **t["rt_cv_multiplier"])
        else:
            cv_s = 0.0

        sdi = f.get("sdi")
        if confidence["rt_confident"] and sdi is not None:
            pos_sdi_s = _lin(sdi, **t["sdi_decline"])
            neg_sdi_s = _lin(sdi, **t["sdi_speed"])
        else:
            pos_sdi_s = 0.0
            neg_sdi_s = 0.0

        # Commission speed: faster-than-baseline failures => more impulsivity.
        crt = f.get("recent_commission_rt_ms")
        base_mean = f.get("baseline_mean_rt_ms")
        if confidence["nogo_confident"] and crt is not None and base_mean and base_mean > 0:
            ratio = crt / base_mean
            comm_speed_s = _lin(ratio, **t["commission_speed"])
        else:
            comm_speed_s = 0.0

        late_s = _lin(f.get("late_rate", 0.0), **t["late_rate"]) if confidence["omission_confident"] else 0.0

        return {
            "omission": om_s,
            "cv": cv_s,
            "sdi_pos": pos_sdi_s,
            "sdi_neg": neg_sdi_s,
            "inhibition": inhf_s,
            "anticipation": ant_s,
            "commission_speed": comm_speed_s,
            "late": late_s,
        }

    # -------------------------------------------------------------- #
    #  PROBLEM-STATE VECTOR (argmax, flow as default)
    # -------------------------------------------------------------- #
    def _problem_vector(self, s):
        w = self.weights
        lapse = w["ATTENTION_LAPSE"]["omission"] * s["omission"] + \
                w["ATTENTION_LAPSE"]["cv"] * s["cv"]
        decline = w["PERFORMANCE_DECLINE"]["sdi_pos"] * s["sdi_pos"] + \
                  w["PERFORMANCE_DECLINE"]["omission"] * s["omission"]
        impuls = w["HIGH_IMPULSIVITY"]["inhibition"] * s["inhibition"] + \
                 w["HIGH_IMPULSIVITY"]["anticipation"] * s["anticipation"] + \
                 w["HIGH_IMPULSIVITY"]["commission_speed"] * s["commission_speed"]
        timing = w["TIMING_PRESSURE"]["late"] * s["late"]
        return {
            "ATTENTION_LAPSE": round(lapse, 4),
            "PERFORMANCE_DECLINE": round(decline, 4),
            "HIGH_IMPULSIVITY": round(impuls, 4),
            "TIMING_PRESSURE": round(timing, 4),
        }

    # -------------------------------------------------------------- #
    #  HEADROOM VECTOR
    # -------------------------------------------------------------- #
    def _headroom_vector(self, s, confidence):
        stability = 1.0 - s["cv"]
        accuracy = min(1.0 - s["omission"], 1.0 - s["inhibition"])

        # Fast AND stable -> room to tighten timing.
        speed = s["sdi_neg"] * stability
        # Accurate but NOT notably fast -> room to make discrimination harder.
        discrimination = accuracy * (1.0 - s["sdi_neg"])
        # Spare motor restraint (clean withholds, no jumping the gun) -> room to raise Go ratio.
        inhibition = (1.0 - s["inhibition"]) * (1.0 - s["anticipation"])

        if not confidence["rt_confident"]:
            speed = 0.0
        if not (confidence["omission_confident"] and confidence["nogo_confident"] and confidence["rt_confident"]):
            discrimination = 0.0
        if not (confidence["nogo_confident"] and confidence["rt_confident"]):
            inhibition = 0.0

        return {
            "speed": round(speed, 4),
            "discrimination": round(discrimination, 4),
            "inhibition": round(inhibition, 4),
        }

    # -------------------------------------------------------------- #
    #  PUBLIC API
    # -------------------------------------------------------------- #
    def estimate(self, features):
        if features.get("baseline_mean_rt_ms") is None:
            self._last_label = "OPTIMAL_FLOW"
            return {
                "problem_state": "OPTIMAL_FLOW",
                "is_flow": True,
                "problem_evidence": {k: 0.0 for k in self.PROBLEM_STATES},
                "headroom": {"speed": 0.0, "discrimination": 0.0, "inhibition": 0.0},
                "feature_scores": {},
                "elevated_states": [],
                "confidence": {"baseline": False, "omission_confident": False,
                               "nogo_confident": False, "rt_confident": False},
                "note": "NO_BASELINE",
            }

        confidence = {
            "baseline": True,
            "omission_confident": features.get("go_trials_seen", 0) >= self.min_go_trials,
            "nogo_confident": features.get("nogo_seen", 0) >= self.min_nogo,
            "rt_confident": features.get("go_hits_seen", 0) >= self.min_hits,
        }

        s = self._feature_scores(features, confidence)
        problem = self._problem_vector(s)
        headroom = self._headroom_vector(s, confidence)

        dominant = max(problem, key=problem.get)
        dom_value = problem[dominant]
        raw_is_flow = dom_value < self.flow_floor
        raw_label = "OPTIMAL_FLOW" if raw_is_flow else dominant

        if (not raw_is_flow and self._last_label != "OPTIMAL_FLOW"
                and raw_label != self._last_label):
            last_label_value = problem.get(self._last_label, 0.0)
            if dom_value < last_label_value + self.hysteresis_margin:
                problem_state = self._last_label
                is_flow = False
            else:
                problem_state = raw_label
                is_flow = False
        else:
            problem_state = raw_label
            is_flow = raw_is_flow

        self._last_label = problem_state

        elevated_states = [s for s in self.PROBLEM_STATES if problem[s] >= self.flow_floor]

        return {
            "problem_state": problem_state,
            "is_flow": is_flow,
            "problem_evidence": problem,
            "headroom": headroom,
            "feature_scores": {k: round(v, 4) for k, v in s.items()},
            "elevated_states": elevated_states,
            "confidence": confidence,
        }