"""
AdaptationPolicyEngine
==========================================================================
Takes the BehavioralStateEstimator's output and decides if and how to
change task difficulty knobs.

Precedence: if any problem state is active and sustained for several
turns in a row, the engine eases difficulty. If nothing is active,
headroom may tighten a knob instead. Otherwise it holds.

Easing uses a rollback-first rule. When a problem state triggers easing,
the engine first checks its own action log for a difficulty-increasing
change within the last cooldown_trials turns. If a suspect knob is
found, that knob gets reverted one step (reason="rollback") and a fresh
cooldown starts. If nothing suspect is found, the state's default
easing response applies (reason="default_ease"). This is a
self-attribution heuristic and needs to be discussed with clinicians.

Knob <-> signal map:
  speed_headroom          -> shorten exposure/ISI         (tighten)
  discrimination_headroom -> increase stimulus similarity (tighten)
  inhibition_headroom     -> raise go_probability          (tighten)
  PERFORMANCE_DECLINE     -> lengthen exposure/ISI          (targeted ease)
  HIGH_IMPULSIVITY        -> lower go_probability            (targeted ease)
  ATTENTION_LAPSE         -> ease exposure/ISI and reduce similarity
                              (global ease, since lapse isn't knob-specific)

All numeric thresholds, steps, and bounds below are placeholder values,
collected in one place so they're easy to revise later.
"""

from copy import deepcopy
from collections import deque


# ====================================================================== #
#  PROVISIONAL CONFIG
# ====================================================================== #
DEFAULT_CONFIG = {
    # --- Gating ---
    "problem_action_threshold": 0.50,   # problem evidence must clear this...
    "headroom_action_threshold": 0.60,  # ...headroom must clear this...
    "sustain_required": 5,              # ...for this many consecutive calls.
    "cooldown_trials": 5,                # trials to wait after ANY knob change
                                          # (also the rollback lookback window).

    # --- Knob bounds & step sizes (provisional) ---
    "exposure_ms":   {"min": 300, "max": 800,  "step": 50,  "start": 500},
    "isi_ms":        {"min": 600, "max": 1500, "step": 100, "start": 1000},
    "go_probability": {"min": 0.60, "max": 0.85, "step": 0.05, "start": 0.75},
    "similarity_level": {"min": 1, "max": 5, "step": 1, "start": 1},  # 1=max distinct, 5=max similar
}


# ====================================================================== #
#  ENGINE
# ====================================================================== #
class AdaptationPolicyEngine:
    def __init__(self, config=None):
        self.cfg = deepcopy(config or DEFAULT_CONFIG)

        # Current knob values.
        self.knobs = {
            "exposure_ms": self.cfg["exposure_ms"]["start"],
            "isi_ms": self.cfg["isi_ms"]["start"],
            "go_probability": self.cfg["go_probability"]["start"],
            "similarity_level": self.cfg["similarity_level"]["start"],
        }

        # Sustain counters: how many consecutive calls each signal has
        # been above its action threshold.
        self._sustain = {
            "ATTENTION_LAPSE": 0, "PERFORMANCE_DECLINE": 0,
            "HIGH_IMPULSIVITY": 0, "TIMING_PRESSURE": 0,
            "speed": 0, "discrimination": 0, "inhibition": 0,
        }

        # Cooldown: trials remaining before ANY new knob change is allowed.
        self._cooldown_remaining = 0

        # Action log: list of dicts, most recent last.
        # {"trial_id", "knob", "direction" ("tighten"/"ease"), "reason"}
        self.action_log = []

        self.trial_count = 0  # internal trial counter, advanced by caller

    # -------------------------------------------------------------- #
    #  KNOB HELPERS
    # -------------------------------------------------------------- #
    def _clamp(self, knob, value):
        b = self.cfg[knob]
        return max(b["min"], min(b["max"], value))

    def _step_knob(self, knob, direction_sign):
        """direction_sign: +1 makes the knob HARDER, -1 makes it EASIER,
        in the knob's own natural units (see _harder_means table)."""
        step = self.cfg[knob]["step"]
        delta = step * direction_sign * self._harder_sign(knob)
        new_val = self._clamp(knob, self.knobs[knob] + delta)
        changed = new_val != self.knobs[knob]
        self.knobs[knob] = new_val
        return changed

    @staticmethod
    def _harder_sign(knob):
        """Which raw direction makes each knob HARDER.
        exposure/isi: shorter = harder (-1). go_probability: higher = harder (+1).
        similarity_level: higher = harder (+1)."""
        return {
            "exposure_ms": -1, "isi_ms": -1,
            "go_probability": +1, "similarity_level": +1,
        }[knob]

    def _log(self, knob, direction, reason):
        self.action_log.append({
            "trial_id": self.trial_count, "knob": knob,
            "direction": direction, "reason": reason,
            "value_after": self.knobs[knob],
            "knobs_snapshot": dict(self.knobs),  # full snapshot AFTER this specific change
        })

    # -------------------------------------------------------------- #
    #  ROLLBACK LOOKUP
    # -------------------------------------------------------------- #
    def _find_recent_tighten(self, knobs_of_interest):
        """Most recent 'tighten' action on any of `knobs_of_interest` within
        the rollback lookback window. Returns the knob name or None.

        The lookback window is cooldown_trials + sustain_required, not just
        cooldown_trials. Detecting a sustained problem takes sustain_required
        consecutive calls, so a genuinely causal tighten can be that many
        trials old by the time the problem is confirmed, on top of the
        cooldown that followed it. A lookback of cooldown_trials alone lets
        the real suspect age out before detection finishes, which breaks
        the rollback path almost entirely."""
        lookback_trial = self.trial_count - (self.cfg["cooldown_trials"] + self.cfg["sustain_required"])
        for entry in reversed(self.action_log):
            if entry["trial_id"] < lookback_trial:
                break
            if entry["direction"] == "tighten" and entry["knob"] in knobs_of_interest:
                return entry["knob"]
        return None

    # -------------------------------------------------------------- #
    #  EASE / TIGHTEN PRIMITIVES (with logging)
    # -------------------------------------------------------------- #
    def _tighten(self, knob, reason="headroom_tighten"):
        if self._step_knob(knob, +1):
            self._log(knob, "tighten", reason)
            self._cooldown_remaining = self.cfg["cooldown_trials"]
            return True
        return False

    def _ease(self, knob, reason):
        if self._step_knob(knob, -1):
            self._log(knob, "ease", reason)
            self._cooldown_remaining = self.cfg["cooldown_trials"]
            return True
        return False

    def _attempt_rollback_or_default(self, candidate_knobs, default_ease_fn):
        """Shared rollback-first logic used by all three problem states."""
        suspect = self._find_recent_tighten(candidate_knobs)
        if suspect is not None:
            return self._ease(suspect, reason="rollback")
        return default_ease_fn()

    def _escalate_to_global_ease(self):
        """Last-resort fallback for DECLINE and IMPULSIVITY: if a problem
        state's own targeted knob(s) are already at their bound and the
        problem persists, fall through to the same broad relief LAPSE uses
        (timing + similarity) instead of holding with no recourse.

        This is a blunt fallback -- it doesn't assume e.g. easing stimulus
        similarity has a precise link to impulsivity, it just gives the
        player some relief when the state-specific knob has run out of
        room. Logged with its own reason ("escalation") so it's
        distinguishable from a normal targeted/default response."""
        changed_any = False
        changed_any |= self._ease("exposure_ms", reason="escalation")
        changed_any |= self._ease("isi_ms", reason="escalation")
        changed_any |= self._ease("similarity_level", reason="escalation")
        return changed_any

    # -------------------------------------------------------------- #
    #  STATE-SPECIFIC RESPONSES
    # -------------------------------------------------------------- #
    def _respond_to_lapse(self):
        # Global ease: timing AND similarity. Rollback-first checks BOTH
        # timing knobs and similarity, since lapse is non-knob-specific.
        candidates = ["exposure_ms", "isi_ms", "similarity_level"]

        def default_global_ease():
            changed_any = False
            changed_any |= self._ease("exposure_ms", reason="default_ease")
            changed_any |= self._ease("isi_ms", reason="default_ease")
            changed_any |= self._ease("similarity_level", reason="default_ease")
            return changed_any

        return self._attempt_rollback_or_default(candidates, default_global_ease)

    def _respond_to_decline(self):
        candidates = ["exposure_ms", "isi_ms"]

        def default_targeted_ease():
            c1 = self._ease("exposure_ms", reason="default_ease")
            c2 = self._ease("isi_ms", reason="default_ease")
            return c1 or c2

        acted = self._attempt_rollback_or_default(candidates, default_targeted_ease)
        if not acted:
            acted = self._escalate_to_global_ease()
        return acted

    def _respond_to_timing_pressure(self):
        # exposure_ms is the specific knob this state targets, since the
        # window boundary is what's catching real motor responses. Rollback
        # only checks exposure_ms, not isi_ms -- we don't want to ease ISI
        # for a pure timing problem. If exposure_ms is already maxed out,
        # fall through to isi_ms (more time per trial still helps even if
        # the window itself can't grow), then escalate globally.
        candidates = ["exposure_ms"]

        def default_targeted_ease():
            return self._ease("exposure_ms", reason="default_ease")

        acted = self._attempt_rollback_or_default(candidates, default_targeted_ease)
        if not acted:
            acted = self._ease("isi_ms", reason="escalation")
        if not acted:
            acted = self._escalate_to_global_ease()
        return acted

    def _respond_to_impulsivity(self):
        candidates = ["go_probability"]

        def default_targeted_ease():
            return self._ease("go_probability", reason="default_ease")

        acted = self._attempt_rollback_or_default(candidates, default_targeted_ease)
        if not acted:
            acted = self._escalate_to_global_ease()
        return acted

    # -------------------------------------------------------------- #
    #  HEADROOM TIGHTENING
    # -------------------------------------------------------------- #
    def _respond_to_headroom(self, headroom):
        """Among headroom signals that have cleared their own sustain bar,
        tighten whichever currently shows the most headroom first, not a
        fixed speed > discrimination > inhibition order. If that signal's
        knob is already at its bound, fall through to the next-highest
        eligible signal instead of giving up -- otherwise a maxed-out
        top-priority knob could silently block lower-priority signals from
        ever being revisited, even at full evidence.

        Re-evaluated fresh each call (gated by cooldown, so it can't thrash
        trial-to-trial); ties broken by fixed name order for determinism."""
        sr = self.cfg["sustain_required"]
        knob_map = {
            "speed": ["exposure_ms", "isi_ms"],
            "discrimination": ["similarity_level"],
            "inhibition": ["go_probability"],
        }
        eligible = [(key, headroom[key]) for key in ("speed", "discrimination", "inhibition")
                    if self._sustain[key] >= sr]
        if not eligible:
            return False
        eligible.sort(key=lambda kv: (-kv[1], kv[0]))  # highest value first, stable tie-break
        for key, _ in eligible:
            for knob in knob_map[key]:
                if self._tighten(knob):
                    return True
        return False

    # -------------------------------------------------------------- #
    #  SUSTAIN BOOKKEEPING
    # -------------------------------------------------------------- #
    def _update_sustain(self, estimate):
        pe = estimate["problem_evidence"]
        hr = estimate["headroom"]
        pt = self.cfg["problem_action_threshold"]
        ht = self.cfg["headroom_action_threshold"]

        for state in ("ATTENTION_LAPSE", "PERFORMANCE_DECLINE", "HIGH_IMPULSIVITY", "TIMING_PRESSURE"):
            if pe[state] >= pt:
                self._sustain[state] += 1
            else:
                self._sustain[state] = 0

        for hkey in ("speed", "discrimination", "inhibition"):
            if hr[hkey] >= ht:
                self._sustain[hkey] += 1
            else:
                self._sustain[hkey] = 0

    # -------------------------------------------------------------- #
    #  PUBLIC API
    # -------------------------------------------------------------- #
    def update(self, estimate):
        """Call once per committed trial with the estimator's output.
        Returns a dict describing what (if anything) happened, including
        an "actions" list of EVERY individual knob change made this call
        (a single call -- e.g. lapse's global ease -- can produce several)."""
        self.trial_count += 1
        self._update_sustain(estimate)
        start_len = len(self.action_log)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return {"action": "COOLDOWN", "knobs": dict(self.knobs),
                    "cooldown_remaining": self._cooldown_remaining, "actions": []}

        sr = self.cfg["sustain_required"]

        # Problem states always beat headroom -- never tighten while
        # something's wrong. Among multiple sustained problem states,
        # respond to whichever has the highest evidence, not a fixed
        # ATTENTION_LAPSE > PERFORMANCE_DECLINE > HIGH_IMPULSIVITY order
        # (a fixed order would let a weaker but higher-priority state mask
        # a stronger one). Falls through to the next-highest eligible
        # state if the top one's response chain is fully exhausted.
        response_map = {
            "ATTENTION_LAPSE": self._respond_to_lapse,
            "PERFORMANCE_DECLINE": self._respond_to_decline,
            "HIGH_IMPULSIVITY": self._respond_to_impulsivity,
            "TIMING_PRESSURE": self._respond_to_timing_pressure,
        }
        pe = estimate["problem_evidence"]
        eligible_problems = [(state, pe[state]) for state in
                              ("ATTENTION_LAPSE", "PERFORMANCE_DECLINE", "HIGH_IMPULSIVITY", "TIMING_PRESSURE")
                              if self._sustain[state] >= sr]

        if eligible_problems:
            eligible_problems.sort(key=lambda kv: (-kv[1], kv[0]))  # highest evidence first
            acted = False
            trigger = None
            for state, _ in eligible_problems:
                trigger = state
                acted = response_map[state]()
                if acted:
                    break
        else:
            # Problem vector quiet -> headroom may tighten.
            acted = self._respond_to_headroom(estimate["headroom"])
            trigger = "HEADROOM" if acted else None

        if not acted:
            return {"action": "HOLD", "knobs": dict(self.knobs), "trigger": trigger, "actions": []}

        new_entries = self.action_log[start_len:]
        last_entry = new_entries[-1]
        return {
            "action": last_entry["direction"].upper(),
            "knob": last_entry["knob"],
            "reason": last_entry["reason"],
            "trigger": trigger,
            "knobs": dict(self.knobs),
            "actions": new_entries,
        }
