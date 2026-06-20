"""
Tests for AdaptationPolicyEngine.

Run with: python -m unittest test_adaptation_policy_engine.py
"""
import unittest
from adaptation_policy_engine import AdaptationPolicyEngine


def make_estimate(lapse=0.0, decline=0.0, impulsivity=0.0, timing=0.0,
                   speed=0.0, discrimination=0.0, inhibition=0.0):
    """Builds a fake BehavioralStateEstimator output for feeding into
    the policy engine directly, without running real trials through it."""
    return {
        "problem_state": "OPTIMAL_FLOW",
        "is_flow": True,
        "problem_evidence": {"ATTENTION_LAPSE": lapse, "PERFORMANCE_DECLINE": decline,
                              "HIGH_IMPULSIVITY": impulsivity, "TIMING_PRESSURE": timing},
        "headroom": {"speed": speed, "discrimination": discrimination, "inhibition": inhibition},
        "feature_scores": {},
        "confidence": {"baseline": True, "omission_confident": True,
                       "nogo_confident": True, "rt_confident": True},
    }


class TestHeadroomAndRollback(unittest.TestCase):
    """Scenario A+B together, since B's rollback depends on A's prior
    tighten still being in the action log -- they aren't independent."""

    def test_sustained_discrimination_headroom_tightens_similarity(self):
        pol = AdaptationPolicyEngine()
        results = [pol.update(make_estimate(discrimination=0.7)) for _ in range(7)]
        self.assertTrue(any(r["action"] == "TIGHTEN" for r in results))
        self.assertGreater(pol.knobs["similarity_level"], 1)
        self.pol_after_tighten = pol

    def test_lapse_after_tighten_rolls_back_similarity_not_global(self):
        pol = AdaptationPolicyEngine()
        for _ in range(7):
            pol.update(make_estimate(discrimination=0.7))
        similarity_after_tighten = pol.knobs["similarity_level"]
        self.assertGreater(similarity_after_tighten, 1)

        rollback_seen = False
        for _ in range(15):
            r = pol.update(make_estimate(lapse=0.6))
            if r.get("reason") == "rollback":
                rollback_seen = True
        self.assertTrue(rollback_seen, "expected a rollback action attributing the lapse to the prior tighten")
        self.assertLess(pol.knobs["similarity_level"], similarity_after_tighten)


class TestDefaultEasing(unittest.TestCase):

    def test_lapse_with_nothing_recently_tightened_uses_default_global_ease(self):
        pol = AdaptationPolicyEngine()
        results = [pol.update(make_estimate(lapse=0.6)) for _ in range(7)]
        eases = [r for r in results if r["action"] == "EASE"]
        self.assertTrue(eases)
        self.assertTrue(all(r.get("reason") == "default_ease" for r in eases))

    def test_sustained_impulsivity_eases_go_probability(self):
        pol = AdaptationPolicyEngine()
        start_go_prob = pol.knobs["go_probability"]
        for _ in range(7):
            pol.update(make_estimate(impulsivity=0.6))
        self.assertLess(pol.knobs["go_probability"], start_go_prob)


class TestPrecedence(unittest.TestCase):

    def test_problem_state_beats_simultaneous_headroom(self):
        # decline and speed-headroom both sustained at once -- the problem
        # state must win, the engine must never tighten while something
        # is actively wrong.
        pol = AdaptationPolicyEngine()
        results = [pol.update(make_estimate(decline=0.6, speed=0.8)) for _ in range(7)]
        self.assertFalse(any(r["action"] == "TIGHTEN" for r in results))

    def test_dynamic_precedence_picks_highest_evidence_not_fixed_order(self):
        # decline (0.9) and lapse (0.6) sustained together. Under a fixed
        # ATTENTION_LAPSE > PERFORMANCE_DECLINE order, lapse would always
        # win regardless of which evidence is actually higher.
        pol = AdaptationPolicyEngine()
        triggers = []
        for _ in range(7):
            r = pol.update(make_estimate(lapse=0.6, decline=0.9))
            triggers.append(r.get("trigger"))
        self.assertIn("PERFORMANCE_DECLINE", triggers)


class TestTimingPressure(unittest.TestCase):
    """Regression test: TIMING_PRESSURE was once missing from the
    _sustain init dict, which raised a KeyError the first time its
    evidence sustained past threshold. This must not silently break again."""

    def test_timing_pressure_sustains_and_eases_exposure_first(self):
        pol = AdaptationPolicyEngine()
        fired_exposure = False
        for _ in range(7):
            r = pol.update(make_estimate(timing=0.8))
            if r["action"] == "EASE" and r.get("knob") == "exposure_ms" and r.get("trigger") == "TIMING_PRESSURE":
                fired_exposure = True
        self.assertTrue(fired_exposure)


if __name__ == "__main__":
    unittest.main()
