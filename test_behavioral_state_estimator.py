"""
Tests for BehavioralStateEstimator.

Run with: python -m unittest test_behavioral_state_estimator.py
"""
import unittest
from behavioral_state_estimator import BehavioralStateEstimator


def make_features(**overrides):
    base = {
        "omission_rate": 0.0,
        "window_rt_cv": 0.12,
        "baseline_rt_cv": 0.12,
        "sdi": 0.0,
        "inhibition_failure_rate": 0.0,
        "anticipation_rate": 0.0,
        "recent_commission_rt_ms": None,
        "baseline_mean_rt_ms": 280.0,
        "nogo_seen": 5,
        "go_trials_seen": 8,
        "go_hits_seen": 8,
        "late_rate": 0.0,
    }
    base.update(overrides)
    return base


class TestProblemStateDetection(unittest.TestCase):
    """Each scenario uses a fresh estimator -- hysteresis is stateful,
    so unrelated scenarios must not leak state into each other."""

    def test_clean_on_baseline_is_optimal_flow(self):
        est = BehavioralStateEstimator()
        r = est.estimate(make_features())
        self.assertEqual(r["problem_state"], "OPTIMAL_FLOW")
        self.assertTrue(r["is_flow"])

    def test_attention_lapse_detected(self):
        est = BehavioralStateEstimator()
        r = est.estimate(make_features(omission_rate=0.35, window_rt_cv=0.22))
        self.assertEqual(r["problem_state"], "ATTENTION_LAPSE")

    def test_performance_decline_detected(self):
        est = BehavioralStateEstimator()
        r = est.estimate(make_features(sdi=1.8, omission_rate=0.10))
        self.assertEqual(r["problem_state"], "PERFORMANCE_DECLINE")

    def test_high_impulsivity_detected(self):
        est = BehavioralStateEstimator()
        r = est.estimate(make_features(inhibition_failure_rate=0.45,
                                        recent_commission_rt_ms=150.0,
                                        anticipation_rate=0.20))
        self.assertEqual(r["problem_state"], "HIGH_IMPULSIVITY")

    def test_timing_pressure_detected_without_double_counting_as_lapse(self):
        # The critical check here: a high late_rate must register as
        # TIMING_PRESSURE, and must NOT also inflate ATTENTION_LAPSE via
        # the omission term -- the two signals are causally distinct and
        # main.py is responsible for not letting a trial count as both.
        est = BehavioralStateEstimator()
        r = est.estimate(make_features(late_rate=0.25, omission_rate=0.0, go_hits_seen=2))
        self.assertEqual(r["problem_state"], "TIMING_PRESSURE")
        self.assertEqual(r["problem_evidence"]["ATTENTION_LAPSE"], 0.0)

    def test_no_baseline_yet_is_optimal_flow(self):
        est = BehavioralStateEstimator()
        r = est.estimate(make_features(baseline_mean_rt_ms=None))
        self.assertEqual(r["problem_state"], "OPTIMAL_FLOW")
        self.assertEqual(r.get("note"), "NO_BASELINE")


class TestHeadroom(unittest.TestCase):

    def test_speed_headroom_shows_when_fast_and_stable(self):
        est = BehavioralStateEstimator()
        r = est.estimate(make_features(sdi=-1.6, window_rt_cv=0.10))
        self.assertGreater(r["headroom"]["speed"], 0.0)

    def test_discrimination_headroom_shows_when_accurate_not_fast(self):
        est = BehavioralStateEstimator()
        r = est.estimate(make_features(sdi=0.1, omission_rate=0.0, inhibition_failure_rate=0.0))
        self.assertGreater(r["headroom"]["discrimination"], 0.0)

    def test_low_nogo_volume_does_not_show_false_inhibition_headroom(self):
        # Regression test: with only 1 NoGo trial seen and 0 observed
        # failures, the OLD behavior let (1 - badness) default to 1.0,
        # i.e. zero data looked like perfect headroom. nogo_confident
        # must gate this to 0 instead.
        est = BehavioralStateEstimator()
        r = est.estimate(make_features(nogo_seen=1, inhibition_failure_rate=0.0, sdi=0.1))
        self.assertEqual(r["headroom"]["inhibition"], 0.0)

    def test_severe_lapse_with_few_hits_still_detected(self):
        # Regression test: go_hits_seen=2 means rt_confident is False, so
        # the CV term can't fire -- but the omission term must still
        # register from go_trials_seen=10 (volume), not go_hits_seen
        # (value), or a severe lapse would be self-defeatingly hidden by
        # the very data sparsity it causes.
        est = BehavioralStateEstimator()
        r = est.estimate(make_features(go_trials_seen=10, go_hits_seen=2,
                                        omission_rate=0.80, window_rt_cv=0.12))
        self.assertEqual(r["problem_state"], "ATTENTION_LAPSE")
        self.assertGreaterEqual(r["problem_evidence"]["ATTENTION_LAPSE"], 0.50)

    def test_genuinely_confident_and_clean_shows_real_headroom(self):
        # Sanity check on the two regression tests above: confidence
        # gating must not be so aggressive that real headroom never shows.
        est = BehavioralStateEstimator()
        r = est.estimate(make_features(nogo_seen=8, go_hits_seen=8,
                                        inhibition_failure_rate=0.0,
                                        anticipation_rate=0.0, sdi=0.1))
        self.assertGreater(r["headroom"]["inhibition"], 0.0)


class TestHysteresis(unittest.TestCase):
    """Uses ONE estimator instance across calls on purpose -- hysteresis
    is stateful, that's the point being tested."""

    def test_close_alternating_evidence_does_not_flicker(self):
        est = BehavioralStateEstimator()

        def close_lapse_vs_decline(lapse_wins):
            if lapse_wins:
                return make_features(omission_rate=0.30, window_rt_cv=0.20, sdi=0.55)
            return make_features(omission_rate=0.20, window_rt_cv=0.15, sdi=0.75)

        labels = []
        for i in range(6):
            r = est.estimate(close_lapse_vs_decline(lapse_wins=(i % 2 == 0)))
            labels.append(r["problem_state"])

        flips = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
        self.assertLessEqual(flips, 1, f"expected flicker to be suppressed, got labels={labels}")

    def test_clear_winner_still_switches_promptly(self):
        est = BehavioralStateEstimator()
        # Warm up on a lapse-favoring signal first.
        est.estimate(make_features(omission_rate=0.30, window_rt_cv=0.20, sdi=0.55))
        # Then feed an unambiguous decline winner -- well clear of the
        # hysteresis margin, so it must win despite lapse being "sticky".
        r = est.estimate(make_features(omission_rate=0.05, window_rt_cv=0.12, sdi=2.5))
        self.assertEqual(r["problem_state"], "PERFORMANCE_DECLINE")


if __name__ == "__main__":
    unittest.main()
