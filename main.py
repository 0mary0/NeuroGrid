"""
MainGame
==========================================================================
This is a cognitive game with adjustable difficulty. Running the code opens 
a PyGame window with an adaptive Go/No-Go session, on exit (using Q) it 
writes the telemetry and scorecard files with the format "sub-name-ses-number-date"

This file includes 2 classes, a)NeuroGridGoNoGo & b)MetricsEngine

a) NeuroGridGoNoGo: Includes the PyGame loop, rendering and input handling.
b) MetricsEngine: Includes trial bookkeeping, feature computation, baseline 
re-anchoring, scorecard export.

adaptation_policy_engine.py and behavioral_state_estimator.py are used for adaptation
and classification logic.

"""
import os
import pygame
import sys
import time
import csv
import random
import json
import hashlib
from collections import deque
from behavioral_state_estimator import BehavioralStateEstimator
from adaptation_policy_engine import AdaptationPolicyEngine

def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


CONFIG_FILENAME = "neurogrid_config.json"

def load_config():
    """Load neurogrid_config.json from the same directory as this script.
    Returns the parsed dict on success, or an empty dict (so all downstream
    .get() calls fall through to their hardcoded defaults) on any failure.
    Never crashes -- a missing or malformed config file is always recoverable."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)
    if not os.path.exists(config_path):
        print(f"[CONFIG] {CONFIG_FILENAME} not found -- using built-in defaults.")
        return {}
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        print(f"[CONFIG] Loaded {CONFIG_FILENAME}.")
        return cfg
    except Exception as e:
        print(f"[CONFIG WARNING] Could not parse {CONFIG_FILENAME}: {e!r} -- using built-in defaults.")
        return {}


# ====================================================================== #
#  Stimulus similarity levels (placeholder, geometric guess)
# ====================================================================== #
# This table is a placeholder pending perceptual validation. The right
# long-term method is a similarity-sorting task with clinicians/patients
# (pairwise or card-sort ratings, ideally analyzed via multidimensional
# scaling) to derive an empirical similarity axis instead of a geometric
# guess. Will be revised after the clinic visit.

GO_COLOR = (0, 114, 178)          # Okabe-Ito blue
NOGO_BASE_COLOR = (230, 159, 0)   # Okabe-Ito orange

SIMILARITY_LEVELS = {
    1: {"border_radius_fraction": 0.00, "color": (230, 159, 0)},
    2: {"border_radius_fraction": 0.20, "color": (184, 150, 36)},
    3: {"border_radius_fraction": 0.45, "color": (127, 139, 80)},
    4: {"border_radius_fraction": 0.70, "color": (69, 128, 125)},
    5: {"border_radius_fraction": 0.88, "color": (46, 123, 142)},
}


# ====================================================================== #
#  BASE GAME ENGINE
# ====================================================================== #
class NeuroGridGoNoGo:
    def __init__(self):
        # --- RESEARCH METADATA ---
        print("\n=== NeuroGrid Go/No-Go v2.5.0 Initialization ===")
        self.participant_id = input("Enter Participant ID (or press Enter for ANONYMOUS): ").strip() or "ANONYMOUS"
        self.session_id = input("Enter Session/Condition ID (or press Enter for 001): ").strip() or "001"
        self.session_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.file_safe_timestamp = time.strftime("%Y%m%d_%H%M%S")

        # --- LOAD EXTERNAL CONFIG ---
        cfg = load_config()
        proto = cfg.get("protocol", {})
        est_cfg = cfg.get("estimator", {})
        pol_cfg = cfg.get("policy", {})

        # --- SEED SEQUENCE ---
        self.rng_seed = random.randint(0, 999999)
        self.research_rng = random.Random(self.rng_seed)
        self.stimulus_type_history = []

        self.file_prefix = f"sub-{self.participant_id}_ses-{self.session_id}_{self.file_safe_timestamp}_"

        # --- DISPLAY ---
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"NeuroGrid Go/No-Go v2.5.0 - Participant: {self.participant_id}")

        # --- CLOCK & RUN STATE ---
        self.clock = pygame.time.Clock()
        self.running = True

        # --- UI FONTS ---
        pygame.font.init()
        self._font_pause_large = pygame.font.SysFont(None, 72)
        self._font_pause_small = pygame.font.SysFont(None, 32)

        # --- PAUSE STATE ---
        self.pause_count = 0
        self._post_pause_buffer_remaining = 0
        self.total_post_pause_buffer_trials = 0

        # --- TELEMETRY DATASTORES ---
        self.event_log = []
        self.trial_log = []
        self.estimator_log = []
        self.policy_log = []

        # --- SYSTEM STATE VARIABLES ---
        self.trial_counter = 0
        self.game_state = "WAITING"
        self.state_timer_ms = pygame.time.get_ticks()
        self.feedback_state_timer_ms = 0

        # --- ACTIVE TASK VARIABLES (from config, fallback to defaults) ---
        self.exposure_duration_ms      = proto.get("exposure_duration_ms",       500)
        self.feedback_duration_ms      = 200   # Not clinically tunable
        self.inter_stimulus_interval_ms = proto.get("inter_stimulus_interval_ms", 1000)
        self.go_probability            = proto.get("go_probability",              0.75)
        self.similarity_level          = 1
        self.late_press_window_ms      = proto.get("late_press_window_ms",        200)

        self.current_stimulus_type = None
        self.target_spawn_time_ms = 0
        self.active_trial_data = None
        self.active_trial_committed = False
        self.response_registered = False
        self.feedback_type = None

        # --- METRICS ENGINE ---
        self.metrics_engine = MetricsEngine(
            rng_seed=self.rng_seed,
            stim_duration=self.exposure_duration_ms,
            isi=self.inter_stimulus_interval_ms,
            config=proto
        )

        # --- BEHAVIORAL STATE ESTIMATOR ---
        self.estimator = BehavioralStateEstimator(
            thresholds=est_cfg.get("thresholds"),
            weights=est_cfg.get("problem_weights"),
            flow_floor=est_cfg.get("flow_floor", 0.30),
            hysteresis_margin=est_cfg.get("hysteresis_margin", 0.10),
            min_nogo=est_cfg.get("min_nogo_for_confidence", 3),
            min_go_trials=est_cfg.get("min_go_trials_for_confidence", 3),
            min_hits=est_cfg.get("min_hits_for_confidence", 3),
        )

        # --- ADAPTATION POLICY ENGINE ---
        from adaptation_policy_engine import DEFAULT_CONFIG as _POL_DEFAULTS
        from copy import deepcopy as _dc
        pol_full = _dc(_POL_DEFAULTS)
        pol_full["problem_action_threshold"] = pol_cfg.get("problem_action_threshold", pol_full["problem_action_threshold"])
        pol_full["headroom_action_threshold"] = pol_cfg.get("headroom_action_threshold", pol_full["headroom_action_threshold"])
        pol_full["sustain_required"]          = pol_cfg.get("sustain_required",          pol_full["sustain_required"])
        pol_full["cooldown_trials"]           = pol_cfg.get("cooldown_trials",           pol_full["cooldown_trials"])
        if "knobs" in pol_cfg:
            for knob, vals in pol_cfg["knobs"].items():
                if knob in pol_full:
                    pol_full[knob].update(vals)

        self.policy = AdaptationPolicyEngine(config=pol_full)
        self.policy.knobs["exposure_ms"]      = self.exposure_duration_ms
        self.policy.knobs["isi_ms"]           = self.inter_stimulus_interval_ms
        self.policy.knobs["go_probability"]   = self.go_probability
        self.policy.knobs["similarity_level"] = self.similarity_level

    # ------------------------------------------------------------------ #
    #  TRIAL LIFECYCLE
    # ------------------------------------------------------------------ #
    def spawn_target(self):
        self.trial_counter += 1

        if self.research_rng.random() < self.go_probability:
            self.current_stimulus_type = "GO"
        else:
            self.current_stimulus_type = "NO_GO"

        self.stimulus_type_history.append(self.current_stimulus_type)
        self.target_spawn_time_ms = pygame.time.get_ticks()
        self.game_state = "STIMULUS_ACTIVE"
        self.active_trial_committed = False
        self.response_registered = False

        default_outcome = "OMISSION" if self.current_stimulus_type == "GO" else "SUCCESSFUL_WITHHOLD"

        self.active_trial_data = {
            "trial_id": self.trial_counter,
            "stimulus_type": self.current_stimulus_type,
            "spawn_time_ms": self.target_spawn_time_ms,
            "accuracy_outcome": default_outcome,
            "primary_rt_ms": -1,
            "extra_spacebar_presses": 0,
            "late_presses": 0,
            "uninstructed_isi_presses": 0
        }

        self.log_raw_event("STIMULUS_ONSET", self.current_stimulus_type)
        print(f"\n[SPAWN] Trial {self.trial_counter}: Type [{self.current_stimulus_type}] active.")

    def log_raw_event(self, event_type, details="-1"):
        self.event_log.append([pygame.time.get_ticks(), self.trial_counter, event_type, details])

    # ------------------------------------------------------------------ #
    #  INPUT
    # ------------------------------------------------------------------ #
    def handle_input(self, event):
        if event.type != pygame.KEYDOWN:
            return

        # --- Pause controls ---
        # P: pause, only when between stimuli (WAITING or FEEDBACK) so no
        # in-flight trial timing is disturbed. STIMULUS_ACTIVE presses are
        # ignored; the participant can wait ~500ms for the stimulus to expire.
        if event.key == pygame.K_p and self.game_state in ("WAITING", "FEEDBACK"):
            self._enter_pause()
            return
        if event.key == pygame.K_r and self.game_state == "PAUSED":
            self._resume_from_pause()
            return
        if event.key == pygame.K_q and self.game_state == "PAUSED":
            print("[QUIT] Quit key pressed during pause -- saving session data.")
            self.running = False
            return

        # --- SPACEBAR: game response ---
        if event.key != pygame.K_SPACE:
            return

        press_time_ms = pygame.time.get_ticks()

        # --- CASE 1: STIMULUS ON SCREEN ---
        if self.game_state == "STIMULUS_ACTIVE":
            if not self.response_registered:
                # First valid response this trial.
                self.response_registered = True
                rt_ms = press_time_ms - self.target_spawn_time_ms
                self.active_trial_data["primary_rt_ms"] = rt_ms

                if self.current_stimulus_type == "GO":
                    self.active_trial_data["accuracy_outcome"] = "HIT"
                    self.log_raw_event("SPACEBAR_PRESS_GO", "VALID")
                    print(f"[LOGGED] Go HIT. RT: {rt_ms} ms")
                else:  # NO_GO
                    self.active_trial_data["accuracy_outcome"] = "COMMISSION"
                    self.log_raw_event("SPACEBAR_PRESS_NOGO", "COMMISSION_ERROR")
                    print(f"[LOGGED] COMMISSION ERROR: failed inhibition. RT: {rt_ms} ms")
            else:
                # Additional press while the stimulus is still up.
                self.active_trial_data["extra_spacebar_presses"] += 1
                self.log_raw_event("SPACEBAR_PRESS_EXTRA", "REDUNDANT")
                print("[LOGGED] Redundant press.")

        # --- CASE 2: INTER-STIMULUS INTERVAL ---
        elif self.game_state == "WAITING":
            if not self.active_trial_data:
                return
            elapsed_since_window_close = press_time_ms - self.state_timer_ms
            if elapsed_since_window_close <= self.late_press_window_ms:
                if (self.active_trial_data["late_presses"] == 0
                        and self.active_trial_data["accuracy_outcome"] == "OMISSION"):
                    self.active_trial_data["accuracy_outcome"] = "LATE_RESPONSE"
                self.active_trial_data["late_presses"] += 1
                self.log_raw_event("SPACEBAR_PRESS_LATE", "LATE_MOTOR_RESPONSE")
                print(f"[LOGGED] Late press ({elapsed_since_window_close}ms after window close) -- plausible motor delay.")
            else:
                self.active_trial_data["uninstructed_isi_presses"] += 1
                self.log_raw_event("SPACEBAR_PRESS_ISI", "NOISE")
                print("[LOGGED] ISI noise press.")

    def _enter_pause(self):
        self.game_state = "PAUSED"
        self.log_raw_event("SESSION_PAUSED", f"pause_number={self.pause_count + 1}")
        self.pause_count += 1
        print(f"[PAUSE] Session paused (pause #{self.pause_count}). R=Resume  Q=Save & Quit")

    def _resume_from_pause(self):
        self.metrics_engine.flush_rolling_windows()
        self.estimator.reset()
        self.policy._cooldown_remaining = 0

        buffer_n = self.metrics_engine.buffer_trials
        self._post_pause_buffer_remaining = buffer_n
        self.game_state = "WAITING"
        self.state_timer_ms = pygame.time.get_ticks()
        self.log_raw_event("SESSION_RESUMED", f"post_pause_buffer_trials={buffer_n}")
        print(f"[RESUME] Resumed. {buffer_n} buffer trials before estimation resumes.")

    # ------------------------------------------------------------------ #
    #  COMMIT  (+ per-trial state estimation)
    # ------------------------------------------------------------------ #
    def commit_current_trial_to_metrics(self):
        if self.active_trial_data and not self.active_trial_committed:
            trial_row = [
                self.active_trial_data["trial_id"],
                self.active_trial_data["stimulus_type"],
                self.active_trial_data["spawn_time_ms"],
                self.active_trial_data["accuracy_outcome"],
                self.active_trial_data["primary_rt_ms"],
                self.active_trial_data["extra_spacebar_presses"],
                self.active_trial_data["late_presses"],
                self.active_trial_data["uninstructed_isi_presses"]
            ]
            self.trial_log.append(trial_row)
            self.metrics_engine.process_trial_data(trial_row)
            self.active_trial_committed = True

            # --- Run the estimator once we're in the active phase with real data ---
            if self.metrics_engine.current_phase == "ACTIVE" and self.metrics_engine.active_trials_n > 0:

                if self._post_pause_buffer_remaining > 0:
                    self._post_pause_buffer_remaining -= 1
                    self.total_post_pause_buffer_trials += 1
                    self.log_raw_event("POST_PAUSE_BUFFER_TRIAL",
                                       f"remaining={self._post_pause_buffer_remaining}")
                    return

                features = self.metrics_engine.get_current_features()
                estimate = self.estimator.estimate(features)
                self.estimator_log.append({
                    "trial_id": self.active_trial_data["trial_id"],
                    "estimate": estimate,
                    "features": features,
                })

                elevated = estimate.get("elevated_states", [])
                elevated_str = (f" | also_elevated={elevated}"
                                if len(elevated) > 1 or
                                   (elevated and elevated[0] != estimate["problem_state"])
                                else "")
                print(f"[ESTIMATE] T{self.active_trial_data['trial_id']}: "
                      f"{estimate['problem_state']} | evid={estimate['problem_evidence']} "
                      f"| headroom={estimate['headroom']}{elevated_str}")

                # --- Adaptation policy: decide on / apply a knob change ---
                policy_result = self.policy.update(estimate)
                actions = policy_result.get("actions", [])
                if actions:
                    # One row per knob change.
                    for act in actions:
                        self.policy_log.append({
                            "trial_id": self.active_trial_data["trial_id"],
                            "result": {
                                "action": act["direction"].upper(),
                                "knob": act["knob"],
                                "reason": act["reason"],
                                "trigger": policy_result.get("trigger"),
                                "knobs": act["knobs_snapshot"],
                            },
                        })
                else:
                    # HOLD / COOLDOWN: no knob changed
                    self.policy_log.append({
                        "trial_id": self.active_trial_data["trial_id"],
                        "result": {
                            "action": policy_result["action"], "knob": None, "reason": None,
                            "trigger": policy_result.get("trigger"),
                            "knobs": policy_result["knobs"],
                        },
                    })

                self.exposure_duration_ms = self.policy.knobs["exposure_ms"]
                self.inter_stimulus_interval_ms = self.policy.knobs["isi_ms"]
                self.go_probability = self.policy.knobs["go_probability"]
                self.similarity_level = self.policy.knobs["similarity_level"]

                if policy_result["action"] not in ("HOLD", "COOLDOWN"):
                    print(f"[POLICY] T{self.active_trial_data['trial_id']}: "
                          f"{policy_result['action']} {policy_result.get('knob')} "
                          f"(reason={policy_result.get('reason')}, trigger={policy_result.get('trigger')}) "
                          f"-> exposure={self.exposure_duration_ms}ms isi={self.inter_stimulus_interval_ms}ms "
                          f"go_prob={self.go_probability}")

    # ------------------------------------------------------------------ #
    #  GAME LOOP LOGIC
    # ------------------------------------------------------------------ #
    def update_game_logic(self):
        now = pygame.time.get_ticks()

        if self.game_state == "STIMULUS_ACTIVE":
            if now - self.target_spawn_time_ms >= self.exposure_duration_ms:
                self.log_raw_event("STIMULUS_TIMEOUT", self.active_trial_data["accuracy_outcome"])
                outcome = self.active_trial_data["accuracy_outcome"]
                self.feedback_type = "CORRECT" if outcome in ("HIT", "SUCCESSFUL_WITHHOLD") else "ERROR"
                self.game_state = "FEEDBACK"
                self.feedback_state_timer_ms = now

        elif self.game_state == "FEEDBACK":
            if now - self.feedback_state_timer_ms >= self.feedback_duration_ms:
                self.game_state = "WAITING"
                self.state_timer_ms = now

        elif self.game_state == "WAITING":
            if now - self.state_timer_ms >= self.inter_stimulus_interval_ms:
                self.commit_current_trial_to_metrics()
                self.spawn_target()

    # ------------------------------------------------------------------ #
    #  RENDER
    # ------------------------------------------------------------------ #
    def draw_stimulus(self):
        self.screen.fill((15, 15, 15))
        cx, cy = self.screen_width // 2, self.screen_height // 2

        if self.game_state == "PAUSED":
            pause_text = self._font_pause_large.render("PAUSED", True, (200, 200, 200))
            hint_text  = self._font_pause_small.render(
                "R  \u2013  Resume     Q  \u2013  Save & Quit", True, (100, 100, 100))
            self.screen.blit(pause_text, pause_text.get_rect(center=(cx, cy - 30)))
            self.screen.blit(hint_text,  hint_text.get_rect(center=(cx, cy + 40)))
            return

        if self.game_state == "WAITING":
            pygame.draw.circle(self.screen, (60, 60, 60), (cx, cy), 4)

        elif self.game_state == "STIMULUS_ACTIVE":
            if self.current_stimulus_type == "GO":
                pygame.draw.circle(self.screen, GO_COLOR, (cx, cy), 50)
            else:
                level = SIMILARITY_LEVELS[self.similarity_level]
                border_radius = round(level["border_radius_fraction"] * 50)
                rect = pygame.Rect(cx - 50, cy - 50, 100, 100)
                pygame.draw.rect(self.screen, level["color"], rect, border_radius=border_radius)

            # Acknowledgment ring
            if self.response_registered:
                pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), 62, 3)

        elif self.game_state == "FEEDBACK":
            # Small colored dot so feedback is non-distracting.
            color = (34, 139, 34) if self.feedback_type == "CORRECT" else (139, 0, 0)
            pygame.draw.circle(self.screen, color, (cx, cy), 4)

    # ------------------------------------------------------------------ #
    #  PERSISTENCE
    # ------------------------------------------------------------------ #
    def save_session_data(self):
        try:
            if not self.active_trial_committed:
                if self.game_state == "STIMULUS_ACTIVE":
                    self.log_raw_event("FORCED_WINDOW_SHUTDOWN_MID_TRIAL")
                self.commit_current_trial_to_metrics()
        except Exception as e:
            print(f"In-flight Trial Flush Error (continuing with what was already committed): {e!r}")

        if self.event_log:
            try:
                filename = f"{self.file_prefix}raw_event_telemetry.csv"
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Monotonic_MS", "Trial_ID", "Event_Type", "Context_Details"])
                    writer.writerows(self.event_log)
            except Exception as e:
                print(f"Event Log Error (export skipped, continuing): {e!r}")

        if self.trial_log:
            try:
                filename = f"{self.file_prefix}trial_summary_telemetry.csv"
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Trial_ID", "Stimulus_Type", "Spawn_MS", "Accuracy_Outcome",
                                     "Primary_RT_MS", "Extra_Presses", "Late_Presses", "Uninstructed_ISI_Presses"])
                    writer.writerows(self.trial_log)
            except Exception as e:
                print(f"Trial Log Error (export skipped, continuing): {e!r}")

        # Per-trial estimator trace (evidence vector + headroom over time).
        if self.estimator_log:
            try:
                filename = f"{self.file_prefix}estimator_trace.csv"
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Trial_ID", "Problem_State", "Is_Flow",
                        "Ev_Attention_Lapse", "Ev_Performance_Decline", "Ev_High_Impulsivity",
                        "Ev_Timing_Pressure",
                        "Headroom_Speed", "Headroom_Discrimination", "Headroom_Inhibition",
                        "Omission_Confident", "NoGo_Confident", "RT_Confident",
                        "Elevated_States"
                    ])
                    for row in self.estimator_log:
                        est = row["estimate"]
                        pe = est["problem_evidence"]
                        hr = est["headroom"]
                        conf = est["confidence"]
                        writer.writerow([
                            row["trial_id"], est["problem_state"], est["is_flow"],
                            pe["ATTENTION_LAPSE"], pe["PERFORMANCE_DECLINE"], pe["HIGH_IMPULSIVITY"],
                            pe["TIMING_PRESSURE"],
                            hr["speed"], hr["discrimination"], hr["inhibition"],
                            conf["omission_confident"], conf["nogo_confident"], conf["rt_confident"],
                            "|".join(est.get("elevated_states", []))
                        ])
            except Exception as e:
                print(f"Estimator Trace Error (export skipped, continuing): {e!r}")

        # Per-trial adaptation-policy trace
        if self.policy_log:
            try:
                filename = f"{self.file_prefix}policy_trace.csv"
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Trial_ID", "Action", "Knob", "Reason", "Trigger",
                        "Exposure_ms", "ISI_ms", "Go_Probability", "Similarity_Level"
                    ])
                    for row in self.policy_log:
                        r = row["result"]
                        knobs = r.get("knobs", {})
                        writer.writerow([
                            row["trial_id"], r["action"], r.get("knob"), r.get("reason"), r.get("trigger"),
                            knobs.get("exposure_ms"), knobs.get("isi_ms"),
                            knobs.get("go_probability"), knobs.get("similarity_level")
                        ])
            except Exception as e:
                print(f"Policy Trace Error (export skipped, continuing): {e!r}")

        try:
            co_occurrence_summary = {"Trials_With_Any_Co_Elevation": 0,
                                     "Trials_With_3_Or_More_Co_Elevated": 0,
                                     "Most_Common_Co_Occurring_Pair": "N/A",
                                     "Note": "Co-elevation = >= 2 problem states above flow_floor simultaneously."}
            if self.estimator_log:
                pair_counts = {}
                co_count = 0
                triple_count = 0
                for entry in self.estimator_log:
                    elevated = entry["estimate"].get("elevated_states", [])
                    if len(elevated) >= 2:
                        co_count += 1
                        if len(elevated) >= 3:
                            triple_count += 1
                        # Count every pair
                        for i in range(len(elevated)):
                            for j in range(i + 1, len(elevated)):
                                pair = f"{elevated[i]}+{elevated[j]}"
                                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                co_occurrence_summary["Trials_With_Any_Co_Elevation"] = co_count
                co_occurrence_summary["Trials_With_3_Or_More_Co_Elevated"] = triple_count
                if pair_counts:
                    co_occurrence_summary["Most_Common_Co_Occurring_Pair"] = max(
                        pair_counts, key=pair_counts.get)
                    co_occurrence_summary["All_Pair_Counts"] = pair_counts

            self.metrics_engine.export_experimental_scorecard(
                self.participant_id, self.session_id, self.session_timestamp,
                self.file_prefix, self.stimulus_type_history, self.late_press_window_ms,
                self.pause_count, self.total_post_pause_buffer_trials,
                co_occurrence_summary
            )
        except Exception as e:
            print(f"Scorecard Export Error: {e!r}")

    def run(self):
        crashed_exception = None
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    self.handle_input(event)
                self.update_game_logic()
                self.draw_stimulus()
                pygame.display.flip()
                self.clock.tick(60)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Ctrl+C detected -- session ended early. Saving collected data...")
        except Exception as e:
            crashed_exception = e
            print(f"\n[CRASH] Unexpected error during session: {e!r}")
            print("[CRASH] Saving whatever data was collected before the crash...")
        finally:
            try:
                self.save_session_data()
                print("[SAVE] Session data saved.")
            except Exception as save_err:
                print(f"[SAVE FAILURE] Could not complete save_session_data(): {save_err!r}")
            pygame.quit()

        if crashed_exception is not None:
            raise crashed_exception   # data is safely saved; still surface the real bug
        sys.exit()


# ====================================================================== #
#  METRICS ENGINE
# ====================================================================== #
class MetricsEngine:
    def __init__(self, rng_seed, stim_duration, isi, config=None):
        self.schema_version = "2.3.0"
        self.rng_seed = rng_seed
        self.stimulus_duration_ms = stim_duration
        self.inter_stimulus_interval_ms = isi

        cfg = config or {}
        self.window_size              = cfg.get("window_size",               10)
        self.nogo_window_size         = cfg.get("nogo_window_size",           6)
        self.buffer_trials            = cfg.get("buffer_trials",             10)
        self.required_baseline_hits   = cfg.get("required_baseline_hits",    10)
        self.max_calibration_trials   = cfg.get("max_calibration_trials",    25)
        self.anticipation_floor_ms    = cfg.get("anticipation_floor_ms",    150)
        self.baseline_std_floor_ms    = 10.0
        self.min_trials_for_vigilance_profile = cfg.get("min_trials_for_vigilance_profile", 15)

        # --- Plateau re-baseline (PROVISIONAL config) ---
        self.rebaseline_enabled       = True
        self.plateau_block_size       = cfg.get("plateau_block_size",        10)
        self.plateau_tolerance_ms     = cfg.get("plateau_tolerance_ms",      25.0)
        self.plateau_required_stable  = cfg.get("plateau_required_stable",    2)
        self._plateau_current_block = []
        self._plateau_recent_blocks = deque(maxlen=self.plateau_required_stable + 1)
        self._plateau_prev_block_mean = None
        self._plateau_stable_count = 0
        self.baseline_rebaselined = False
        self.baseline_rebaseline_trial_id = None
        self.provisional_baseline_mean_ms = None
        self.provisional_baseline_std_ms = None
        self.provisional_baseline_cv = None

        # Sliding windows
        self.recent_outcomes = deque(maxlen=self.window_size)
        self.recent_is_go = deque(maxlen=self.window_size)
        self.recent_hit_rts_ms = deque(maxlen=self.window_size)
        self.recent_hit_is_anticipation = deque(maxlen=self.window_size)
        self.recent_nogo_outcomes = deque(maxlen=self.nogo_window_size)
        self.recent_commission_rts_ms = deque(maxlen=self.nogo_window_size)

        # Accumulators
        self.calibration_hit_rts_ms = []
        self.active_hit_rts_ms = []
        self.active_commission_rts_ms = []
        self.active_phase_trials = []

        # Counters
        self.longest_omission_streak = 0
        self.current_omission_streak = 0
        self.active_omissions_n = 0
        self.active_commissions_n = 0
        self.active_successful_withholds_n = 0
        self.active_anticipations_n = 0
        self.active_trials_n = 0
        self.global_extra_presses = 0
        self.global_late_presses = 0
        self.global_isi_presses = 0

        self.total_trials_processed = 0
        self.calibration_trials_seen = 0
        self.current_phase = "BUFFER"
        self.calibration_status = "PENDING"
        self.baseline_confidence_rating = "UNVALUATED"

        # Baseline / window stats
        self.baseline_mean_ms = None
        self.baseline_std_ms = None
        self.baseline_cv = None
        self.current_window_mean_ms = None
        self.current_window_std_ms = None
        self.current_window_cv = None
        self.standardized_drift_index = None

        # Alert latches
        self.omission_alert_count = 0
        self.omission_latch_active = False
        self.inhibition_alert_count = 0
        self.inhibition_latch_active = False

    def _execute_phase_migration(self):
        if len(self.calibration_hit_rts_ms) >= 3:
            self.baseline_mean_ms = sum(self.calibration_hit_rts_ms) / len(self.calibration_hit_rts_ms)
            variance_sum = sum((rt - self.baseline_mean_ms) ** 2 for rt in self.calibration_hit_rts_ms)
            calculated_std = (variance_sum / (len(self.calibration_hit_rts_ms) - 1)) ** 0.5 if len(self.calibration_hit_rts_ms) > 1 else self.baseline_std_floor_ms
            self.baseline_std_ms = max(calculated_std, self.baseline_std_floor_ms)
            self.baseline_cv = self.baseline_std_ms / self.baseline_mean_ms

            if len(self.calibration_hit_rts_ms) >= self.required_baseline_hits:
                self.calibration_status = "OPTIMAL"
                self.baseline_confidence_rating = "HIGH_ANCHORED_STABLE"
            else:
                self.calibration_status = "DEGRADED_FORCED_TERMINATION"
                self.baseline_confidence_rating = "LOW_INTERPRET_DRIFT_WITH_CAUTION"
        else:
            self.baseline_mean_ms = self.baseline_std_ms = self.baseline_cv = None
            self.calibration_status = "FAILED_INSUFFICIENT_HITS"
            self.baseline_confidence_rating = "ZERO_UNUSABLE"

        print(f"[METRICS] Migration finalized. Baseline status: {self.calibration_status}")
        self.current_phase = "ACTIVE"

    # ------------------------------------------------------------------ #
    #  PLATEAU RE-BASELINE
    # ------------------------------------------------------------------ #
    def _update_plateau_tracker(self, rt_ms, trial_id):
        if not self.rebaseline_enabled or self.baseline_rebaselined:
            return
        if self.baseline_mean_ms is None:   # no provisional baseline
            return

        self._plateau_current_block.append(rt_ms)
        if len(self._plateau_current_block) < self.plateau_block_size:
            return

        block = list(self._plateau_current_block)
        self._plateau_current_block = []
        block_mean = sum(block) / len(block)
        self._plateau_recent_blocks.append(block)

        if self._plateau_prev_block_mean is not None:
            if abs(block_mean - self._plateau_prev_block_mean) <= self.plateau_tolerance_ms:
                self._plateau_stable_count += 1
            else:
                self._plateau_stable_count = 0
        self._plateau_prev_block_mean = block_mean

        if (self._plateau_stable_count >= self.plateau_required_stable
                and len(self._plateau_recent_blocks) >= self.plateau_required_stable + 1):
            self._rebaseline_from_plateau(trial_id)

    def _rebaseline_from_plateau(self, trial_id):
        pooled = [rt for block in self._plateau_recent_blocks for rt in block]
        if len(pooled) < 3:
            return
        new_mean = sum(pooled) / len(pooled)
        var = sum((rt - new_mean) ** 2 for rt in pooled) / (len(pooled) - 1)
        new_std = max(var ** 0.5, self.baseline_std_floor_ms)

        # Preserve the original calibration anchor for transparency.
        self.provisional_baseline_mean_ms = self.baseline_mean_ms
        self.provisional_baseline_std_ms = self.baseline_std_ms
        self.provisional_baseline_cv = self.baseline_cv

        self.baseline_mean_ms = new_mean
        self.baseline_std_ms = new_std
        self.baseline_cv = new_std / new_mean
        self.baseline_rebaselined = True
        self.baseline_rebaseline_trial_id = trial_id

        print(f"[METRICS] Plateau detected at trial {trial_id}. Re-baselined "
              f"{self.provisional_baseline_mean_ms:.1f} -> {new_mean:.1f} ms "
              f"(SD {self.provisional_baseline_std_ms:.1f} -> {new_std:.1f}).")

    def process_trial_data(self, trial_row):
        self.total_trials_processed += 1
        trial_id = safe_int(trial_row[0], default=0)
        stim_type = trial_row[1]
        spawn_ts = safe_int(trial_row[2], default=0)
        accuracy = trial_row[3]
        primary_rt_ms = safe_int(trial_row[4], default=-1)
        extra_cnt = safe_int(trial_row[5], default=0)
        late_cnt = safe_int(trial_row[6], default=0)
        isi_cnt = safe_int(trial_row[7], default=0)

        is_go = (stim_type == "GO")

        if self.current_phase == "BUFFER":
            if self.total_trials_processed <= self.buffer_trials:
                return
            self.current_phase = "CALIBRATION"

        if self.current_phase == "CALIBRATION":
            self.calibration_trials_seen += 1
            if is_go and accuracy == "HIT" and primary_rt_ms > 0:
                self.calibration_hit_rts_ms.append(primary_rt_ms)
            if (len(self.calibration_hit_rts_ms) == self.required_baseline_hits
                    or self.calibration_trials_seen >= self.max_calibration_trials):
                self._execute_phase_migration()
            return

        if self.current_phase == "ACTIVE":
            self.active_trials_n += 1
            self.global_extra_presses += extra_cnt
            self.global_late_presses += late_cnt
            self.global_isi_presses += isi_cnt

            self.active_phase_trials.append({
                "trial_id": trial_id,
                "stimulus_type": stim_type,
                "timestamp_ms": spawn_ts,
                "accuracy": accuracy,
                "primary_rt_ms": primary_rt_ms
            })

            if accuracy == "OMISSION":
                self.active_omissions_n += 1
            elif accuracy == "COMMISSION":
                self.active_commissions_n += 1
                if primary_rt_ms > 0:
                    self.active_commission_rts_ms.append(primary_rt_ms)
                    self.recent_commission_rts_ms.append(primary_rt_ms)
            elif accuracy == "SUCCESSFUL_WITHHOLD":
                self.active_successful_withholds_n += 1
            elif accuracy == "HIT" and primary_rt_ms > 0:
                self.active_hit_rts_ms.append(primary_rt_ms)
                self._update_plateau_tracker(primary_rt_ms, trial_id)

            # --- Sliding windows ---
            self.recent_outcomes.append(accuracy)
            self.recent_is_go.append(1 if is_go else 0)
            if accuracy == "HIT" and primary_rt_ms > 0:
                self.recent_hit_rts_ms.append(primary_rt_ms)
                is_anticip = 1 if primary_rt_ms < self.anticipation_floor_ms else 0
                self.recent_hit_is_anticipation.append(is_anticip)
                if is_anticip:
                    self.active_anticipations_n += 1
            if not is_go:
                self.recent_nogo_outcomes.append(1 if accuracy == "COMMISSION" else 0)

            # --- Omission streak ---
            if accuracy == "OMISSION":
                self.current_omission_streak += 1
                self.longest_omission_streak = max(self.longest_omission_streak, self.current_omission_streak)
            else:
                self.current_omission_streak = 0

            # --- Omission rate over GO trials in the window ---
            go_in_window = sum(self.recent_is_go)
            rolling_omissions = self.recent_outcomes.count("OMISSION")
            omission_rate = (rolling_omissions / go_in_window) * 100.0 if go_in_window > 0 else 0.0

            # --- Inhibition failure rate over No-Go trials in the window ---
            nogo_seen = len(self.recent_nogo_outcomes)
            inhibition_failure_rate = (sum(self.recent_nogo_outcomes) / nogo_seen) if nogo_seen > 0 else 0.0

            # --- RT drift vs baseline ---
            if len(self.recent_hit_rts_ms) >= 3 and self.calibration_status != "FAILED_INSUFFICIENT_HITS":
                self.current_window_mean_ms = sum(self.recent_hit_rts_ms) / len(self.recent_hit_rts_ms)
                w_var = sum((rt - self.current_window_mean_ms) ** 2 for rt in self.recent_hit_rts_ms)
                self.current_window_std_ms = max((w_var / (len(self.recent_hit_rts_ms) - 1)) ** 0.5 if len(self.recent_hit_rts_ms) > 1 else self.baseline_std_floor_ms, self.baseline_std_floor_ms)
                self.current_window_cv = self.current_window_std_ms / self.current_window_mean_ms
                self.standardized_drift_index = (self.current_window_mean_ms - self.baseline_mean_ms) / self.baseline_std_ms

            drift_threshold = 1.5 if self.calibration_status == "OPTIMAL" else 2.0

            # --- Omission / drift latch (INTERIM) ---
            omission_exceeded = ((omission_rate >= 30.0 and go_in_window >= 3)
                                 or (self.standardized_drift_index is not None and self.standardized_drift_index > drift_threshold))
            if omission_exceeded:
                if not self.omission_latch_active:
                    self.omission_alert_count += 1
                    self.omission_latch_active = True
                    sdi = round(self.standardized_drift_index, 2) if self.standardized_drift_index is not None else "N/A"
                    print(f"[ALERT] Attentional drift flagged. SDI: {sdi}")
            else:
                self.omission_latch_active = False

            # --- Inhibition failure latch ---
            if nogo_seen >= 3 and inhibition_failure_rate >= 0.50:
                if not self.inhibition_latch_active:
                    self.inhibition_alert_count += 1
                    self.inhibition_latch_active = True
                    print(f"[ALERT] Inhibition-failure burst. Rate: {round(inhibition_failure_rate * 100, 1)}%")
            elif inhibition_failure_rate < 0.34:
                self.inhibition_latch_active = False

    # ------------------------------------------------------------------ #
    #  FEATURE ASSEMBLY FOR THE ESTIMATOR
    # ------------------------------------------------------------------ #
    def flush_rolling_windows(self):
        """Clear all rolling window deques. Called on pause/resume so
        pre-pause behavioural data doesn't contaminate post-pause estimates.
        Does not touch baseline parameters (baseline_mean_ms, baseline_std_ms,
        SDI history) -- the participant's own RT reference is still valid
        after a short break, only the recency windows need clearing."""
        self.recent_outcomes.clear()
        self.recent_is_go.clear()
        self.recent_hit_rts_ms.clear()
        self.recent_hit_is_anticipation.clear()
        self.recent_nogo_outcomes.clear()
        self.recent_commission_rts_ms.clear()

    def get_current_features(self):
        """Package current rolling features into the dict the
        BehavioralStateEstimator expects (see estimator.required_features()).
        Rates are returned as proportions 0..1, not percentages."""
        go_in_window = sum(self.recent_is_go)
        rolling_omissions = self.recent_outcomes.count("OMISSION")
        omission_rate = (rolling_omissions / go_in_window) if go_in_window > 0 else 0.0

        nogo_seen = len(self.recent_nogo_outcomes)
        inhibition_failure_rate = (sum(self.recent_nogo_outcomes) / nogo_seen) if nogo_seen > 0 else 0.0

        anticipation_rate = (sum(self.recent_hit_is_anticipation) / len(self.recent_hit_is_anticipation)) \
            if self.recent_hit_is_anticipation else 0.0

        recent_commission_rt = (sum(self.recent_commission_rts_ms) / len(self.recent_commission_rts_ms)) \
            if self.recent_commission_rts_ms else None

        late_rate = (self.recent_outcomes.count("LATE_RESPONSE") / go_in_window) if go_in_window > 0 else 0.0

        return {
            "omission_rate": omission_rate,
            "window_rt_cv": self.current_window_cv,
            "baseline_rt_cv": self.baseline_cv,
            "sdi": self.standardized_drift_index,
            "inhibition_failure_rate": inhibition_failure_rate,
            "anticipation_rate": anticipation_rate,
            "recent_commission_rt_ms": recent_commission_rt,
            "baseline_mean_rt_ms": self.baseline_mean_ms,
            "nogo_seen": nogo_seen,
            "go_trials_seen": go_in_window,
            "go_hits_seen": len(self.recent_hit_rts_ms),
            "late_rate": late_rate,
        }

    # ------------------------------------------------------------------ #
    #  ANALYSIS HELPERS
    # ------------------------------------------------------------------ #
    def _compute_rt_percentiles(self, rts_list):
        if not rts_list:
            return {"P10": "N/A", "P25": "N/A", "P50_Median": "N/A", "P75": "N/A", "P90": "N/A"}
        sorted_rts = sorted(rts_list)
        n = len(sorted_rts)

        def get_perc(p):
            idx = (n - 1) * p
            low = int(idx)
            high = min(low + 1, n - 1)
            weight = idx - low
            return round(sorted_rts[low] * (1 - weight) + sorted_rts[high] * weight, 2)

        return {"P10": get_perc(0.10), "P25": get_perc(0.25), "P50_Median": get_perc(0.50),
                "P75": get_perc(0.75), "P90": get_perc(0.90)}

    def _compute_vigilance_decrement(self):
        n = len(self.active_phase_trials)
        if n < self.min_trials_for_vigilance_profile:
            return f"INSUFFICIENT_ACTIVE_TRIALS (have {n}, need {self.min_trials_for_vigilance_profile})"
        t = n // 3
        blocks = {
            "Early_Tertile": self.active_phase_trials[:t],
            "Mid_Tertile": self.active_phase_trials[t:2 * t],
            "Late_Tertile": self.active_phase_trials[2 * t:]
        }

        profile = {}
        for name, trials in blocks.items():
            b_total = len(trials)
            b_omissions = sum(1 for x in trials if x["accuracy"] == "OMISSION")
            b_commissions = sum(1 for x in trials if x["accuracy"] == "COMMISSION")
            b_rts = [x["primary_rt_ms"] for x in trials if x["accuracy"] == "HIT" and x["primary_rt_ms"] > 0]

            b_mean = sum(b_rts) / len(b_rts) if b_rts else None
            b_cv = None
            if len(b_rts) > 1:
                b_var = sum((rt - b_mean) ** 2 for rt in b_rts)
                b_std = (b_var / (len(b_rts) - 1)) ** 0.5
                b_cv = b_std / b_mean if b_mean > 0 else 0

            profile[name] = {
                "Total_Presented": b_total,
                "Omission_Count": b_omissions,
                "Inhibition_Failure_Count": b_commissions,
                "Valid_Go_Hits_Count": len(b_rts),
                "Mean_Hit_RT_ms": round(b_mean, 2) if b_mean is not None else "NO_HITS",
                "RT_CV_Sigma": round(b_cv, 4) if b_cv is not None else "INSUFFICIENT_HITS",
                "Distributional_Percentiles": self._compute_rt_percentiles(b_rts)
            }

        early_m = profile["Early_Tertile"]["Mean_Hit_RT_ms"]
        late_m = profile["Late_Tertile"]["Mean_Hit_RT_ms"]
        if isinstance(early_m, (int, float)) and isinstance(late_m, (int, float)):
            profile["Fatigue_Slope_Decline_ms"] = round(late_m - early_m, 2)
        else:
            profile["Fatigue_Slope_Decline_ms"] = "INSUFFICIENT_TERTILE_HITS"
        return profile

    def _compute_successive_variability(self):
        n = len(self.active_hit_rts_ms)
        if n < 2:
            return "INSUFFICIENT_DATA"
        sq = [(self.active_hit_rts_ms[i] - self.active_hit_rts_ms[i - 1]) ** 2 for i in range(1, n)]
        return round(sum(sq) / len(sq), 2)

    def _compute_post_error_slowing(self):
        post_error, post_correct = [], []
        for i in range(1, len(self.active_phase_trials)):
            prev = self.active_phase_trials[i - 1]
            curr = self.active_phase_trials[i]
            if curr["accuracy"] == "HIT" and curr["primary_rt_ms"] > 0:
                if prev["accuracy"] == "COMMISSION":
                    post_error.append(curr["primary_rt_ms"])
                elif prev["accuracy"] == "HIT":
                    post_correct.append(curr["primary_rt_ms"])
        if not post_error or not post_correct:
            return "INSUFFICIENT_ERROR_SEQUENCES"
        return round((sum(post_error) / len(post_error)) - (sum(post_correct) / len(post_correct)), 2)

    # ------------------------------------------------------------------ #
    #  SCORECARD
    # ------------------------------------------------------------------ #
    def export_experimental_scorecard(self, participant_id, session_id, timestamp, file_prefix, type_history, late_press_window_ms, pause_count=0, total_post_pause_buffer_trials=0, co_occurrence_summary=None):
        total_active_hits = len(self.active_hit_rts_ms)
        active_mean = sum(self.active_hit_rts_ms) / total_active_hits if total_active_hits > 0 else None
        active_std = active_cv = None
        if total_active_hits > 1:
            s_var = sum((rt - active_mean) ** 2 for rt in self.active_hit_rts_ms)
            active_std = max((s_var / (total_active_hits - 1)) ** 0.5, self.baseline_std_floor_ms)
            active_cv = active_std / active_mean

        total_nogo = self.active_commissions_n + self.active_successful_withholds_n
        mean_commission_rt = (round(sum(self.active_commission_rts_ms) / len(self.active_commission_rts_ms), 2)
                              if self.active_commission_rts_ms else "NO_COMMISSIONS")

        seq_str = ",".join(type_history)
        sequence_sha256 = hashlib.sha256(seq_str.encode("utf-8")).hexdigest()

        reached_active = self.current_phase == "ACTIVE"
        completeness_caveats = []
        if not reached_active:
            completeness_caveats.append(
                f"Session ended during {self.current_phase} phase, before calibration completed -- "
                f"no behavioral estimates or adaptation occurred."
            )
        elif total_active_hits < self.min_trials_for_vigilance_profile:
            completeness_caveats.append(
                f"Only {total_active_hits} active GO hits collected (recommend >= "
                f"{self.min_trials_for_vigilance_profile} for the vigilance tertile profile to be "
                f"statistically meaningful, not just crash-safe)."
            )

        scorecard = {
            "Session_Completeness_Check": {
                "Reached_Active_Phase": reached_active,
                "Active_Trials_Collected": self.active_trials_n,
                "Valid_Go_Hits_Collected": total_active_hits,
                "Sufficient_For_Vigilance_Profile": reached_active and total_active_hits >= self.min_trials_for_vigilance_profile,
                "Caveats": completeness_caveats if completeness_caveats else ["None -- session reached sufficient volume for full analysis."]
            },
            "Session_Pause_Log": {
                "Total_Pauses": pause_count,
                "Total_Post_Pause_Buffer_Trials": total_post_pause_buffer_trials,
                "Note": ("Each resume starts a fresh buffer equal to the protocol "
                         "buffer_trials setting, during which estimation and adaptation "
                         "are suspended and rolling windows are cleared. Buffer trials "
                         "are still logged to trial_summary_telemetry.csv with "
                         "POST_PAUSE_BUFFER_TRIAL events in raw_event_telemetry.csv.")
            },
            "Protocol_Metadata_Header": {
                "Participant_Identifier": participant_id,
                "Session_Identifier": session_id,
                "Execution_Timestamp": timestamp,
                "Telemetry_Schema_Version": self.schema_version,
                "Stimulus_Sequence_Seed": self.rng_seed,
                "Stimulus_Sequence_SHA256_Hash": sequence_sha256
            },
            "Protocol_Configuration_Logs": {
                "Paradigm_Type": "Fixed-Location Go/No-Go",
                "Stimulus_Exposure_Duration_START_ms": self.stimulus_duration_ms,
                "Inter_Stimulus_Interval_START_ms": self.inter_stimulus_interval_ms,
                "Initial_Skipped_Buffer_Trials": self.buffer_trials,
                "Target_Baseline_Hits_Required": self.required_baseline_hits,
                "Sliding_Analysis_Window_Size": self.window_size,
                "NoGo_Inhibition_Window_Size": self.nogo_window_size,
                "Anticipation_Floor_ms": self.anticipation_floor_ms,
                "Late_Press_Window_ms": late_press_window_ms,
                "Timing_Is_Adaptive": True,
                "Timing_Note": ("Exposure/ISI are starting values only; the "
                                 "AdaptationPolicyEngine can change them mid-session. "
                                 "SOA is fixed WITHIN each unchanged-knob segment, not "
                                 "across the whole session -- see policy_trace.csv for "
                                 "exact trial-by-trial knob values and EEG epoch segmentation.")
            },
            "Data_Quality_Verification": {
                "Total_Global_Stimuli_Presented": self.total_trials_processed,
                "Calibration_Quality_Status": self.calibration_status,
                "Baseline_Anchor_Confidence_Rating": self.baseline_confidence_rating,
                "Engine_Terminal_Phase": self.current_phase
            },
            "Baseline_Reanchoring": {
                "Rebaseline_Enabled": self.rebaseline_enabled,
                "Plateau_Block_Size": self.plateau_block_size,
                "Plateau_Tolerance_ms": self.plateau_tolerance_ms,
                "Plateau_Required_Stable_Blocks": self.plateau_required_stable,
                "Rebaselined": self.baseline_rebaselined,
                "Rebaseline_Trial_ID": self.baseline_rebaseline_trial_id if self.baseline_rebaselined else "NOT_TRIGGERED",
                "Provisional_Calibration_Mean_ms": round(self.provisional_baseline_mean_ms, 2) if self.provisional_baseline_mean_ms is not None else "N/A_NOT_REBASELINED",
                "Provisional_Calibration_SD_ms": round(self.provisional_baseline_std_ms, 2) if self.provisional_baseline_std_ms is not None else "N/A_NOT_REBASELINED"
            },
            "Baseline_Phenotype_Anchor": {
                "Effective_Mean_Processing_Speed_ms": round(self.baseline_mean_ms, 2) if self.baseline_mean_ms is not None else "UNAVAILABLE",
                "Effective_RT_Sigma_SD_ms": round(self.baseline_std_ms, 2) if self.baseline_std_ms is not None else "UNAVAILABLE",
                "Effective_Normalized_IIV_CV": round(self.baseline_cv, 4) if self.baseline_cv is not None else "UNAVAILABLE",
                "Anchor_Source": "PLATEAU_REANCHORED" if self.baseline_rebaselined else "CALIBRATION_ONLY"
            },
            "Active_Phase_Performance_Profile": {
                "Active_Trials_Total_N": self.active_trials_n,
                "Valid_Go_Hits_N": total_active_hits,
                "Omissions_N": self.active_omissions_n,
                "Inhibition_Failures_Commissions_N": self.active_commissions_n,
                "Successful_Withholds_N": self.active_successful_withholds_n,
                "Anticipatory_Hits_N": self.active_anticipations_n,
                "Anticipation_Rate_Pct": round((self.active_anticipations_n / total_active_hits) * 100.0, 2) if total_active_hits > 0 else "NO_HITS",
                "Inhibition_Failure_Rate_Pct": round((self.active_commissions_n / total_nogo) * 100.0, 2) if total_nogo > 0 else "NO_NOGO_TRIALS",
                "Mean_Commission_RT_ms": mean_commission_rt,
                "Global_Redundant_Extra_Presses": self.global_extra_presses,
                "Global_Late_Presses_Motor_Delay_Plausible": self.global_late_presses,
                "Global_ISI_Noise_Presses": self.global_isi_presses,
                "Mean_Hit_RT_ms": round(active_mean, 2) if active_mean is not None else "INSUFFICIENT_DATA",
                "RT_Sigma_SD_ms": round(active_std, 2) if active_std is not None else "INSUFFICIENT_DATA",
                "Normalized_IIV_CV": round(active_cv, 4) if active_cv is not None else "INSUFFICIENT_DATA",
                "Hit_RT_Distribution_Percentiles": self._compute_rt_percentiles(self.active_hit_rts_ms),
                "Longest_Consecutive_Omission_Streak": self.longest_omission_streak
            },
            "Vigilance_Decrement_Chronological_Profile": self._compute_vigilance_decrement(),
            "Advanced_Attentional_Biomarkers": {
                "Trial_To_Trial_Successive_Variability_MSSD": self._compute_successive_variability(),
                "Exploratory_Post_Error_Slowing_PES_ms": self._compute_post_error_slowing()
            },
            "Co_Occurring_State_Elevations": co_occurrence_summary or {
                "Note": "No estimator data available for this session."},
            "Terminal_Active_Window_Snapshot": {
                "Final_Rolling_Mean_RT_ms": round(self.current_window_mean_ms, 2) if self.current_window_mean_ms is not None else "INSUFFICIENT_DATA",
                "Standardized_Drift_Index": round(self.standardized_drift_index, 2) if self.standardized_drift_index is not None else "INSUFFICIENT_DATA"
            },
            "Experimental_Alerts_Summary": {
                "Total_Distinct_Lapse_Episodes": self.omission_alert_count,
                "Total_Distinct_Inhibition_Failure_Bursts": self.inhibition_alert_count
            }
        }

        try:
            filename = f"{file_prefix}experimental_scorecard_summary.json"
            with open(filename, "w") as f:
                json.dump(scorecard, f, indent=4)
            print(f"[SUCCESS] v2.5.0 scorecard written to {filename}")
        except Exception as e:
            print(f"Serialization Error: {e!r}")


if __name__ == "__main__":
    game = NeuroGridGoNoGo()
    game.run()