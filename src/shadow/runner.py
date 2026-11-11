"""Run a challenger in shadow mode next to the serving model, and act on the verdict.

Phases:
  idle:   no second model is running
  shadow:  a challenger scores every reading, silently, next to the serving model
  watch:   the challenger was just promoted: it now serves, and the previous model
          keeps shadowing it, warm and ready to take over again

The shadow model must see EVERY reading the serving model accepts, scored or
not, because its own history has to stay in step. Whatever it does, it can never
change the serving model's answer. Without a `serving` model the runner only
reports and never swaps anything.
"""

import logging
import time
from datetime import datetime, timezone

from src.shadow.compare import ShadowStats, decide, decide_watch

log = logging.getLogger("shadow")


def _round(value, digits=4):
    return None if value is None else round(value, digits)


def _utc_now():
    return datetime.now(timezone.utc)


class ShadowRunner:
    def __init__(self, challenger, serving=None, clock=time.perf_counter, now=_utc_now):
        self.serving = serving
        self.shadow = challenger  # the challenger, or the previous model during the watch
        self.phase = "shadow" if challenger is not None else "idle"
        self.stats = ShadowStats()
        self.events = []
        self._clock = clock
        self._now = now
        self._held = False  # "held for review" is logged once, not on every reading
        if challenger is not None:
            self._event("shadow_started", [])

    def observe(self, reading, champion_answer, champion_seconds):
        """Compare the shadow model with the serving model on one reading, then act. Never raises."""
        if self.shadow is None:
            return
        self._record(reading, champion_answer, champion_seconds)
        self._act()

    def _record(self, reading, champion_answer, champion_seconds):
        try:
            started = self._clock()
            answer = self.shadow.score(dict(reading))  # its own copy of the reading
            seconds = self._clock() - started
        except Exception as error:  # the shadow model must never break the serving one
            log.warning("shadow model failed: %s", error)
            self.stats.record_failure()
            return

        if champion_answer["status"] != "ok" or answer["status"] == "warming_up":
            return  # nothing to compare yet
        if answer["status"] != "ok":
            self.stats.record_failure()
            return
        self.stats.record_pair(
            champion_answer["is_anomaly"] is True,
            answer["is_anomaly"] is True,
            champion_seconds,
            seconds,
        )

    def _act(self):
        if self.serving is None:
            return  # report-only mode
        if self.phase == "shadow":
            verdict = decide(self.stats)
            if verdict["action"] == "rollback":
                self._event("challenger_rejected", verdict["reasons"])
                self._finish()
            elif verdict["action"] == "promote":
                self._event("promoted", verdict["reasons"])
                self.serving, self.shadow = self.shadow, self.serving  # the old model is the way back
                self.phase = "watch"
                self.stats = ShadowStats()
                self._held = False
            elif verdict["action"] == "hold" and not self._held:
                self._held = True
                self._event("held_for_review", verdict["reasons"])
        elif self.phase == "watch":
            verdict = decide_watch(self.stats)
            if verdict["action"] == "rollback":
                self._event("rolled_back", verdict["reasons"])
                self.serving, self.shadow = self.shadow, self.serving  # the previous model serves again
                self._finish()
            elif verdict["action"] == "confirm":
                self._event("promotion_confirmed", verdict["reasons"])
                self._finish()

    def _finish(self):
        self.shadow = None
        self.phase = "idle"
        self.stats = ShadowStats()
        self._held = False

    def _event(self, kind, reasons):
        entry = {
            "at": self._now().isoformat(),
            "event": kind,
            "reasons": list(reasons),
            "serving_run_id": getattr(self.serving, "run_id", None),
            "shadow_run_id": getattr(self.shadow, "run_id", None),
        }
        self.events.append(entry)
        log.info("rollout: %s %s", kind, "; ".join(reasons))

    def report(self):
        s = self.stats
        if self.phase == "shadow":
            decision = decide(s)
        elif self.phase == "watch":
            decision = decide_watch(s)
        else:
            decision = None
        return {
            "phase": self.phase,
            "serving_run_id": getattr(self.serving, "run_id", None),
            "challenger_run_id": getattr(self.shadow, "run_id", None),
            "paired": s.paired,
            "failures": s.failures,
            "champion_alerts": s.champion_alerts,
            "challenger_alerts": s.challenger_alerts,
            "both": s.both,
            "only_champion": s.only_champion,
            "only_challenger": s.only_challenger,
            "overlap": _round(s.overlap),
            "error_rate": _round(s.error_rate),
            "challenger_alert_rate": _round(s.challenger_alert_rate),
            "latency_ratio": _round(s.latency_ratio),
            "decision": decision,
            "events": self.events,
        }