"""Run a challenger model in shadow mode next to the champion."""

import logging
import time

from src.shadow.compare import ShadowStats, decide

log = logging.getLogger("shadow")


def _round(value, digits=4):
    return None if value is None else round(value, digits)


class ShadowRunner:
    """Feeds every reading to the challenger and keeps the comparison statistics.

    The challenger must see EVERY reading the champion accepts, scored or not,
    because its own history (the last 50 readings per sensor) has to stay in
    step. Whatever it does, it can never change the champion's answer.
    """

    def __init__(self, challenger, clock=time.perf_counter):
        self.challenger = challenger
        self.stats = ShadowStats()
        self._clock = clock

    def observe(self, reading, champion_answer, champion_seconds):
        """Score `reading` with the challenger and record how it compares. Never raises."""
        try:
            started = self._clock()
            answer = self.challenger.score(dict(reading))  # its own copy of the reading
            seconds = self._clock() - started
        except Exception as error:  # the challenger must never break the champion
            log.warning("challenger failed: %s", error)
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

    def report(self):
        s = self.stats
        return {
            "challenger_run_id": getattr(self.challenger, "run_id", None),
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
            "decision": decide(s),
        }