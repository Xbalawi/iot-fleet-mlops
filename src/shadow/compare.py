"""Compare a challenger model with the champion on the same live readings.

In shadow mode the challenger scores every reading but never raises an alert:
only the champion's answer is used. This module turns what was observed into a
decision. It holds no models and does no I/O, so it is tested in isolation.

A live feed has no labels, so these rules judge SAFETY and CONSISTENCY, not
accuracy. Accuracy was already checked offline, by the promotion rule in
src/model/retrain.py.

A "pair" is a reading that both models scored. A "failure" is a reading the
champion scored and the challenger could not (no score), or any exception the
challenger raised.
"""

from dataclasses import dataclass

MIN_PAIRED = 300  # paired readings needed before a promotion
MIN_CHAMPION_ALERTS = 5  # a quiet feed cannot validate a model
MIN_OVERLAP = 0.5  # share of the champion's alerts the challenger must also raise
ROLLBACK_MIN_READINGS = 100  # guardrails are judged once there are this many readings
MAX_ERROR_RATE = 0.01
MAX_ALERT_RATE = 0.10  # ceiling on the share of readings the challenger flags
MAX_ALERT_RATIO = 3.0  # challenger alerts compared with the champion's
MAX_LATENCY_RATIO = 3.0  # challenger scoring time compared with the champion's
WATCH_PAIRED = 300  # paired readings a freshly promoted model is watched for



@dataclass
class ShadowStats:
    paired: int = 0
    both: int = 0  # both models flagged the reading
    only_champion: int = 0
    only_challenger: int = 0
    failures: int = 0
    champion_seconds: float = 0.0
    challenger_seconds: float = 0.0

    def record_pair(self, champion_flagged, challenger_flagged, champion_seconds=0.0, challenger_seconds=0.0):
        self.paired += 1
        if champion_flagged and challenger_flagged:
            self.both += 1
        elif champion_flagged:
            self.only_champion += 1
        elif challenger_flagged:
            self.only_challenger += 1
        self.champion_seconds += champion_seconds
        self.challenger_seconds += challenger_seconds

    def record_failure(self):
        self.failures += 1

    @property
    def champion_alerts(self):
        return self.both + self.only_champion

    @property
    def challenger_alerts(self):
        return self.both + self.only_challenger

    @property
    def attempts(self):
        return self.paired + self.failures

    @property
    def error_rate(self):
        return self.failures / self.attempts if self.attempts else 0.0

    @property
    def challenger_alert_rate(self):
        return self.challenger_alerts / self.paired if self.paired else 0.0

    @property
    def overlap(self):
        """Share of the champion's alerts that the challenger also raised (None if there are none)."""
        return self.both / self.champion_alerts if self.champion_alerts else None

    @property
    def latency_ratio(self):
        """Challenger scoring time over champion scoring time (None if nothing was timed)."""
        return self.challenger_seconds / self.champion_seconds if self.champion_seconds else None


def decide(
    stats,
    min_paired=MIN_PAIRED,
    min_champion_alerts=MIN_CHAMPION_ALERTS,
    min_overlap=MIN_OVERLAP,
    rollback_min_readings=ROLLBACK_MIN_READINGS,
    max_error_rate=MAX_ERROR_RATE,
    max_alert_rate=MAX_ALERT_RATE,
    max_alert_ratio=MAX_ALERT_RATIO,
    max_latency_ratio=MAX_LATENCY_RATIO,
):
    """Return {"action": "continue" | "promote" | "hold" | "rollback", "reasons": [...]}."""
    # 1. Rollback: any broken guardrail, as soon as there is enough to judge.
    problems = []
    if stats.attempts >= rollback_min_readings and stats.error_rate > max_error_rate:
        problems.append(
            f"the challenger failed on {stats.failures} of {stats.attempts} readings ({stats.error_rate:.1%})"
        )
    if stats.paired >= rollback_min_readings:
        if stats.challenger_alert_rate > max_alert_rate:
            problems.append(
                f"alert storm: the challenger flagged {stats.challenger_alert_rate:.1%} of readings "
                f"(limit {max_alert_rate:.0%})"
            )
        if (
            stats.champion_alerts >= min_champion_alerts
            and stats.challenger_alerts > max_alert_ratio * stats.champion_alerts
        ):
            problems.append(
                f"the challenger raised {stats.challenger_alerts} alerts against the champion's "
                f"{stats.champion_alerts} (limit {max_alert_ratio:g}x)"
            )
        ratio = stats.latency_ratio
        if ratio is not None and ratio > max_latency_ratio:
            problems.append(f"the challenger is {ratio:.1f}x slower than the champion (limit {max_latency_ratio:g}x)")
    if problems:
        return {"action": "rollback", "reasons": problems}

    # 2. Not enough evidence yet.
    if stats.paired < min_paired:
        return {
            "action": "continue",
            "reasons": [f"collecting evidence: {stats.paired} of {min_paired} paired readings"],
        }
    if stats.champion_alerts < min_champion_alerts:
        return {
            "action": "continue",
            "reasons": [
                f"quiet feed: the champion raised {stats.champion_alerts} alerts, "
                f"at least {min_champion_alerts} are needed to compare"
            ],
        }

    # 3. Safe and enough evidence: does the challenger agree with the champion's alerts?
    if stats.overlap < min_overlap:
        return {
            "action": "hold",
            "reasons": [
                f"the challenger shares only {stats.overlap:.0%} of the champion's alerts "
                f"(needs {min_overlap:.0%}); labels or a person must decide"
            ],
        }
    return {
        "action": "promote",
        "reasons": [
            f"safe over {stats.paired} readings: {stats.failures} failures, "
            f"alert rate {stats.challenger_alert_rate:.1%}, "
            f"{stats.overlap:.0%} of the champion's alerts shared"
        ],
    }

def decide_watch(
    stats,
    watch_paired=WATCH_PAIRED,
    min_previous_alerts=MIN_CHAMPION_ALERTS,
    rollback_min_readings=ROLLBACK_MIN_READINGS,
    max_alert_rate=MAX_ALERT_RATE,
    max_alert_ratio=MAX_ALERT_RATIO,
    max_latency_ratio=MAX_LATENCY_RATIO,
):
    """Judge a freshly promoted model that is now SERVING, against the previous model still shadowing it.

    The stats are read the other way round from decide(): "champion" is the model
    that serves (the new one) and "challenger" is the previous model running in
    shadow, kept warm as the way back.
    Returns {"action": "continue" | "confirm" | "rollback", "reasons": [...]}.
    """
    problems = []
    if stats.paired >= rollback_min_readings:
        serving_rate = stats.champion_alerts / stats.paired
        if serving_rate > max_alert_rate:
            problems.append(
                f"alert storm: the new model flagged {serving_rate:.1%} of readings (limit {max_alert_rate:.0%})"
            )
        if (
            stats.challenger_alerts >= min_previous_alerts
            and stats.champion_alerts > max_alert_ratio * stats.challenger_alerts
        ):
            problems.append(
                f"the new model raised {stats.champion_alerts} alerts against the previous model's "
                f"{stats.challenger_alerts} (limit {max_alert_ratio:g}x)"
            )
        if stats.challenger_seconds:
            ratio = stats.champion_seconds / stats.challenger_seconds
            if ratio > max_latency_ratio:
                problems.append(
                    f"the new model is {ratio:.1f}x slower than the previous one (limit {max_latency_ratio:g}x)"
                )
    if problems:
        return {"action": "rollback", "reasons": problems}

    if stats.paired < watch_paired:
        return {
            "action": "continue",
            "reasons": [f"watching the new model: {stats.paired} of {watch_paired} paired readings"],
        }
    return {
        "action": "confirm",
        "reasons": [
            f"the new model behaved over {stats.paired} readings: alert rate "
            f"{stats.champion_alerts / stats.paired:.1%}, {stats.champion_alerts} alerts against "
            f"the previous model's {stats.challenger_alerts}"
        ],
    }