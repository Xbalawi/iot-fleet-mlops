import pytest

from src.shadow.compare import ShadowStats, decide, decide_watch

def test_record_pair_counts_every_case():
    s = ShadowStats()
    s.record_pair(True, True, 0.01, 0.02)
    s.record_pair(True, False)
    s.record_pair(False, True)
    s.record_pair(False, False)
    s.record_failure()
    assert (s.paired, s.both, s.only_champion, s.only_challenger, s.failures) == (4, 1, 1, 1, 1)
    assert s.champion_alerts == 2 and s.challenger_alerts == 2
    assert s.attempts == 5
    assert s.champion_seconds == pytest.approx(0.01)


def test_empty_stats_have_no_rates_to_divide():
    s = ShadowStats()
    assert s.error_rate == 0.0
    assert s.challenger_alert_rate == 0.0
    assert s.overlap is None
    assert s.latency_ratio is None


def test_not_enough_readings_keeps_collecting():
    result = decide(ShadowStats(paired=50))
    assert result["action"] == "continue"
    assert "50 of 300" in result["reasons"][0]


def test_a_quiet_feed_cannot_validate_a_challenger():
    result = decide(ShadowStats(paired=500))
    assert result["action"] == "continue"
    assert "quiet" in result["reasons"][0]


def test_a_consistent_challenger_is_promoted():
    stats = ShadowStats(
        paired=400, both=8, only_champion=2, only_challenger=3,
        champion_seconds=10.0, challenger_seconds=12.0,
    )
    assert decide(stats)["action"] == "promote"


def test_a_challenger_that_misses_most_of_the_champions_alerts_is_held():
    result = decide(ShadowStats(paired=500, both=1, only_champion=9))
    assert result["action"] == "hold"
    assert "10%" in result["reasons"][0]


def test_failures_trigger_a_rollback():
    result = decide(ShadowStats(paired=90, failures=10))
    assert result["action"] == "rollback"
    assert "10 of 100" in result["reasons"][0]


def test_a_few_failures_are_tolerated():
    stats = ShadowStats(paired=495, both=8, only_champion=2, only_challenger=2, failures=5)
    assert decide(stats)["action"] == "promote"  # 5 of 500 is exactly the 1% limit


def test_an_alert_storm_triggers_a_rollback():
    result = decide(ShadowStats(paired=200, both=10, only_challenger=40))
    assert result["action"] == "rollback"
    assert any("alert storm" in reason for reason in result["reasons"])


def test_far_more_alerts_than_the_champion_triggers_a_rollback():
    result = decide(ShadowStats(paired=1000, both=10, only_challenger=40))  # 5% of readings: no storm
    assert result["action"] == "rollback"
    assert len(result["reasons"]) == 1
    assert "50 alerts" in result["reasons"][0]


def test_a_slow_challenger_triggers_a_rollback():
    stats = ShadowStats(paired=200, both=5, champion_seconds=2.0, challenger_seconds=8.0)
    result = decide(stats)
    assert result["action"] == "rollback"
    assert "4.0x slower" in result["reasons"][0]


def test_guardrails_wait_for_enough_readings():
    result = decide(ShadowStats(paired=50, only_challenger=20))  # 40% alert rate, but only 50 readings
    assert result["action"] == "continue"


def test_every_broken_guardrail_is_reported():
    result = decide(ShadowStats(paired=100, only_challenger=30, failures=10))
    assert result["action"] == "rollback"
    assert len(result["reasons"]) == 2

def test_a_watched_model_that_storms_is_rolled_back():
    result = decide_watch(ShadowStats(paired=200, both=10, only_champion=40))  # 25% of readings flagged
    assert result["action"] == "rollback"
    assert any("alert storm" in reason for reason in result["reasons"])


def test_a_watched_model_with_far_more_alerts_than_the_previous_one_is_rolled_back():
    result = decide_watch(ShadowStats(paired=1000, both=10, only_champion=40))  # 5%: no storm
    assert result["action"] == "rollback"
    assert len(result["reasons"]) == 1
    assert "50 alerts" in result["reasons"][0]


def test_a_slow_watched_model_is_rolled_back():
    stats = ShadowStats(paired=200, both=5, champion_seconds=8.0, challenger_seconds=2.0)
    result = decide_watch(stats)
    assert result["action"] == "rollback"
    assert "4.0x slower" in result["reasons"][0]


def test_the_watch_keeps_watching_until_there_are_enough_readings():
    result = decide_watch(ShadowStats(paired=100, both=2))
    assert result["action"] == "continue"
    assert "100 of 300" in result["reasons"][0]


def test_the_watch_confirms_after_enough_calm_readings():
    result = decide_watch(ShadowStats(paired=300, both=8, only_champion=1, only_challenger=1))
    assert result["action"] == "confirm"