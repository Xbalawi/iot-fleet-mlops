from src.shadow.runner import ShadowRunner

READING = {"sensor_id": "s", "type": "temperature", "value": 1.0}
WARMING = {"status": "warming_up", "is_anomaly": None, "score": None}


def ok(flag):
    return {"status": "ok", "is_anomaly": flag, "score": 0.5}


class FakeChallenger:
    """Answers from a script; an Exception in the script is raised."""

    run_id = "cand1"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.seen = []

    def score(self, reading):
        self.seen.append(reading)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_pairs_are_counted_by_what_each_model_flagged():
    runner = ShadowRunner(FakeChallenger(ok(True), ok(False)))
    runner.observe(READING, ok(True), 0.01)
    runner.observe(READING, ok(True), 0.01)
    assert runner.stats.paired == 2
    assert runner.stats.both == 1
    assert runner.stats.only_champion == 1


def test_the_challenger_sees_every_reading_even_when_the_champion_did_not_score():
    challenger = FakeChallenger(ok(False))
    runner = ShadowRunner(challenger)
    runner.observe(READING, WARMING, 0.001)
    assert len(challenger.seen) == 1
    assert runner.stats.paired == 0


def test_a_warming_up_challenger_is_not_a_failure():
    runner = ShadowRunner(FakeChallenger(WARMING))
    runner.observe(READING, ok(False), 0.01)
    assert runner.stats.failures == 0
    assert runner.stats.paired == 0


def test_a_challenger_that_cannot_score_counts_as_a_failure():
    no_features = {"status": "no_features", "is_anomaly": None, "score": None}
    not_modeled = {"status": "not_modeled", "is_anomaly": None, "score": None}
    runner = ShadowRunner(FakeChallenger(no_features, not_modeled))
    runner.observe(READING, ok(False), 0.01)
    runner.observe(READING, ok(False), 0.01)
    assert runner.stats.failures == 2


def test_a_crashing_challenger_never_raises():
    runner = ShadowRunner(FakeChallenger(RuntimeError("boom")))
    runner.observe(READING, ok(False), 0.01)  # must not raise
    assert runner.stats.failures == 1


def test_the_challenger_gets_its_own_copy_of_the_reading():
    challenger = FakeChallenger(ok(False))
    ShadowRunner(challenger).observe(READING, ok(False), 0.01)
    assert challenger.seen[0] == READING
    assert challenger.seen[0] is not READING


def test_both_scoring_times_are_recorded():
    runner = ShadowRunner(FakeChallenger(ok(False)), clock=iter([10.0, 10.5]).__next__)
    runner.observe(READING, ok(False), 0.25)
    assert runner.stats.champion_seconds == 0.25
    assert runner.stats.challenger_seconds == 0.5


def test_the_report_carries_the_decision():
    runner = ShadowRunner(FakeChallenger(ok(True)))
    runner.observe(READING, ok(True), 0.01)
    report = runner.report()
    assert report["challenger_run_id"] == "cand1"
    assert report["paired"] == 1
    assert report["decision"]["action"] == "continue"