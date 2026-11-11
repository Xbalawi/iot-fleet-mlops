from src.shadow.runner import ShadowRunner


class Ticker:
    """A clock that advances 5 ms per call, so every scoring takes exactly 5 ms."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.005
        return self.t


class Model:
    """A scripted model: flags the readings whose number satisfies `flags`, or raises `error`."""

    def __init__(self, run_id, flags=lambda i: False, error=None):
        self.run_id = run_id
        self.flags = flags
        self.error = error

    def score(self, reading):
        if self.error:
            raise self.error
        return {"status": "ok", "is_anomaly": self.flags(int(reading["value"])), "score": 0.5}


def every(n):
    return lambda i: i % n == 0


def reading(i):
    return {"sensor_id": "s", "type": "temperature", "value": float(i)}


def feed(runner, count):
    """What the service does for each request: the serving model answers, then the runner observes."""
    for i in range(count):
        r = reading(i)
        runner.observe(r, runner.serving.score(r), 0.005)


def kinds(runner):
    return [event["event"] for event in runner.events]


def test_a_consistent_challenger_is_promoted_and_the_old_model_becomes_the_way_back():
    old, new = Model("old", every(40)), Model("new", every(40))
    runner = ShadowRunner(new, serving=old, clock=Ticker())

    feed(runner, 300)

    assert runner.phase == "watch"
    assert runner.serving is new
    assert runner.shadow is old
    assert kinds(runner) == ["shadow_started", "promoted"]


def test_an_alert_storm_gets_the_challenger_rejected():
    old, new = Model("old", every(40)), Model("new", lambda i: True)
    runner = ShadowRunner(new, serving=old, clock=Ticker())

    feed(runner, 150)

    assert runner.phase == "idle"
    assert runner.serving is old
    assert runner.shadow is None
    assert kinds(runner) == ["shadow_started", "challenger_rejected"]
    assert any("alert storm" in reason for reason in runner.events[-1]["reasons"])


def test_a_crashing_challenger_is_rejected_and_the_service_keeps_answering():
    old, new = Model("old", every(40)), Model("new", error=RuntimeError("boom"))
    runner = ShadowRunner(new, serving=old, clock=Ticker())

    feed(runner, 150)  # must not raise

    assert runner.serving is old
    assert kinds(runner) == ["shadow_started", "challenger_rejected"]
    assert "100 of 100" in runner.events[-1]["reasons"][0]


def test_a_challenger_that_misses_the_champions_alerts_is_held_once():
    old, new = Model("old", every(40)), Model("new")  # the new model flags nothing
    runner = ShadowRunner(new, serving=old, clock=Ticker())

    feed(runner, 450)

    assert runner.phase == "shadow"
    assert runner.serving is old
    assert kinds(runner).count("held_for_review") == 1


def test_a_promoted_model_that_misbehaves_is_rolled_back_to_the_previous_one():
    old = Model("old", every(40))
    new = Model("new", lambda i: i % 40 == 0 if i < 300 else True)  # fine in shadow, a storm once it serves
    runner = ShadowRunner(new, serving=old, clock=Ticker())

    feed(runner, 400)

    assert kinds(runner) == ["shadow_started", "promoted", "rolled_back"]
    assert runner.phase == "idle"
    assert runner.serving is old
    assert runner.shadow is None


def test_a_promotion_that_stays_calm_is_confirmed():
    old, new = Model("old", every(40)), Model("new", every(40))
    runner = ShadowRunner(new, serving=old, clock=Ticker())

    feed(runner, 600)

    assert kinds(runner) == ["shadow_started", "promoted", "promotion_confirmed"]
    assert runner.phase == "idle"
    assert runner.serving is new
    assert runner.shadow is None


def test_without_a_serving_model_the_runner_only_reports():
    runner = ShadowRunner(Model("new", every(40)), clock=Ticker())

    for i in range(300):
        champion_answer = {"status": "ok", "is_anomaly": i % 40 == 0, "score": 0.5}
        runner.observe(reading(i), champion_answer, 0.005)

    assert runner.phase == "shadow"
    assert kinds(runner) == ["shadow_started"]
    assert runner.report()["decision"]["action"] == "promote"


def test_the_report_shows_the_phase_the_decision_and_the_events():
    runner = ShadowRunner(Model("new", every(40)), serving=Model("old", every(40)), clock=Ticker())
    feed(runner, 10)
    report = runner.report()
    assert report["phase"] == "shadow"
    assert report["serving_run_id"] == "old"
    assert report["challenger_run_id"] == "new"
    assert report["decision"]["action"] == "continue"
    assert [event["event"] for event in report["events"]] == ["shadow_started"]

    # once the challenger is rejected the rollout is over: no decision left to show
    bad = ShadowRunner(Model("new", lambda i: True), serving=Model("old", every(40)), clock=Ticker())
    feed(bad, 150)
    done = bad.report()
    assert done["phase"] == "idle"
    assert done["decision"] is None
    assert [event["event"] for event in done["events"]] == ["shadow_started", "challenger_rejected"]