"""Assemble the folder that is pushed to the Hugging Face Space.

  python -m scripts.make_space

Writes space_build/ with: a Dockerfile for Spaces (port 7860, user 1000), the
Space's README (its header configures the Space), requirements-serve.txt, src/
and model_store/. A .git folder already inside space_build/ is left alone, so
you can run this again after every change and push.
"""

import argparse
import shutil
from pathlib import Path

DOCKERFILE = """FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1

# Hugging Face runs the container as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Libraries first: this layer is cached until requirements-serve.txt changes.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --default-timeout=120 -r requirements-serve.txt

COPY --chown=user src ./src
COPY --chown=user model_store ./model_store

ENV MODEL_DIR=/home/user/app/model_store
USER user

EXPOSE 7860

# One worker on purpose: the scorer keeps each sensor's recent readings in memory.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

README = """---
title: IoT Fleet MLOps
emoji: 📡
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# IoT Fleet MLOps: anomaly detection service

A live demo of the service from [iot-fleet-mlops](https://github.com/Xbalawi/iot-fleet-mlops):
one Isolation Forest per sensor type, scoring one reading per request.

- `/docs`: interactive API page, try `POST /predict` from here
- `/health`: which model run is serving
- `/metrics`: Prometheus metrics
- `/shadow`: rollout state (shadow mode is off in this demo)

The service keeps the last 50 readings of every sensor in memory, so a sensor needs
50 readings before it is scored (`warming_up` until then). Timestamps must include a
timezone and be newer than the sensor's previous reading. State lives in memory only:
restarting the Space resets it.
"""


def copy_tree(source, target):
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="space_build")
    args = parser.parse_args(argv)

    for needed in ("src", "model_store", "requirements-serve.txt"):
        if not Path(needed).exists():
            raise SystemExit(f"{needed} not found: run this from the repository root")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    copy_tree(Path("src"), out / "src")
    copy_tree(Path("model_store"), out / "model_store")
    shutil.copy("requirements-serve.txt", out / "requirements-serve.txt")
    (out / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8", newline="\n")
    (out / "README.md").write_text(README, encoding="utf-8", newline="\n")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()