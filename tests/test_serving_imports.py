import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_service_does_not_import_mlflow():
    """The Docker image ships without MLflow, so importing the service must not need it."""
    code = "import sys; import src.api.app; print('mlflow' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"