from pathlib import Path

import pytest

from nsqa.pipeline import Pipeline

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def pipe() -> Pipeline:
    return Pipeline(data_dir=DATA, backend="offline").build()
