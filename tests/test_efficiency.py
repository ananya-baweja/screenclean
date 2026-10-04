import json
import zipfile
from pathlib import Path

import pytest
import yaml

from screenclean import jobs

pytest.importorskip("torch")


def test_efficiency_task_on_cpu(tmp_path, capsys):
    repo = tmp_path / "repo"
    (repo / "jobs" / "queue").mkdir(parents=True)
    (repo / "configs").mkdir()
    cfg = {
        "sizes": {"64x96": [64, 96]},
        "cpu_repeats": 1,
        "models": [{"label": "ScreenCleanNet tiny (untrained)", "name": "scnet", "untrained": "scnet_tiny"}],
    }
    (repo / "configs" / "eff.yaml").write_text(yaml.safe_dump(cfg))
    spec = {"id": "0023_efficiency", "task": "efficiency", "runtime": "cpu", "config": "configs/eff.yaml"}
    (repo / "jobs" / "queue" / "0023_efficiency.yaml").write_text(yaml.safe_dump(spec))
    marker = tmp_path / "m.txt"
    assert jobs.run("auto", tmp_path / "drive", repo, marker, runtime="cpu") == 0
    assert "JOB FINISHED" in capsys.readouterr().out
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        (row,) = json.loads(zf.read("summary.json"))["rows"]
    assert row["params_m"] == pytest.approx(0.79, abs=0.01)  # scnet_tiny (P6)
    assert row["gflops_64x96"] > 0 and row["cpu_s_64x96"] > 0
