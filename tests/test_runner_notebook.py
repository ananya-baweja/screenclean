import importlib.util
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_runner_nb", ROOT / "tools" / "make_runner_nb.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_notebook_matches_generator():
    gen = _load_generator()
    assert (ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8") == gen.render()


def test_notebook_structure():
    nb = nbformat.reads(_load_generator().render(), as_version=4)
    nbformat.validate(nb)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert len(code) == 4
    assert '#@param {type:"string"}' in code[0]
    assert "drive.mount" in code[1] and 'pip install -q -e ".[colab]"' in code[1]
    assert "python -m screenclean jobs run" in code[2]
    assert "last_results_zip.txt" in code[3]
    joined = "\n".join(code).lower()
    for word in ("token", "secret", "userdata"):
        assert word not in joined
