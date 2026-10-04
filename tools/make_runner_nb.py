"""Generate ``notebooks/colab_runner.ipynb``.

The notebook only calls the package; all logic lives in ``screenclean.jobs``.
Regenerate after editing this file (never hand-edit the notebook JSON):

    python tools/make_runner_nb.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "colab_runner.ipynb"

INTRO = """\
# ScreenClean job runner

Runs the next job from `jobs/queue/` in the repo. No logins or keys are needed except
Google Drive access.

1. **Runtime → Change runtime type** → pick **T4 GPU** or **CPU**, whichever the job needs.
2. **Runtime → Run all**, and allow Google Drive access.
3. Keep this tab open until cell 3 prints `JOB FINISHED`, `TIME BUDGET REACHED` or `JOB FAILED`.
4. Cell 4 downloads `<job>.zip`. Move it from Downloads into your `results_inbox` folder.
   If the page lost its connection, cell 4 can show an old message: click its ▶ button to run it again.
   Every zip is also kept in Google Drive → `screenclean/results_zips/`.

**Is it still running?** The page can stop showing new lines while the job keeps working. Open
Google Drive → `screenclean/job_status/<job>.json`: its `progress` and `progress_utc` fields update
after every step.
"""

SETTINGS = """\
# Cell 1 — Settings (Colab form fields)
GH_USER = "ananya-baweja"   #@param {type:"string"}
REPO = "screenclean"        #@param {type:"string"}
BRANCH = "main"             #@param {type:"string"}
JOB = "auto"                #@param {type:"string"}
DRIVE_ROOT = "/content/drive/MyDrive/screenclean"  #@param {type:"string"}
"""

SETUP = """\
# Cell 2 — Setup (public repo, no login)
import os, subprocess
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_ROOT, exist_ok=True)
repo_dir = f"/content/{REPO}"
if os.path.isdir(repo_dir):
    subprocess.run(["git", "-C", repo_dir, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", repo_dir, "checkout", "-B", BRANCH, f"origin/{BRANCH}"], check=True)
else:
    url = f"https://github.com/{GH_USER}/{REPO}.git"
    subprocess.run(["git", "clone", "-b", BRANCH, url, repo_dir], check=True)
%cd {repo_dir}
!git log --oneline -1
!pip install -q -e ".[colab]"
"""

RUN = """\
# Cell 3 — Run the next job (prints JOB FINISHED / TIME BUDGET REACHED / JOB FAILED)
!python -m screenclean jobs run --job "{JOB}" --drive-root "{DRIVE_ROOT}"
"""

DOWNLOAD = """\
# Cell 4 — Download the results zip, then move it into your results_inbox folder
import pathlib, subprocess
from google.colab import files
marker = pathlib.Path("/content/last_results_zip.txt")
if marker.exists():
    zip_path = marker.read_text().strip()
else:  # e.g. cell 3 ran again: take the zip of the job that ended last, if it finished
    cmd = ["python", "-m", "screenclean", "jobs", "last-zip", "--drive-root", DRIVE_ROOT]
    zip_path = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
if zip_path:
    name = pathlib.Path(zip_path).name
    print(f"Downloading {name}. Move it from Downloads into results_inbox.")
    files.download(zip_path)
    print(f"If nothing downloaded, it is also in Google Drive: screenclean/results_zips/{name}")
else:
    print("No results zip this time. If it said TIME BUDGET REACHED, run the notebook again later.")
"""


def build() -> nbformat.NotebookNode:
    nb = new_notebook(
        cells=[
            new_markdown_cell(INTRO),
            new_code_cell(SETTINGS),
            new_code_cell(SETUP),
            new_code_cell(RUN),
            new_code_cell(DOWNLOAD),
        ],
        metadata={
            "colab": {"name": "colab_runner.ipynb", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
    )
    # Stable cell ids keep the generated file identical between runs.
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i}"
    return nb


def render() -> str:
    return nbformat.writes(build()) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed notebook is out of date")
    args = parser.parse_args()
    text = render()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        raise SystemExit(0 if current == text else "notebooks/colab_runner.ipynb is out of date")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
