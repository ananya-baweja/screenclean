"""Command-line interface: ``python -m screenclean <command>``.

Commands so far:

- ``jobs run``        run the next queued job (used by the Colab runner notebook)
- ``jobs list``       show the queue and each job's state on Drive
- ``results ingest``  unpack a downloaded results zip into ``results/jobs/<id>/``
- ``env``             print the environment report as JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from screenclean import __version__


def _cmd_jobs_run(args: argparse.Namespace) -> int:
    from screenclean import jobs

    return jobs.run(args.job, args.drive_root, args.repo_root, args.marker, args.runtime)


def _cmd_jobs_list(args: argparse.Namespace) -> int:
    from screenclean import jobs

    for line in jobs.describe_queue(args.repo_root, args.drive_root):
        print(line)
    return 0


def _cmd_results_ingest(args: argparse.Namespace) -> int:
    from screenclean.utils.results_zip import IngestError, ingest_zip

    status = 0
    for z in args.zips:
        try:
            dest = ingest_zip(z, args.repo_root, forbid_pattern=args.forbid, overwrite=args.overwrite)
            print(f"ingested {Path(z).name} -> {dest}")
        except IngestError as e:
            print(f"NOT ingested {Path(z).name}: {e}", file=sys.stderr)
            status = 1
    return status


def _cmd_tables(args: argparse.Namespace) -> int:
    from screenclean.eval.tables import build_tables

    for path in build_tables(args.repo_root):
        print(f"wrote {path}")
    return 0


def _cmd_capture_kit(args: argparse.Namespace) -> int:
    from screenclean.render.capture_kit import build_kit

    meta = build_kit(args.out)
    n_lines = sum(len(p["lines"]) for p in meta["pages"])
    print(f"wrote {len(meta['pages'])} pages ({n_lines} text lines) to {args.out}")
    return 0


def _cmd_env(args: argparse.Namespace) -> int:
    from screenclean.utils.env import collect_env

    print(json.dumps(collect_env(args.repo_root), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="screenclean", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"screenclean {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    jobs_p = sub.add_parser("jobs", help="Colab job queue").add_subparsers(dest="jobs_command", required=True)
    run_p = jobs_p.add_parser("run", help="run the next job (or --job <id>)")
    run_p.add_argument("--job", default="auto", help='job id, or "auto" for the next runnable job')
    run_p.add_argument("--drive-root", required=True, type=Path, help="project folder on Google Drive")
    run_p.add_argument("--repo-root", default=Path("."), type=Path)
    run_p.add_argument(
        "--marker",
        default=Path("/content/last_results_zip.txt"),
        type=Path,
        help="file that receives the results zip path (read by the notebook)",
    )
    run_p.add_argument("--runtime", choices=["cpu", "gpu"], default=None, help="override runtime detection")
    run_p.set_defaults(func=_cmd_jobs_run)

    list_p = jobs_p.add_parser("list", help="show the queue and job states")
    list_p.add_argument("--drive-root", type=Path, default=None)
    list_p.add_argument("--repo-root", default=Path("."), type=Path)
    list_p.set_defaults(func=_cmd_jobs_list)

    res_p = sub.add_parser("results", help="job results")
    res_p = res_p.add_subparsers(dest="results_command", required=True)
    ing_p = res_p.add_parser("ingest", help="unpack results zips into results/jobs/<id>/")
    ing_p.add_argument("zips", nargs="+", type=Path)
    ing_p.add_argument("--repo-root", default=Path("."), type=Path)
    ing_p.add_argument("--forbid", default=None, help="regex that must not appear in any file (ignores case)")
    ing_p.add_argument("--overwrite", action="store_true", help="replace an existing results folder")
    ing_p.set_defaults(func=_cmd_results_ingest)

    tab_p = sub.add_parser("tables", help="rebuild results/tables/*.md from ingested job results")
    tab_p.add_argument("--repo-root", default=Path("."), type=Path)
    tab_p.set_defaults(func=_cmd_tables)

    kit_p = sub.add_parser("capture-kit", help="build the capture kit (pages with markers + slideshow)")
    kit_p.add_argument("--out", default=Path("capture_kit"), type=Path)
    kit_p.set_defaults(func=_cmd_capture_kit)

    env_p = sub.add_parser("env", help="print the environment report")
    env_p.add_argument("--repo-root", default=Path("."), type=Path)
    env_p.set_defaults(func=_cmd_env)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    return args.func(args)
