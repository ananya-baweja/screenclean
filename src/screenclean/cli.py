"""Command-line interface: ``python -m screenclean <command>``.

Commands so far:

- ``jobs run``        run the next queued job (used by the Colab runner notebook)
- ``jobs list``       show the queue and each job's state on Drive
- ``results ingest``  unpack a downloaded results zip into ``results/jobs/<id>/``
- ``scan``            photos of screens -> clean upright pages, a searchable PDF and Markdown
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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def expand_inputs(items: list[str]) -> list[Path]:
    """Files, folders (their images, sorted) and wildcards (``photos/*.jpg``; PowerShell doesn't
    expand them), without duplicates, in the order given."""
    import glob

    out: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif any(ch in item for ch in "*?["):
            found = sorted(Path(f) for f in glob.glob(item) if Path(f).suffix.lower() in IMAGE_EXTS)
        else:
            found = [p]
        out += [f for f in found if f not in out]
    return out


def _cmd_scan(args: argparse.Namespace) -> int:
    from screenclean.eval import tesseract
    from screenclean.product.export import searchable_pdf, to_markdown, to_text
    from screenclean.product.pipeline import scan
    from screenclean.utils.io import write_image

    photos = expand_inputs(args.inputs)
    missing = [p for p in photos if not p.is_file()]
    if missing or not photos:
        print(
            f"no such photo(s): {', '.join(map(str, missing))}" if missing else "no photos found",
            file=sys.stderr,
        )
        return 2
    if args.ocr == "tesseract" and not tesseract.available():
        print("Tesseract not found: install it or set TESSERACT_CMD (or use --ocr none)", file=sys.stderr)
        return 2
    results = []
    for i, photo in enumerate(photos, 1):
        res = scan(photo, cleaner=args.cleaner, ocr=args.ocr, mode=args.mode)
        results.append(res)
        found = (
            f"screen found ({res.detection['method']})" if res.detected else "screen NOT found: whole photo"
        )
        print(
            f"[{i}/{len(photos)}] {photo.name}: {found}, {len(res.lines)} lines, {res.timings['total']:.1f} s"
        )
        if args.pages:
            write_image(Path(args.pages) / f"{photo.stem}_page.jpg", res.page, quality=92)
    title = args.title or "Scanned screens"
    outputs = {
        args.pdf: lambda: searchable_pdf(results, title=title),
        args.md: lambda: to_markdown(results, title).encode("utf-8"),
        args.txt: lambda: to_text(results).encode("utf-8"),
        args.json: lambda: json.dumps([r.summary() for r in results], indent=2, ensure_ascii=False).encode(
            "utf-8"
        ),
    }
    for path, make in outputs.items():
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(make())
            print(f"wrote {path}")
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

    scan_p = sub.add_parser(
        "scan",
        help="photos of screens -> searchable PDF + Markdown",
        description="Find the screen in each photo, remove moire, straighten it, read the text. "
        "Without --pdf/--md/--txt/--json, writes scan.pdf and scan.md here.",
    )
    scan_p.add_argument("inputs", nargs="+", help="photos, folders of photos, or wildcards like photos/*.jpg")
    scan_p.add_argument("--pdf", default=None, help="searchable PDF: one page per photo")
    scan_p.add_argument("--md", default=None, help="Markdown with the text of every photo")
    scan_p.add_argument("--txt", default=None, help="plain text")
    scan_p.add_argument("--json", default=None, help="details per photo: corners, lines with boxes, timings")
    scan_p.add_argument("--pages", default=None, help="folder for the cleaned, straightened page images")
    scan_p.add_argument("--cleaner", default="fft_notch_local", choices=["fft_notch_local", "none"])
    scan_p.add_argument("--ocr", default="tesseract", choices=["tesseract", "none"])
    scan_p.add_argument("--mode", default="document", choices=["document", "photo"],
                        help="document: even out lighting and contrast; photo: keep the look")  # fmt: skip
    scan_p.add_argument("--title", default=None, help="title for the Markdown and PDF")
    scan_p.set_defaults(func=_cmd_scan)

    env_p = sub.add_parser("env", help="print the environment report")
    env_p.add_argument("--repo-root", default=Path("."), type=Path)
    env_p.set_defaults(func=_cmd_env)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan" and not any((args.pdf, args.md, args.txt, args.json)):
        args.pdf, args.md = "scan.pdf", "scan.md"
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    return args.func(args)
