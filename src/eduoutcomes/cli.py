"""Interfaz de línea de comandos: ``eduoutcomes <comando>`` (o ``python -m eduoutcomes``)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import __version__


def _setup_logging(verbose: int, log_file: Path | None = None) -> None:
    level = logging.DEBUG if verbose > 1 else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)
    for noisy in ("matplotlib", "PIL", "fontTools", "numba", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", default="configs/default.yaml", help="YAML file (default: %(default)s)")
    p.add_argument("--quick", action="store_true", help="structural test only (tiny grids); NOT reportable")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any configuration field, e.g. --set n_workers=4 (repeatable)")


def _load(args):
    from .config import load_config
    return load_config(args.config, args.overrides, quick=True if args.quick else None)


def cmd_run(args) -> int:
    import matplotlib
    cfg = _load(args)
    if not cfg.show_figures:
        matplotlib.use("Agg")
    _setup_logging(args.verbose, Path(cfg.output_dir) / "metadata" / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    from .pipeline import run
    run(cfg, until=args.until)
    return 0


def cmd_download(args) -> int:
    _setup_logging(args.verbose)
    from .constants import REFERENCE_SHA256
    from .data import download_uci, sha256
    path = download_uci(args.dest)
    digest = sha256(path.read_bytes())
    ok = digest == REFERENCE_SHA256
    logging.getLogger(__name__).info("SHA-256 %s — %s", digest, "matches reference" if ok else "DIFFERS from reference")
    return 0 if ok else 1


def cmd_synthetic(args) -> int:
    _setup_logging(args.verbose)
    from .data import make_synthetic
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic(args.n, args.seed).to_csv(dest, sep=";", index=False)
    logging.getLogger(__name__).info("Synthetic data (same schema as UCI 697; NOT real) → %s", dest)
    return 0


def cmd_verify(args) -> int:
    _setup_logging(args.verbose)
    from .metrics import verify_metrics
    logging.getLogger(__name__).info("Vectorised metrics match scikit-learn (max |Δ| = %.1e)", verify_metrics())
    return 0


def cmd_config(args) -> int:
    print(_load(args).to_json())
    return 0


def cmd_stages(_args) -> int:
    from .pipeline import STAGES
    for sid, name, _ in STAGES:
        print(f"{sid:12s} {name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="eduoutcomes", description="Calibrated and explainable multiclass prediction "
                                                                 "of student academic outcomes (UCI 697).")
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("-v", "--verbose", action="count", default=1, help="-vv shows tables in the log")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the whole pipeline")
    _add_config_args(p)
    p.add_argument("--until", default=None, help="stop after this stage (see `eduoutcomes stages`)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("download-data", help="download UCI 697 and verify its SHA-256")
    p.add_argument("--dest", default="data/raw/data.csv")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("make-synthetic", help="write synthetic data with the UCI 697 schema (tests/CI only)")
    p.add_argument("--dest", default="data/synthetic/data.csv")
    p.add_argument("--n", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synthetic)

    p = sub.add_parser("verify-metrics", help="check vectorised metrics against scikit-learn")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("show-config", help="print the resolved configuration")
    _add_config_args(p)
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("stages", help="list pipeline stages")
    p.set_defaults(func=cmd_stages)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
