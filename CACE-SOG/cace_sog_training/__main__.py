"""Command line entry: python -m cace_sog_training --help."""
import argparse
import json
import math
from pathlib import Path
import sys

from . import __version__
from .data import DEFAULT_CHANNELS


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def selection(parser):
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--spin-policy", choices=("exclude", "consistent-state"), default="exclude")
    parser.add_argument("--spin-policy-note", help="record the actual reviewed electronic-state policy; this is not a spin-input model")


def model_options(parser, grid=False):
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--cutoff", type=finite, default=5.)
    if grid:
        parser.add_argument("--n-dl", type=finite, nargs="+", default=[2., 1., .5])
    else:
        parser.add_argument("--n-dl", type=finite, default=1.)
    parser.add_argument("--max-modes", type=positive, default=250000)
    parser.add_argument("--atom-basis", type=positive, default=4)
    parser.add_argument("--radial-basis", type=positive, default=6)
    parser.add_argument("--max-l", type=int, default=3)
    parser.add_argument("--max-nu", type=positive, default=3)
    parser.add_argument("--message-passing", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--batch-size", type=positive, default=8)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Full-retention disk-backed ELEMENTA CACE-SOG training (single device).")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="validate exported frames and build/resume a disk index")
    index.add_argument("--export-dir", required=True, type=Path)
    index.add_argument("--database", required=True, type=Path)
    index.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    index.add_argument("--resume", action="store_true")
    index.add_argument("--commit-every", type=positive, default=1000)
    index.add_argument("--limit-records", type=positive, help="debug prefix only; leaves index partial and unusable for train")
    train = sub.add_parser("train", help="train all selected train frames; choose best.pt using val")
    selection(train)
    model_options(train)
    train.add_argument("--run-dir", required=True, type=Path)
    train.add_argument("--model-kind", choices=("sog", "sr"), default="sog")
    train.add_argument("--epochs", type=positive, default=50)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--lr", type=finite, default=1e-3)
    train.add_argument("--energy-weight", type=finite, default=1.)
    train.add_argument("--force-weight", type=finite, default=10.)
    train.add_argument("--max-grad-norm", type=finite, default=10.)
    train.add_argument("--checkpoint-every", type=positive, default=1000)
    train.add_argument("--validate-every-steps", type=int, default=0, help="0: full val each epoch; positive: also validate at step intervals")
    train.add_argument("--shuffle", choices=("affine", "sequential"), default="affine")
    train.add_argument("--resume", type=Path)
    train.add_argument("--max-steps", type=positive, help="explicit debug stop only; status remains partial")
    train.add_argument("--progress-seconds", type=finite, default=30.)
    for name in ("smoke", "converge"):
        command = sub.add_parser(name)
        selection(command)
        model_options(command, grid=name == "converge")
        command.add_argument("--output", required=True, type=Path)
        if name == "converge":
            command.add_argument("--samples", type=positive, default=4)
            command.add_argument("--checkpoint", type=Path, help="repeat the grid check on an existing trained model")
        else:
            command.add_argument("--steps", type=positive, default=10, help="optimizer smoke steps; records a bounded memory trace")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--database", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--channels", nargs="+", help="default: the checkpoint training channels")
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=positive, default=8)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.add_argument("--progress-seconds", type=finite, default=30.)
    env = sub.add_parser("environment")
    env.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            from .data import build_index
            result = build_index(args.export_dir, args.database, channels=args.channels,
                                 resume=args.resume, commit_every=args.commit_every,
                                 limit_records=args.limit_records)
        else:
            from . import runtime
            if args.command == "environment":
                result = runtime.environment()
                if args.output:
                    runtime.write_json(args.output, result)
            else:
                if hasattr(args, "workers") and args.workers < 0:
                    raise ValueError("workers must be nonnegative")
                if hasattr(args, "progress_seconds") and args.progress_seconds <= 0:
                    raise ValueError("progress-seconds must be positive")
                result = {"train": runtime.run_train, "evaluate": runtime.run_evaluate,
                          "smoke": runtime.run_smoke, "converge": runtime.run_converge}[args.command](args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Resume index with --resume; resume train from the last completed optimizer-step checkpoint.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
