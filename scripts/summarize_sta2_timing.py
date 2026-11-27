"""Save measured full-epoch timing and quote-derived credit estimates (stdlib only)."""
import argparse
import json
import math
from pathlib import Path


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--quoted-credits", type=positive, required=True)
    parser.add_argument("--quoted-hours", type=positive, default=12)
    parser.add_argument("--allocation-seconds", type=positive,
                        help="Completed top-level job ElapsedRaw from sacct; includes startup and shutdown")
    parser.add_argument("--actual-credits", type=positive,
                        help="Actual charge from VSC accounting, when available")
    args = parser.parse_args()
    history = json.loads((args.run_dir / "history.json").read_text())
    if not isinstance(history, list) or not history:
        parser.error("no completed epoch; inspect the log and accounting before retrying")
    rate = args.quoted_credits / args.quoted_hours
    epochs = []
    for row in history:
        if "epoch_seconds" not in row:
            parser.error("history lacks full epoch timing; use the updated trainer")
        seconds = float(row["epoch_seconds"])
        if not math.isfinite(seconds) or seconds <= 0:
            parser.error("invalid epoch timing")
        epochs.append({key: row[key] for key in (
            "epoch", "train_seconds", "validation_seconds", "checkpoint_seconds", "epoch_seconds",
        )} | {"estimated_credits": seconds / 3600 * rate})
    report = {
        "run_dir": str(args.run_dir.resolve()),
        "quoted_credits": args.quoted_credits, "quoted_hours": args.quoted_hours,
        "quote_derived_credits_per_hour": rate,
        "epochs": epochs,
        "allocation_seconds": args.allocation_seconds,
        "estimated_allocation_credits": (
            args.allocation_seconds / 3600 * rate if args.allocation_seconds else None
        ),
        "actual_credits": args.actual_credits,
        "note": "Quote-derived costs are estimates, not charges. Epoch timing includes checkpoint saves but excludes startup, history publication and final synchronization. First-epoch timing includes cold caches and warmup.",
    }
    output = args.run_dir / "timing-report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
