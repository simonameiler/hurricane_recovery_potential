#!/usr/bin/env python3
"""Combine aggregated CSV outputs into per-event raw/scaled files.

Simple CLI that wraps modules.impact_utils.combine_aggregated_outputs_to_per_event.
Designed to run on the cluster; it will add the repository root to sys.path so the
module import works when executed from the repo or from the batch script.
"""
import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Combine aggregated CSVs into per-event raw/scaled outputs")
    parser.add_argument("--base-out-dir", default="/home/groups/bakerjw/smeiler/climada_data/data/results/hrp_impacts_out", help="Base output directory (default ./impacts_out)")
    parser.add_argument("--source-subdirs", default=None, help="Comma-separated list of source subdirs (default: per_event,per_state_aggregated)")
    parser.add_argument("--dest-subdir", default="by_event", help="Destination subdir under base_out_dir (default 'by_event')")
    parser.add_argument("--raw-dir-name", default="raw", help="Name of raw output folder (default 'raw')")
    parser.add_argument("--scaled-dir-name", default="scaled", help="Name of scaled output folder (default 'scaled')")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Disable verbose prints")
    args = parser.parse_args()

    # ensure repo root is importable (script lives in scripts/)
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    try:
        from modules.impact_utils import combine_aggregated_outputs_to_per_event
    except Exception as e:
        print(f"Failed to import combine_aggregated_outputs_to_per_event: {e}", file=sys.stderr)
        raise

    if args.source_subdirs is None:
        source_subs = None
    else:
        source_subs = [s.strip() for s in args.source_subdirs.split(",") if s.strip()]

    written = combine_aggregated_outputs_to_per_event(
        base_out_dir=args.base_out_dir,
        source_subdirs=source_subs,
        dest_subdir=args.dest_subdir,
        raw_dir_name=args.raw_dir_name,
        scaled_dir_name=args.scaled_dir_name,
        verbose=args.verbose,
    )

    print("Wrote files summary:")
    for k, v in written.items():
        print(f"  {k}: {len(v)} files")
        if args.verbose:
            for p in v:
                print(f"    {p}")


if __name__ == "__main__":
    main()
