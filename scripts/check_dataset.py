from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_schema import check_required_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--required",
        nargs="*",
        default=[],
        help="Optional explicit list of required column names.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    report = check_required_columns(df, args.required)

    print(f"Rows: {report.n_rows}")
    print(f"Columns: {report.n_columns}")
    if args.required:
        print(f"Present required columns: {list(report.present_required)}")
        print(f"Missing required columns: {list(report.missing_required)}")


if __name__ == "__main__":
    main()
