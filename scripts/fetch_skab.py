"""Download SKAB (Skoltech Anomaly Benchmark) and print a summary of every file.

  python -m scripts.fetch_skab

The data is GPL-3.0 licensed, so it is downloaded to data/skab/ (git-ignored)
and never committed.
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://github.com/waico/SKAB/archive/refs/heads/master.zip"


def download(out_dir):
    """Download the repository archive and extract only its data/ CSV files. Returns the file count."""
    out = Path(out_dir)
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    count = 0
    for name in archive.namelist():
        parts = name.split("/")
        # SKAB-master/data/<folder>/<file>.csv
        if len(parts) >= 3 and parts[1] == "data" and name.endswith(".csv"):
            target = out.joinpath(*parts[2:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            count += 1
    return count


def summarize(out_dir):
    """One line per CSV: rows, sampling step, anomaly share, and the columns of the first file."""
    rows = []
    columns = None
    for path in sorted(Path(out_dir).rglob("*.csv")):
        frame = pd.read_csv(path, sep=";", parse_dates=["datetime"])
        if columns is None:
            columns = list(frame.columns)
        step = frame["datetime"].diff().dropna().median()
        rows.append(
            {
                "file": path.relative_to(out_dir).as_posix(),
                "rows": len(frame),
                "step_s": step.total_seconds() if pd.notna(step) else None,
                "anomaly_%": round(100 * frame["anomaly"].mean(), 1) if "anomaly" in frame else None,
                "sorted": bool(frame["datetime"].is_monotonic_increasing),
                "dups": int(frame["datetime"].duplicated().sum()),
            }
        )
    return pd.DataFrame(rows), columns


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/skab")
    args = parser.parse_args(argv)

    if not list(Path(args.out).rglob("*.csv")):
        print(f"extracted {download(args.out)} files to {args.out}")
    table, columns = summarize(args.out)
    print("columns:", columns)
    print(table.to_string(index=False))
    print(f"\n{len(table)} files, {table['rows'].sum()} rows")


if __name__ == "__main__":
    main()