import argparse
import ast
from pathlib import Path

import pandas as pd


METRICS = [
    "PCC_F",
    "PCC_S",
    "PCC_M",
    "MI_F",
    "MI_M",
    "AUC_0vNZ",
    "AUC_Q50",
    "NRMSE_F",
    "NRMSE_M",
    "JSDIV_M",
]


def read_metrics(log_path: Path):
    metrics = None
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if "PCC_F" not in line or not line.startswith("{"):
            continue
        metrics = ast.literal_eval(line)
    return metrics


def collect_rows(base_dir: Path, loss_index: int, allowed_projects: set[str]):
    rows = []
    for project_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        if allowed_projects and project_dir.name not in allowed_projects:
            continue
        for run_dir in sorted(path for path in project_dir.iterdir() if path.is_dir()):
            log_path = run_dir / "log.txt"
            if not log_path.exists():
                continue
            metrics = read_metrics(log_path)
            if metrics is None:
                continue
            parts = run_dir.name.split("_")
            if len(parts) <= loss_index:
                continue
            row = {
                "project": project_dir.name,
                "model": parts[0],
                "loss": parts[loss_index],
                "run": run_dir.name,
            }
            row.update(metrics)
            rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--loss-index", type=int, default=1)
    parser.add_argument("--project", action="append", default=[])
    return parser.parse_args()


def main(args):
    rows = collect_rows(args.base_dir, args.loss_index, set(args.project))
    if not rows:
        raise RuntimeError(f"No completed run was found under {args.base_dir}")

    dataframe = pd.DataFrame(rows)
    metric_columns = [name for name in METRICS if name in dataframe.columns]
    dataframe = dataframe[["project", "model", "loss", "run", *metric_columns]]
    dataframe = dataframe.sort_values(["project", "model", "loss", "run"])

    summary = (
        dataframe.groupby(["model", "loss"], as_index=False)[metric_columns]
        .mean()
        .sort_values(["model", "loss"])
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.summary_csv}")


if __name__ == "__main__":
    main(parse_args())
