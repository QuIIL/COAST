from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch


PROJECTS = [
    "abalo_human_squamous_cell_carcinoma",
    "erickson_human_prostate_cancer_p1",
    "mirzazadeh_mouse_bone",
    "mirzazadeh_mouse_brain_p1",
    "mirzazadeh_mouse_brain_p2",
    "vicari_mouse_brain",
    "villacampa_lung_organoid",
]


RESULT_FILES = [
    "group_pcc_long.csv",
    "group_pcc_wide_gene_mean.csv",
    "group_pcc_wide_flat.csv",
    "per_gene_pcc_long.csv",
    "overall_pcc.csv",
    "load_report.csv",
]


DEFAULT_GROUP_ORDER = {

    "tumor_marker": 1,
    "tumor_markers": 1,
    "cancer_marker": 1,
    "cancer_markers": 1,
    "oncogene": 1,
    "oncogenic": 1,
    "tumor": 1,

    "lineage_marker": 2,
    "lineage_markers": 2,
    "tumor_lineage": 2,
    "tissue_lineage": 2,
    "epithelial_lineage": 2,

    "epithelial": 3,
    "epithelial_marker": 3,
    "epithelial_markers": 3,


    "tumor_stroma_boundary": 4,
    "tumor-stroma_boundary": 4,
    "tumor_stroma": 4,
    "boundary": 4,
    "spatial_boundary": 4,
    "spatial_gradient": 4,
    "gradient": 4,

    "stroma_ecm": 5,
    "stroma": 5,
    "ecm": 5,
    "extracellular_matrix": 5,
    "matrix": 5,

    "caf_fibrosis": 6,
    "caf": 6,
    "fibrosis": 6,
    "fibroblast": 6,
    "fibroblasts": 6,
    "collagen": 6,


    "immune": 7,
    "immune_marker": 7,
    "immune_markers": 7,
    "immune_cell": 7,
    "immune_cells": 7,

    "inflammation": 8,
    "inflammatory": 8,
    "cytokine": 8,
    "chemokine": 8,

    "antigen_presentation": 9,
    "mhc": 9,
    "hla": 9,


    "hypoxia": 10,
    "hypoxic": 10,

    "metabolism": 11,
    "metabolic": 11,
    "glycolysis": 11,
    "lipid_metabolism": 11,

    "proliferation": 12,
    "cell_cycle": 12,
    "growth": 12,

    "emt_invasion": 13,
    "emt": 13,
    "invasion": 13,
    "migration": 13,
    "metastasis": 13,

    "vascular": 14,
    "angiogenesis": 14,
    "endothelial": 14,


    "housekeeping": 90,
    "ribosomal": 91,
    "mitochondrial": 92,
    "other": 99,
    "unknown": 99,
    "misc": 99,
    "miscellaneous": 99,
}


def normalize_group_name(x: str) -> str:
    x = str(x).strip().lower()
    x = x.replace("-", "_")
    x = x.replace("/", "_")
    x = x.replace(" ", "_")
    while "__" in x:
        x = x.replace("__", "_")
    return x


def load_group_order_map(group_order_csv: str | None = None) -> Dict[str, int]:
    order_map = {}

    for k, v in DEFAULT_GROUP_ORDER.items():
        order_map[normalize_group_name(k)] = int(v)

    if group_order_csv is not None:
        df = pd.read_csv(group_order_csv)
        required = {"group", "group_order"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"group_order_csv missing columns: {sorted(missing)}")

        for row in df.itertuples(index=False):
            g = normalize_group_name(getattr(row, "group"))
            o = int(getattr(row, "group_order"))
            order_map[g] = o

    return order_map


def get_group_order(group_name: str, group_order_map: Dict[str, int]) -> int:
    return int(group_order_map.get(normalize_group_name(group_name), 99))


def strip_sftp_url(path_or_url: str) -> str:


    if path_or_url.startswith("sftp://"):
        s = path_or_url[len("sftp://") :]
        first_slash = s.find("/")
        if first_slash < 0:
            raise ValueError(f"Invalid sftp URL: {path_or_url}")
        return s[first_slash:]
    return path_or_url


def script_supports_arg(script_path: Path, arg_name: str) -> bool:


    try:
        txt = script_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return arg_name in txt


def get_meta_slide_and_expr(meta: Any) -> tuple[str, torch.Tensor]:


    if isinstance(meta, dict):
        slide = (
            meta.get("slide")
            or meta.get("slide_id")
            or meta.get("sample")
            or meta.get("sample_id")
        )
        if slide is None:
            raise KeyError(f"Could not find slide key in metadata dict keys={list(meta.keys())}")

        y = (
            meta.get("gene_exp")
            if "gene_exp" in meta
            else meta.get("expr")
            if "expr" in meta
            else meta.get("X")
            if "X" in meta
            else None
        )
        if y is None:
            raise KeyError(f"Could not find expression key in metadata dict keys={list(meta.keys())}")

        return str(slide), torch.as_tensor(y, dtype=torch.float32)


    slide = str(meta[3])
    y = torch.as_tensor(meta[4], dtype=torch.float32)
    return slide, y


def load_cache_metadata(cache_path: Path, split: str):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    if split not in cache:
        raise KeyError(f"Split '{split}' not found in {cache_path}. Available keys={list(cache.keys())}")

    data = cache[split]

    if "metadata" not in data:
        raise KeyError(f"cache[{split}] does not contain metadata.")

    return data["metadata"]


def convert_peka_yhat_to_slide_pt(
    cache_path: Path,
    split: str,
    yhat_npy: Path,
    out_dir: Path,
    ytrue_npy: Path | None = None,
    validate_tol: float = 1e-4,
    strict_ytrue: bool = False,
) -> dict:


    metadata = load_cache_metadata(cache_path, split=split)

    print(f"[LOAD] PEKA Yhat: {yhat_npy}")
    yhat = np.load(yhat_npy, mmap_mode="r")

    if yhat.ndim != 2:
        raise ValueError(f"{yhat_npy}: expected 2D [N, G], got shape={yhat.shape}")

    n_meta = len(metadata)

    if yhat.shape[0] != n_meta:
        raise ValueError(
            f"{yhat_npy}: row count mismatch. "
            f"Yhat N={yhat.shape[0]}, cache metadata N={n_meta}. "
            f"Check split={split} and project."
        )


    slides = []
    true_rows = []

    for meta in metadata:
        slide, y = get_meta_slide_and_expr(meta)
        slides.append(slide)
        true_rows.append(y)

    ytrue_cache = torch.stack(true_rows, dim=0).numpy()

    if yhat.shape[1] != ytrue_cache.shape[1]:
        raise ValueError(
            f"{yhat_npy}: gene dimension mismatch. "
            f"Yhat G={yhat.shape[1]}, cache G={ytrue_cache.shape[1]}"
        )

    ytrue_status = "not_checked"
    ytrue_max_abs = np.nan
    ytrue_mean_abs = np.nan

    if ytrue_npy is not None and ytrue_npy.exists():
        print(f"[CHECK] PEKA gene file: {ytrue_npy}")
        ytrue_peka = np.load(ytrue_npy, mmap_mode="r")

        if ytrue_peka.shape != ytrue_cache.shape:
            msg = (
                f"{ytrue_npy}: shape mismatch. "
                f"gene file={ytrue_peka.shape}, cache={ytrue_cache.shape}"
            )
            if strict_ytrue:
                raise ValueError(msg)
            print(f"[WARN] {msg}")
            ytrue_status = "shape_mismatch"

        else:
            diff = np.asarray(ytrue_peka, dtype=np.float32) - ytrue_cache.astype(np.float32)
            abs_diff = np.abs(diff)
            ytrue_max_abs = float(np.nanmax(abs_diff))
            ytrue_mean_abs = float(np.nanmean(abs_diff))

            if ytrue_max_abs <= validate_tol:
                ytrue_status = "matched"
                print(
                    f"[OK] gene_test.npy matches cache order. "
                    f"max_abs={ytrue_max_abs:.6g}, mean_abs={ytrue_mean_abs:.6g}"
                )
            else:
                ytrue_status = "mismatch"
                msg = (
                    f"gene_test.npy does not match cache order/value. "
                    f"max_abs={ytrue_max_abs:.6g}, mean_abs={ytrue_mean_abs:.6g}. "
                    f"This may indicate wrong split, wrong cache, or order mismatch."
                )
                if strict_ytrue:
                    raise ValueError(msg)
                print(f"[WARN] {msg}")

    else:
        print("[WARN] gene_test.npy not provided/found; order validation skipped.")


    idx_by_slide = defaultdict(list)

    for idx, slide in enumerate(slides):
        idx_by_slide[slide].append(idx)

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for slide_id in sorted(idx_by_slide.keys()):
        idxs = np.asarray(idx_by_slide[slide_id], dtype=np.int64)
        pred_slide = np.asarray(yhat[idxs], dtype=np.float32)
        out_path = out_dir / f"{slide_id}.pt"

        torch.save(torch.from_numpy(pred_slide), out_path)

        rows.append(
            {
                "slide": slide_id,
                "n_spots": int(pred_slide.shape[0]),
                "n_genes": int(pred_slide.shape[1]),
                "path": str(out_path),
            }
        )

    slide_report = pd.DataFrame(rows)
    slide_report_path = out_dir / "peka_slide_conversion_report.csv"
    slide_report.to_csv(slide_report_path, index=False)

    print(f"[SAVE] Converted PEKA slide-wise predictions: {out_dir}")
    print(f"[SAVE] Slide report: {slide_report_path}")

    return {
        "n_spots": int(yhat.shape[0]),
        "n_genes": int(yhat.shape[1]),
        "n_slides": int(len(idx_by_slide)),
        "ytrue_status": ytrue_status,
        "ytrue_max_abs": ytrue_max_abs,
        "ytrue_mean_abs": ytrue_mean_abs,
        "converted_dir": str(out_dir),
    }


def load_membership_order_table(
    membership_csv: Path,
    projects: list[str],
    group_order_map: Dict[str, int],
    min_priority_applied: int | None = None,
    max_priority_applied: int | None = None,
) -> pd.DataFrame:


    df = pd.read_csv(membership_csv)

    required = {"project", "gene_index", "gene", "group"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"membership_csv missing columns: {sorted(missing)}")

    df = df[df["project"].astype(str).isin([str(p) for p in projects])].copy()

    df["project"] = df["project"].astype(str)
    df["group"] = df["group"].astype(str)
    df["gene"] = df["gene"].astype(str)
    df["gene_index"] = df["gene_index"].astype(int)

    if "priority" in df.columns:
        df["gene_priority"] = pd.to_numeric(df["priority"], errors="coerce")
    else:
        df["gene_priority"] = np.nan

    if min_priority_applied is not None:
        df = df[df["gene_priority"].fillna(-999) >= float(min_priority_applied)].copy()

    if max_priority_applied is not None:
        df = df[df["gene_priority"].fillna(999) <= float(max_priority_applied)].copy()

    df["gene_priority_sort"] = df["gene_priority"].fillna(999)
    df["group_order"] = df["group"].map(lambda x: get_group_order(x, group_order_map)).astype(int)

    df = (
        df.drop_duplicates(subset=["project", "group", "gene_index"])
        .sort_values(
            ["project", "group_order", "group", "gene_priority_sort", "gene_index"],
            ascending=[True, True, True, True, True],
        )
        .reset_index(drop=True)
    )

    return df


def add_group_order(df: pd.DataFrame, group_order_map: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()

    if "group" in df.columns:
        df["group_order"] = df["group"].map(lambda x: get_group_order(x, group_order_map)).astype(int)

    return df


def attach_gene_order_info(
    df: pd.DataFrame,
    membership_order: pd.DataFrame,
) -> pd.DataFrame:


    if not {"project", "group"}.issubset(df.columns):
        return df

    if membership_order.empty:
        return df

    gene_list_df = (
        membership_order
        .groupby(["project", "group"], as_index=False)
        .agg(
            genes=("gene", lambda x: ";".join([str(v) for v in x])),
            gene_indices=("gene_index", lambda x: ";".join([str(int(v)) for v in x])),
            gene_priorities=(
                "gene_priority",
                lambda x: ";".join(
                    [
                        "" if pd.isna(v) else str(int(v)) if float(v).is_integer() else str(float(v))
                        for v in x
                    ]
                ),
            ),
        )
    )

    df = df.copy()


    for c in ["genes", "gene_indices", "gene_priorities"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df.merge(gene_list_df, on=["project", "group"], how="left")

    return df


def attach_per_gene_priority(
    df: pd.DataFrame,
    membership_order: pd.DataFrame,
) -> pd.DataFrame:


    if not {"project", "group", "gene_index"}.issubset(df.columns):
        return df

    if membership_order.empty:
        return df

    key = membership_order[
        ["project", "group", "gene_index", "gene_priority", "group_order"]
    ].drop_duplicates(subset=["project", "group", "gene_index"])

    df = df.copy()

    for c in ["gene_priority", "group_order"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df.merge(key, on=["project", "group", "gene_index"], how="left")

    if "group_order" not in df.columns:
        df["group_order"] = 99

    if "gene_priority" not in df.columns:
        df["gene_priority"] = np.nan

    return df


def sort_result_df(
    df: pd.DataFrame,
    fname: str,
    group_order_map: Dict[str, int],
    membership_order: pd.DataFrame,
    performance_order: bool = False,
) -> pd.DataFrame:


    df = df.copy()

    if "project" in df.columns:
        df["project"] = df["project"].astype(str)

    if "group" in df.columns:
        df = add_group_order(df, group_order_map)


    if fname.startswith("group_pcc_long"):
        df = attach_gene_order_info(df, membership_order)

        if performance_order and "pcc_gene_mean" in df.columns:
            sort_cols = []
            asc = []

            if "project" in df.columns:
                sort_cols.append("project")
                asc.append(True)
            if "method" in df.columns:
                sort_cols.append("method")
                asc.append(True)

            sort_cols += ["pcc_gene_mean", "n_valid_genes", "group_order", "group"]
            asc += [False, False, True, True]

            df = df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)

        else:
            sort_cols = []
            asc = []

            if "project" in df.columns:
                sort_cols.append("project")
                asc.append(True)
            if "method" in df.columns:
                sort_cols.append("method")
                asc.append(True)

            sort_cols += ["group_order", "group"]
            asc += [True, True]

            df = df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)

        return df


    if fname.startswith("group_pcc_wide"):
        sort_cols = []
        asc = []

        if "project" in df.columns:
            sort_cols.append("project")
            asc.append(True)

        if "group_order" in df.columns:
            sort_cols.append("group_order")
            asc.append(True)

        if "group" in df.columns:
            sort_cols.append("group")
            asc.append(True)

        if sort_cols:
            df = df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)

        return df


    if fname.startswith("per_gene_pcc_long"):
        df = attach_per_gene_priority(df, membership_order)

        df["gene_priority_sort"] = pd.to_numeric(
            df.get("gene_priority", np.nan), errors="coerce"
        ).fillna(999)

        sort_cols = []
        asc = []

        if "project" in df.columns:
            sort_cols.append("project")
            asc.append(True)
        if "method" in df.columns:
            sort_cols.append("method")
            asc.append(True)
        if "group_order" in df.columns:
            sort_cols.append("group_order")
            asc.append(True)
        if "group" in df.columns:
            sort_cols.append("group")
            asc.append(True)

        sort_cols += ["gene_priority_sort", "gene_index"]
        asc += [True, True]

        df = df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
        df = df.drop(columns=["gene_priority_sort"])

        return df


    sort_cols = []
    asc = []

    if "project" in df.columns:
        sort_cols.append("project")
        asc.append(True)
    if "method" in df.columns:
        sort_cols.append("method")
        asc.append(True)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)

    return df


def postprocess_project_outputs(
    project_out: Path,
    project: str,
    group_order_map: Dict[str, int],
    membership_order: pd.DataFrame,
) -> None:


    for fname in RESULT_FILES:
        p = project_out / fname

        if not p.exists():
            continue

        df = pd.read_csv(p)

        if "project" not in df.columns:
            df.insert(0, "project", project)

        df_sorted = sort_result_df(
            df=df,
            fname=fname,
            group_order_map=group_order_map,
            membership_order=membership_order,
            performance_order=False,
        )

        df_sorted.to_csv(p, index=False)

        if fname == "group_pcc_long.csv":
            perf = sort_result_df(
                df=df,
                fname=fname,
                group_order_map=group_order_map,
                membership_order=membership_order,
                performance_order=True,
            )
            perf_path = project_out / "group_pcc_long_performance_order.csv"
            perf.to_csv(perf_path, index=False)


def read_with_project(csv_path: Path, project: str) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    if "project" not in df.columns:
        df.insert(0, "project", project)
    else:
        cols = ["project"] + [c for c in df.columns if c != "project"]
        df = df[cols]

    return df


def merge_outputs(
    out_root: Path,
    projects: list[str],
    group_order_map: Dict[str, int],
    membership_order: pd.DataFrame,
) -> None:


    merged_dir = out_root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    for fname in RESULT_FILES:
        dfs = []

        for project in projects:
            p = out_root / project / fname
            df = read_with_project(p, project)

            if df is not None:
                dfs.append(df)

        if not dfs:
            print(f"[WARN] No files found for {fname}; skip.")
            continue

        merged = pd.concat(dfs, ignore_index=True)

        merged = sort_result_df(
            df=merged,
            fname=fname,
            group_order_map=group_order_map,
            membership_order=membership_order,
            performance_order=False,
        )

        merged_path = merged_dir / fname.replace(".csv", "_all_projects.csv")
        merged.to_csv(merged_path, index=False)
        print(f"[MERGED] {merged_path}")

        if fname == "group_pcc_long.csv":
            perf = sort_result_df(
                df=merged,
                fname=fname,
                group_order_map=group_order_map,
                membership_order=membership_order,
                performance_order=True,
            )
            perf_path = merged_dir / "group_pcc_long_all_projects_performance_order.csv"
            perf.to_csv(perf_path, index=False)
            print(f"[MERGED] {perf_path}")

    group_long_path = merged_dir / "group_pcc_long_all_projects.csv"

    if group_long_path.exists():
        group_long = pd.read_csv(group_long_path)

        if {"project", "method", "group", "pcc_gene_mean", "pcc_flat"}.issubset(group_long.columns):
            group_long = add_group_order(group_long, group_order_map)

            summary = (
                group_long
                .groupby(["method", "group_order", "group"], as_index=False)
                .agg(
                    n_projects=("project", "nunique"),
                    mean_pcc_gene_mean=("pcc_gene_mean", "mean"),
                    std_pcc_gene_mean=("pcc_gene_mean", "std"),
                    mean_pcc_flat=("pcc_flat", "mean"),
                    std_pcc_flat=("pcc_flat", "std"),
                )
                .sort_values(["method", "group_order", "group"], ascending=[True, True, True])
                .reset_index(drop=True)
            )

            summary_path = merged_dir / "group_pcc_summary_across_projects.csv"
            summary.to_csv(summary_path, index=False)
            print(f"[SUMMARY] {summary_path}")

            summary_perf = summary.sort_values(
                ["method", "mean_pcc_gene_mean", "group_order", "group"],
                ascending=[True, False, True, True],
            ).reset_index(drop=True)

            summary_perf_path = merged_dir / "group_pcc_summary_across_projects_performance_order.csv"
            summary_perf.to_csv(summary_perf_path, index=False)
            print(f"[SUMMARY] {summary_perf_path}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--compute_script", default="./scripts/analysis/compute_group_pcc.py")
    ap.add_argument("--membership_csv", default="/data1/SpaRED/gene_group/spared_project_group_membership_strict.csv")
    ap.add_argument("--cache_dir", default="/data1/SpaRED_cache")

    ap.add_argument("--nh2st_root", default="/data4/SpaRED_pred/NH2ST")
    ap.add_argument("--coast_root", default="/data4/SpaRED_pred/COAST")
    ap.add_argument("--coast_suffix", default="COAST_N18F36S1D1")

    ap.add_argument("--peka_root", default="/home/compu/keunho/lib/PEKA/dump")
    ap.add_argument("--peka_run_name", default="K100_img224_lora256a32_kd0.5_tau2.0")
    ap.add_argument("--peka_yhat_name", default="Yhat_test.npy")
    ap.add_argument("--peka_gene_name", default="gene_test.npy")
    ap.add_argument("--peka_converted_root", default="/data4/SpaRED_pred/PEKA_slide_pt")

    ap.add_argument("--split", default="test")
    ap.add_argument("--out_root", default="./group_pcc_out_7projects_with_peka")

    ap.add_argument("--min_genes", type=int, default=1)


    ap.add_argument("--min_priority", type=int, default=None)


    ap.add_argument("--max_priority", type=int, default=None)


    ap.add_argument(
        "--group_order_csv",
        default=None,
        help="Optional CSV with columns: group,group_order",
    )

    ap.add_argument("--skip_peka_conversion", action="store_true")
    ap.add_argument("--strict_peka_ytrue", action="store_true")
    ap.add_argument("--validate_tol", type=float, default=1e-4)

    ap.add_argument("--allow_missing_slides", action="store_true")
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    compute_script = Path(strip_sftp_url(args.compute_script))
    membership_csv = Path(strip_sftp_url(args.membership_csv))
    cache_dir = Path(strip_sftp_url(args.cache_dir))

    nh2st_root = Path(strip_sftp_url(args.nh2st_root))
    coast_root = Path(strip_sftp_url(args.coast_root))

    peka_root = Path(strip_sftp_url(args.peka_root))
    peka_converted_root = Path(strip_sftp_url(args.peka_converted_root))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not compute_script.exists():
        raise FileNotFoundError(f"compute_script not found: {compute_script}")

    if not membership_csv.exists():
        raise FileNotFoundError(f"membership_csv not found: {membership_csv}")

    group_order_map = load_group_order_map(args.group_order_csv)

    compute_supports_max_priority = script_supports_arg(compute_script, "--max_priority")
    compute_supports_group_order_csv = script_supports_arg(compute_script, "--group_order_csv")


    min_priority_applied = args.min_priority
    max_priority_applied = args.max_priority if compute_supports_max_priority else None

    if args.max_priority is not None and not compute_supports_max_priority:
        print(
            "[WARN] compute_script does not appear to support --max_priority. "
            "Therefore --max_priority will NOT be forwarded and will NOT affect metric computation. "
            "Use --min_priority if your current compute script only supports that."
        )

    if args.group_order_csv is not None and not compute_supports_group_order_csv:
        print(
            "[INFO] compute_script does not support --group_order_csv. "
            "This is okay: wrapper will apply group ordering during CSV post-processing."
        )

    membership_order = load_membership_order_table(
        membership_csv=membership_csv,
        projects=PROJECTS,
        group_order_map=group_order_map,
        min_priority_applied=min_priority_applied,
        max_priority_applied=max_priority_applied,
    )

    succeeded = []
    failed = []
    conversion_rows = []

    for project in PROJECTS:
        print("\n" + "=" * 120)
        print(f"[PROJECT] {project}")

        cache_path = cache_dir / f"{project}.pt"

        nh2st_pred = nh2st_root / project
        coast_pred = coast_root / project / args.coast_suffix

        peka_dump_dir = peka_root / project / args.peka_run_name
        peka_yhat = peka_dump_dir / args.peka_yhat_name
        peka_gene = peka_dump_dir / args.peka_gene_name
        peka_pt_dir = peka_converted_root / project

        project_out = out_root / project
        project_out.mkdir(parents=True, exist_ok=True)

        try:
            if not args.skip_peka_conversion:
                if not cache_path.exists():
                    raise FileNotFoundError(f"cache not found: {cache_path}")

                if not peka_yhat.exists():
                    raise FileNotFoundError(f"PEKA Yhat not found: {peka_yhat}")

                if args.dry_run:
                    print(f"[DRY] Would convert PEKA: {peka_yhat} -> {peka_pt_dir}")
                    conv = {
                        "n_spots": np.nan,
                        "n_genes": np.nan,
                        "n_slides": np.nan,
                        "ytrue_status": "dry_run",
                        "ytrue_max_abs": np.nan,
                        "ytrue_mean_abs": np.nan,
                        "converted_dir": str(peka_pt_dir),
                    }
                else:
                    conv = convert_peka_yhat_to_slide_pt(
                        cache_path=cache_path,
                        split=args.split,
                        yhat_npy=peka_yhat,
                        out_dir=peka_pt_dir,
                        ytrue_npy=peka_gene if peka_gene.exists() else None,
                        validate_tol=args.validate_tol,
                        strict_ytrue=args.strict_peka_ytrue,
                    )

                conversion_rows.append(
                    {
                        "project": project,
                        "cache_path": str(cache_path),
                        "peka_yhat": str(peka_yhat),
                        "peka_gene": str(peka_gene),
                        **conv,
                    }
                )

            else:
                print(f"[SKIP] PEKA conversion skipped. Use existing: {peka_pt_dir}")

            cmd = [
                sys.executable,
                str(compute_script),
                "--cache_path",
                str(cache_path),
                "--membership_csv",
                str(membership_csv),
                "--project",
                project,
                "--pred",
                f"NH2ST={nh2st_pred}",
                "--pred",
                f"COAST={coast_pred}",
                "--pred",
                f"PEKA={peka_pt_dir}",
                "--split",
                args.split,
                "--out_dir",
                str(project_out),
                "--min_genes",
                str(args.min_genes),
            ]

            if args.min_priority is not None:
                cmd += ["--min_priority", str(args.min_priority)]

            if args.max_priority is not None and compute_supports_max_priority:
                cmd += ["--max_priority", str(args.max_priority)]

            if args.group_order_csv is not None and compute_supports_group_order_csv:
                cmd += ["--group_order_csv", str(args.group_order_csv)]

            if args.allow_missing_slides:
                cmd += ["--allow_missing_slides"]

            print("[CMD]")
            print(" ".join(cmd))

            if not args.dry_run:
                subprocess.run(cmd, check=True)

                postprocess_project_outputs(
                    project_out=project_out,
                    project=project,
                    group_order_map=group_order_map,
                    membership_order=membership_order,
                )

            succeeded.append(project)

        except Exception as e:
            failed.append(project)
            print(f"[ERROR] {project}: {repr(e)}")

            if not args.continue_on_error:
                raise

    if not args.dry_run:
        if conversion_rows:
            conv_df = pd.DataFrame(conversion_rows)
            conv_path = out_root / "peka_conversion_report_all_projects.csv"
            conv_df.to_csv(conv_path, index=False)
            print(f"[SAVE] {conv_path}")

        merge_outputs(
            out_root=out_root,
            projects=succeeded,
            group_order_map=group_order_map,
            membership_order=membership_order,
        )

        status_df = pd.DataFrame(
            [{"project": p, "status": "succeeded"} for p in succeeded]
            + [{"project": p, "status": "failed"} for p in failed]
        )
        status_path = out_root / "run_status.csv"
        status_df.to_csv(status_path, index=False)
        print(f"[SAVE] {status_path}")

    print("\nDone.")
    print(f"Succeeded: {len(succeeded)} / {len(PROJECTS)}")

    if failed:
        print("Failed:")
        for p in failed:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
