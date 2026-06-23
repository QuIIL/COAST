from __future__ import annotations

import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


DEFAULT_GROUP_ORDER = {


    "tumor_marker": 1,
    "tumor markers": 1,
    "cancer_marker": 1,
    "cancer markers": 1,
    "oncogene": 1,
    "oncogenic": 1,
    "tumor": 1,

    "lineage_marker": 2,
    "lineage markers": 2,
    "tumor_lineage": 2,
    "tissue_lineage": 2,
    "epithelial_lineage": 2,

    "epithelial": 3,
    "epithelial_marker": 3,
    "epithelial markers": 3,


    "tumor_stroma_boundary": 4,
    "tumor-stroma boundary": 4,
    "tumor stroma boundary": 4,
    "boundary": 4,
    "spatial_boundary": 4,
    "spatial gradient": 4,
    "gradient": 4,

    "stroma_ecm": 5,
    "stroma": 5,
    "ecm": 5,
    "extracellular_matrix": 5,
    "extracellular matrix": 5,
    "matrix": 5,

    "caf_fibrosis": 6,
    "caf": 6,
    "fibrosis": 6,
    "fibroblast": 6,
    "fibroblasts": 6,
    "collagen": 6,


    "immune": 7,
    "immune_marker": 7,
    "immune markers": 7,
    "immune_cell": 7,
    "immune cells": 7,

    "inflammation": 8,
    "inflammatory": 8,
    "cytokine": 8,
    "chemokine": 8,

    "antigen_presentation": 9,
    "antigen presentation": 9,
    "mhc": 9,
    "hla": 9,


    "hypoxia": 10,
    "hypoxic": 10,

    "metabolism": 11,
    "metabolic": 11,
    "glycolysis": 11,
    "lipid_metabolism": 11,
    "lipid metabolism": 11,

    "proliferation": 12,
    "cell_cycle": 12,
    "cell cycle": 12,
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
    g = normalize_group_name(group_name)
    return int(group_order_map.get(g, 99))


def safe_corr(a: np.ndarray, b: np.ndarray, eps_std: float = 1e-6) -> float:

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]

    if a.size < 3:
        return np.nan
    if np.std(a) < eps_std or np.std(b) < eps_std:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def per_gene_pcc(pred: np.ndarray, true: np.ndarray, eps_std: float = 1e-6) -> np.ndarray:


    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, true={true.shape}")

    vals = np.full(pred.shape[1], np.nan, dtype=np.float64)
    for g in range(pred.shape[1]):
        vals[g] = safe_corr(pred[:, g], true[:, g], eps_std=eps_std)

    return vals


def nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def nanmedian(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def nanstd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=0)) if x.size else np.nan


def parse_pred_specs(pred_specs: List[str]) -> Dict[str, str]:


    if not pred_specs:
        raise ValueError("At least one --pred must be provided.")

    out = {}

    for spec in pred_specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
            name = name.strip()
            path = path.strip()
        else:
            path = spec.strip()
            name = os.path.basename(os.path.normpath(path))

        if not name:
            raise ValueError(f"Invalid --pred spec: {spec}")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Prediction directory not found for method '{name}': {path}")
        if name in out:
            raise ValueError(f"Duplicated method name: {name}")

        out[name] = path

    return out


def load_membership(
    membership_csv: str,
    project: str,
    max_priority: int | None = None,
    min_priority: int | None = None,
) -> pd.DataFrame:


    if max_priority is not None and min_priority is not None:
        raise ValueError("Use either --max_priority or --min_priority, not both.")

    df = pd.read_csv(membership_csv)
    required = {"project", "gene_index", "gene", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"membership_csv missing columns: {sorted(missing)}")

    sub = df[df["project"].astype(str) == str(project)].copy()

    if sub.empty:
        available = sorted(df["project"].astype(str).unique().tolist())[:30]
        raise ValueError(
            f"No membership rows for project='{project}'. "
            f"First available projects: {available}"
        )

    sub["gene_index"] = sub["gene_index"].astype(int)
    sub["gene"] = sub["gene"].astype(str)
    sub["group"] = sub["group"].astype(str)

    if "gene_norm" in sub.columns:
        sub["gene_norm"] = sub["gene_norm"].astype(str)

    if "priority" in sub.columns:
        sub["priority"] = pd.to_numeric(sub["priority"], errors="coerce")
    else:
        sub["priority"] = np.nan


    if max_priority is not None:
        sub = sub[sub["priority"].fillna(999) <= float(max_priority)].copy()


    if min_priority is not None:
        sub = sub[sub["priority"].fillna(-999) >= float(min_priority)].copy()


    sub = sub.drop_duplicates(subset=["gene_index", "group"]).reset_index(drop=True)

    if sub.empty:
        raise ValueError(
            f"Membership became empty after priority filtering. "
            f"project={project}, max_priority={max_priority}, min_priority={min_priority}"
        )

    return sub


def get_meta_slide_and_expr(meta):


    if isinstance(meta, dict):
        slide = (
            meta.get("slide")
            or meta.get("slide_id")
            or meta.get("sample")
            or meta.get("sample_id")
        )
        if slide is None:
            raise KeyError(f"Could not find slide key in metadata dict keys={list(meta.keys())}")

        if "gene_exp" in meta:
            y = meta["gene_exp"]
        elif "expr" in meta:
            y = meta["expr"]
        elif "X" in meta:
            y = meta["X"]
        else:
            raise KeyError(f"Could not find expression key in metadata dict keys={list(meta.keys())}")

        return str(slide), torch.as_tensor(y, dtype=torch.float32)

    slide_id = str(meta[3])
    y = torch.as_tensor(meta[4], dtype=torch.float32)
    return slide_id, y


def build_true_by_slide(cache_path: str, split: str = "test") -> Tuple[Dict[str, torch.Tensor], int]:


    cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    if split not in cache:
        raise KeyError(f"Split '{split}' not found in cache. Available keys: {list(cache.keys())}")

    data = cache[split]

    if "metadata" not in data:
        raise KeyError(f"cache[{split}] must contain 'metadata'.")

    buckets = defaultdict(list)

    for idx, meta in enumerate(data["metadata"]):
        slide_id, y = get_meta_slide_and_expr(meta)
        buckets[slide_id].append((idx, y))

    true_by_slide = {}
    num_genes = None

    for slide_id, items in buckets.items():
        items.sort(key=lambda x: x[0])
        y_slide = torch.stack([x[1] for x in items], dim=0)
        true_by_slide[slide_id] = y_slide

        if num_genes is None:
            num_genes = int(y_slide.shape[1])
        elif int(y_slide.shape[1]) != num_genes:
            raise ValueError("Inconsistent gene dimension across slides.")

    if num_genes is None:
        raise RuntimeError(f"No metadata found in cache[{split}].")

    return true_by_slide, num_genes


def load_method_pred_and_true(
    pred_dir: str,
    true_by_slide: Dict[str, torch.Tensor],
    strict: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:


    pred_all = []
    true_all = []
    used_slides = []
    missing_slides = []

    for slide_id in sorted(true_by_slide.keys()):
        pred_path = os.path.join(pred_dir, f"{slide_id}.pt")

        if not os.path.exists(pred_path):
            missing_slides.append(slide_id)
            if strict:
                raise FileNotFoundError(f"Missing prediction file: {pred_path}")
            continue

        pred = torch.load(pred_path, map_location="cpu")

        if isinstance(pred, dict):

            for key in ["pred", "preds", "y_hat", "y_hat_spot", "output", "outputs"]:
                if key in pred:
                    pred = pred[key]
                    break
            else:
                raise ValueError(
                    f"{pred_path} is a dict but no known prediction key was found. "
                    f"Available keys: {list(pred.keys())}"
                )

        pred = torch.as_tensor(pred, dtype=torch.float32)
        true = true_by_slide[slide_id]

        if pred.ndim != 2:
            raise ValueError(f"{pred_path}: expected [N_slide, G], got {tuple(pred.shape)}")

        if pred.shape != true.shape:
            raise ValueError(
                f"{slide_id}: pred/true shape mismatch: "
                f"pred={tuple(pred.shape)}, true={tuple(true.shape)}. "
                "Check cache, split, project, and prediction directory."
            )

        pred_all.append(pred)
        true_all.append(true)
        used_slides.append(slide_id)

    if not pred_all:
        raise RuntimeError(f"No predictions loaded from {pred_dir}")

    pred_cat = torch.cat(pred_all, dim=0).numpy()
    true_cat = torch.cat(true_all, dim=0).numpy()

    return pred_cat, true_cat, used_slides, missing_slides


def make_gene_table_from_membership(membership: pd.DataFrame, num_genes: int) -> pd.DataFrame:


    gene_df = (
        membership[["gene_index", "gene"]]
        .drop_duplicates(subset=["gene_index"])
        .sort_values("gene_index")
        .reset_index(drop=True)
    )

    max_gene_index = int(gene_df["gene_index"].max())
    if max_gene_index >= num_genes:
        raise ValueError(
            f"membership has gene_index={max_gene_index} but prediction has G={num_genes}. "
            "Check whether the membership file matches the cache/project."
        )

    return gene_df


def sort_group_genes(gdf: pd.DataFrame, G: int) -> Tuple[List[int], List[str], List[float]]:


    gdf2 = gdf.copy()
    gdf2["gene_index"] = gdf2["gene_index"].astype(int)

    if "priority" in gdf2.columns:
        gdf2["priority_sort"] = pd.to_numeric(gdf2["priority"], errors="coerce").fillna(999)
    else:
        gdf2["priority_sort"] = 999

    gdf2 = (
        gdf2[gdf2["gene_index"].between(0, G - 1)]
        .drop_duplicates(subset=["gene_index"])
        .sort_values(["priority_sort", "gene_index"], ascending=[True, True])
        .reset_index(drop=True)
    )

    idxs = gdf2["gene_index"].astype(int).tolist()
    genes = gdf2["gene"].astype(str).tolist()
    priorities = gdf2["priority_sort"].astype(float).tolist()

    return idxs, genes, priorities


def summarize_groups(
    method: str,
    pred: np.ndarray,
    true: np.ndarray,
    membership: pd.DataFrame,
    group_order_map: Dict[str, int],
    eps_std: float = 1e-6,
    min_genes: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:


    G = pred.shape[1]
    pcc_g = per_gene_pcc(pred, true, eps_std=eps_std)

    overall = {
        "method": method,
        "n_spots": int(pred.shape[0]),
        "n_genes": int(G),
        "overall_pcc_gene_mean": nanmean(pcc_g),
        "overall_pcc_flat": safe_corr(pred.reshape(-1), true.reshape(-1), eps_std=eps_std),
    }


    per_gene_rows = []

    for row in membership.itertuples(index=False):
        gi = int(row.gene_index)

        if gi < 0 or gi >= G:
            continue

        group_name = str(row.group)
        priority_value = getattr(row, "priority", np.nan)
        group_order = get_group_order(group_name, group_order_map)

        per_gene_rows.append(
            {
                "method": method,
                "project": row.project,
                "group_order": group_order,
                "group": group_name,
                "gene_priority": priority_value,
                "gene_index": gi,
                "gene": row.gene,
                "pcc": pcc_g[gi],
            }
        )

    per_gene_df = pd.DataFrame(per_gene_rows)

    if not per_gene_df.empty:
        per_gene_df["gene_priority_sort"] = pd.to_numeric(
            per_gene_df["gene_priority"], errors="coerce"
        ).fillna(999)

        per_gene_df = per_gene_df.sort_values(
            ["method", "group_order", "group", "gene_priority_sort", "gene_index"],
            ascending=[True, True, True, True, True],
        ).drop(columns=["gene_priority_sort"]).reset_index(drop=True)


    group_rows = []

    for group_name, gdf in membership.groupby("group", sort=False):
        group_order = get_group_order(group_name, group_order_map)

        idxs, genes, priorities = sort_group_genes(gdf, G=G)

        if len(idxs) < min_genes:
            group_rows.append(
                {
                    "method": method,
                    "group_order": group_order,
                    "group": group_name,
                    "n_genes": len(idxs),
                    "n_valid_genes": 0,
                    "pcc_gene_mean": np.nan,
                    "pcc_gene_median": np.nan,
                    "pcc_gene_std": np.nan,
                    "pcc_flat": np.nan,
                    "genes": ";".join(genes),
                    "gene_priorities": ";".join([str(x) for x in priorities]),
                }
            )
            continue

        group_pcc = pcc_g[idxs]
        valid = np.isfinite(group_pcc)

        group_rows.append(
            {
                "method": method,
                "group_order": group_order,
                "group": group_name,
                "n_genes": len(idxs),
                "n_valid_genes": int(valid.sum()),
                "pcc_gene_mean": nanmean(group_pcc),
                "pcc_gene_median": nanmedian(group_pcc),
                "pcc_gene_std": nanstd(group_pcc),
                "pcc_flat": safe_corr(
                    pred[:, idxs].reshape(-1),
                    true[:, idxs].reshape(-1),
                    eps_std=eps_std,
                ),
                "genes": ";".join(genes),
                "gene_priorities": ";".join([str(x) for x in priorities]),
            }
        )

    group_df = pd.DataFrame(group_rows)


    group_df = group_df.sort_values(
        ["method", "group_order", "group"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return group_df, per_gene_df, overall


def make_bio_ordered_wide(
    group_long: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:


    if value_col not in group_long.columns:
        raise KeyError(f"value_col not found: {value_col}")

    order_df = (
        group_long[["group", "group_order"]]
        .drop_duplicates(subset=["group"])
        .sort_values(["group_order", "group"], ascending=[True, True])
        .reset_index(drop=True)
    )

    wide = group_long.pivot_table(
        index="group",
        columns="method",
        values=value_col,
        aggfunc="first",
    ).reset_index()

    wide = order_df.merge(wide, on="group", how="left")
    wide = wide.sort_values(["group_order", "group"], ascending=[True, True]).reset_index(drop=True)

    return wide


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_path", required=True, help="Project cache .pt path")
    parser.add_argument(
        "--membership_csv",
        required=True,
        help="spared_project_group_membership_strict.csv or overlap version",
    )
    parser.add_argument("--project", required=True, help="Project name in membership CSV")

    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="NAME=/pred/dir or just /pred/dir. Can be repeated.",
    )

    parser.add_argument("--split", default="test", help="Cache split to evaluate, default: test")
    parser.add_argument("--out_dir", default="group_pcc_out")
    parser.add_argument("--min_genes", type=int, default=1)
    parser.add_argument("--eps_std", type=float, default=1e-6)

    parser.add_argument(
        "--allow_missing_slides",
        action="store_true",
        help="Skip missing slide predictions instead of raising error",
    )

    parser.add_argument(
        "--max_priority",
        type=int,
        default=None,
        help=(
            "Keep genes with priority <= max_priority. "
            "Recommended convention: priority 1 is highest."
        ),
    )
    parser.add_argument(
        "--min_priority",
        type=int,
        default=None,
        help=(
            "Backward-compatible filter: keep genes with priority >= min_priority. "
            "Do not use together with --max_priority."
        ),
    )

    parser.add_argument(
        "--group_order_csv",
        default=None,
        help=(
            "Optional CSV with columns: group,group_order. "
            "Overrides default biological group ordering."
        ),
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    group_order_map = load_group_order_map(args.group_order_csv)

    pred_specs = parse_pred_specs(args.pred)

    membership = load_membership(
        membership_csv=args.membership_csv,
        project=args.project,
        max_priority=args.max_priority,
        min_priority=args.min_priority,
    )

    true_by_slide, num_genes = build_true_by_slide(args.cache_path, split=args.split)
    make_gene_table_from_membership(membership, num_genes=num_genes)

    all_group = []
    all_gene = []
    all_overall = []
    load_rows = []

    for method, pred_dir in pred_specs.items():
        pred, true, used_slides, missing_slides = load_method_pred_and_true(
            pred_dir=pred_dir,
            true_by_slide=true_by_slide,
            strict=not args.allow_missing_slides,
        )

        group_df, per_gene_df, overall = summarize_groups(
            method=method,
            pred=pred,
            true=true,
            membership=membership,
            group_order_map=group_order_map,
            eps_std=args.eps_std,
            min_genes=args.min_genes,
        )

        all_group.append(group_df)
        all_gene.append(per_gene_df)
        all_overall.append(overall)

        load_rows.append(
            {
                "method": method,
                "pred_dir": pred_dir,
                "n_used_slides": len(used_slides),
                "n_missing_slides": len(missing_slides),
                "missing_slides": ";".join(missing_slides),
                "used_slides": ";".join(used_slides),
                "n_spots": pred.shape[0],
                "n_genes": pred.shape[1],
            }
        )

    group_long = pd.concat(all_group, ignore_index=True)
    per_gene_long = pd.concat(all_gene, ignore_index=True)
    overall_df = pd.DataFrame(all_overall)
    load_df = pd.DataFrame(load_rows)


    group_long_bio_order = group_long.sort_values(
        ["method", "group_order", "group"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


    group_long_perf_order = group_long.sort_values(
        ["method", "pcc_gene_mean", "n_valid_genes", "group_order", "group"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)


    group_wide_mean = make_bio_ordered_wide(group_long_bio_order, value_col="pcc_gene_mean")
    group_wide_flat = make_bio_ordered_wide(group_long_bio_order, value_col="pcc_flat")


    group_long_path = os.path.join(args.out_dir, "group_pcc_long.csv")
    group_long_bio_path = os.path.join(args.out_dir, "group_pcc_long_bio_order.csv")
    group_long_perf_path = os.path.join(args.out_dir, "group_pcc_long_performance_order.csv")

    group_wide_mean_path = os.path.join(args.out_dir, "group_pcc_wide_gene_mean.csv")
    group_wide_flat_path = os.path.join(args.out_dir, "group_pcc_wide_flat.csv")

    per_gene_path = os.path.join(args.out_dir, "per_gene_pcc_long.csv")
    overall_path = os.path.join(args.out_dir, "overall_pcc.csv")
    load_path = os.path.join(args.out_dir, "load_report.csv")


    group_long_bio_order.to_csv(group_long_path, index=False)
    group_long_bio_order.to_csv(group_long_bio_path, index=False)
    group_long_perf_order.to_csv(group_long_perf_path, index=False)

    group_wide_mean.to_csv(group_wide_mean_path, index=False)
    group_wide_flat.to_csv(group_wide_flat_path, index=False)

    per_gene_long.to_csv(per_gene_path, index=False)
    overall_df.to_csv(overall_path, index=False)
    load_df.to_csv(load_path, index=False)


    print("\n===== Input =====")
    print(f"project          : {args.project}")
    print(f"cache_path       : {args.cache_path}")
    print(f"membership_csv   : {args.membership_csv}")
    print(f"group_order_csv  : {args.group_order_csv}")
    print(f"split            : {args.split}")
    print(f"out_dir          : {args.out_dir}")
    print(f"groups           : {membership['group'].nunique()}")
    print(f"membership rows  : {len(membership)}")
    print(f"test slides      : {len(true_by_slide)}")
    print(f"num genes        : {num_genes}")
    print(f"max_priority     : {args.max_priority}")
    print(f"min_priority     : {args.min_priority}")

    print("\n===== Overall PCC =====")
    print(overall_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n===== Group PCC_F mean, biological order =====")
    display_cols = [
        "method",
        "group_order",
        "group",
        "n_genes",
        "n_valid_genes",
        "pcc_gene_mean",
        "pcc_flat",
        "genes",
    ]

    print(
        group_long_bio_order[display_cols]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n===== Group PCC_F mean, performance order =====")
    display_cols_perf = [
        "method",
        "group_order",
        "group",
        "n_genes",
        "n_valid_genes",
        "pcc_gene_mean",
        "pcc_flat",
    ]

    print(
        group_long_perf_order[display_cols_perf]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nSaved:")
    for p in [
        group_long_path,
        group_long_bio_path,
        group_long_perf_path,
        group_wide_mean_path,
        group_wide_flat_path,
        per_gene_path,
        overall_path,
        load_path,
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
