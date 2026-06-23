from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import pandas as pd


def norm_gene(g: str) -> str:
    return str(g).strip().upper()


def read_spared_gene_list(path: str) -> Dict[str, List[str]]:
    projects: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [x.strip() for x in row]
            if not row or not row[0]:
                continue
            project = row[0]
            genes = [g for g in row[1:] if g]

            seen = set()
            uniq = []
            for g in genes:
                ng = norm_gene(g)
                if ng not in seen:
                    uniq.append(g.strip())
                    seen.add(ng)
            projects[project] = uniq
    return projects


GROUPS: Dict[str, Dict[str, object]] = {
    "Housekeeping_basal": {
        "priority": 90,
        "description": "Basal/housekeeping-like genes frequently used as broadly expressed controls or core cellular genes.",
        "genes": [
            "ACTB", "GAPDH", "B2M", "UBC", "MALAT1", "TMSB4X", "TMSB10",
            "EEF1A1", "EEF1G", "EEF2", "HSP90AA1", "HSP90AB1", "HSP90B1",
            "HSPA5", "HSPA8", "CALR", "CANX", "PPIA", "RPLP1", "RPS27",
            "RPS27A", "RPL17", "RPL4", "RPL5", "RPL8", "RPL13", "RPL13A",
            "RPL22", "RPL24", "RPL27A", "RPL28", "RPL36", "RPL37A", "RPL38",
            "RPS8", "RPS12", "RPS14", "RPS15", "RPS15A", "RPS17", "RPS18",
            "RPS19", "RPS20", "FTL", "FTH1",
        ],
    },
    "Immune_TME": {
        "priority": 10,
        "description": "Immune and tumor-microenvironment related genes including antigen presentation, immunoglobulin, myeloid/complement, and chemokine markers.",
        "genes": [
            "CD74", "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "HLA-DRB5",
            "HLA-DPA1", "HLA-DPB1", "HLA-E", "B2M", "TAPBP", "TNFSF10",
            "IGKC", "IGHG1", "IGHG3", "IGHG4", "IGHM", "IGHA1", "IGLC1", "IGLC2",
            "IGLC3", "JCHAIN", "LYZ", "LSP1", "C1QA", "C1QB", "C1R", "C1S", "C3",
            "CFB", "APOE", "CXCL9", "CXCL12", "CCL19", "IFITM2", "IFITM3", "ISG15",
            "IFI6", "IFI27", "TYMP", "FCGRT", "CX3CL1",
        ],
    },
    "Stromal_ECM_TumorStroma": {
        "priority": 20,
        "description": "Stromal, extracellular matrix, and tumor-stroma boundary related genes.",
        "genes": [
            "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "COL6A1", "COL6A2",
            "COL6A3", "COL9A1", "COL9A3", "COL11A1", "COL11A2", "COL16A1", "COL18A1",
            "COL22A1", "DCN", "BGN", "FN1", "FBLN1", "EMILIN1", "ELN", "LAMB1", "LAMB2",
            "LAMB3", "LAMC1", "HSPG2", "SPARC", "VIM", "AEBP1", "IGFBP4", "IGFBP5",
            "IGFBP7", "MGP", "THBS1", "THBS2", "THBS3", "THBS4", "TIMP1", "TIMP2",
            "MMP9", "MMP14", "LOX", "VCAN", "TNXB", "CTGF", "CCN2", "SOD3", "LRP1",
        ],
    },
    "SmoothMuscle_Perivascular": {
        "priority": 25,
        "description": "Smooth muscle, myofibroblast, and perivascular/stromal contractile markers.",
        "genes": [
            "ACTA2", "ACTG2", "TAGLN", "MYL9", "MYL6", "MYH11", "MYLK", "CNN1", "CNN3",
            "CALD1", "DES", "TPM1", "TPM2", "SMTN", "SYNPO2", "CSRP1", "CSRP3",
            "ADIRF", "RGS5", "CAVIN1", "FLNA",
        ],
    },
    "Epithelial_Tumor": {
        "priority": 30,
        "description": "Epithelial and carcinoma-associated markers across breast/prostate/squamous/intestinal datasets.",
        "genes": [
            "KRT1", "KRT5", "KRT6A", "KRT6B", "KRT6C", "KRT8", "KRT10", "KRT14", "KRT16",
            "KRT17", "KRT18", "KRT19", "KRT37", "KRTDAP", "MUC1", "MUC2", "MUC3", "MUC5B",
            "EPCAM", "TACSTD2", "AGR2", "TFF1", "TFF3", "SCGB2A2", "SCGB1D2", "SNCG",
            "WFDC2", "S100A2", "S100A6", "S100A7", "S100A8", "S100A9", "S100A10", "S100A11",
            "S100A14", "S100A16", "CALML5", "KLK4", "KLK6", "AZGP1", "NKX3-1", "STEAP1",
            "STEAP2", "TMPRSS2", "AMACR", "MSMB", "HOXB13", "SPDEF", "FOXA1", "SLC45A3",
            "CRISP3", "CPB1", "MCCD1", "MUC5B", "PHGR1", "ZG16", "FABP1", "FABP2", "FABP6",
            "PIGR", "CLCA1", "DEFA5", "DEFA6", "REG3B", "REG4", "GUCA2A", "CAR1", "CAR2",
        ],
    },
    "Hypoxia_Metabolism": {
        "priority": 40,
        "description": "Hypoxia, glycolysis, and metabolism-associated genes.",
        "genes": [
            "VEGFA", "SLC2A1", "SLC2A3", "LDHA", "LDHB", "HK2", "ENO1", "ENO2", "ENO3",
            "HIGD1A", "BNIP3", "DDIT4", "PGK1", "TPI1", "GPI", "ALDOA", "ALDOC", "HMGCS1",
            "HMGCS2", "UQCRQ", "UQCRFS1", "COX4I1", "COX6A2", "COX8B", "NDUFA4", "ATP5F1D",
            "ATP5F1E", "ATP5IF1", "ATP5MPL", "SLC16A1", "NAMPT", "FASN", "SCD", "SCD1", "SCD2",
        ],
    },
    "Proliferation_CellCycle": {
        "priority": 15,
        "description": "Proliferation and cell-cycle related genes.",
        "genes": [
            "MKI67", "TOP2A", "CDK1", "CDK2AP1", "CCNB1", "CCND1", "CCND2", "PCNA", "ZWINT",
            "KIF15", "OIP5", "OIP5OS1", "STMN1", "STMN2", "STMN3", "STMN4", "PTTG1IP",
            "NCAPD3", "HIST1H1A", "HIST1H1B", "HIST1H2AG", "HIST1H2BD", "HIST1H2BH",
            "HIST1H2BG", "HIST1H3D", "HIST1H4C", "HIST1H4E", "HIST1H4H", "H2AFJ", "H2AFV", "H2AFZ",
        ],
    },
    "Neuronal_Synaptic": {
        "priority": 50,
        "description": "Neuronal and synaptic markers common in brain datasets.",
        "genes": [
            "SNAP25", "SYT1", "SYT7", "SYT11", "SYN1", "SYN2", "SYP", "SYNPR", "SYNGR1",
            "SLC17A6", "SLC17A7", "SLC32A1", "GAD1", "GAD2", "NRGN", "CAMK2A", "CAMK2B",
            "CAMK2N1", "CAMK2N2", "MAP1A", "MAP1B", "MAP2", "GAP43", "UCHL1", "NEFL",
            "NEFM", "NEFH", "NPTX1", "NPTXR", "HPCA", "HPCAL1", "HPCAL4", "CCK", "PENK",
            "PVALB", "CALB1", "CALB2", "TH", "PCP4", "PCP4L1", "OLFM1", "GRIA1", "GRIA2",
            "GRIA4", "GRIN2B", "GRIN2C", "GABRA1", "GABRA4", "GABRD", "VAMP1", "VAMP2",
            "CPLX1", "CPLX2", "STX1A", "STXBP1", "VSNL1", "NCALD", "DCLK1", "BASP1",
            "RTN1", "RTN3", "RTN4", "TUBB2A", "TUBB2B", "TUBB3", "TUBA1A", "TUBA1B",
        ],
    },
    "Oligodendrocyte_Myelin": {
        "priority": 55,
        "description": "Oligodendrocyte and myelin-associated markers.",
        "genes": [
            "MBP", "MOBP", "PLP1", "CNP", "MAG", "MOG", "MAL", "CLDN11", "OLIG1", "BCAS1",
            "ERMN", "QKI", "TSPAN2", "GPR37", "UGT8A", "FA2H", "CAR2", "SIRT2", "ABCA2",
            "ENPP2", "TF", "TRF",
        ],
    },
    "Astrocyte_Glial": {
        "priority": 60,
        "description": "Astrocyte/glial and neuro-supportive markers.",
        "genes": [
            "GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDOC", "GLUL", "APOE", "APOD", "CLU",
            "S100B", "NDRG1", "NDRG2", "AGT", "PTGDS", "SPARCL1", "FABP7", "VIM", "TTYH1",
            "ATP1A2", "ATP1B1", "ATP1A3", "GPM6B", "DBI", "GATM", "SELENOP",
        ],
    },
    "Development_Stem_Organoid": {
        "priority": 70,
        "description": "Developmental, stem/progenitor, and organoid-associated markers.",
        "genes": [
            "SOX2", "SOX2-OT", "SOX4", "SOX11", "NES", "DCX", "EOMES", "HES5", "NOTCH2",
            "NOTCH3", "WT1", "PODXL", "MAFB", "FOXC2", "PAX8", "GPC2", "GPC3", "MDK",
            "PTN", "IGF2", "DLK1", "CRABP1", "CRABP2", "NR2F1", "NR2F2", "ID3", "ID4",
            "HMGA1", "HMGA2", "CD24", "CD24A", "PROM1",
        ],
    },
}


PRIORITY = {k: int(v["priority"]) for k, v in GROUPS.items()}


def make_definition_df(projects: Dict[str, List[str]]) -> pd.DataFrame:
    union_norm_to_original: Dict[str, str] = {}
    for genes in projects.values():
        for g in genes:
            ng = norm_gene(g)
            union_norm_to_original.setdefault(ng, g)

    rows = []
    for group, spec in GROUPS.items():
        for gene in spec["genes"]:
            ng = norm_gene(gene)
            if ng in union_norm_to_original:
                rows.append({
                    "group": group,
                    "priority": PRIORITY[group],
                    "gene": union_norm_to_original[ng],
                    "gene_norm": ng,
                    "description": spec["description"],
                })
    return pd.DataFrame(rows).sort_values(["priority", "group", "gene_norm"])


def make_membership(projects: Dict[str, List[str]], definition_df: pd.DataFrame) -> pd.DataFrame:
    gene_to_groups: Dict[str, List[str]] = defaultdict(list)
    for _, r in definition_df.iterrows():
        gene_to_groups[r["gene_norm"]].append(r["group"])

    rows = []
    for project, genes in projects.items():
        for idx, g in enumerate(genes):
            ng = norm_gene(g)
            for group in gene_to_groups.get(ng, []):
                rows.append({
                    "project": project,
                    "gene_index": idx,
                    "gene": g,
                    "gene_norm": ng,
                    "group": group,
                    "priority": PRIORITY[group],
                })
    return pd.DataFrame(rows).sort_values(["project", "gene_index", "priority", "group"])


def make_strict_membership(membership_df: pd.DataFrame) -> pd.DataFrame:
    if membership_df.empty:
        return membership_df.copy()

    strict = (
        membership_df.sort_values(["project", "gene_index", "priority", "group"])
        .drop_duplicates(["project", "gene_index"], keep="first")
        .reset_index(drop=True)
    )
    return strict


def make_overlap_report(definition_df: pd.DataFrame) -> pd.DataFrame:
    if definition_df.empty:
        return pd.DataFrame(columns=["gene", "gene_norm", "n_groups", "groups"])
    rows = []
    for gene_norm, sub in definition_df.groupby("gene_norm"):
        groups = sorted(sub["group"].unique().tolist(), key=lambda g: PRIORITY[g])
        if len(groups) > 1:
            rows.append({
                "gene": sub.iloc[0]["gene"],
                "gene_norm": gene_norm,
                "n_groups": len(groups),
                "groups": ";".join(groups),
            })
    return pd.DataFrame(rows).sort_values(["n_groups", "gene_norm"], ascending=[False, True])


def make_coverage(projects: Dict[str, List[str]], membership_df: pd.DataFrame) -> pd.DataFrame:
    all_groups = list(GROUPS.keys())
    rows = []
    for project, genes in projects.items():
        total = len(genes)
        sub = membership_df[membership_df["project"] == project]
        for group in all_groups:
            sg = sub[sub["group"] == group].sort_values("gene_index")
            matched = sg["gene"].drop_duplicates().tolist()
            rows.append({
                "project": project,
                "group": group,
                "n_total_genes": total,
                "n_group_genes": len(matched),
                "coverage_frac": len(matched) / total if total else 0.0,
                "genes": ";".join(matched),
            })
    return pd.DataFrame(rows).sort_values(["project", "group"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene_list_csv", required=True)
    ap.add_argument("--out_dir", default="spared_gene_groups")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    projects = read_spared_gene_list(args.gene_list_csv)

    definition_df = make_definition_df(projects)
    membership_df = make_membership(projects, definition_df)
    strict_df = make_strict_membership(membership_df)
    overlap_df = make_overlap_report(definition_df)
    coverage_df = make_coverage(projects, membership_df)
    coverage_strict_df = make_coverage(projects, strict_df)

    paths = {
        "definitions": os.path.join(args.out_dir, "spared_gene_group_definitions.csv"),
        "overlap": os.path.join(args.out_dir, "spared_gene_group_overlap_report.csv"),
        "membership": os.path.join(args.out_dir, "spared_project_group_membership.csv"),
        "membership_strict": os.path.join(args.out_dir, "spared_project_group_membership_strict.csv"),
        "coverage": os.path.join(args.out_dir, "spared_project_group_coverage.csv"),
        "coverage_strict": os.path.join(args.out_dir, "spared_project_group_coverage_strict.csv"),
    }

    definition_df.to_csv(paths["definitions"], index=False)
    overlap_df.to_csv(paths["overlap"], index=False)
    membership_df.to_csv(paths["membership"], index=False)
    strict_df.to_csv(paths["membership_strict"], index=False)
    coverage_df.to_csv(paths["coverage"], index=False)
    coverage_strict_df.to_csv(paths["coverage_strict"], index=False)

    print("===== SpaRED gene grouping complete =====")
    print(f"projects                 : {len(projects)}")
    print(f"union genes in definitions: {definition_df['gene_norm'].nunique() if not definition_df.empty else 0}")
    print(f"membership rows          : {len(membership_df)}")
    print(f"strict membership rows   : {len(strict_df)}")
    print(f"overlap genes            : {len(overlap_df)}")
    print("\nSaved:")
    for k, p in paths.items():
        print(f"  {k:18s}: {p}")

    print("\n===== Group coverage summary, all projects aggregated =====")

    cov = coverage_df.copy()
    summary = []
    for group, sub in cov.groupby("group"):
        covered = sub[sub["n_group_genes"] > 0]
        summary.append({
            "group": group,
            "n_projects_covered": len(covered),
            "median_n_genes_when_covered": float(covered["n_group_genes"].median()) if len(covered) else 0.0,
            "max_n_genes": int(sub["n_group_genes"].max()) if len(sub) else 0,
        })
    print(pd.DataFrame(summary).sort_values(["n_projects_covered", "max_n_genes"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
