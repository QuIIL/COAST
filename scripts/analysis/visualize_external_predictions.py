import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from coast.data import TCGAExternalSlideDataset

LOWER_BETTER = ["NKX3-1", "AZGP1", "HLA-B", "B2M"]

crop = {
    "TCGA-44-6776-01Z-00-DX1.551e200f-5f40-4b95-8b2f-7c6000a21be0": 0.33,
    "TCGA-38-4626-01Z-00-DX1.142bc018-bd79-4db9-84e2-f83a617ea92a": 0.33,
}
LUNG_LIST = [

    "TCGA-44-6776-01Z-00-DX1.551e200f-5f40-4b95-8b2f-7c6000a21be0",
    "TCGA-78-8655-01Z-00-DX1.DC81D146-A4A6-411E-A5DC-C0689A8D445F",
    "TCGA-97-7547-01Z-00-DX1.3563966f-21ee-4405-9106-9e0bf8df9f1a",
    "TCGA-38-4626-01Z-00-DX1.142bc018-bd79-4db9-84e2-f83a617ea92a",
    "TCGA-62-A471-01Z-00-DX1.51A235BB-7F67-4C0D-8B15-2E26116D8822",

    "TCGA-64-5775-01Z-00-DX1.9A338C1A-CDB6-41AF-B6D5-EDE608B5E3E1",
    "TCGA-78-7158-01Z-00-DX1.74db3dcd-2715-4ce7-8e48-30c8b7946ba0",
    "TCGA-49-4490-01Z-00-DX6.0a04ad8e-7f33-4496-bd9e-0110ec33f887",
    "TCGA-L9-A50W-01Z-00-DX1.21FAAB58-58EF-4A6A-8F0A-16D9AB9B1056",
]


def _safe_read_tsv_or_csv(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t")
    elif ext in [".csv"]:
        return pd.read_csv(path)
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path)


def create_lut(colormap_type=cv2.COLORMAP_JET):
    lut_input = np.arange(256, dtype=np.uint8).reshape(256, 1)
    lut_color = cv2.applyColorMap(lut_input, colormap_type)
    return lut_color.reshape(256, 3)


def get_slide_paths(slide_root):
    slide_paths = {}
    for path, dir, files in os.walk(slide_root):
        for file in files:
            if file.endswith(".svs"):
                slide_id = os.path.splitext(file)[0]

                if slide_id not in LUNG_LIST:
                    continue
                slide_paths[slide_id] = os.path.join(path, file)
    return slide_paths


def extract_features(data) -> np.ndarray:
    feat = None
    if isinstance(data, torch.Tensor):
        feat = data
    elif isinstance(data, dict):
        for k in ["preds_img", "gene_exp"]:
            if k in data and isinstance(data[k], torch.Tensor):
                feat = data[k]
                break
        if feat is None:
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    feat = v
                    break
    if feat is not None and feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    return feat.detach().cpu().numpy() if feat is not None else None


def calculate_global_stats(dataset, pred_dir, target_genes):
    print(">>> Calculating Global Statistics (1st/99th percentile)...")
    gene_values_buffer = {idx: [] for idx, _ in target_genes}

    for i in tqdm(range(len(dataset)), desc="Collecting Stats"):
        slide_id, _, _, _ = dataset[i]
        pred_path = os.path.join(pred_dir, f"{slide_id}.pt")

        if not os.path.exists(pred_path): continue
        try:
            pred_data = torch.load(pred_path, map_location='cpu')
            feat = extract_features(pred_data)
            if feat is None: continue

            for g_idx, _ in target_genes:
                if g_idx < feat.shape[1]:
                    vals = feat[:, g_idx]
                    vals = np.nan_to_num(vals)
                    vals = np.clip(vals, 0, 1000)
                    gene_values_buffer[g_idx].append(vals)
        except:
            continue

    global_stats = {}
    for g_idx, g_name in target_genes:
        if not gene_values_buffer[g_idx]:
            global_stats[g_idx] = (0, 1)
            continue
        all_vals = np.concatenate(gene_values_buffer[g_idx])
        p_low = np.percentile(all_vals, 1)
        p_high = np.percentile(all_vals, 99)
        global_stats[g_idx] = (p_low, p_high)
        print(f"  - {g_name:>10} | Range: {p_low:.4f} ~ {p_high:.4f}")

    return global_stats


def visualize_external_heatmap(slide_path, coords, values, save_path, vmin=None, vmax=None, title="", local_norm=False, patient_time_dict=None, patient_event_dict=None):
    import openslide

    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        print(f"[ERROR] 슬라이드 로드 실패: {slide_path} / {e}")
        return

    slide_w, slide_h = slide.dimensions
    downsample_factor = 80
    thumb_size = (slide_w // downsample_factor, slide_h // downsample_factor)

    slide_id = os.path.basename(slide_path).replace(".svs", "")
    parts = slide_id.split("-")
    pid = "-".join(parts[:3])
    t = int(patient_time_dict[pid])
    e = int(patient_event_dict[pid])
    save_path = save_path.replace("_heatmap/", "_heatmap_{}/{}_".format(e, t))

    img = slide.get_thumbnail(thumb_size)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    thumbnail_path = os.path.join(os.path.dirname(save_path), "thumbnail.png")

    thumb = img.copy()
    if "TCGA-44-6776-01Z-00-DX1.551e200f-5f40-4b95-8b2f-7c6000a21be0" in save_path:
        thumb = thumb[:, :thumb.shape[1] * 33 // 100]
    if "TCGA-38-4626-01Z-00-DX1.142bc018-bd79-4db9-84e2-f83a617ea92a" in save_path:
        thumb = thumb[:, :thumb.shape[1] * 33 // 100]
    if "TCGA-05-4245-01Z-00-DX1.36ff5403-d4bb-4415-b2c5-7c750d655cde" in save_path:
        thumb = thumb[thumb.shape[0] * 6 // 10:]
    if "TCGA-49-4490-01Z-00-DX6.0a04ad8e-7f33-4496-bd9e-0110ec33f887" in save_path:
        thumb = thumb[:, :thumb.shape[1] * 5 // 10]
    os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
    cv2.imwrite(thumbnail_path, thumb)

    H, W = img.shape[:2]
    scale_x = W / slide_w
    scale_y = H / slide_h

    values = np.nan_to_num(values)
    values = np.clip(values, 0, 1000)

    if vmin is None or vmax is None or local_norm:
        p_low = np.percentile(values, 1)
        p_high = np.percentile(values, 99)
    else:
        p_low = vmin
        p_high = vmax

    clipped_values = np.clip(values, p_low, p_high)

    v_diff = p_high - p_low
    if v_diff <= 1e-9:
        norm_values = np.zeros_like(clipped_values)
    else:
        norm_values = (clipped_values - p_low) / v_diff

    lut = create_lut(cv2.COLORMAP_JET)
    for i in range(len(coords)):
        x, y = coords[i]
        x_img = int(x * scale_x)
        y_img = int(y * scale_y)
        color_idx = int(norm_values[i] * 255)
        bgr = lut[color_idx].tolist()
        cv2.circle(img, (x_img, y_img), 2, bgr, -1)

    if "TCGA-44-6776-01Z-00-DX1.551e200f-5f40-4b95-8b2f-7c6000a21be0" in save_path:
        img = img[:, :img.shape[1] * 33 // 100]
    if "TCGA-38-4626-01Z-00-DX1.142bc018-bd79-4db9-84e2-f83a617ea92a" in save_path:
        img = img[:, :img.shape[1] * 33 // 100]
    if "TCGA-05-4245-01Z-00-DX1.36ff5403-d4bb-4415-b2c5-7c750d655cde" in save_path:
        img = img[img.shape[0] * 6 // 10:]
    if "TCGA-49-4490-01Z-00-DX6.0a04ad8e-7f33-4496-bd9e-0110ec33f887" in save_path:
        img = img[:, :img.shape[1] * 5 // 10]


    for lb in LOWER_BETTER:
        if lb in save_path:
            save_path = save_path.replace(lb, "{}(LB)".format(lb))


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def load_stats_from_cache(cache_path, dataset, pred_dir, target_genes):
    if not os.path.exists(cache_path):
        print(f"\n[CACHE] No cache found at {cache_path}. Calculating...")
        global_stats = calculate_global_stats(dataset, pred_dir, target_genes)
        torch.save(global_stats, cache_path)
        print(f"[CACHE] Saved statistics to {cache_path}")

    print(f"\n[CACHE] Found stats cache: {cache_path}")
    global_stats = torch.load(cache_path, weights_only=False)
    print("[CACHE] Loaded statistics successfully.")
    return global_stats


def main(setting):
    if setting == "prostate":
        prj_name = "TCGA-PRAD-prostate"
        target_genes = [
            (30, "TMPRSS2"), (107, "HOXB13"), (100, "FOXA1"), (46, "FASN"),
            (106, "STEAP1"), (77, "IGFBP2"), (20, "NKX3-1"), (3, "AZGP1"),
        ]
        annotation_path = "/home/work/.home/data/TCGA-PRAD-prostate_features/annotation/TCGA-PRAD-OS-labels.tsv"
    elif setting == "lung":
        prj_name = "TCGA-LUAD-lung"
        target_genes = [
            (51, "MDK"), (15, "SLC2A1"), (123, "HK2"), (111, "CXCR4"),
            (42, "SOX4"), (4, "HLA-B"), (16, "B2M"), (48, "LDHA"), (77, "YWHAZ")
        ]
        annotation_path = "/home/work/.home/data/TCGA-LUAD-lung_features/annotation/TCGA-LUAD-OS-labels.tsv"
    else:
        raise AssertionError("Unknown setting")

    ann = _safe_read_tsv_or_csv(annotation_path)
    required = {"patient_id", "time_days", "event"}
    if not required.issubset(set(ann.columns)):
        raise ValueError(f"annotation must contain columns {required}")

    ann = ann[["patient_id", "time_days", "event"]].copy()
    ann["patient_id"] = ann["patient_id"].astype(str)
    ann["time_days"] = pd.to_numeric(ann["time_days"], errors="coerce")
    ann["event"] = pd.to_numeric(ann["event"], errors="coerce")
    ann = ann.dropna(subset=["patient_id", "time_days", "event"])
    ann = ann[(ann["time_days"] > 0)]
    ann["time_days"] = ann["time_days"].astype(int)
    ann["event"] = ann["event"].astype(int)
    patient_time_dict = dict(zip(ann["patient_id"], ann["time_days"]))
    patient_event_dict = dict(zip(ann["patient_id"], ann["event"]))

    base_dir = "/home/work/.home/data/{}_features/20x_256px_0px_overlap/features_uni_v1".format(prj_name)
    slide_root = "/home/work/.home/data/{}".format(prj_name)
    pred_dir = "/home/work/.home/data/{}_pred/COAST_N18F72S1D1".format(prj_name)


    save_base_dir = "/home/work/.home/data/{}_pred_selected/COAST_N18F72S1D1_gene_heatmap".format(prj_name)
    os.makedirs(save_base_dir, exist_ok=True)

    cache_path = pred_dir.rstrip('/') + "_stats.pt"

    dataset = TCGAExternalSlideDataset(base_dir, None)
    slide_paths = get_slide_paths(slide_root)
    print(f"\n # Found Slides in Dataset: {len(dataset)}")

    print(f"\n # Loading stats from cache: {cache_path}")
    global_stats = load_stats_from_cache(cache_path, dataset, pred_dir, target_genes)


    stats_records = []

    print("\n>>> Starting Visualization with Global Ranges...")
    for i in tqdm(range(len(dataset)), desc="Visualizing"):
        slide_id, coords, _, label = dataset[i]
        coords = np.asarray(coords, dtype=np.float32)

        if slide_id not in slide_paths: continue

        svs_path = slide_paths[slide_id]
        pred_path = os.path.join(pred_dir, f"{slide_id}.pt")

        if not os.path.exists(pred_path): continue

        pred_data = torch.load(pred_path, map_location='cpu')
        pred_feat_all = extract_features(pred_data)

        if pred_feat_all is None: continue

        min_len = min(len(coords), len(pred_feat_all))
        coords = coords[:min_len]
        pred_feat_all = pred_feat_all[:min_len]

        for gene_idx, gene_name in target_genes:
            if gene_idx >= pred_feat_all.shape[1]: continue

            pred_vals = pred_feat_all[:, gene_idx]


            clean_vals = np.nan_to_num(pred_vals)
            clean_vals = np.clip(clean_vals, 0, 1000)

            val_mean = np.mean(clean_vals)
            val_std = np.std(clean_vals)
            val_min = np.min(clean_vals)
            val_max = np.max(clean_vals)

            stat_str = f"{val_mean:.2f}±{val_std:.2f}({val_min:.2f}-{val_max:.2f})"

            stats_records.append({
                "Dataset": prj_name,
                "Slide": slide_id,
                "Gene": gene_name,
                "Stats": stat_str
            })

            g_min, g_max = global_stats.get(gene_idx, (None, None))

            out_file = os.path.join(save_base_dir, slide_id, f"{gene_name}.png")
            visualize_external_heatmap(
                slide_path=svs_path,
                coords=coords,
                values=pred_vals,
                save_path=out_file,
                vmin=g_min,
                vmax=g_max,
                title=f"{gene_name}",
                local_norm=False,
                patient_time_dict=patient_time_dict,
                patient_event_dict=patient_event_dict
            )


    if stats_records:
        df_stats = pd.DataFrame(stats_records)
        os.makedirs(save_base_dir, exist_ok=True)
        csv_path = os.path.join(save_base_dir, f"{prj_name}_gene_statistics.csv")
        df_stats.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n[DONE] Statistics saved to: {csv_path}")

    print("[DONE] Visualization Complete.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["prostate", "lung"], default="lung")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().setting)
