import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import torch
from tqdm import tqdm

def _safe_load_entry(entry):
    file_path, annotation = entry
    try:
        arr = torch.load(file_path, map_location="cpu")
        return arr, annotation
    except Exception as e:
        print(f"[WARN] Failed to load {file_path}: {e}")
        return None, annotation


def preload_uni_feats(feature_dir, img_dir, anno_csv_path, coords_csv_path, split, num_workers=32):
    annotation = {}
    num_genes = None

    with open(anno_csv_path, "r") as rf:
        rf.readline()
        for line in rf:
            sid, slide, sp_split, *gene = line.strip().split(",")
            gene = list(map(float, gene))
            if num_genes is None:
                num_genes = len(gene)
            annotation[sid] = [sp_split, slide, gene]

    with open(coords_csv_path, "r") as rf:
        rf.readline()
        for line in rf:
            sid, x, y = line.strip().split(",")
            if sid in annotation:
                annotation[sid].append([int(x), int(y)])

    data = []
    for path, _, files in os.walk(feature_dir):
        for fname in files:
            if not fname.endswith(".pt"):
                continue
            sid = fname.replace(".pt", "")
            if sid not in annotation:
                continue

            sp_split, slide, gene, coords = annotation[sid]
            if sp_split != split:
                continue

            fpath = os.path.join(path, fname)
            img_path = fpath.replace(feature_dir, img_dir).replace(".pt", ".png")
            data.append((fpath, [img_path, sid, sp_split, slide, gene, coords]))

    feats, meta = [], []
    print(f"[UNI] {split} | N={len(data)}")

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for f, m in tqdm(ex.map(_safe_load_entry, data), total=len(data)):
            if f is not None:
                feats.append(f)
                meta.append(m)
    if len(feats) == 0:
        return None
    return torch.stack(feats), meta, num_genes


def merge_scfoundation(project, preload_feats, preload_meta, scfm_root):
    h5_path = os.path.join(scfm_root, f"{project}_patch_gene_info_01B-resolution_bulk_cell_embedding_t4_resolution.h5")

    with h5py.File(h5_path, "r") as f:
        emb = f["embedding"][:]
        uids = np.array([u.decode() for u in f["unique_id"][:]])

    uid_to_idx = {u: i for i, u in enumerate(uids)}

    keep = []
    idxs = []
    for i in range(len(preload_meta)):
        uid = preload_meta[i][1]
        if uid in uid_to_idx:
            j = uid_to_idx[uid]
            if not np.allclose(emb[j], 0):
                keep.append(True)
                idxs.append(j)
                continue
        keep.append(False)

    keep = np.array(keep)
    idxs = np.array(idxs)

    uni_new = preload_feats[keep]
    gene_new = torch.tensor(emb[idxs], dtype=torch.float32)
    meta_new = [m for m, k in zip(preload_meta, keep) if k]

    return uni_new, gene_new, meta_new


def build_slide_knn_indices_deterministic(meta, k=36, exclude_self=True):


    N = len(meta)
    if N == 0:
        return torch.empty((0, k), dtype=torch.long)

    slide_to_indices = {}
    for i, m in enumerate(meta):
        slide = m[3]
        slide_to_indices.setdefault(slide, []).append(i)

    nn_idx = torch.empty((N, k), dtype=torch.long)

    BIG = torch.iinfo(torch.int64).max

    for slide, idxs in slide_to_indices.items():
        idxs_t = torch.tensor(idxs, dtype=torch.long)
        coords = torch.tensor([meta[i][-1] for i in idxs], dtype=torch.int64)
        n = coords.shape[0]

        if n == 1:
            nn_idx[idxs_t] = idxs_t[0].repeat(k).unsqueeze(0)
            continue


        dx = coords[:, 0].unsqueeze(1) - coords[:, 0].unsqueeze(0)
        dy = coords[:, 1].unsqueeze(1) - coords[:, 1].unsqueeze(0)
        sqdist = dx * dx + dy * dy


        j = torch.arange(n, dtype=torch.int64).unsqueeze(0).expand(n, n)
        key = sqdist * (n + 1) + j

        if exclude_self:
            key.fill_diagonal_(BIG)

        kk = min(k, n - (1 if exclude_self else 0))

        nn_local = torch.topk(key, k=kk, dim=1, largest=False).indices


        idxs_list = torch.tensor(idxs, dtype=torch.long)
        nn_global = idxs_list[nn_local]

        if kk < k:
            pad = nn_global[:, -1:].repeat(1, k - kk)
            nn_global = torch.cat([nn_global, pad], dim=1)

        nn_idx[idxs_t] = nn_global

    return nn_idx


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    splits = ["train", "val", "test"]

    for project in sorted(os.listdir(args.uni_root)):
        project_dir = os.path.join(args.uni_root, project)
        if not os.path.isdir(project_dir):
            continue

        save_path = os.path.join(args.out_dir, f"{project}.pt")
        if args.skip_existing and os.path.exists(save_path):
            continue

        scfm_path = os.path.join(
            args.scfm_root,
            f"{project}_patch_gene_info_01B-resolution_bulk_cell_embedding_t4_resolution.h5",
        )
        if not os.path.isfile(scfm_path):
            continue

        print(f"[PROJECT] {project}")
        project_cache = {}
        num_genes_global = None

        for split in splits:
            result = preload_uni_feats(
                os.path.join(args.uni_root, project),
                os.path.join(args.patch_root, project),
                os.path.join(args.csv_root, f"{project}_patch_gene_info.csv"),
                os.path.join(args.coords_root, f"{project}.csv"),
                split,
                num_workers=args.num_workers,
            )
            if result is None:
                continue

            feats_uni, metadata, num_genes = result
            feats_uni, feats_gene, metadata = merge_scfoundation(
                project,
                feats_uni,
                metadata,
                args.scfm_root,
            )

            if num_genes_global is None:
                num_genes_global = num_genes

            nn_idx = build_slide_knn_indices_deterministic(
                metadata,
                k=args.knn_k,
                exclude_self=True,
            )

            project_cache[split] = {
                "UNI_feats": feats_uni,
                "scFM_feats": feats_gene,
                "metadata": metadata,
                f"nn{args.knn_k}_idx": nn_idx,
            }
            print(f"[OK] {project} {split} | N={feats_uni.shape[0]} | nn={tuple(nn_idx.shape)}")

        project_cache["num_genes"] = num_genes_global
        torch.save(project_cache, save_path)
        print(f"[SAVED] {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uni-root", required=True)
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--csv-root", required=True)
    parser.add_argument("--coords-root", required=True)
    parser.add_argument("--scfm-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--knn-k", type=int, default=36)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
