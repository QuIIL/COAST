import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch

from coast.data import STSpotCtxDataset


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def draw_spot_ctx_on_slide(
    slide_png: str,
    all_coords: np.ndarray,
    center_idx: int,
    near_idx: np.ndarray,
    far_idx: np.ndarray,
    slide_w: float,
    slide_h: float,
    out_path: str,
    downsample: float = 2.0,
    r_all: int = 3,
    r_ctx: int = 5,
    r_center: int = 7,
):


    img = cv2.imread(slide_png)
    if img is None:
        raise FileNotFoundError(slide_png)

    if downsample != 1.0:
        img = cv2.resize(img, (int(img.shape[1] / downsample), int(img.shape[0] / downsample)))

    H, W = img.shape[:2]
    scale_x = W / float(slide_w)
    scale_y = H / float(slide_h)


    C_ALL = (170, 170, 170)
    C_NEAR = (255, 0, 0)
    C_FAR = (0, 255, 0)
    C_CENTER = (0, 0, 255)
    C_CENTER_BORDER = (255, 255, 255)


    for (x, y) in all_coords:
        cv2.circle(img, (int(x * scale_x), int(y * scale_y)), r_all, C_ALL, -1)


    for j in near_idx.tolist():
        x, y = all_coords[j]
        cv2.circle(img, (int(x * scale_x), int(y * scale_y)), r_ctx, C_NEAR, -1)


    for j in far_idx.tolist():
        x, y = all_coords[j]
        cv2.circle(img, (int(x * scale_x), int(y * scale_y)), r_ctx+ 2, C_FAR, -1)


    cx, cy = all_coords[center_idx]
    cv2.circle(img, (int(cx * scale_x), int(cy * scale_y)), r_center + 2, C_CENTER_BORDER, -1)
    cv2.circle(img, (int(cx * scale_x), int(cy * scale_y)), r_center, C_CENTER, -1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--slide-root", required=True)
    parser.add_argument("--slide-info-csv", required=True)
    parser.add_argument("--out-dir", default="ctx_vis")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--near-k", type=int, default=18)
    parser.add_argument("--far-k", type=int, default=36)
    parser.add_argument("--nn-key", default="nn36_idx")
    parser.add_argument("--far-scope", default="slide", choices=["slide", "global"])
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--downsample", type=float, default=2.0)
    return parser.parse_args()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    slide_info = pd.read_csv(args.slide_info_csv)
    rng = np.random.default_rng(args.seed)

    for project_file in sorted(os.listdir(args.cache_dir)):
        if not project_file.endswith(".pt"):
            continue

        cache_path = os.path.join(args.cache_dir, project_file)
        dataset = STSpotCtxDataset(
            cache_path=cache_path,
            split=args.split,
            near_k=args.near_k,
            far_k=args.far_k,
            nn_key=args.nn_key,
            far_scope=args.far_scope,
            seed=args.seed,
        )
        sample_count = min(args.num_examples, len(dataset))
        picks = rng.choice(len(dataset), size=sample_count, replace=False).tolist()

        slide_to_indices = {}
        for index, metadata in enumerate(dataset.metadata):
            slide_to_indices.setdefault(metadata[3], []).append(index)

        project_name = os.path.splitext(project_file)[0]
        for index in picks:
            example = dataset[index]
            slide_id = dataset.metadata[index][3]
            slide_png = os.path.join(args.slide_root, project_name, f"{slide_id}.png")
            if not os.path.exists(slide_png):
                print(f"Missing slide image: {slide_png}")
                continue

            row = slide_info[
                (slide_info["project"] == project_name)
                & (slide_info["slide"] == slide_id)
            ]
            if row.empty:
                print(f"Missing slide metadata: {project_name} {slide_id}")
                continue

            near_idx = (
                dataset.nn_full[index][: dataset.near_k].clone()
                if dataset.near_k > 0
                else torch.empty(0, dtype=torch.long)
            )
            if dataset.far_k > 0:
                far_idx = (
                    dataset._sample_far_train_uniform(index, slide_id, near_idx)
                    if dataset.is_train
                    else dataset._sample_far_eval_stratified(index, slide_id, near_idx)
                )
            else:
                far_idx = torch.empty(0, dtype=torch.long)

            slide_indices = slide_to_indices[slide_id]
            all_coords = np.asarray(
                [dataset.metadata[i][5] for i in slide_indices],
                dtype=np.float32,
            )
            global_to_local = {global_index: local_index for local_index, global_index in enumerate(slide_indices)}
            center_local = global_to_local[index]
            near_local = np.asarray(
                [global_to_local[int(i)] for i in near_idx.tolist() if int(i) in global_to_local],
                dtype=np.int64,
            )
            far_local = np.asarray(
                [global_to_local[int(i)] for i in far_idx.tolist() if int(i) in global_to_local],
                dtype=np.int64,
            )

            out_path = os.path.join(
                args.out_dir,
                project_name,
                f"{slide_id}_idx{index}_scope{dataset.far_scope}_{args.split}.png",
            )
            draw_spot_ctx_on_slide(
                slide_png=slide_png,
                all_coords=all_coords,
                center_idx=center_local,
                near_idx=near_local,
                far_idx=far_local,
                slide_w=float(row["slide_w"].iloc[0]),
                slide_h=float(row["slide_h"].iloc[0]),
                out_path=out_path,
                downsample=args.downsample,
            )
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main(parse_args())
