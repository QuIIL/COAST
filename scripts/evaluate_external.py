from __future__ import annotations
import os
import argparse
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from coast.data import TCGAExternalCOASTSpotInSlideDataset, TCGAExternalSlideListDataset
from coast.models import create_model, load_model_weights


@torch.no_grad()
def run_one_slide(args, model, device, slide_id: str, feat_path: str):
    out_path = os.path.join(args.save_dir, f"{slide_id}.pt")

    if os.path.exists(out_path) and args.skip_exist:
        return


    spot_ds = TCGAExternalCOASTSpotInSlideDataset(
        slide_id=slide_id,
        feature_path=feat_path,
        nn_dir=args.nn_dir,
        nn_key=args.nn_key,
        split="test",
        near_k=args.near_k,
        far_k=args.far_k,
        num_genes=args.num_genes,
        infer_mode=True,
        fix_bad_slides=args.fix_bad_slides,
        seed=args.seed,
    )

    spot_loader = DataLoader(
        spot_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    N = len(spot_ds)
    preds_all = torch.empty((N, args.num_genes), dtype=torch.float32, device="cpu")


    K = args.near_k + args.far_k
    ctx_type_row = None
    if args.add_type_embed:
        ctx_type_row = torch.zeros(K, dtype=torch.long, device=device)
        if args.far_k > 0:
            ctx_type_row[args.near_k:] = 1

    for batch in spot_loader:

        x_img = batch["x_img"].to(device, non_blocking=True)
        x_ctx = batch["x_ctx"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, dtype=torch.float32, non_blocking=True)
        coords_ctx = batch["coords_ctx"].to(device, dtype=torch.float32, non_blocking=True)
        idx = batch["idx"].detach().cpu().long()

        y_hat_spot, _ = model(
            x_img=x_img,
            x_ctx_img=x_ctx,
            coords=coords,
            coords_ctx=coords_ctx,
            ctx_type=ctx_type_row.unsqueeze(0).expand(x_img.shape[0], -1) if ctx_type_row is not None else None,
            ctx_mask=None,
        )
        preds_all[idx] = y_hat_spot.detach().cpu()

    torch.save(preds_all, out_path)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)


    slide_ds = TCGAExternalSlideListDataset(
        base_dir=args.base_dir,
        annotation_path=args.annotation_path,
        require_dx=args.require_dx,
    )

    model = create_model(
        args.model_name,
        input_dim=args.img_dim,
        hidden_dim=args.hidden_dim,
        feat_dim=args.feat_dim,
        num_genes=args.num_genes,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        add_type_embed=args.add_type_embed,
    ).to(device)
    missing, unexpected = load_model_weights(model, args.weights)
    if missing or unexpected:
        if missing:
            print("Missing keys (non-strict):", missing[:30], "..." if len(missing) > 30 else "")
        if unexpected:
            print("Unexpected keys (non-strict):", unexpected[:30], "..." if len(unexpected) > 30 else "")

    model.eval()

    failed: List[str] = []
    for slide_id, feat_path, gg in tqdm(slide_ds, total=len(slide_ds), dynamic_ncols=True):
        try:
            run_one_slide(args, model, device, slide_id, feat_path)
        except Exception as e:
            print(f"[ERROR] {repr(e)} | slide_id={slide_id}", flush=True)
            failed.append(slide_id)
            with open(os.path.join(args.save_dir, "Failed_All_append.txt"), "a") as f:
                f.write(f"{slide_id}\n")

    if failed:
        failed = sorted(set(failed))
        with open(os.path.join(args.save_dir, "Failed.txt"), "w") as f:
            for sid in failed:
                f.write(f"{sid}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()


    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--annotation_path", type=str, default=None)
    p.add_argument("--nn_dir", type=str, required=True)
    p.add_argument("--nn_key", type=str, default="nn36_idx")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)


    p.add_argument("--require_dx", action="store_true")
    p.add_argument("--fix_bad_slides", action="store_true")


    p.add_argument("--model_name", type=str, default="COAST", choices=["COAST", "COASTEmbAbl"])
    p.add_argument("--img_dim", type=int, default=1024)
    p.add_argument("--feat_dim", type=int, default=1024)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_genes", type=int, required=True)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--add_type_embed", action="store_true")


    p.add_argument("--near_k", type=int, default=18)
    p.add_argument("--far_k", type=int, default=72)


    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)


    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--skip_exist", action="store_true")

    args = p.parse_args()
    main(args)
