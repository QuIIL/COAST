from __future__ import annotations

import os
import argparse
from typing import List

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from coast.data import TCGAExternalSlideDataset, far_eval_grid_regular
from coast.models import create_model, load_model_weights


def ddp_env():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def ddp_init(rank, world_size, local_rank):
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()


def ddp_cleanup(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def safe_tqdm(loader, rank):
    return tqdm(loader, dynamic_ncols=True) if rank == 0 else loader


def load_precomputed_nn(nn_dir: str, slide_id: str, nn_key: str) -> np.ndarray:
    nn_path = os.path.join(nn_dir, f"{slide_id}.h5")
    if not os.path.exists(nn_path):
        raise FileNotFoundError(f"nn file not found: {nn_path}")
    with h5py.File(nn_path, "r") as f:
        if nn_key not in f:
            raise KeyError(f"Expected '{nn_key}' in {nn_path}. keys={list(f.keys())}")
        nn = f[nn_key][:]
    return nn.astype(np.int64)


@torch.no_grad()
def eval_external_ddp(args, ddp_model, loader, device, rank, world_size):
    ddp_model.eval()
    os.makedirs(args.save_dir, exist_ok=True)

    failed_local: List[str] = []

    for batch in safe_tqdm(loader, rank):
        slide_ids, coords_np, feats_np, gg = batch
        slide_id = slide_ids[0] if isinstance(slide_ids, (list, tuple)) else str(slide_ids)

        try:

            feats = feats_np[0] if isinstance(feats_np, torch.Tensor) and feats_np.ndim == 3 else feats_np
            coords_arr = np.array(coords_np)
            coords_arr = coords_arr[0] if coords_arr.ndim == 3 else coords_arr

            feats = torch.as_tensor(feats, dtype=torch.float32)
            coords = torch.as_tensor(coords_arr, dtype=torch.float32)
            N, D = feats.shape


            nn_full = load_precomputed_nn(args.nn_dir, slide_id, nn_key=args.nn_key)
            if nn_full.shape[0] != N:
                raise ValueError(
                    f"[{slide_id}] nn rows mismatch: nn={nn_full.shape[0]} vs feats={N} "
                    f"(mask 적용 상태가 precompute와 inference에서 일치해야 함)"
                )
            if nn_full.shape[1] < args.near_k:
                raise ValueError(f"[{slide_id}] nn cols too small: nnK={nn_full.shape[1]} < near_k={args.near_k}")

            near_idx = torch.as_tensor(nn_full[:, :args.near_k], dtype=torch.long)


            if args.far_k > 0:
                far_idx = torch.empty((N, args.far_k), dtype=torch.long)
                for i in range(N):
                    excl = set([i]) | set(map(int, near_idx[i].tolist()))
                    far_list = far_eval_grid_regular(coords, args.far_k, excl)
                    far_idx[i] = torch.as_tensor(far_list, dtype=torch.long)
                ctx_idx = torch.cat([near_idx, far_idx], dim=1)
            else:
                ctx_idx = near_idx
            K = ctx_idx.shape[1]


            if args.add_type_embed:
                ctx_type_row = torch.zeros(K, dtype=torch.long, device=device)
                if args.far_k > 0:
                    ctx_type_row[args.near_k:] = 1
            else:
                ctx_type_row = None


            m = ddp_model.module if hasattr(ddp_model, "module") else ddp_model


            preds_all = torch.empty((N, args.num_genes), dtype=torch.float32, device="cpu")

            feats_d = feats.to(device, non_blocking=True)
            coords_d = coords.to(device, non_blocking=True)


            for s in range(0, N, args.infer_bs):
                e = min(N, s + args.infer_bs)
                b = e - s

                x_img = feats_d[s:e]

                idx_b = ctx_idx[s:e].to(device, non_blocking=True)
                x_ctx = feats_d.index_select(0, idx_b.reshape(-1)).view(b, K, D)

                c = coords_d[s:e]
                coords_ctx = coords_d.index_select(0, idx_b.reshape(-1)).view(b, K, 2)

                y_hat_spot, y_hat_diff = m(
                    x_img=x_img,
                    x_ctx_img=x_ctx,
                    coords=c,
                    coords_ctx=coords_ctx,
                    ctx_type=ctx_type_row.unsqueeze(0).expand(b, -1) if ctx_type_row is not None else None,
                    ctx_mask=None,
                )

                preds_all[s:e] = y_hat_spot.detach().cpu()

            out_path = os.path.join(args.save_dir, f"{slide_id}.pt")
            torch.save(preds_all, out_path)

        except Exception as e:
            msg = f"[rank{rank}] {repr(e)} | slide_id={slide_id}"
            print(msg, flush=True)
            failed_local.append(str(slide_id))
            with open(os.path.join(args.save_dir, "Failed_All_append.txt"), "a") as f:
                f.write(f"{slide_id}\n")


    fail_path = os.path.join(args.save_dir, f"Failed.rank{rank}.txt")
    if failed_local:
        with open(fail_path, "w") as f:
            for sid in failed_local:
                f.write(f"{sid}\n")

    if world_size == 1:
        if failed_local:
            with open(os.path.join(args.save_dir, "Failed.txt"), "w") as f:
                for sid in failed_local:
                    f.write(f"{sid}\n")
        return

    dist.barrier()


    if is_main(rank):
        merged: List[str] = []
        for r in range(world_size):
            p = os.path.join(args.save_dir, f"Failed.rank{r}.txt")
            if os.path.exists(p):
                with open(p, "r") as f:
                    merged.extend([line.strip() for line in f if line.strip()])

        if merged:
            merged = sorted(set(merged))
            with open(os.path.join(args.save_dir, "Failed.txt"), "w") as f:
                for sid in merged:
                    f.write(f"{sid}\n")

        if args.cleanup_failed_parts:
            for r in range(world_size):
                p = os.path.join(args.save_dir, f"Failed.rank{r}.txt")
                if os.path.exists(p):
                    os.remove(p)

    dist.barrier()


def main(args):
    rank, world_size, local_rank = ddp_env()
    ddp_init(rank, world_size, local_rank)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    test_ds = TCGAExternalSlideDataset(
        base_dir=args.base_dir,
        annotation_path=args.annotation_path,
        require_dx=args.require_dx,
        fix_bad_slides=args.fix_bad_slides,
    )

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            test_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=(args.num_workers > 0),
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
    if is_main(rank) and (missing or unexpected):
        if missing:
            print("[rank0] Missing keys (non-strict):", missing[:30], "..." if len(missing) > 30 else "")
        if unexpected:
            print("[rank0] Unexpected keys (non-strict):", unexpected[:30], "..." if len(unexpected) > 30 else "")

    if world_size > 1:
        ddp_model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        class _Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.module = m
            def forward(self, *a, **k):
                return self.module(*a, **k)
        ddp_model = _Wrap(model)

    if sampler is not None:
        sampler.set_epoch(0)

    eval_external_ddp(args, ddp_model, test_loader, device, rank, world_size)
    ddp_cleanup(world_size)


if __name__ == "__main__":
    p = argparse.ArgumentParser()


    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--annotation_path", type=str, default=None)
    p.add_argument("--nn_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)


    p.add_argument("--nn_key", type=str, default="nn36_idx")


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


    p.add_argument("--infer_bs", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)


    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--cleanup_failed_parts", action="store_true")

    args = p.parse_args()
    main(args)
