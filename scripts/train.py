import argparse
import os
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from coast.data import STSpotCtxDataset
from coast.losses import SpotMSEPlusDiffMSELoss
from coast.metrics import MetricCalculator
from coast.models import create_model
from coast.utils import WarmupCosineScheduler, load_config, log, set_seed


def train_epoch(args, model, loader, criterion, optimizer, device):
    model.train()

    loss_sum = 0.0
    loss_reg_sum, loss_diff_sum = 0.0, 0.0
    all_preds, all_y = [], []

    cnt, log_cnt = 0, 0
    for batch_idx, batch in enumerate(loader):

        x_img = batch["x_img"].to(device)
        x_ctx = batch["x_ctx"].to(device)
        coords = batch["coords"].to(device, dtype=torch.float32)
        coords_ctx = batch["coords_ctx"].to(device, dtype=torch.float32)
        ctx_type = batch["ctx_type"].to(device)
        y = batch["y"].to(device)
        y_ctx = batch["y_ctx"].to(device)

        y_hat_spot, y_hat_diff = model(
            x_img=x_img,
            x_ctx_img=x_ctx,
            coords=coords,
            coords_ctx=coords_ctx,
            ctx_type=ctx_type if args.model.add_type_embed else None,
            ctx_mask=None,
        )

        loss, loss_spot, loss_diff = criterion(
            y_hat_spot=y_hat_spot,
            y_true_spot=y,
            y_hat_diff=y_hat_diff,
            y_true_ctx=y_ctx,
            pair_mask=None,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        all_preds.append(y_hat_spot.detach().cpu())
        all_y.append(y.detach().cpu())
        loss_sum += float(loss.item()) * y.shape[0]
        loss_reg_sum += float(loss_spot.item()) * y.shape[0]
        loss_diff_sum += float(loss_diff.item()) * y.shape[0]

        cnt += y.shape[0]
        log_cnt += y.shape[0]
        if log_cnt > 1000:
            log_cnt = 0
            log(args.data.log_path, f"[{batch_idx}] loss={loss_sum / cnt:.6f} spot={loss_reg_sum / cnt:.6f} diff={loss_diff_sum / cnt:.6f}")

    all_preds = torch.cat(all_preds, 0)
    all_y = torch.cat(all_y, 0)

    total_loss = loss_sum / all_preds.shape[0]
    train_metrics = MetricCalculator.get_metrics(all_preds, all_y)

    return total_loss, train_metrics


@torch.no_grad()
def eval_epoch(args, model, loader, criterion, device, save_pred=False):
    model.eval()

    loss_sum = 0.0
    loss_reg_sum, loss_diff_sum = 0.0, 0.0
    all_preds, all_y = [], []


    pred_buckets = defaultdict(list)

    cnt, log_cnt = 0, 0
    for batch_idx, batch in enumerate(loader):
        x_img = batch["x_img"].to(device)
        x_ctx = batch["x_ctx"].to(device)
        coords = batch["coords"].to(device, dtype=torch.float32)
        coords_ctx = batch["coords_ctx"].to(device, dtype=torch.float32)
        ctx_type = batch["ctx_type"].to(device)
        y = batch["y"].to(device)
        y_ctx = batch["y_ctx"].to(device)

        slide_ids = batch["slide_id"]
        idxs = batch["idx"]
        idxs = idxs.detach().cpu().tolist()

        y_hat_spot, y_hat_diff = model(
            x_img=x_img,
            x_ctx_img=x_ctx,
            coords=coords,
            coords_ctx=coords_ctx,
            ctx_type=ctx_type if args.model.add_type_embed else None,
            ctx_mask=None,
        )

        loss, loss_spot, loss_diff = criterion(
            y_hat_spot=y_hat_spot,
            y_true_spot=y,
            y_hat_diff=y_hat_diff,
            y_true_ctx=y_ctx,
            pair_mask=None,
        )


        pred_cpu = y_hat_spot.detach().cpu()
        y_cpu = y.detach().cpu()

        all_preds.append(pred_cpu)
        all_y.append(y_cpu)

        bs = y.shape[0]
        loss_sum += float(loss.item()) * bs
        loss_reg_sum += float(loss_spot.item()) * bs
        loss_diff_sum += float(loss_diff.item()) * bs

        cnt += bs
        log_cnt += bs
        if log_cnt > 1000:
            log_cnt = 0
            log(args.data.log_path, f"[Eval {batch_idx}] loss={loss_sum / cnt:.6f} spot={loss_reg_sum / cnt:.6f} diff={loss_diff_sum / cnt:.6f}")


        if save_pred:

            for i in range(bs):
                pred_buckets[slide_ids[i]].append((idxs[i], pred_cpu[i]))


    all_preds_cat = torch.cat(all_preds, 0)
    all_y_cat = torch.cat(all_y, 0)

    total_loss = loss_sum / all_preds_cat.shape[0]
    eval_metrics = MetricCalculator.get_metrics(all_preds_cat, all_y_cat)


    if save_pred:
        os.makedirs(args.data.save_pred_dir, exist_ok=True)

        for slide_id, items in pred_buckets.items():

            items.sort(key=lambda t: t[0])

            preds_slide = torch.stack([t[1] for t in items], dim=0)
            out_path = os.path.join(args.data.save_pred_dir, f"{slide_id}.pt")
            torch.save(preds_slide, out_path)

        log(args.data.log_path, f"[TEST] Saved predictions per slide to: {args.data.save_pred_dir}")

    return total_loss, eval_metrics


def main(args):
    set_seed(args.train.seed)

    exp_tag = f"{args.model.name}_{args.loss.name}"
    exp_tag = exp_tag + "_" + args.tag if args.tag is not None and len(args.tag) > 0 else exp_tag

    args.data.base_path = os.path.join(args.data.base_path, args.data.project) + ".pt"
    args.data.save_dir = os.path.join(args.data.save_dir, args.data.project, exp_tag)
    args.data.save_pred_dir = os.path.join(args.data.save_pred_dir, args.data.project, exp_tag)
    args.data.log_path = os.path.join(args.data.save_dir, "log.txt")

    os.makedirs(args.data.save_dir, exist_ok=True)
    log(args.data.log_path, str(args))


    train_ds = STSpotCtxDataset(
        cache_path=args.data.base_path,
        split="train",
        near_k=args.data.near_k,
        far_k=args.data.far_k,
        nn_key=args.data.nn_key,
        far_scope=args.data.far_scope,
    )
    val_ds = STSpotCtxDataset(
        cache_path=args.data.base_path,
        split="val",
        near_k=args.data.near_k,
        far_k=args.data.far_k,
        nn_key=args.data.nn_key,
        far_scope=args.data.far_scope,
    )
    test_ds = STSpotCtxDataset(
        cache_path=args.data.base_path,
        split="test",
        near_k=args.data.near_k,
        far_k=args.data.far_k,
        nn_key=args.data.nn_key,
        far_scope=args.data.far_scope,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.data.batch_size,
        num_workers=args.data.num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.data.batch_size,
        num_workers=args.data.num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.data.batch_size,
        num_workers=args.data.num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )


    args.model.num_genes = train_ds.num_genes if getattr(args.model, "num_genes", None) is None else args.model.num_genes
    log(args.data.log_path, f"[{args.data.project}] Spots: Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = create_model(
        args.model.name,
        input_dim=args.model.img_dim,
        hidden_dim=args.model.hidden_dim,
        feat_dim=args.model.feat_dim,
        num_genes=args.model.num_genes,
        num_layers=args.model.num_layers,
        num_heads=args.model.num_heads,
        dropout=args.model.dropout,
        add_type_embed=args.model.add_type_embed,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay)

    scheduler = None
    if args.train.cosine_scheduler:
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=args.train.warmup_epochs,
            max_epochs=args.train.epochs,
            min_lr=args.train.lr / 10,
        )


    criterion = SpotMSEPlusDiffMSELoss(
        w_spot=args.loss.w_spot,
        w_diff=args.loss.w_diff,
        gene_mean=train_ds.gene_mean.to(device),
        gene_var=train_ds.gene_var.to(device),
        gene_std=train_ds.gene_std.to(device),
        norm_reg=args.train.norm_reg,
        norm_diff=args.train.norm_diff,
    )

    best_pcc = float("-inf")
    last_model_path = os.path.join(args.data.save_dir, "model_last.pth")
    best_model_path = os.path.join(args.data.save_dir, "model_best.pth")

    for epoch in range(1, args.train.epochs + 1):
        train_loss, _ = train_epoch(args, model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = eval_epoch(args, model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        log(args.data.log_path, f"[Epoch {epoch}/{args.train.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        log(args.data.log_path, "PCC M {:.4f} S {:.4f} F {:.4f}  MI M {:.4f} F {:.4f}\n".format(
            val_metrics["PCC_M"], val_metrics["PCC_S"], val_metrics["PCC_F"], val_metrics["MI_M"], val_metrics["MI_F"]
        ))

        torch.save(model.state_dict(), last_model_path)
        if val_metrics["PCC_F"] > best_pcc:
            best_pcc = val_metrics["PCC_F"]
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_metrics = eval_epoch(args, model, test_loader, criterion, device, save_pred=args.loss.name == "N18F72S1D1")

    log(args.data.log_path, f"[TEST] Loss={test_loss:.4f}")
    log(args.data.log_path, str(test_metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="./configs/baseline.yaml")
    parser.add_argument("--data_config", type=str, default=None)
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--loss_config", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")

    cfg = parser.parse_args()


    args = load_config([cfg.base_config, cfg.data_config, cfg.model_config, cfg.loss_config])


    args.tag = cfg.tag

    main(args)
