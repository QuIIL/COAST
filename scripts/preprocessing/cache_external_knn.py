import os
import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm


def build_knn_gridhash_deterministic(coords_int: np.ndarray, k: int = 36, step: int = 256, max_r: int = 50):


    N = coords_int.shape[0]
    if N == 0:
        return np.zeros((0, k), dtype=np.int64)
    if N == 1:
        return np.zeros((1, k), dtype=np.int64)


    qx = coords_int[:, 0] // step
    qy = coords_int[:, 1] // step


    cell2idx = {}
    for i in range(N):
        key = (int(qx[i]), int(qy[i]))
        cell2idx.setdefault(key, []).append(i)


    offsets = []
    for r in range(0, max_r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if r == 0:
                    if dy == 0 and dx == 0:
                        offsets.append((dy, dx))
                else:

                    if max(abs(dx), abs(dy)) == r:
                        offsets.append((dy, dx))

    nn = np.empty((N, k), dtype=np.int64)

    coords64 = coords_int.astype(np.int64)

    for i in range(N):
        xi, yi = coords64[i, 0], coords64[i, 1]
        cxi, cyi = int(qx[i]), int(qy[i])

        cand = []

        for dy, dx in offsets:
            key = (cxi + dx, cyi + dy)
            if key in cell2idx:
                cand.extend(cell2idx[key])

            if len(cand) >= (k + 8):
                break


        cand = [j for j in cand if j != i]


        if len(cand) < k:
            cand = [j for j in range(N) if j != i]


        cj = np.array(cand, dtype=np.int64)
        dx = coords64[cj, 0] - xi
        dy = coords64[cj, 1] - yi
        sq = dx * dx + dy * dy


        key = sq * (N + 1) + cj
        order = np.argsort(key, kind="mergesort")
        picked = cj[order][:k]


        if picked.shape[0] < k:
            pad = np.repeat(picked[-1] if picked.shape[0] > 0 else i, k - picked.shape[0])
            picked = np.concatenate([picked, pad], axis=0)

        nn[i] = picked

    return nn


@torch.no_grad()
def build_knn_cdist_blockwise(coords_int: np.ndarray, k: int = 36, block: int = 2048, device: str = "cuda"):


    coords = torch.as_tensor(coords_int, dtype=torch.int64)
    N = coords.shape[0]
    if N == 0:
        return np.zeros((0, k), dtype=np.int64)
    if N == 1:
        return np.zeros((1, k), dtype=np.int64)

    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    coords_d = coords.to(dev)

    out = torch.empty((N, k), dtype=torch.int64, device="cpu")
    BIG = torch.iinfo(torch.int64).max
    j_all = torch.arange(N, dtype=torch.int64, device=dev).unsqueeze(0)

    for s in range(0, N, block):
        e = min(N, s + block)
        q = coords_d[s:e]

        dx = q[:, 0].unsqueeze(1) - coords_d[:, 0].unsqueeze(0)
        dy = q[:, 1].unsqueeze(1) - coords_d[:, 1].unsqueeze(0)
        sq = dx * dx + dy * dy

        key = sq * (N + 1) + j_all.expand(e - s, N)


        rows = torch.arange(e - s, device=dev)
        cols = torch.arange(s, e, device=dev)
        key[rows, cols] = BIG

        kk = min(k, N - 1)
        nn = torch.topk(key, k=kk, dim=1, largest=False).indices

        if kk < k:
            pad = nn[:, -1:].repeat(1, k - kk)
            nn = torch.cat([nn, pad], dim=1)

        out[s:e] = nn.detach().cpu()

    return out.numpy()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)


    feature_paths = []
    for root, _, files in os.walk(args.base_dir):
        for fn in files:
            if fn.endswith(".h5"):
                slide_id = fn[:-3]
                if args.require_dx and ("DX" not in slide_id):
                    continue
                feature_paths.append((slide_id, os.path.join(root, fn)))

    feature_paths.sort()
    print(f"[FOUND] slides={len(feature_paths)}")

    for slide_id, h5_path in tqdm(feature_paths):
        out_path = os.path.join(args.out_dir, f"{slide_id}.h5" if args.write_new_h5 else f"{slide_id}.nn{args.k}.pt")

        if args.skip_existing and os.path.exists(out_path):
            continue

        try:
            with h5py.File(h5_path, "r") as f:
                coords = f["coords"][:]


            if slide_id == "TCGA-J4-A67L-01Z-00-DX1.4B2B89CD-B390-488F-AE3F-9E81E6D860AD":
                mask = coords[:, 0] > 81011
                coords = coords[mask]
            if slide_id == "TCGA-J2-8192-01Z-00-DX1.A784F381-7906-480F-99A1-0B88005953A0":
                mask = coords[:, 0] < 95676
                coords = coords[mask]
            if slide_id == "TCGA-49-6744-01Z-00-DX3.fa64d744-6caa-4f62-8689-33da9c312838":
                mask = coords[:, 0] < 86332
                coords = coords[mask]

            coords_int = coords.astype(np.int64)

            if args.backend == "grid":
                nn = build_knn_gridhash_deterministic(coords_int, k=args.k, step=args.step, max_r=args.max_r)
            else:
                nn = build_knn_cdist_blockwise(coords_int, k=args.k, block=args.block, device=args.device)

            if args.write_new_h5:

                with h5py.File(out_path, "w") as fo:
                    fo.create_dataset(f"nn{args.k}_idx", data=nn, compression="gzip")
                    if args.store_coords:
                        fo.create_dataset("coords", data=coords_int, compression="gzip")
            else:
                torch.save({"slide_id": slide_id, "nn_idx": torch.as_tensor(nn, dtype=torch.long)}, out_path)

        except Exception as e:
            print(f"[WARN] {slide_id} failed: {repr(e)}", flush=True)
            if args.fail_log:
                with open(args.fail_log, "a") as f:
                    f.write(f"{slide_id}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="dir containing TCGA *.h5 with coords/features")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--k", type=int, default=36)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--require_dx", action="store_true", help="filter slide_id contains 'DX'")


    ap.add_argument("--backend", type=str, default="grid", choices=["grid", "cdist"])
    ap.add_argument("--step", type=int, default=256, help="grid stride in pixels (for grid backend)")
    ap.add_argument("--max_r", type=int, default=50, help="max grid radius search (grid backend)")
    ap.add_argument("--block", type=int, default=2048, help="cdist block rows")
    ap.add_argument("--device", type=str, default="cuda")


    ap.add_argument("--write_new_h5", action="store_true", help="write per-slide h5 containing nn_idx")
    ap.add_argument("--store_coords", action="store_true")
    ap.add_argument("--fail_log", type=str, default=None)

    args = ap.parse_args()
    main(args)
