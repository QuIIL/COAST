import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def find_h5ad_files(h5ad_root: Path):

    files = []
    for d in sorted([p for p in h5ad_root.iterdir() if p.is_dir()]):
        cand_adata = d / "adata.h5ad"
        if cand_adata.exists():
            files.append(cand_adata)
            continue
        stars = sorted(d.glob("*.h5ad"))
        if stars:
            files.append(stars[0])
    return files

def load_coords(adata):

    if "spatial" in adata.obsm:
        arr = adata.obsm["spatial"]
    elif "X_spatial" in adata.obsm:
        arr = adata.obsm["X_spatial"]
    else:
        raise KeyError("No 'spatial' or 'X_spatial' in ad.obsm")
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad spatial shape: {arr.shape}")

    return arr[:, :2]

def export_dataset(h5ad_path: Path, out_root: Path, overwrite: bool = False) -> bool:
    import scanpy as sc

    dataset = h5ad_path.parent.name
    out_path = out_root / f"{dataset}.csv"
    if out_path.exists() and not overwrite:
        print(f"[skip] {dataset}: exists -> {out_path}")
        return True

    print(f"[load] {h5ad_path}")
    ad = sc.read_h5ad(str(h5ad_path))

    try:
        coords = load_coords(ad)
    except Exception as e:
        print(f"[warn] {dataset}: spatial coords missing or invalid ({e}); skipped.", file=sys.stderr)
        return False

    if coords.shape[0] != ad.n_obs:
        print(f"[warn] {dataset}: coords rows ({coords.shape[0]}) != n_obs ({ad.n_obs}); skipped.", file=sys.stderr)
        return False


    spots = ad.obs.index.astype(str).to_list()
    df = pd.DataFrame({
        "unique_id": spots,
        "x": coords[:, 0],
        "y": coords[:, 1],
    })
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[ok]   {dataset}: saved -> {out_path}  (N={len(df)})")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad-root", type=str, default="/data3/SpaRED/h5ad",
                    help="데이터셋 폴더들이 모여있는 상위 경로")
    ap.add_argument("--out-root", type=str, default="/data1/SpaRED/coords",
                    help="좌표 CSV를 저장할 경로(데이터셋별 파일명)")
    ap.add_argument("--overwrite", action="store_true", help="기존 파일이 있어도 덮어쓰기")
    args = ap.parse_args()

    h5ad_root = Path(args.h5ad_root)
    out_root  = Path(args.out_root)
    if not h5ad_root.exists():
        print(f"[error] h5ad root not found: {h5ad_root}", file=sys.stderr)
        sys.exit(1)

    h5ad_files = find_h5ad_files(h5ad_root)
    if not h5ad_files:
        print(f"[error] no .h5ad found under: {h5ad_root}", file=sys.stderr)
        sys.exit(1)

    ok = 0
    for f in h5ad_files:
        try:
            ok += int(export_dataset(f, out_root, overwrite=args.overwrite))
        except Exception as e:
            print(f"[fail] {f}: {e}", file=sys.stderr)

    print(f"\nDone. {ok}/{len(h5ad_files)} datasets exported.")

if __name__ == "__main__":
    main()
