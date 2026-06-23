import argparse
from pathlib import Path

import pandas as pd


def collect_slide_info(base_dir: Path):
    import scanpy as sc

    records = []
    for project_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        h5ad_path = project_dir / "adata.h5ad"
        if not h5ad_path.exists():
            continue

        adata = sc.read_h5ad(h5ad_path)
        slide_key = next((key for key in adata.obs_keys() if "slide" in key.lower()), None)
        if slide_key is None:
            continue

        for slide_id in adata.obs[slide_key].unique().tolist():
            mask = adata.obs[slide_key] == slide_id
            width = None
            height = None
            try:
                spatial_meta = adata.uns["spatial"].get(slide_id, {})
                hires_image = spatial_meta.get("images", {}).get("hires")
                scale_factor = spatial_meta.get("scalefactors", {}).get("tissue_hires_scalef")
                if hires_image is not None and scale_factor is not None:
                    hires_height, hires_width = hires_image.shape[:2]
                    width = int(hires_width / scale_factor)
                    height = int(hires_height / scale_factor)
            except (KeyError, TypeError, ValueError):
                pass

            records.append(
                {
                    "project": project_dir.name,
                    "slide": slide_id,
                    "slide_w": width,
                    "slide_h": height,
                    "gene_count": adata.n_vars,
                    "spot_count": int(mask.sum()),
                }
            )
    return pd.DataFrame(records)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main(args):
    dataframe = collect_slide_info(args.base_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output, index=False)
    print(f"Saved {len(dataframe)} rows to {args.output}")


if __name__ == "__main__":
    main(parse_args())
