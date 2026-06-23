from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class TCGAExternalSlideListDataset(Dataset):


    def __init__(self, base_dir: str, annotation_path: Optional[str] = None, require_dx: bool = True):
        super().__init__()
        self.data: List[Tuple[str, str, int]] = []

        feature_paths = []
        for path, _, files in os.walk(base_dir):
            for filename in files:
                if not filename.endswith(".h5"):
                    continue
                slide_id = filename[:-3]
                if require_dx and ("DX" not in slide_id):
                    continue
                feature_paths.append((slide_id, os.path.join(path, filename)))
        feature_paths.sort()

        cnt = [0, 0, 0, 0, 0]
        if annotation_path is not None and os.path.exists(annotation_path):
            with h5py.File(annotation_path, "r") as h5:
                for slide_id, feature_path in feature_paths:
                    g = h5["slides"][slide_id]
                    gg = int(g["gleason_grade_group"][()]) - 1
                    self.data.append((slide_id, feature_path, gg))
                    if 0 <= gg < len(cnt):
                        cnt[gg] += 1
        else:
            for slide_id, feature_path in feature_paths:
                self.data.append((slide_id, feature_path, 0))

        print(f"Loaded {len(self.data)} slides ({cnt})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]



class TCGAExternalSlideDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        annotation_path: Optional[str] = None,
        require_dx: bool = True,
        fix_bad_slides: bool = True,
    ):
        super().__init__()
        self.slides = TCGAExternalSlideListDataset(
            base_dir=base_dir,
            annotation_path=annotation_path,
            require_dx=require_dx,
        ).data
        self.fix_bad_slides = fix_bad_slides

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, index: int):
        slide_id, feature_path, grade_group = self.slides[index]
        with h5py.File(feature_path, "r") as handle:
            coords = handle["coords"][:]
            features = handle["features"][:]
        if self.fix_bad_slides:
            coords, features = apply_known_masks(slide_id, coords, features)
        return slide_id, coords, features, grade_group

def apply_known_masks(slide_id: str, coords: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if slide_id == "TCGA-J4-A67L-01Z-00-DX1.4B2B89CD-B390-488F-AE3F-9E81E6D860AD":
        mask = coords[:, 0] > 81011
        coords, features = coords[mask], features[mask]
    if slide_id == "TCGA-J2-8192-01Z-00-DX1.A784F381-7906-480F-99A1-0B88005953A0":
        mask = coords[:, 0] < 95676
        coords, features = coords[mask], features[mask]
    if slide_id == "TCGA-49-6744-01Z-00-DX3.fa64d744-6caa-4f62-8689-33da9c312838":
        mask = coords[:, 0] < 86332
        coords, features = coords[mask], features[mask]
    return coords, features


def far_eval_grid_regular(coords: torch.Tensor, K: int, exclude: set[int]) -> List[int]:
    N = coords.shape[0]
    if K <= 0:
        return []
    candidates = [j for j in range(N) if j not in exclude]
    if len(candidates) == 0:
        candidates = [j for j in range(N) if j != next(iter(exclude))] if N > 1 else [0]

    cand_coords = coords[candidates]
    x = cand_coords[:, 0].contiguous()
    y = cand_coords[:, 1].contiguous()

    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)
    aspect = w / h

    gx = max(1, int(round((K * aspect) ** 0.5)))
    gy = max(1, int(np.ceil(K / gx)))
    while gx * gy < K:
        gy += 1

    xs = torch.linspace(x0, x1, steps=gx + 1)
    ys = torch.linspace(y0, y1, steps=gy + 1)

    cx = torch.bucketize(x, xs[1:-1]).clamp(0, gx - 1)
    cy = torch.bucketize(y, ys[1:-1]).clamp(0, gy - 1)
    cell = cy * gx + cx
    ncells = gx * gy

    x_cent = (xs[:-1] + xs[1:]) / 2.0
    y_cent = (ys[:-1] + ys[1:]) / 2.0
    cell_cx = x_cent[cx]
    cell_cy = y_cent[cy]
    d2 = (x - cell_cx).pow(2) + (y - cell_cy).pow(2)

    best = [-1] * ncells
    best_d2 = [float("inf")] * ncells
    second = [-1] * ncells
    second_d2 = [float("inf")] * ncells

    for i, c in enumerate(cell.tolist()):
        gi = candidates[i]
        di = float(d2[i].item())
        if di < best_d2[c]:
            second[c], second_d2[c] = best[c], best_d2[c]
            best[c], best_d2[c] = gi, di
        elif di < second_d2[c]:
            second[c], second_d2[c] = gi, di

    picked = [g for g in best if g != -1]
    if len(picked) > K:
        picked = picked[:K]

    if len(picked) < K:
        add2 = [g for g in second if g != -1 and g not in picked]
        need = K - len(picked)
        picked.extend(add2[:need])

    if len(picked) < K:
        remain = [g for g in candidates if g not in set(picked)]
        remain.sort()
        need = K - len(picked)
        picked.extend(remain[:need])

    if len(picked) < K:
        picked.extend([picked[-1] if picked else 0] * (K - len(picked)))

    return picked[:K]


class TCGAExternalCOASTSpotInSlideDataset(Dataset):


    def __init__(
        self,
        slide_id: str,
        feature_path: str,
        nn_dir: str,
        nn_key: str = "nn36_idx",
        split: str = "test",
        near_k: int = 18,
        far_k: int = 72,
        num_genes: int = 128,
        infer_mode: bool = True,
        fix_bad_slides: bool = True,
        seed: int = 47,
    ):
        super().__init__()
        self.slide_id = slide_id
        self.feature_path = feature_path
        self.nn_dir = nn_dir
        self.nn_key = nn_key
        self.split = split
        self.is_train = (split == "train")
        self.near_k = int(near_k)
        self.far_k = int(far_k)
        self.K = self.near_k + self.far_k
        self.num_genes = int(num_genes)
        self.infer_mode = infer_mode
        self.fix_bad_slides = fix_bad_slides
        self.seed = int(seed)

        self.ctx_type = torch.zeros(self.K, dtype=torch.long)
        if self.far_k > 0:
            self.ctx_type[self.near_k:] = 1


        self._loaded = False
        self._coords = None
        self._feats = None
        self._nn = None

        with h5py.File(self.feature_path, "r") as f:
            coords_raw = f["coords"][:]

        if self.fix_bad_slides:
            temp_feats = np.zeros((coords_raw.shape[0], 1))
            coords_masked, _ = apply_known_masks(self.slide_id, coords_raw, temp_feats)
            self._n = int(coords_masked.shape[0])
        else:
            self._n = int(coords_raw.shape[0])

        self._n_raw = int(coords_raw.shape[0])
        self._loaded = False

    def __len__(self):
        return int(self._n)

    def _ensure_loaded(self):
        if self._loaded:
            return

        with h5py.File(self.feature_path, "r") as f:
            coords = f["coords"][:]
            feats = f["features"][:]

        if self.fix_bad_slides:
            coords, feats = apply_known_masks(self.slide_id, coords, feats)

        coords_t = torch.as_tensor(coords, dtype=torch.float32)
        feats_t = torch.as_tensor(feats, dtype=torch.float32)

        nn_path = os.path.join(self.nn_dir, f"{self.slide_id}.h5")
        if not os.path.exists(nn_path):
            raise FileNotFoundError(f"nn file not found: {nn_path}")

        with h5py.File(nn_path, "r") as f:
            if self.nn_key not in f:
                raise KeyError(f"Expected '{self.nn_key}' in {nn_path}. keys={list(f.keys())}")
            nn = f[self.nn_key][:].astype(np.int64)

        if nn.shape[0] != feats_t.shape[0]:
            raise ValueError(
                f"[{self.slide_id}] nn rows mismatch: nn={nn.shape[0]} vs feats={feats_t.shape[0]} "
                f"(mask/precompute mismatch?)"
            )
        if self.near_k > 0 and nn.shape[1] < self.near_k:
            raise ValueError(f"[{self.slide_id}] nn cols too small: nnK={nn.shape[1]} < near_k={self.near_k}")

        self._coords = coords_t
        self._feats = feats_t
        self._nn = nn
        self._n = int(feats_t.shape[0])
        self._loaded = True

    def __getitem__(self, local_i: int) -> Dict[str, torch.Tensor]:
        self._ensure_loaded()

        coords: torch.Tensor = self._coords
        feats: torch.Tensor = self._feats
        nn: np.ndarray = self._nn

        N, D = feats.shape
        if not (0 <= local_i < N):
            raise IndexError(f"local_i out of range: {local_i} / N={N} slide={self.slide_id}")

        near = nn[local_i, : self.near_k] if self.near_k > 0 else np.zeros((0,), dtype=np.int64)
        near_idx = torch.as_tensor(near, dtype=torch.long)

        if self.far_k > 0:
            excl = set([int(local_i)]) | set(map(int, near_idx.tolist()))
            if self.is_train:
                wi = torch.utils.data.get_worker_info()
                wid = wi.id if wi is not None else 0
                g = torch.Generator()
                g.manual_seed(self.seed + 1000 * wid + int(local_i))

                pool = [j for j in range(N) if j not in excl]
                if len(pool) == 0:
                    pool = [j for j in range(N) if j != int(local_i)]
                if len(pool) == 0:
                    pool = [int(local_i)]

                ridx = torch.randint(0, len(pool), (self.far_k,), generator=g)
                far_idx = torch.tensor([pool[int(t)] for t in ridx.tolist()], dtype=torch.long)
            else:
                far_list = far_eval_grid_regular(coords, self.far_k, excl)
                far_idx = torch.as_tensor(far_list, dtype=torch.long)
        else:
            far_idx = torch.empty(0, dtype=torch.long)

        ctx_idx = torch.cat([near_idx, far_idx], dim=0)

        x_img = feats[local_i]
        x_ctx = feats[ctx_idx]
        c = coords[local_i]
        coords_ctx = coords[ctx_idx]

        if self.infer_mode:
            y = torch.zeros(self.num_genes, dtype=torch.float32)
            y_ctx = torch.zeros(self.K, self.num_genes, dtype=torch.float32)
        else:
            raise ValueError("infer_mode=False not supported without labels.")

        return {
            "x_img": x_img,
            "x_ctx": x_ctx,
            "coords": c,
            "y": y,
            "y_ctx": y_ctx,
            "coords_ctx": coords_ctx,
            "ctx_type": self.ctx_type.clone(),
            "slide_id": self.slide_id,
            "idx": torch.tensor(local_i, dtype=torch.long),
        }
