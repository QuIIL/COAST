from __future__ import annotations
from typing import Dict, Tuple, List, DefaultDict
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset


class STSpotCtxDataset(Dataset):


    def __init__(
        self,
        cache_path: str,
        split: str = "train",
        near_k: int = 18,
        far_k: int = 18,
        nn_key: str = "nn36_idx",
        far_scope: str = "slide",
        eval_far_strata: int = 4,
        seed: int = 47,
        compute_gene_stats: bool = True,
        eps_stats: float = 1e-8,
    ):
        super().__init__()
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        data = cache[split]

        self.split = split
        self.is_train = (split == "train")

        self.x_uni = data["UNI_feats"]
        self.metadata = data["metadata"]
        if nn_key not in data:
            raise KeyError(f"Expected '{nn_key}' in cache[{split}].")
        self.nn_full = data[nn_key]

        self.near_k = int(near_k)
        self.far_k = int(far_k)
        self.K = self.near_k + self.far_k

        if self.near_k < 0 or self.far_k < 0:
            raise ValueError("near_k and far_k must be non-negative.")
        if self.K == 0:
            raise ValueError("Need at least one context (near_k+far_k > 0).")
        if self.near_k > 0 and self.nn_full.shape[1] < self.near_k:
            raise ValueError(
                f"nn_full has only {self.nn_full.shape[1]} neighbors, need near_k={self.near_k}"
            )

        self.far_scope = far_scope.lower()
        if self.far_scope not in {"slide", "global"}:
            raise ValueError("far_scope must be one of {'slide','global'}")

        self.eval_far_strata = int(eval_far_strata)
        if self.eval_far_strata <= 0:
            raise ValueError("eval_far_strata must be > 0.")
        self.seed = int(seed)


        self.num_genes = cache.get("num_genes", None)


        self.slide_to_indices: DefaultDict[str, List[int]] = defaultdict(list)
        for i, meta in enumerate(self.metadata):
            self.slide_to_indices[meta[3]].append(i)


        self.global_indices: List[int] = list(range(len(self.metadata)))


        if compute_gene_stats and len(self.metadata) > 0:
            g0 = torch.as_tensor(self.metadata[0][4], dtype=torch.float32)
            G = int(g0.numel())
            self.num_genes = G if self.num_genes is None else self.num_genes

            mean = torch.zeros(G, dtype=torch.float32)
            m2 = torch.zeros(G, dtype=torch.float32)
            n = 0

            for meta in self.metadata:
                y = torch.as_tensor(meta[4], dtype=torch.float32)
                if y.numel() != G:
                    raise ValueError(f"Inconsistent gene dim: expected {G}, got {y.numel()}")
                n += 1
                delta = y - mean
                mean += delta / n
                delta2 = y - mean
                m2 += delta * delta2

            var = m2 / max(n - 1, 1)
            std = torch.sqrt(var + eps_stats)

            self.gene_mean = mean
            self.gene_var = var
            self.gene_std = std
        else:
            self.gene_mean = None
            self.gene_var = None
            self.gene_std = None

    def __len__(self) -> int:
        return len(self.metadata)

    def _get_gene_and_coords(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        meta = self.metadata[idx]
        slide_id = meta[3]
        gene_exp = torch.as_tensor(meta[4], dtype=torch.float32)
        coords = torch.as_tensor(meta[5], dtype=torch.float32)
        return gene_exp, coords, slide_id

    def _get_far_pool(self, slide_id: str) -> List[int]:
        if self.far_scope == "slide":
            return self.slide_to_indices[slide_id]
        return self.global_indices

    def _sample_far_train_uniform(self, idx: int, slide_id: str, near_idx: torch.Tensor) -> torch.Tensor:

        if self.far_k == 0:
            return torch.empty(0, dtype=torch.long)

        pool = self._get_far_pool(slide_id)
        exclude = set([int(idx)]) | set(map(int, near_idx.tolist()))
        candidates = [j for j in pool if j not in exclude]

        if len(candidates) == 0:
            candidates = [j for j in pool if j != int(idx)]
            if len(candidates) == 0:
                candidates = [int(idx)]

        rand = torch.randint(0, len(candidates), (self.far_k,))
        far_idx = torch.tensor([candidates[i] for i in rand.tolist()], dtype=torch.long)
        return far_idx

    def _sample_far_eval_stratified(self, idx: int, slide_id: str, near_idx: torch.Tensor) -> torch.Tensor:


        if self.far_k == 0:
            return torch.empty(0, dtype=torch.long)

        if self.far_scope != "slide":
            raise ValueError("grid_regular eval far requires far_scope='slide'")

        pool = self.slide_to_indices[slide_id]
        exclude = set([int(idx)]) | set(map(int, near_idx.tolist()))
        candidates = [j for j in pool if j not in exclude]

        if len(candidates) == 0:
            candidates = [j for j in pool if j != int(idx)]
            if len(candidates) == 0:
                return torch.tensor([int(idx)] * self.far_k, dtype=torch.long)


        cand_coords = torch.tensor([self.metadata[j][5] for j in candidates], dtype=torch.float32)
        x = cand_coords[:, 0].contiguous()
        y = cand_coords[:, 1].contiguous()


        x0, x1 = float(x.min()), float(x.max())
        y0, y1 = float(y.min()), float(y.max())
        w = max(x1 - x0, 1e-6)
        h = max(y1 - y0, 1e-6)
        aspect = w / h


        K = int(self.far_k)


        gx = max(1, int(round((K * aspect) ** 0.5)))
        gy = max(1, int(np.ceil(K / gx)))

        while gx * gy < K:
            gy += 1


        xs = torch.linspace(x0, x1, steps=gx + 1).contiguous()
        ys = torch.linspace(y0, y1, steps=gy + 1).contiguous()


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


        picked: List[int] = [g for g in best if g != -1]


        if len(picked) > K:
            picked = picked[:K]


        if len(picked) < K:
            add2 = [g for g in second if g != -1 and g not in picked]
            need = K - len(picked)
            picked.extend(add2[:need])


        if len(picked) < K:
            remain = [g for g in candidates if g not in set(picked)]
            need = K - len(picked)

            remain.sort()
            picked.extend(remain[:need])


        if len(picked) < K:
            picked.extend([int(idx)] * (K - len(picked)))

        return torch.tensor(picked[:K], dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        x_img = self.x_uni[idx]
        y, coords, slide_id = self._get_gene_and_coords(idx)


        near_idx = self.nn_full[idx][: self.near_k].clone() if self.near_k > 0 else torch.empty(0, dtype=torch.long)


        if self.far_k > 0:
            if self.is_train:
                far_idx = self._sample_far_train_uniform(idx, slide_id, near_idx)
            else:
                far_idx = self._sample_far_eval_stratified(idx, slide_id, near_idx)
        else:
            far_idx = torch.empty(0, dtype=torch.long)

        ctx_idx = torch.cat([near_idx, far_idx], dim=0)


        x_ctx = self.x_uni[ctx_idx]


        y_ctx = torch.stack([torch.as_tensor(self.metadata[i][4], dtype=torch.float32) for i in ctx_idx.tolist()], dim=0)
        coords_ctx = torch.stack([torch.as_tensor(self.metadata[i][5], dtype=torch.float32) for i in ctx_idx.tolist()], dim=0)


        ctx_type = torch.zeros(self.K, dtype=torch.long)
        if self.far_k > 0:
            ctx_type[self.near_k:] = 1

        return {
            "x_img": x_img,
            "x_ctx": x_ctx,
            "coords": coords,
            "y": y,
            "y_ctx": y_ctx,
            "coords_ctx": coords_ctx,
            "ctx_type": ctx_type,
            "slide_id": slide_id,
            "idx": idx,
        }
