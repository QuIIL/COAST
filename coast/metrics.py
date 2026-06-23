import numpy as np
import torch
from scipy.ndimage import convolve
from scipy.spatial.distance import jensenshannon
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score


class MetricCalculator:


    @staticmethod
    def _finite_mask(a: np.ndarray, b: np.ndarray):
        m = np.isfinite(a) & np.isfinite(b)
        return m

    @staticmethod
    def _infer_step_1d(vals: np.ndarray, eps: float = 1e-12) -> float:


        u = np.unique(vals.astype(np.float64))
        if u.size <= 1:
            return 1.0
        d = np.diff(np.sort(u))
        d = d[np.isfinite(d) & (d > eps)]
        if d.size == 0:
            return 1.0
        return float(np.median(d))

    @staticmethod
    def _coords_to_grid_indices(coords: np.ndarray, step: tuple | None = None):


        coords = MetricCalculator._to_numpy(coords)
        assert coords.ndim == 2 and coords.shape[1] == 2, "coords must be [N,2]"

        x = coords[:, 0]
        y = coords[:, 1]

        x0 = float(np.nanmin(x))
        y0 = float(np.nanmin(y))

        if step is None:
            sx = MetricCalculator._infer_step_1d(x)
            sy = MetricCalculator._infer_step_1d(y)
        else:
            sx, sy = float(step[0]), float(step[1])

        sx = sx if np.isfinite(sx) and sx > 0 else 1.0
        sy = sy if np.isfinite(sy) and sy > 0 else 1.0

        ix = np.rint((x - x0) / sx).astype(np.int64)
        iy = np.rint((y - y0) / sy).astype(np.int64)

        ix -= ix.min()
        iy -= iy.min()

        W = int(ix.max() + 1) if ix.size else 0
        H = int(iy.max() + 1) if iy.size else 0
        return ix, iy, W, H, (sx, sy), (x0, y0)

    @staticmethod
    def _scatter_to_grid(values: np.ndarray, ix: np.ndarray, iy: np.ndarray, H: int, W: int):


        grid = np.zeros((H, W), dtype=np.float32)
        cnt = np.zeros((H, W), dtype=np.float32)

        for v, xg, yg in zip(values.astype(np.float32), ix, iy):
            if not np.isfinite(v):
                continue
            grid[yg, xg] += v
            cnt[yg, xg] += 1.0

        mask = (cnt > 0).astype(np.float32)
        grid = np.where(cnt > 0, grid / np.maximum(cnt, 1e-8), 0.0).astype(np.float32)
        return grid, mask

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray, eps_std: float = 1e-6) -> float:
        if np.std(a) < eps_std or np.std(b) < eps_std:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def _hist_range(x: np.ndarray, y: np.ndarray, clip_quantiles=(0.0, 1.0)):

        loq, hiq = clip_quantiles
        if loq > 0.0 or hiq < 1.0:
            xy = np.concatenate([x.reshape(-1), y.reshape(-1)])
            lo = float(np.quantile(xy, loq))
            hi = float(np.quantile(xy, hiq))
        else:
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
            lo, hi = float(x.min()), float(x.min() + 1.0)
        return lo, hi


    @staticmethod
    def PCC_F(x, y, eps_std: float = 1e-6, skip_constant: bool = False) -> float:


        x = MetricCalculator._to_numpy(x)
        y = MetricCalculator._to_numpy(y)
        vals = []
        for g in range(x.shape[1]):
            a = x[:, g]
            b = y[:, g]
            if np.std(a) < eps_std or np.std(b) < eps_std:
                if not skip_constant:
                    vals.append(0.0)
                continue
            vals.append(float(np.corrcoef(a, b)[0, 1]))
        return float(np.mean(vals)) if len(vals) else 0.0

    @staticmethod
    def PCC_S(x, y, eps_std: float = 1e-6, skip_constant: bool = False) -> float:


        x = MetricCalculator._to_numpy(x)
        y = MetricCalculator._to_numpy(y)
        vals = []
        for i in range(x.shape[0]):
            a = x[i]
            b = y[i]
            if np.std(a) < eps_std or np.std(b) < eps_std:
                if not skip_constant:
                    vals.append(0.0)
                continue
            vals.append(float(np.corrcoef(a, b)[0, 1]))
        return float(np.mean(vals)) if len(vals) else 0.0

    @staticmethod
    def PCC_M(x, y, eps_std: float = 1e-6) -> float:


        x = MetricCalculator._to_numpy(x).reshape(-1)
        y = MetricCalculator._to_numpy(y).reshape(-1)
        if np.std(x) < eps_std or np.std(y) < eps_std:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])


    @staticmethod
    def MI_F(
            x,
            y,
            n_neighbors: int = 5,
            random_state: int = 0,
            max_samples: int | None = None,
            add_jitter: float = 0.0,
    ) -> float:


        x = MetricCalculator._to_numpy(x)
        y = MetricCalculator._to_numpy(y)

        N, G = y.shape
        mi_list = []

        rng = np.random.RandomState(random_state)

        for g in range(G):
            xv = x[:, g].astype(np.float64)
            yv = y[:, g].astype(np.float64)

            m = MetricCalculator._finite_mask(xv, yv)
            xv = xv[m]
            yv = yv[m]


            if (max_samples is not None) and (xv.size > max_samples):
                idx = rng.choice(xv.size, size=max_samples, replace=False)
                xv = xv[idx]
                yv = yv[idx]


            if add_jitter and add_jitter > 0:
                xv = xv + rng.normal(0.0, add_jitter, size=xv.shape)
                yv = yv + rng.normal(0.0, add_jitter, size=yv.shape)


            mi_g = mutual_info_regression(
                X=xv.reshape(-1, 1),
                y=yv,
                n_neighbors=n_neighbors,
                random_state=random_state,
                discrete_features=False,
            )[0]
            mi_list.append(float(mi_g))

        return float(np.mean(mi_list)) if len(mi_list) else 0.0

    @staticmethod
    def MI_M(
            x,
            y,
            n_neighbors: int = 5,
            random_state: int = 0,
            max_samples: int | None = 20000,
            add_jitter: float = 0.0,
    ) -> float:


        x = MetricCalculator._to_numpy(x).reshape(-1).astype(np.float64)
        y = MetricCalculator._to_numpy(y).reshape(-1).astype(np.float64)

        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]

        if x.size < max(50, n_neighbors + 5):
            return 0.0

        rng = np.random.RandomState(random_state)

        if (max_samples is not None) and (x.size > max_samples):
            idx = rng.choice(x.size, size=max_samples, replace=False)
            x = x[idx]
            y = y[idx]

        if add_jitter and add_jitter > 0:
            x = x + rng.normal(0.0, add_jitter, size=x.shape)
            y = y + rng.normal(0.0, add_jitter, size=y.shape)

        mi = mutual_info_regression(
            X=x.reshape(-1, 1),
            y=y,
            n_neighbors=n_neighbors,
            random_state=random_state,
            discrete_features=False,
        )[0]
        return float(mi)


    @staticmethod
    def NRMSE_F(x, y, eps_range: float = 1e-6) -> float:


        x = MetricCalculator._to_numpy(x)
        y = MetricCalculator._to_numpy(y)

        vals = []
        for g in range(y.shape[1]):
            rmse = np.sqrt(np.mean((x[:, g] - y[:, g]) ** 2))
            denom = float(np.max(y[:, g]) - np.min(y[:, g]))
            if denom < eps_range:
                vals.append(0.0)
            else:
                vals.append(float(rmse / denom))
        return float(np.mean(vals)) if len(vals) else 0.0

    @staticmethod
    def NRMSE_M(x, y, eps_range: float = 1e-6) -> float:


        x = MetricCalculator._to_numpy(x)
        y = MetricCalculator._to_numpy(y)
        rmse = np.sqrt(np.mean((x - y) ** 2))
        denom = float(np.max(y) - np.min(y))
        if denom < eps_range:
            return 0.0
        return float(rmse / denom)


    @staticmethod
    def AUC_zero_vs_nonzero(x, y) -> float:


        x = MetricCalculator._to_numpy(x)
        y = MetricCalculator._to_numpy(y)

        auc_list = []
        for g in range(y.shape[1]):
            labels = (y[:, g] > 0).astype(np.int32)
            if len(np.unique(labels)) < 2:
                auc_list.append(0.0)
                continue
            try:
                auc_list.append(float(roc_auc_score(labels, x[:, g])))
            except Exception:
                auc_list.append(0.0)
        return float(np.mean(auc_list)) if len(auc_list) else 0.0

    @staticmethod
    def AUC_quantile(x, y, q: float = 0.5) -> float:


        x = MetricCalculator._to_numpy(x)
        y = MetricCalculator._to_numpy(y)

        auc_list = []
        for g in range(y.shape[1]):
            thresh = float(np.quantile(y[:, g], q))
            labels = (y[:, g] > thresh).astype(np.int32)
            if len(np.unique(labels)) < 2:
                auc_list.append(0.0)
                continue
            try:
                auc_list.append(float(roc_auc_score(labels, x[:, g])))
            except Exception:
                auc_list.append(0.0)
        return float(np.mean(auc_list)) if len(auc_list) else 0.0


    @staticmethod
    def JSDIV_M(x, y, bins: int = 64, eps: float = 1e-8, clip_quantiles=(0.0, 1.0)) -> float:


        x = MetricCalculator._to_numpy(x).reshape(-1)
        y = MetricCalculator._to_numpy(y).reshape(-1)

        lo, hi = MetricCalculator._hist_range(x, y, clip_quantiles=clip_quantiles)

        p_cnt, _ = np.histogram(x, bins=bins, range=(lo, hi), density=False)
        q_cnt, _ = np.histogram(y, bins=bins, range=(lo, hi), density=False)

        p = p_cnt.astype(np.float64) + eps
        q = q_cnt.astype(np.float64) + eps
        p /= p.sum()
        q /= q.sum()

        return float(jensenshannon(p, q) ** 2)

    @staticmethod
    def SSIM_spatial_per_gene(
            x,
            y,
            coords,
            win_size: int = 7,
            K1: float = 0.01,
            K2: float = 0.03,
            eps: float = 1e-8,
            step: tuple | None = None,
            clip_quantiles: tuple = (0.0, 1.0),
            return_per_gene: bool = False,
    ):


        x = MetricCalculator._to_numpy(x)
        y = MetricCalculator._to_numpy(y)
        coords = np.asarray(coords)

        assert x.shape == y.shape, "x and y must have same shape [N,G]"
        assert coords.shape[0] == x.shape[0], "coords must have same N as x/y"
        assert coords.shape[1] == 2, "coords must be [N,2]"

        N, G = x.shape
        if N == 0 or G == 0:
            return (0.0, np.zeros((0,), dtype=np.float32)) if return_per_gene else 0.0


        if win_size < 1:
            raise ValueError("win_size must be >= 1")
        if win_size % 2 == 0:
            win_size += 1


        ix, iy, W, H, (sx, sy), (x0, y0) = MetricCalculator._coords_to_grid_indices(coords, step=step)
        if H == 0 or W == 0:
            return (0.0, np.zeros((G,), dtype=np.float32)) if return_per_gene else 0.0


        kernel = np.ones((win_size, win_size), dtype=np.float32)

        per_gene = np.zeros((G,), dtype=np.float32)

        for g in range(G):
            Xg, Mg = MetricCalculator._scatter_to_grid(x[:, g], ix, iy, H, W)
            Yg, _ = MetricCalculator._scatter_to_grid(y[:, g], ix, iy, H, W)


            m_sum = convolve(Mg, kernel, mode="constant", cval=0.0)
            valid = m_sum > 0.0

            if not np.any(valid):
                per_gene[g] = 0.0
                continue


            X_m = Xg * Mg
            Y_m = Yg * Mg
            XY_m = (Xg * Yg) * Mg
            X2_m = (Xg * Xg) * Mg
            Y2_m = (Yg * Yg) * Mg

            sum_x = convolve(X_m, kernel, mode="constant", cval=0.0)
            sum_y = convolve(Y_m, kernel, mode="constant", cval=0.0)
            sum_xy = convolve(XY_m, kernel, mode="constant", cval=0.0)
            sum_x2 = convolve(X2_m, kernel, mode="constant", cval=0.0)
            sum_y2 = convolve(Y2_m, kernel, mode="constant", cval=0.0)

            mu_x = sum_x / (m_sum + eps)
            mu_y = sum_y / (m_sum + eps)

            var_x = sum_x2 / (m_sum + eps) - mu_x * mu_x
            var_y = sum_y2 / (m_sum + eps) - mu_y * mu_y
            cov_xy = sum_xy / (m_sum + eps) - mu_x * mu_y

            var_x = np.maximum(var_x, 0.0)
            var_y = np.maximum(var_y, 0.0)


            xv = Xg[Mg > 0]
            yv = Yg[Mg > 0]
            if xv.size == 0 or yv.size == 0:
                per_gene[g] = 0.0
                continue

            loq, hiq = clip_quantiles
            if loq > 0.0 or hiq < 1.0:
                xyv = np.concatenate([xv.reshape(-1), yv.reshape(-1)])
                lo = float(np.quantile(xyv, loq))
                hi = float(np.quantile(xyv, hiq))
                L = max(hi - lo, 1e-6)
            else:
                L = float(max(xv.max(), yv.max()) - min(xv.min(), yv.min()))
                L = max(L, 1e-6)

            C1 = (K1 * L) ** 2
            C2 = (K2 * L) ** 2

            num = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
            den = (mu_x * mu_x + mu_y * mu_y + C1) * (var_x + var_y + C2)

            ssim_map = np.zeros_like(mu_x, dtype=np.float32)
            ssim_map[valid] = (num[valid] / (den[valid] + eps)).astype(np.float32)


            per_gene[g] = float(np.mean(ssim_map[valid]))

        mean_ssim = float(np.mean(per_gene)) if G > 0 else 0.0
        return (mean_ssim, per_gene) if return_per_gene else mean_ssim

    @staticmethod
    def get_metrics(all_preds, all_y):
        return {
            "PCC_F": MetricCalculator.PCC_F(all_preds, all_y),
            "PCC_S": MetricCalculator.PCC_S(all_preds, all_y),
            "PCC_M": MetricCalculator.PCC_M(all_preds, all_y),
            "MI_F": MetricCalculator.MI_F(all_preds, all_y),
            "MI_M": MetricCalculator.MI_M(all_preds, all_y),
            "NRMSE_F": MetricCalculator.NRMSE_F(all_preds, all_y),
            "NRMSE_M": MetricCalculator.NRMSE_M(all_preds, all_y),
            "AUC_0vNZ": MetricCalculator.AUC_zero_vs_nonzero(all_preds, all_y),
            "AUC_Q50": MetricCalculator.AUC_quantile(all_preds, all_y, q=0.5),
            "JSDIV_M": MetricCalculator.JSDIV_M(all_preds, all_y, bins=64, clip_quantiles=(0.01, 0.99)),
        }
