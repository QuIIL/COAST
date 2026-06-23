import os
import argparse
import numpy as np

from coast.utils import concordance_index


def cindex_permutation_pvalue_pooled(
    risks,
    times,
    events,
    observed_cindex,
    num_perm=10000,
    seed=42,
    alternative="greater",
):


    risks = np.asarray(risks, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)

    if len(risks) < 3 or int(events.sum()) < 1:
        return float("nan"), np.asarray([], dtype=np.float64)

    rng = np.random.RandomState(seed)
    null_vals = np.empty(num_perm, dtype=np.float64)

    for b in range(num_perm):
        perm_risks = rng.permutation(risks)
        null_vals[b] = concordance_index(perm_risks, times, events)

    if alternative == "greater":
        p = (np.sum(null_vals >= observed_cindex) + 1.0) / (num_perm + 1.0)
    elif alternative == "two-sided":
        obs_dist = abs(observed_cindex - 0.5)
        null_dist = np.abs(null_vals - 0.5)
        p = (np.sum(null_dist >= obs_dist) + 1.0) / (num_perm + 1.0)
    else:
        raise ValueError("alternative must be 'greater' or 'two-sided'")

    return float(p), null_vals


def _as_str_array(x):
    out = []
    for v in x:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return np.asarray(out, dtype=object)


def load_oof_predictions(pred_root, n_folds=5):
    risks, times, events = [], [], []
    slide_ids, patient_ids, fold_ids = [], [], []

    for fold in range(n_folds):
        path = os.path.join(pred_root, f"fold_{fold}", "test_predictions.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing fold prediction file: {path}")

        d = np.load(path, allow_pickle=True)

        r = np.asarray(d["risk"], dtype=np.float64)
        t = np.asarray(d["time"], dtype=np.float64)
        e = np.asarray(d["event"], dtype=np.int64)

        if "slide_id" in d:
            sid = _as_str_array(d["slide_id"])
        else:
            sid = np.asarray([f"fold{fold}_sample{i}" for i in range(len(r))], dtype=object)

        if "patient_id" in d:
            pid = _as_str_array(d["patient_id"])
        else:
            pid = sid.copy()

        if "fold_id" in d:
            fid = np.asarray(d["fold_id"], dtype=np.int64)
        else:
            fid = np.full(len(r), fold, dtype=np.int64)

        risks.append(r)
        times.append(t)
        events.append(e)
        slide_ids.append(sid)
        patient_ids.append(pid)
        fold_ids.append(fid)

    return (
        np.concatenate(risks, axis=0),
        np.concatenate(times, axis=0),
        np.concatenate(events, axis=0),
        np.concatenate(slide_ids, axis=0),
        np.concatenate(patient_ids, axis=0),
        np.concatenate(fold_ids, axis=0),
    )


def aggregate_by_patient(
    risks,
    times,
    events,
    slide_ids,
    patient_ids,
    fold_ids,
    method="mean",
):


    if method == "none":
        return risks, times, events, patient_ids, fold_ids

    out_r, out_t, out_e, out_pid, out_fold = [], [], [], [], []

    unique_pids = np.unique(patient_ids)

    for pid in unique_pids:
        idx = np.where(patient_ids == pid)[0]

        r_i = risks[idx]
        t_i = times[idx]
        e_i = events[idx]
        f_i = fold_ids[idx]


        if not np.allclose(t_i, t_i[0]):
            raise ValueError(f"Inconsistent survival time for patient={pid}: {t_i}")
        if not np.all(e_i == e_i[0]):
            raise ValueError(f"Inconsistent event label for patient={pid}: {e_i}")


        if len(np.unique(f_i)) > 1:
            raise ValueError(f"Patient={pid} appears in multiple folds: {np.unique(f_i)}")

        if method == "mean":
            r_agg = float(np.mean(r_i))
        elif method == "median":
            r_agg = float(np.median(r_i))
        elif method == "max":
            r_agg = float(np.max(r_i))
        else:
            raise ValueError("patient_agg must be one of: none, mean, median, max")

        out_r.append(r_agg)
        out_t.append(float(t_i[0]))
        out_e.append(int(e_i[0]))
        out_pid.append(str(pid))
        out_fold.append(int(f_i[0]))

    return (
        np.asarray(out_r, dtype=np.float64),
        np.asarray(out_t, dtype=np.float64),
        np.asarray(out_e, dtype=np.int64),
        np.asarray(out_pid, dtype=object),
        np.asarray(out_fold, dtype=np.int64),
    )


def main(args):
    risks, times, events, slide_ids, patient_ids, fold_ids = load_oof_predictions(
        pred_root=args.pred_root,
        n_folds=args.n_folds,
    )

    n_rows_before = len(risks)
    n_patients_before = len(np.unique(patient_ids))

    risks, times, events, patient_ids, fold_ids = aggregate_by_patient(
        risks=risks,
        times=times,
        events=events,
        slide_ids=slide_ids,
        patient_ids=patient_ids,
        fold_ids=fold_ids,
        method=args.patient_agg,
    )

    cidx = concordance_index(risks, times, events)

    p_value, null_vals = cindex_permutation_pvalue_pooled(
        risks=risks,
        times=times,
        events=events,
        observed_cindex=cidx,
        num_perm=args.pvalue_permutations,
        seed=args.seed,
        alternative=args.cindex_p_alternative,
    )

    n = len(risks)
    n_evt = int(events.sum())
    n_cens = int((events == 0).sum())

    line = (
        f"[OOF_POOLED_TEST] model={args.model_type} "
        f"cindex={cidx:.4f} cindex_p_perm={p_value:.6g} "
        f"alternative={args.cindex_p_alternative} "
        f"n_patients={n} events={n_evt} censored={n_cens} "
        f"n_rows_before_patient_agg={n_rows_before} "
        f"n_unique_patients_before_patient_agg={n_patients_before} "
        f"patient_agg={args.patient_agg} "
        f"n_perm={args.pvalue_permutations} seed={args.seed}"
    )

    os.makedirs(args.pred_root, exist_ok=True)

    summary_path = os.path.join(args.pred_root, "oof_pooled_summary.txt")
    with open(summary_path, "a") as f:
        f.write(line + "\n")

    out_npz = os.path.join(args.pred_root, "oof_pooled_predictions_and_null.npz")
    np.savez(
        out_npz,
        risk=risks,
        time=times,
        event=events,
        patient_id=patient_ids,
        fold_id=fold_ids,
        cindex=np.asarray([cidx], dtype=np.float64),
        p_value=np.asarray([p_value], dtype=np.float64),
        null_cindex=null_vals,
        patient_agg=np.asarray([args.patient_agg], dtype=object),
    )

    print(line)
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {out_npz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Aggregate 5-fold OOF survival predictions and compute pooled C-index p-value")

    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--n_folds", type=int, default=5)

    parser.add_argument("--patient_agg", type=str, default="mean",
                        choices=["none", "mean", "median", "max"])

    parser.add_argument("--pvalue_permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cindex_p_alternative", type=str, default="greater",
                        choices=["greater", "two-sided"])

    args = parser.parse_args()
    main(args)
