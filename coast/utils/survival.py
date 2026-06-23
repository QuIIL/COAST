import numpy as np


def concordance_index(risks, times, events):
    risks = np.asarray(risks, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)

    if not (risks.shape == times.shape == events.shape):
        raise ValueError("risks, times, and events must have the same shape")

    concordant = 0.0
    comparable = 0.0

    for i in range(len(risks) - 1):
        for j in range(i + 1, len(risks)):
            if times[i] < times[j] and events[i] == 1:
                comparable += 1.0
                if risks[i] > risks[j]:
                    concordant += 1.0
                elif risks[i] == risks[j]:
                    concordant += 0.5
            elif times[j] < times[i] and events[j] == 1:
                comparable += 1.0
                if risks[j] > risks[i]:
                    concordant += 1.0
                elif risks[i] == risks[j]:
                    concordant += 0.5
            elif times[i] == times[j] and events[i] != events[j]:
                event_idx, censored_idx = (i, j) if events[i] == 1 else (j, i)
                comparable += 1.0
                if risks[event_idx] > risks[censored_idx]:
                    concordant += 1.0
                elif risks[event_idx] == risks[censored_idx]:
                    concordant += 0.5

    return float(concordant / comparable) if comparable > 0 else float("nan")
