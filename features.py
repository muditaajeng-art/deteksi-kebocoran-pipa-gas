import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
import config as C

NAMES = [f'{n}_s{i}' for i in (1, 2) for n in
         ('mean', 'std', 'min', 'max', 'last', 'slope', 'delta_last20_first20', 'max_abs_diff')] + \
        ['mean_r1_minus_r2', 'corr_r1_r2', 'std_r1_minus_r2']

def window_features(r):
    """r: (2, T) residual. Return X (n_win, 19) dan indeks akhir tiap jendela."""
    W = C.WINDOW
    w = swv(r, W, axis=1)[:, ::C.STRIDE, :]              # (2, n, W)
    ends = np.arange(W - 1, r.shape[1], C.STRIDE)[:w.shape[1]]
    i = np.arange(W) - (W - 1) / 2
    f = []
    for s in range(2):
        x = w[s]
        f += [x.mean(1), x.std(1), x.min(1), x.max(1), x[:, -1], (x * i).sum(1) / (i**2).sum(),
              x[:, -20:].mean(1) - x[:, :20].mean(1), np.abs(np.diff(x, axis=1)).max(1)]
    d = w[0] - w[1]
    a = w[0] - w[0].mean(1, keepdims=True); b = w[1] - w[1].mean(1, keepdims=True)
    corr = (a * b).sum(1) / (np.sqrt((a**2).sum(1) * (b**2).sum(1)) + 1e-12)
    f += [d.mean(1), corr, d.std(1)]
    return np.stack(f, 1), ends

def threshold_alarm(r, th=C.THRESH, n=C.NCONSEC):
    """Status alarm per sampel: max|r| > th selama n sampel berturut-turut."""
    c = (np.abs(r).max(0) > th)
    out = c.copy()
    for j in range(1, n):
        out[j:] &= c[:-j] if j else c
    out[:n - 1] = False
    return out
