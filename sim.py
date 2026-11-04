import numpy as np
from scipy.signal import lfilter
import config as C

def lowpass(x, tau):
    a = 1.0 - np.exp(-C.DT / tau)
    return lfilter([a], [1.0, -(1.0 - a)], x, zi=[(1.0 - a) * x[0]])[0]

def p_ss(pin, q, k):
    return np.stack([np.sqrt(np.maximum(pin**2 - k * q**2 * x, 1.0)) for x in C.X_SENSOR])

def unit_noise(rng, T):
    out = np.empty((2, T))
    for i in range(2):
        a = np.empty(T); a[0] = rng.normal()
        e = rng.normal(size=T)
        for n in range(1, T):
            a[n] = C.NOISE_AR_RHO * a[n-1] + np.sqrt(1 - C.NOISE_AR_RHO**2) * e[n]
        out[i] = C.NOISE_AR_STD * a + C.NOISE_WHITE_STD * rng.normal(size=T)
    return out

def make_episode(rng, leak_drop=None, T=C.EPISODE_LEN):
    pin = np.full(T, C.PIN0); q = np.full(T, C.Q0)
    for _ in range(rng.integers(C.N_TRANSIENTS[0], C.N_TRANSIENTS[1] + 1)):
        kind = rng.choice(['pin_step', 'pin_ramp', 'q_step', 'q_pulse'])
        if kind == 'pin_step':
            t0 = rng.integers(100, T - 100)
            pin[t0:] += rng.choice([-1, 1]) * rng.uniform(*C.PIN_STEP)
        elif kind == 'pin_ramp':
            dur = int(rng.integers(*C.PIN_RAMP_DUR)); t0 = rng.integers(100, T - dur - 100)
            tot = rng.choice([-1, 1]) * rng.uniform(*C.PIN_RAMP_TOTAL)
            pin[t0:t0 + dur] += np.linspace(0, tot, dur); pin[t0 + dur:] += tot
        elif kind == 'q_step':
            q[rng.integers(100, T - 100):] *= rng.uniform(*C.Q_STEP)
        else:
            dur = int(rng.integers(*C.Q_PULSE_DUR)); t0 = rng.integers(100, T - dur - 100)
            q[t0:t0 + dur] *= rng.uniform(*C.Q_PULSE_MULT)
    pin = np.clip(pin, 18.0, 22.0); q = np.clip(q, 0.5, 1.5)
    kappa = 1.0 + C.KAPPA_SD * rng.normal()
    p_true = np.stack([lowpass(s, C.TAU_TRUE) for s in p_ss(pin, q, C.K_PIPE * kappa)])
    p_twin = np.stack([lowpass(s, C.TAU_TWIN) for s in p_ss(pin, q, C.K_PIPE)])
    leak = np.zeros((2, T)); label = np.zeros(T, bool); meta = dict(leak=False)
    if leak_drop is not None:
        t0 = int(rng.integers(*C.ONSET_RANGE)); dur = int(rng.integers(*C.DURATION_RANGE))
        gamma = rng.uniform(*C.GAMMA_RANGE); lag2 = int(rng.integers(C.LAG2_RANGE[0], C.LAG2_RANGE[1] + 1))
        trep = min(t0 + dur, T - 1)
        for i, (d, lag) in enumerate([(leak_drop, 0), (gamma * leak_drop, lag2)]):
            tgt = np.zeros(T); tgt[t0 + lag:trep + lag] = d
            leak[i] = lowpass(tgt, C.TAU_LEAK)
        label[t0:trep] = True
        meta = dict(leak=True, t0=t0, trep=trep, drop=float(leak_drop), gamma=float(gamma), lag2=lag2)
    return dict(p_true=p_true, p_twin=p_twin, leak_eff=leak, noise=unit_noise(rng, T), label=label, meta=meta)

def make_dataset(seed, n_ep, n_leak, n_small):
    rng = np.random.default_rng(seed)
    drops = np.concatenate([rng.uniform(*C.SMALL_RANGE, n_small),
                            rng.uniform(*C.LARGE_RANGE, n_leak - n_small)])
    rng.shuffle(drops)
    spec = list(drops) + [None] * (n_ep - n_leak)
    rng.shuffle(spec)
    return [make_episode(np.random.default_rng(rng.integers(2**32)), s) for s in spec]

def residual(ep, k=1):
    return ep['p_true'] - ep['leak_eff'] + k * ep['noise'] - ep['p_twin']
