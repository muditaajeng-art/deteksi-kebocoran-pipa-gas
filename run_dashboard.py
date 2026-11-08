import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as C, sim, features

def drop_of(o): return C.DROP_10 + (o - 10) * (C.DROP_100 - C.DROP_10) / 90.0

def run(rng, o, T, leak=True, valve_close=None):
    """Satu simulasi deteksi digital twin: kondisi tunak tanpa transien; katup uji dibuka pada detik ke-60."""
    pin = np.full(T, C.PIN0); q = np.full(T, C.Q0)
    kappa = 1.0 + C.KAPPA_SD * rng.normal()
    p_true = np.stack([sim.lowpass(s, C.TAU_TRUE) for s in sim.p_ss(pin, q, C.K_PIPE * kappa)])
    p_twin = np.stack([sim.lowpass(s, C.TAU_TWIN) for s in sim.p_ss(pin, q, C.K_PIPE)])
    eff = np.zeros((2, T))
    if leak:
        D = drop_of(o); end = T if valve_close is None else valve_close
        for i, (d, lag) in enumerate([(D, 0), (C.DASH_GAMMA * D, C.DASH_LAG2)]):
            tgt = np.zeros(T); tgt[C.DASH_LEAK_AT + lag:end + lag] = d
            eff[i] = sim.lowpass(tgt, C.DASH_TAU_VALVE)
    meas = p_true - eff + sim.unit_noise(rng, T)
    return meas, p_twin, meas - p_twin

def rng_for(*a): return np.random.default_rng(np.random.SeedSequence([C.MAIN_SEED, 2, *a]))
def first_alarm(r, th):
    al = features.threshold_alarm(r, th); idx = np.where(al[C.DASH_LEAK_AT:])[0]
    return None if len(idx) == 0 else int(idx[0]) + 1     # detik sejak katup dibuka (>= 3)

def merge_count(t):
    t = np.sort(t); return 0 if len(t) == 0 else 1 + int((np.diff(t) > C.MERGE_GAP).sum())

out = dict(by_opening={}, threshold={}, normal={})
for o in C.OPENINGS:
    t = []
    for j in range(C.DASH_RUNS):
        r = rng_for(o, j); _, _, res = run(r, o, C.DASH_T)
        a = first_alarm(res, C.THRESH)
        t.append(a)
    ok = np.array([x for x in t if x is not None], float)
    out['by_opening'][o] = dict(drop_s1=drop_of(o), detected=int(len(ok)), n=C.DASH_RUNS,
                                mean=float(ok.mean()), sd=float(ok.std(ddof=1)), min=int(ok.min()), max=int(ok.max()))
    print(o, out['by_opening'][o], flush=True)
# --- Simulasi tanpa kebocoran: alarm palsu per ambang
normal_res = [run(rng_for(0, j), 0, C.DASH_NORMAL_LEN, leak=False)[2] for j in range(C.DASH_RUNS)]
hours = C.DASH_RUNS * C.DASH_NORMAL_LEN / 3600
for th in C.DASH_THRESHOLDS:
    n_al = [merge_count(np.where(features.threshold_alarm(r, th))[0]) for r in normal_res]
    # --- sensitivitas ambang pada bukaan 10%
    t = [first_alarm(run(rng_for(10, j), 10, C.DASH_T)[2], th) for j in range(C.DASH_RUNS)]
    ok = np.array([x for x in t if x is not None], float)
    out['threshold'][str(th)] = dict(detected=int(len(ok)), n=C.DASH_RUNS,
        mean=float(ok.mean()) if len(ok) else None, sd=float(ok.std(ddof=1)) if len(ok) > 1 else None,
        median=float(np.median(ok)) if len(ok) else None,
        false_alarms=int(sum(n_al)), runs_with_alarm=int(sum(x > 0 for x in n_al)), fa_per_hour=sum(n_al) / hours)
    print('th', th, out['threshold'][str(th)], flush=True)
out['normal'] = dict(runs=C.DASH_RUNS, seconds_each=C.DASH_NORMAL_LEN, hours=hours)

# --- Gambar 2
o_ = list(C.OPENINGS); m_ = [out['by_opening'][o]['mean'] for o in o_]; s_ = [out['by_opening'][o]['sd'] for o in o_]
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
fig, ax = plt.subplots(figsize=(5.5 * 1.3, 3.19 * 1.3), dpi=200)
ax.errorbar(o_, m_, yerr=s_, fmt='o-', color='#1f4e79', capsize=3, lw=1.4)
ax.axhline(C.NCONSEC, color='gray', ls=':', lw=1); ax.text(12, C.NCONSEC - 0.38, 'batas minimum (3 sampel)', fontsize=8, color='gray')
ax.set_xlabel('Bukaan katup uji (%)'); ax.set_ylabel('Waktu deteksi (detik)'); ax.set_xticks(o_); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig('figures/fig2_waktu_deteksi.png'); plt.close(fig)

# --- Gambar 3-4: contoh satu simulasi (bukaan 50%, katup dibuka detik 60-180)
OPEN, T, CLOSE = 50, 300, 180
meas, twin, res = run(rng_for(777), OPEN, T, valve_close=CLOSE)
a = first_alarm(res, C.THRESH); t_alarm = C.DASH_LEAK_AT + a - 1
demo = dict(opening=OPEN, valve_open=C.DASH_LEAK_AT, valve_close=CLOSE, detection_time_s=a, alarm_sample=int(t_alarm), sensors={})
for i, (nm, w, h) in enumerate([('PT-M621', 5.36, 3.65), ('PT-M622', 5.28, 3.54)]):
    fig, ax = plt.subplots(figsize=(w * 1.3, h * 1.3), dpi=200); t = np.arange(T)
    ax.axvspan(C.DASH_LEAK_AT, CLOSE, color='#c8a27a', alpha=.35, label='Katup uji terbuka')
    ax.plot(t, meas[i], color='#2e9e44', lw=1.1, label='Tekanan terukur')
    ax.plot(t, twin[i], color='#7b3fa0', ls='--', lw=1.4, label='Estimasi tekanan normal (twin)')
    ax.axvline(t_alarm, color='red', lw=1.3, label='Anomali terdeteksi')
    ax.set_xlabel('Waktu (detik)'); ax.set_ylabel('Tekanan (bar)'); ax.set_title(f'{nm} (data simulasi)'); ax.grid(alpha=.25)
    ax.legend(fontsize=7.5, loc='lower left'); fig.tight_layout(); fig.savefig(f'figures/fig{3+i}_{nm}.png'); plt.close(fig)
    pre = slice(0, C.DASH_LEAK_AT); post = slice(C.DASH_LEAK_AT, CLOSE)
    demo['sensors'][nm] = dict(p_pre_mean=float(meas[i][pre].mean()), p_pre_max_abs_residual=float(np.abs(res[i][pre]).max()),
        p_min_after_open=float(meas[i][post].min()), twin_at_min=float(twin[i][post][np.argmin(meas[i][post])]),
        residual_min=float(res[i][post].min()), residual_at_alarm=float(res[i][t_alarm]),
        twin_mean_post=float(twin[i][post].mean()))
out['demo'] = demo
print(demo)
json.dump(out, open('results/dashboard.json', 'w'), indent=2)
