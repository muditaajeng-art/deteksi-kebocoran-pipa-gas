import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '1'
import json, time, platform, itertools, warnings
import numpy as np, sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import config as C, sim, features
warnings.filterwarnings('ignore')
SEED = C.MAIN_SEED

def build(ds, k):
    X, y, g, ends, mask, ep_i = [], [], [], [], [], []
    for i, ep in enumerate(ds):
        r = sim.residual(ep, k); x, e = features.window_features(r)
        lab = ep['label'][e]; m = np.ones(len(e), bool)
        if ep['meta']['leak']:
            m &= ~((e >= ep['meta']['trep']) & (e < ep['meta']['trep'] + C.GRACE))
        X.append(x); y.append(lab); g.append(np.full(len(e), i)); ends.append(e); mask.append(m)
    return [np.concatenate(a) for a in (X, y, g, ends, mask)]

def bw(y):  # bobot kelas seimbang
    n, npos = len(y), y.sum()
    return np.where(y, n / (2 * npos), n / (2 * (n - npos)))

MODELS = {
 'SVM': (lambda **p: SVC(kernel='rbf', class_weight='balanced', random_state=SEED, **p),
         {'C': [1, 10, 100], 'gamma': ['scale', 0.01, 0.1]}, False),
 'RF':  (lambda **p: RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=1, **p),
         {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_leaf': [1, 5]}, False),
 'GB':  (lambda **p: GradientBoostingClassifier(random_state=SEED, **p),
         {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [2, 3]}, True),
 'MLP': (lambda **p: MLPClassifier(solver='adam', activation='relu', learning_rate_init=1e-3, max_iter=500,
                                   early_stopping=True, random_state=SEED, **p),
         {'hidden_layer_sizes': [(32,), (64, 32)], 'alpha': [1e-4, 1e-2]}, True),
}
def pipe(name, **p): return Pipeline([('sc', StandardScaler()), ('clf', MODELS[name][0](**p))])
def fit(name, params, X, y):
    m = pipe(name, **params)
    if MODELS[name][2]: m.fit(X, y, clf__sample_weight=bw(y))
    else: m.fit(X, y)
    return m
def score(m, X):
    return m.decision_function(X) if hasattr(m, 'decision_function') else m.predict_proba(X)[:, 1]

# ---------- 1. Penyetelan hiperparameter pada dataset independen (seed berbeda)
t0 = time.time()
tune = sim.make_dataset(C.TUNE_SEED, C.TUNE_EPISODES, C.TUNE_LEAK, C.TUNE_SMALL)
Xt, yt, gt, _, mt = build(tune, 1); Xt, yt, gt = Xt[mt], yt[mt], gt[mt]
best, tuning = {}, {}
for name, (_, grid, use_sw) in MODELS.items():
    gs = GridSearchCV(pipe(name), {f'clf__{k}': v for k, v in grid.items()}, scoring='f1',
                      cv=GroupKFold(n_splits=C.N_FOLDS), n_jobs=2)
    gs.fit(Xt, yt, groups=gt, **({'clf__sample_weight': bw(yt)} if use_sw else {}))
    best[name] = {k.replace('clf__', ''): (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()}
    tuning[name] = dict(best_cv_f1=float(gs.best_score_), n_configs=len(gs.cv_results_['params']))
    print(name, best[name], round(gs.best_score_, 4), f'{time.time()-t0:.0f}s', flush=True)
for n in best:
    if 'hidden_layer_sizes' in best[n]: best[n]['hidden_layer_sizes'] = tuple(best[n]['hidden_layer_sizes'])

# ---------- 2. Dataset utama + validasi silang berkelompok per episode
main = sim.make_dataset(SEED, C.N_EPISODES, C.N_LEAK, C.N_SMALL)
is_leak = np.array([e['meta']['leak'] for e in main])
rng = np.random.default_rng(C.FOLD_SEED)
fold_of = np.zeros(len(main), int)
for grp in (np.where(is_leak)[0], np.where(~is_leak)[0]):
    p = rng.permutation(grp)
    for j, idx in enumerate(p): fold_of[idx] = j % C.N_FOLDS
data = {k: build(main, k) for k in C.NOISE_LEVELS}
X1, y1, g1, e1, m1 = data[1]
fold_w = fold_of[g1]
names = list(MODELS)
pred = {k: {n: np.zeros(len(y1), bool) for n in names} for k in C.NOISE_LEVELS}
sc = {n: np.zeros(len(y1)) for n in names}
fit_time = {}
for f in range(C.N_FOLDS):
    tr, te = (fold_w != f) & m1, fold_w == f
    for n in names:
        ts = time.time(); m = fit(n, best[n], X1[tr], y1[tr]); fit_time.setdefault(n, []).append(time.time() - ts)
        for k in C.NOISE_LEVELS:
            Xk = data[k][0]; pred[k][n][te] = m.predict(Xk[te])
        sc[n][te] = score(m, X1[te])
    print('fold', f, f'{time.time()-t0:.0f}s', flush=True)

# aturan ambang (tanpa pelatihan)
def base_alarm(k):
    return [features.threshold_alarm(sim.residual(ep, k)) for ep in main]
base = {k: base_alarm(k) for k in C.NOISE_LEVELS}
for k in C.NOISE_LEVELS:
    pred[k]['Ambang'] = np.concatenate([a[e] for a, e in zip(base[k], [np.arange(C.WINDOW - 1, C.EPISODE_LEN, C.STRIDE)] * len(main))])

def window_metrics(p, y, m, s=None):
    d = dict(f1=f1_score(y[m], p[m]), precision=precision_score(y[m], p[m]), recall=recall_score(y[m], p[m]),
             fp_windows=int((p[m] & ~y[m]).sum()), fn_windows=int((~p[m] & y[m]).sum()))
    fl = [f1_score(y[m & (fold_w == f)], p[m & (fold_w == f)]) for f in range(C.N_FOLDS)]
    d['f1_fold_mean'], d['f1_fold_sd'] = float(np.mean(fl)), float(np.std(fl, ddof=1))
    if s is not None:
        d['auc'] = roc_auc_score(y[m], s[m])
        d['auc_fold_mean'] = float(np.mean([roc_auc_score(y[m & (fold_w == f)], s[m & (fold_w == f)]) for f in range(C.N_FOLDS)]))
    return d

def merge_count(times):
    if len(times) == 0: return 0
    t = np.sort(times); return 1 + int((np.diff(t) > C.MERGE_GAP).sum())

def event_metrics(name):
    det, delays, small_det, fa, normal_s = 0, [], 0, 0, 0.0
    for i, ep in enumerate(main):
        mt_ = ep['meta']
        if name == 'Ambang':
            al = base[1][i]; t_all = np.arange(C.EPISODE_LEN); pos = t_all[al]; valid = t_all >= C.WINDOW - 1; step = 1.0
        else:
            sel = g1 == i; pos = e1[sel][pred[1][name][sel]]; t_all = e1[sel]; valid = np.ones(len(t_all), bool); step = C.STRIDE
        lab = ep['label'][t_all]
        grace = np.zeros(len(t_all), bool)
        if mt_['leak']: grace = (t_all >= mt_['trep']) & (t_all < mt_['trep'] + C.GRACE)
        normal = valid & ~lab & ~grace
        normal_s += normal.sum() * step
        if mt_['leak']:
            hit = pos[(pos >= mt_['t0']) & (pos < mt_['trep'])]
            if len(hit):
                det += 1; delays.append(float(hit.min() - mt_['t0'])); small_det += int(mt_['drop'] <= 0.30)
        fp_t = pos[np.isin(pos, t_all[normal])]
        fa += merge_count(fp_t)
    return dict(detected=det, of=C.N_LEAK, small_detected=small_det, small_of=C.N_SMALL,
                false_alarms=fa, normal_hours=normal_s / 3600, fa_per_hour=fa / (normal_s / 3600),
                median_delay=float(np.median(delays)) if delays else None,
                mean_delay=float(np.mean(delays)) if delays else None)

res = dict(best_params={k: {a: (list(b) if isinstance(b, tuple) else b) for a, b in v.items()} for k, v in best.items()},
           tuning=tuning, windows=dict(total=int(len(y1)), evaluated=int(m1.sum()), leak=int((y1 & m1).sum()),
                                       normal=int((~y1 & m1).sum())), models={})
for n in names + ['Ambang']:
    d = window_metrics(pred[1][n], y1, m1, sc.get(n))
    d.update(event_metrics(n))
    d['f1_by_noise'] = {str(k): f1_score(y1[m1], pred[k][n][m1]) for k in C.NOISE_LEVELS}
    res['models'][n] = d
    print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != 'f1_by_noise'}, d['f1_by_noise'], flush=True)

# ---------- 3. Benchmark inferensi (model dilatih pada seluruh data k=1)
bench = {}
Xb = X1[m1]
for n in names:
    m = fit(n, best[n], Xb, y1[m1]); N = 5000
    Xs = Xb[np.random.default_rng(0).integers(0, len(Xb), N)]
    batch = []
    for _ in range(7):
        s = time.perf_counter(); m.predict(Xs); batch.append((time.perf_counter() - s) / N * 1e3)
    single = []
    for j in range(300):
        s = time.perf_counter(); m.predict(Xs[j:j + 1]); single.append((time.perf_counter() - s) * 1e3)
    bench[n] = dict(batch_ms_per_window_median=float(np.median(batch)), single_ms_median=float(np.median(single)),
                    train_s_mean_per_fold=float(np.mean(fit_time[n])))
    print('bench', n, bench[n], flush=True)
res['inference'] = bench
mem = [l for l in open('/proc/meminfo') if l.startswith('MemTotal')][0].split()[1]
cpu = [l for l in open('/proc/cpuinfo') if l.startswith('model name')][0].split(':')[1].strip()
res['environment'] = dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                          os=platform.platform(), cpu=cpu, cores=os.cpu_count(), ram_gb=round(int(mem) / 1e6, 1), threads_per_model=1)
res['dataset'] = dict(n_episodes=C.N_EPISODES, hours=C.N_EPISODES * C.EPISODE_LEN / 3600, leak_episodes=int(is_leak.sum()),
                      leak_drops_small=int(sum(e['meta']['drop'] <= 0.30 for e in main if e['meta']['leak'])),
                      leak_drop_min=float(min(e['meta']['drop'] for e in main if e['meta']['leak'])),
                      leak_drop_max=float(max(e['meta']['drop'] for e in main if e['meta']['leak'])),
                      leak_samples=int(sum(e['label'].sum() for e in main)),
                      total_samples=int(C.N_EPISODES * C.EPISODE_LEN),
                      mean_leak_duration=float(np.mean([e['meta']['trep'] - e['meta']['t0'] for e in main if e['meta']['leak']])))
res['runtime_s'] = time.time() - t0
json.dump(res, open('results/experiment.json', 'w'), indent=2, default=float)
print('selesai', f'{time.time()-t0:.0f}s')
