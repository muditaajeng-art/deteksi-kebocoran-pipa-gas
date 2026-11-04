"""Keterlambatan aturan ambang menurut ukuran kebocoran (mendukung pembahasan Subbab 4.3)."""
import json, numpy as np, sim, features, config as C
ds = sim.make_dataset(C.MAIN_SEED, C.N_EPISODES, C.N_LEAK, C.N_SMALL)
small, large, missed = [], [], []
for ep in ds:
    m = ep['meta']
    if not m['leak']: continue
    hit = np.where(features.threshold_alarm(sim.residual(ep, 1))[m['t0']:m['trep']])[0]
    if len(hit) == 0: missed.append(m['drop']); continue
    (small if m['drop'] <= 0.30 else large).append(int(hit[0]))
out = dict(small=dict(n=len(small), mean=float(np.mean(small)), median=float(np.median(small))),
           large=dict(n=len(large), mean=float(np.mean(large)), median=float(np.median(large))), missed_drops=missed)
json.dump(out, open('results/baseline_delay.json', 'w'), indent=2); print(out)
