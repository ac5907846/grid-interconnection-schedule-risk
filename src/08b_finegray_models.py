"""Fine-Gray subdistribution hazard models via the IPCW time-varying construction."""
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter

from utils import DERIVED, TAB, df_to_md

warnings.filterwarnings('ignore')

d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')
BASE = {'tech': 'Solar', 'region': 'MISO', 'cohort': '2014-17'}
GRID = np.arange(1.0, 15.0, 1.0)
WMIN = 0.08


def fit_finegray(event_of_interest):
    km = KaplanMeierFitter().fit(d['T_years'], (d['event'] == 0).astype(int))
    G = lambda t: float(np.clip(km.predict(t), 1e-6, 1.0))
    rows = []
    for i, (T, E) in zip(d.index, zip(d['T_years'].values, d['event'].values)):
        interest = int(E == event_of_interest)
        rows.append((i, 0.0, T, interest, 1.0))
        if E != 0 and not interest:
            gi = G(T); prev = T
            for t in GRID[GRID > T]:
                w = G(t) / gi
                if w < WMIN:
                    break
                rows.append((i, prev, t, 0, w))
                prev = t
    L = pd.DataFrame(rows, columns=['id', 'start', 'stop', 'ev', 'w'])
    L = L[L['stop'] > L['start']]
    X = pd.get_dummies(d[['log_mw', 'log_backlog', 'dc_market', 'tech', 'region', 'cohort']],
                       columns=['tech', 'region', 'cohort']).astype(float)
    for k, v in BASE.items():
        X = X.drop(columns=f'{k}_{v}', errors='ignore')
    L = L.join(X, on='id')
    m = CoxTimeVaryingFitter(penalizer=0.01)
    m.fit(L, id_col='id', event_col='ev', start_col='start', stop_col='stop',
          weights_col='w', robust=False, show_progress=False)
    print(f'  expanded rows: {len(L):,}')
    return m


out = {}
for ev, name in [(1, 'completion'), (2, 'withdrawal')]:
    print(f'Fitting Fine-Gray for {name} ...')
    m = fit_finegray(ev)
    s = m.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    s.columns = ['coef', 'SHR', 'SHR lower 95%', 'SHR upper 95%', 'p']
    df_to_md(s.round(4), TAB / f'table_finegray_{name}.md', floatfmt='.4f')
    out[name] = s
    print(s.round(3).to_string(), '\n')

print('done')
