"""County-cluster bootstrap and surrogate test for the third structural break."""
import argparse

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from utils import C, DERIVED, TAB, save_figure, set_style, df_to_md

OUT = TAB.parent / 'bootstrap'
OUT.mkdir(exist_ok=True)
set_style()

ap = argparse.ArgumentParser()
ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--perm', type=int, default=2000)
ap.add_argument('--jobs', type=int, default=-1)
args = ap.parse_args()

d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')
d['fips'] = d['fips'].astype(str)
BASE = {'tech': 'Solar', 'region': 'MISO', 'cohort': '2014-17'}
COVS = ['log_mw', 'log_backlog', 'dc_market']
KEEP = ['cohort_2018-20', 'cohort_2021-22', 'tech_Gas', 'region_ERCOT',
        'region_ISO-NE', 'log_mw']


def design(df):
    X = pd.get_dummies(df[['T_years'] + COVS + ['tech', 'region', 'cohort']],
                       columns=['tech', 'region', 'cohort'], drop_first=False)
    for k, v in BASE.items():
        col = f'{k}_{v}'
        if col in X:
            X = X.drop(columns=col)
    return X.astype(float)


def one_draw(seed):
    rng = np.random.default_rng(seed)
    counties = d['fips'].unique()
    take = rng.choice(counties, size=len(counties), replace=True)
    parts = [d[d['fips'] == c] for c in take]
    b = pd.concat(parts, ignore_index=True)
    row = {'seed': seed, 'n': len(b)}
    try:
        for ev, tag in [(1, 'comp'), (2, 'wd')]:
            X = design(b)
            X['E'] = (b['event'] == ev).astype(int).values
            m = CoxPHFitter(penalizer=0.01).fit(X, duration_col='T_years', event_col='E')
            for k in KEEP:
                if k in m.summary.index:
                    row[f'{tag}:{k}'] = float(np.exp(m.summary.loc[k, 'coef']))
        b2 = b.copy()
        Xc = pd.get_dummies(b2[['T_years'] + COVS + ['tech', 'region']],
                            columns=['tech', 'region'], drop_first=False)
        for k, v in [('tech', 'Solar'), ('region', 'MISO')]:
            col = f'{k}_{v}'
            if col in Xc:
                Xc = Xc.drop(columns=col)
        Xc = Xc.astype(float)
        Xc['q_year_c'] = (b2['q_year'] - 2014).values
        Xc['E'] = (b2['event'] == 1).astype(int).values
        mc = CoxPHFitter(penalizer=0.01).fit(Xc, duration_col='T_years', event_col='E')
        row['comp:per_year'] = float(np.exp(mc.summary.loc['q_year_c', 'coef']))
    except Exception as e:
        row['error'] = str(e)[:80]
    return row


boot_path = OUT / 'boot_results.csv'
done = 0
if boot_path.exists():
    done = len(pd.read_csv(boot_path))
print(f'cluster bootstrap: {done} draws done, target {args.boot}')

if done < args.boot:
    from joblib import Parallel, delayed
    todo = list(range(done, args.boot))
    CHUNK = 50
    for i in range(0, len(todo), CHUNK):
        seeds = todo[i:i + CHUNK]
        rows = Parallel(n_jobs=args.jobs, verbose=0)(delayed(one_draw)(s) for s in seeds)
        df = pd.DataFrame(rows)
        df.to_csv(boot_path, mode='a', header=not boot_path.exists(), index=False)
        print(f'  {min(i + CHUNK, len(todo)) + done}/{args.boot} draws written')

res = pd.read_csv(boot_path)
res = res[res.get('error').isna()] if 'error' in res.columns else res
rows = []
for col in res.columns:
    if ':' not in col:
        continue
    v = res[col].dropna()
    if len(v) < 20:
        continue
    rows.append([col, len(v), v.median(),
                 v.quantile(0.025), v.quantile(0.975)])
cis = pd.DataFrame(rows, columns=['Quantity', 'Draws', 'Median HR',
                                  'CI 2.5%', 'CI 97.5%']).set_index('Quantity')
df_to_md(cis.round(3), OUT / 'bootstrap_cis.md', floatfmt='.3f')
print(cis.round(3).to_string())

import ruptures as rpt

comp = d[d['event'] == 1].copy()
comp['cod_q'] = comp['end_date'].dt.to_period('Q')
qs = comp.groupby('cod_q')['T_years'].median()
qs = qs[(qs.index >= pd.Period('2005Q1')) & (qs.index <= pd.Period('2025Q4'))]
y = qs.values


def gain_of_kth_break(series, k=3):
    """Cost reduction achieved by allowing the kth break over k-1 breaks."""
    algo = rpt.Binseg(model='l2').fit(series)
    def total_cost(n_bkps):
        bkps = algo.predict(n_bkps=n_bkps)
        c, start = 0.0, 0
        for b in bkps:
            seg = series[start:b]
            c += ((seg - seg.mean()) ** 2).sum()
            start = b
        return c
    return total_cost(k - 1) - total_cost(k)


obs_gain = gain_of_kth_break(y, 3)
obs_break = rpt.Binseg(model='l2').fit(y).predict(n_bkps=3)[:-1]

bk2 = rpt.Binseg(model='l2').fit(y).predict(n_bkps=2)
mu = np.empty_like(y)
start = 0
for b in bk2:
    mu[start:b] = y[start:b].mean()
    start = b
resid = y - mu
rng = np.random.default_rng(11)
L = 8
n = len(y)
perm_gains, perm_dates = [], []
for _ in range(args.perm):
    idx = rng.integers(0, n - L, size=n // L + 1)
    rb = np.concatenate([resid[i:i + L] for i in idx])[:n]
    yp = mu + rb
    perm_gains.append(gain_of_kth_break(yp, 3))
    perm_dates.append(rpt.Binseg(model='l2').fit(yp).predict(n_bkps=3)[2])
perm_gains = np.array(perm_gains)
pval = float((perm_gains >= obs_gain).mean())

with open(OUT / 'permutation_break.md', 'w') as f:
    f.write('# Test of the third structural break against a two-break null\n\n')
    f.write(f'Observed gain of the third break: {obs_gain:.3f}\n\n')
    f.write(f'Surrogates under the two-break null (moving blocks of {L} quarters): {args.perm}\n\n')
    f.write(f'p-value (share of permutations with gain >= observed): {pval:.4f}\n\n')
    f.write(f'Observed break quarters (indices): {obs_break}; series starts 2005Q1.\n')
print(f'permutation p-value for third break: {pval:.4f}')

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

ax = axes[0]
for col, color, lab in [('comp:cohort_2018-20', C['graph'], '2018-20'),
                        ('comp:cohort_2021-22', C['blue'], '2021-22')]:
    if col in res.columns:
        ax.hist(res[col].dropna(), bins=40, color=color, alpha=0.75, label=lab)
ax.axvline(1.0, color=C['char'], lw=0.9, ls='--')
ax.set_xlabel('Completion hazard ratio vs 2014-17')
ax.set_title('(a) Cluster bootstrap, cohort effects', loc='left')
ax.legend(frameon=False, fontsize=7)
ax.grid(axis='y')

ax = axes[1]
if 'comp:per_year' in res.columns:
    v = res['comp:per_year'].dropna()
    ax.hist(v, bins=40, color=C['green'], alpha=0.8)
    ax.axvline(v.median(), color=C['char'], lw=1.2)
    ax.set_title('(b) Per-year hazard decline', loc='left')
    ax.set_xlabel('HR per additional entry year')
ax.grid(axis='y')

ax = axes[2]
ax.hist(perm_gains, bins=40, color=C['silver'], alpha=0.85)
ax.axvline(obs_gain, color=C['blue'], lw=1.6)
ax.annotate(f'observed\np = {pval:.3f}', xy=(obs_gain, ax.get_ylim()[1] * 0.55),
            fontsize=8, color=C['blue'], ha='right')
ax.set_title('(c) Third-break gain vs permutations', loc='left')
ax.set_xlabel('Segmentation gain')
ax.grid(axis='y')

plt.tight_layout()
save_figure(fig, 'figR_cluster_bootstrap')
plt.close()
print('outputs in', OUT)
